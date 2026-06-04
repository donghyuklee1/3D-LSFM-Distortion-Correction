"""Train LSFM bead distortion-correction models.

Example:
    python scripts/prepare_dataset.py --cfg configs/lsfm_beads.yaml \\
        --out cached/lsfm_beads --seed 0
    python train.py --cfg configs/train_vit_inr.yaml \\
        --out runs/lsfm_beads/vit_multihead_inr
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from torch.utils.data import Subset

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from data import CachedAOStackDataset, LSFMDistortionBeadsDataset
from losses.lsfm_bead_physics import (
    LSFMBeadPhysicsPrior,
    background_sparsity_loss,
    bead_spherical_prior_loss,
    foreground_anisotropic_tv_loss,
    gradient_match_loss,
)
from models.restoration import build_restoration_model
from utils import viz


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _collate_train(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    factors = []
    max_beads = max(b["bead_centers"].shape[0] for b in batch)
    centers = torch.zeros(len(batch), max_beads, 3, dtype=torch.float32)
    radii = torch.zeros(len(batch), max_beads, dtype=torch.float32)
    bead_valid = torch.zeros(len(batch), max_beads, dtype=torch.bool)
    per_bead_factors = torch.zeros(len(batch), max_beads, dtype=torch.float32)
    for i, b in enumerate(batch):
        n = b["bead_centers"].shape[0]
        centers[i, :n] = b["bead_centers"]
        radii[i, :n] = b["bead_radii"]
        bead_valid[i, :n] = True
        f = b["distortion_factor"]
        per_bead = b.get("distortion_factors_per_bead")
        if per_bead is not None:
            per_bead_factors[i, :n] = per_bead
            f = torch.maximum(f, per_bead.max())
        factors.append(f if f.ndim else f.reshape(()))
    return {
        "stack_distorted": torch.stack([b["stack_distorted"] for b in batch], dim=0),
        "stack_corrected": torch.stack([b["stack_corrected"] for b in batch], dim=0),
        "distortion_factor": torch.stack(factors, dim=0),
        "bead_centers": centers,
        "bead_radii": radii,
        "bead_valid": bead_valid,
        "distortion_factors_per_bead": per_bead_factors,
    }


def _sample_distortion_factor(sample: dict) -> float:
    f = float(sample["distortion_factor"])
    per_bead = sample.get("distortion_factors_per_bead")
    if per_bead is not None:
        f = max(f, float(per_bead.max()))
    return f


def _select_hard_indices(dataset: CachedAOStackDataset, min_factor: float) -> list[int]:
    indices: list[int] = []
    for i in range(len(dataset)):
        if _sample_distortion_factor(dataset[i]) >= min_factor:
            indices.append(i)
    return indices


def _distortion_sample_weight(
    factor: torch.Tensor,
    *,
    min_factor: float = 1.8,
    max_factor: float = 10.0,
    alpha: float = 0.0,
) -> torch.Tensor:
    if alpha <= 0:
        return torch.ones_like(factor)
    norm = (factor - min_factor) / max(max_factor - min_factor, 1e-6)
    return 1.0 + alpha * norm.clamp(0.0, 1.0)


def _linear_ramp(epoch: int, start_epoch: int, ramp_epochs: int, target: float) -> float:
    """Linearly ramp a loss weight from 0 -> target over the first ramp_epochs."""
    if target <= 0 or ramp_epochs <= 0:
        return target
    progress = (epoch - start_epoch + 1) / float(ramp_epochs)
    return target * min(1.0, max(0.0, progress))


def _foreground_weights(target: torch.Tensor, gamma: float = 8.0) -> torch.Tensor:
    """Per-voxel weights for sparse bead volumes.

    The target is mostly black background. Plain MSE therefore reports a small
    loss even when beads are missed, and its per-batch value changes strongly
    with bead count. A smooth intensity-based foreground weight gives bead
    voxels stable influence while keeping background suppression active.
    """
    fg = (target / target.amax(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-8)).clamp(0.0, 1.0)
    return 1.0 + gamma * fg


def _projection_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match max-intensity projections along z/y/x to stabilise 3D shape."""
    loss = 0.0
    for dim in (-3, -2, -1):
        loss = loss + F.smooth_l1_loss(pred.amax(dim=dim), target.amax(dim=dim))
    return loss / 3.0


def _soft_dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Foreground overlap loss for sparse bead volumes."""
    p = pred.clamp(0.0, 1.0)
    t = (target / target.amax(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-8)).clamp(0.0, 1.0)
    dims = tuple(range(1, p.ndim))
    inter = (p * t).sum(dim=dims)
    denom = p.square().sum(dim=dims) + t.square().sum(dim=dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def _wmse_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    foreground_gamma: float,
) -> torch.Tensor:
    w = _foreground_weights(target, gamma=foreground_gamma)
    diff2 = (pred - target).pow(2)
    num = (diff2 * w).flatten(1).sum(dim=1)
    den = w.flatten(1).sum(dim=1).clamp_min(1e-8)
    return num / den


def _axial_compactness_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize excess z-axis spread relative to the compact bead target."""
    p_z = pred.sum(dim=(-2, -1))
    t_z = target.sum(dim=(-2, -1))
    p_prob = p_z / p_z.sum(dim=-1, keepdim=True).clamp_min(eps)
    t_prob = t_z / t_z.sum(dim=-1, keepdim=True).clamp_min(eps)
    z = torch.linspace(0.0, 1.0, pred.shape[-3], device=pred.device).view(1, 1, -1)
    p_mean = (p_prob * z).sum(dim=-1)
    t_mean = (t_prob * z).sum(dim=-1)
    p_var = (p_prob * (z - p_mean.unsqueeze(-1)).pow(2)).sum(dim=-1)
    t_var = (t_prob * (z - t_mean.unsqueeze(-1)).pow(2)).sum(dim=-1)
    spread_penalty = F.relu(p_var - t_var).mean()
    moment_match = F.smooth_l1_loss(p_var, t_var)
    return spread_penalty + moment_match


def _weighted_mean(values: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    if weight is None:
        return values.mean()
    w = weight.to(values.device)
    return (values * w).sum() / w.sum().clamp_min(1e-8)


def restoration_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    foreground_gamma: float = 8.0,
    lambda_l1: float = 0.25,
    lambda_proj: float = 0.50,
    lambda_dice: float = 0.0,
    lambda_axial: float = 0.0,
    sample_weight: torch.Tensor | None = None,
    *,
    distorted: torch.Tensor | None = None,
    distortion_factor: torch.Tensor | None = None,
    bead_centers: torch.Tensor | None = None,
    bead_radii: torch.Tensor | None = None,
    bead_valid: torch.Tensor | None = None,
    physics: LSFMBeadPhysicsPrior | None = None,
    lambda_physics: float = 0.0,
    lambda_spherical: float = 0.0,
    lambda_sparsity: float = 0.0,
    lambda_tv: float = 0.0,
    lambda_grad_match: float = 0.0,
    tv_wz: float = 3.0,
    tv_wy: float = 1.0,
    tv_wx: float = 1.0,
    tv_huber_delta: float = 0.025,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    wmse_ps = _wmse_per_sample(pred, target, foreground_gamma)
    loss_wmse = _weighted_mean(wmse_ps, sample_weight)
    l1_ps = F.smooth_l1_loss(pred, target, reduction="none").flatten(1).mean(dim=1)
    loss_l1 = _weighted_mean(l1_ps, sample_weight)
    loss_proj = _projection_loss(pred, target)
    loss_dice = _soft_dice_loss(pred, target)
    loss = loss_wmse + lambda_l1 * loss_l1 + lambda_proj * loss_proj + lambda_dice * loss_dice
    parts = {
        "wmse": loss_wmse.detach(),
        "smooth_l1": loss_l1.detach(),
        "projection": loss_proj.detach(),
        "dice": loss_dice.detach(),
        "plain_mse": F.mse_loss(pred.detach(), target.detach()),
    }
    if lambda_axial > 0:
        loss_axial = _axial_compactness_loss(pred, target)
        loss = loss + lambda_axial * loss_axial
        parts["axial"] = loss_axial.detach()
    if lambda_physics > 0 and physics is not None and distorted is not None and distortion_factor is not None:
        loss_phys = physics.consistency_loss(pred, distorted, distortion_factor)
        loss = loss + lambda_physics * loss_phys
        parts["physics"] = loss_phys.detach()
    if lambda_spherical > 0 and bead_centers is not None and bead_radii is not None and bead_valid is not None:
        loss_sph = bead_spherical_prior_loss(pred, bead_centers, bead_radii, bead_valid)
        loss = loss + lambda_spherical * loss_sph
        parts["spherical"] = loss_sph.detach()
    if lambda_sparsity > 0:
        loss_sparse = background_sparsity_loss(pred, target, gamma=foreground_gamma)
        loss = loss + lambda_sparsity * loss_sparse
        parts["sparsity"] = loss_sparse.detach()
    if lambda_tv > 0:
        loss_tv = foreground_anisotropic_tv_loss(
            pred,
            target,
            wz=tv_wz,
            wy=tv_wy,
            wx=tv_wx,
            huber_delta=tv_huber_delta,
            gamma=foreground_gamma,
        )
        loss = loss + lambda_tv * loss_tv
        parts["tv"] = loss_tv.detach()
    if lambda_grad_match > 0:
        loss_grad = gradient_match_loss(pred, target, gamma=foreground_gamma)
        loss = loss + lambda_grad_match * loss_grad
        parts["grad_match"] = loss_grad.detach()
    return loss, parts


def _build_dataset(cfg: dict):
    t = cfg.get("train", {})
    cache_dir = t.get("cache_dir")
    if cache_dir:
        print(f"[data] cached dataset: {cache_dir}")
        ds = CachedAOStackDataset(cache_dir)
        min_factor = float(t.get("min_distortion_factor", 0.0))
        if min_factor > 0:
            hard_idx = _select_hard_indices(ds, min_factor)
            if not hard_idx:
                raise RuntimeError(
                    f"no samples with distortion >= {min_factor} under {cache_dir}"
                )
            print(f"[data] hard-distortion subset: {len(hard_idx)} / {len(ds)} samples")
            ds = Subset(ds, hard_idx)
        return ds
    length = int(t.get("online_length", 512))
    seed = int(t.get("seed", 0))
    print(f"[data] online LSFMDistortionBeadsDataset: {length}")
    return LSFMDistortionBeadsDataset(cfg, length=length, seed=seed)


def train(
    cfg_path: str,
    model_name: str | None = None,
    out_dir: str | None = None,
    resume: str | None = None,
):
    cfg = yaml.safe_load(open(cfg_path))
    t = cfg.get("train", {})
    m = cfg.get("model", {})
    log_cfg = cfg.get("logging", {})
    device = _device()
    print(f"[device] {device}")

    dataset = _build_dataset(cfg)
    max_train_samples = t.get("max_train_samples")
    if max_train_samples is None:
        max_train_samples = cfg.get("dataset", {}).get("n_samples")
    if max_train_samples is not None:
        n = min(int(max_train_samples), len(dataset))
        dataset = Subset(dataset, list(range(n)))
        print(f"[data] using subset: {n} samples")
    loader = DataLoader(
        dataset,
        batch_size=int(t.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(t.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        drop_last=True,
        collate_fn=_collate_train,
    )

    name = model_name or m.get("name", "skip_autoencoder_3d")
    bd = cfg.get("bead_dataset", {})
    model_kwargs = {k: v for k, v in m.items() if k not in {"name", "base_channels"}}
    model = build_restoration_model(
        name,
        base_channels=int(m.get("base_channels", 16)),
        Nz=int(bd.get("Nz", 64)),
        Ny=int(bd.get("Ny", 64)),
        Nx=int(bd.get("Nx", 64)),
        **model_kwargs,
    ).to(device)
    print(f"[model] {name}: {sum(p.numel() for p in model.parameters())/1e6:.2f} M params")

    resume_path = resume or t.get("resume")
    start_epoch = 0
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        print(f"[resume] {resume_path} -> start_epoch={start_epoch}")

    run_name = t.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(out_dir) if out_dir else Path(log_cfg.get("log_dir", "runs/lsfm_beads")) / name / run_name
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "config.yaml", "w") as f:
        yaml.safe_dump({**cfg, "model": {**m, "name": name}}, f)
    writer = SummaryWriter(out_root.as_posix())
    print(f"[run] log dir = {out_root}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(t.get("lr", 1e-3)),
        weight_decay=float(t.get("weight_decay", 1e-5)),
    )
    if resume_path and t.get("resume_optimizer", False):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
            print("[resume] optimizer state restored")

    epochs = int(t.get("epochs", 40))
    peak_lr = float(t.get("lr", 1e-3))
    lr_schedule = str(t.get("lr_schedule", "constant")).lower()
    lr_min = float(t.get("lr_min", peak_lr * 0.01))
    lr_warmup_epochs = max(0, int(t.get("lr_warmup_epochs", 0)))
    scheduler = None
    if lr_schedule == "cosine":
        cosine_epochs = max(epochs - lr_warmup_epochs, 1)
        if lr_warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                opt,
                start_factor=max(lr_min / peak_lr, 1e-3),
                end_factor=1.0,
                total_iters=lr_warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=cosine_epochs, eta_min=lr_min
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warmup, cosine], milestones=[lr_warmup_epochs]
            )
            print(
                f"[lr] warmup {lr_warmup_epochs} epochs, then cosine "
                f"{peak_lr:.2e} -> {lr_min:.2e} over {cosine_epochs} epochs"
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=epochs, eta_min=lr_min
            )
            print(f"[lr] cosine decay: peak={peak_lr:.2e} -> min={lr_min:.2e} over {epochs} epochs")
    scalar_every = int(log_cfg.get("scalar_every", 10))
    image_every = int(log_cfg.get("image_every", 100))
    foreground_gamma = float(t.get("foreground_gamma", 8.0))
    lambda_l1 = float(t.get("lambda_l1", 0.25))
    lambda_proj = float(t.get("lambda_proj", 0.50))
    lambda_dice = float(t.get("lambda_dice", 0.0))
    lambda_axial = float(t.get("lambda_axial", 0.0))
    lambda_physics = float(t.get("lambda_physics", 0.0))
    lambda_spherical = float(t.get("lambda_spherical", 0.0))
    lambda_sparsity = float(t.get("lambda_sparsity", 0.0))
    lambda_tv = float(t.get("lambda_tv", 0.0))
    lambda_grad_match = float(t.get("lambda_grad_match", 0.0))
    prior_ramp_epochs = int(t.get("prior_ramp_epochs", t.get("physics_ramp_epochs", 0)))
    grad_clip = float(t.get("grad_clip", 1.0))
    save_every = max(1, int(t.get("save_every", 1)))
    tv_wz = float(t.get("tv_wz", 3.0))
    tv_wy = float(t.get("tv_wy", 1.0))
    tv_wx = float(t.get("tv_wx", 1.0))
    tv_huber_delta = float(t.get("tv_huber_delta", 0.025))
    physics_prior = LSFMBeadPhysicsPrior.from_mapping(cfg) if (
        lambda_physics > 0 or lambda_spherical > 0 or lambda_sparsity > 0
    ) else None
    if (
        physics_prior is not None
        or lambda_tv > 0
        or lambda_grad_match > 0
    ):
        print(
            "[physics] priors: "
            f"physics={lambda_physics}, spherical={lambda_spherical}, "
            f"sparsity={lambda_sparsity}, tv={lambda_tv}, grad_match={lambda_grad_match}, "
            f"prior_ramp_epochs={prior_ramp_epochs}"
        )
    distortion_alpha = float(t.get("distortion_loss_alpha", 0.0))
    distortion_min = float(t.get("distortion_min_weight", 1.8))
    distortion_max = float(t.get("distortion_max_weight", 10.0))

    step = 0
    ema_loss: float | None = None
    ema_psnr: float | None = None
    for epoch in range(start_epoch, start_epoch + epochs):
        ramp_scale = _linear_ramp(epoch, start_epoch, prior_ramp_epochs, 1.0)
        eff_lambda_physics = lambda_physics * ramp_scale
        eff_lambda_spherical = lambda_spherical * ramp_scale
        eff_lambda_sparsity = lambda_sparsity * ramp_scale
        eff_lambda_tv = lambda_tv * ramp_scale
        eff_lambda_grad_match = lambda_grad_match * ramp_scale
        eff_lambda_axial = lambda_axial * ramp_scale
        if prior_ramp_epochs > 0 and epoch == start_epoch:
            print(f"[ramp] prior weights scale={ramp_scale:.3f} (epoch {epoch})")
        pbar = tqdm(loader, desc=f"epoch {epoch:04d} [{name}]")
        for batch in pbar:
            x = batch["stack_distorted"].to(device, non_blocking=True)
            y = batch["stack_corrected"].to(device, non_blocking=True)

            pred = model(x)
            sample_weight = _distortion_sample_weight(
                batch["distortion_factor"],
                min_factor=distortion_min,
                max_factor=distortion_max,
                alpha=distortion_alpha,
            )
            loss, loss_parts = restoration_loss(
                pred,
                y,
                foreground_gamma=foreground_gamma,
                lambda_l1=lambda_l1,
                lambda_proj=lambda_proj,
                lambda_dice=lambda_dice,
                lambda_axial=eff_lambda_axial,
                sample_weight=sample_weight,
                distorted=x,
                distortion_factor=batch["distortion_factor"].to(device),
                bead_centers=batch["bead_centers"].to(device),
                bead_radii=batch["bead_radii"].to(device),
                bead_valid=batch["bead_valid"].to(device),
                physics=physics_prior,
                lambda_physics=eff_lambda_physics,
                lambda_spherical=eff_lambda_spherical,
                lambda_sparsity=eff_lambda_sparsity,
                lambda_tv=eff_lambda_tv,
                lambda_grad_match=eff_lambda_grad_match,
                tv_wz=tv_wz,
                tv_wy=tv_wy,
                tv_wx=tv_wx,
                tv_huber_delta=tv_huber_delta,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            with torch.no_grad():
                loss_mse = loss_parts["plain_mse"]
                psnr = -10.0 * torch.log10(loss_mse.clamp_min(1e-10))
                ema_loss = loss.item() if ema_loss is None else 0.97 * ema_loss + 0.03 * loss.item()
                ema_psnr = psnr.item() if ema_psnr is None else 0.97 * ema_psnr + 0.03 * psnr.item()
            if step % scalar_every == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/loss_ema", ema_loss, step)
                writer.add_scalar("train/lr", opt.param_groups[0]["lr"], step)
                if prior_ramp_epochs > 0:
                    writer.add_scalar("train/prior_ramp_scale", ramp_scale, step)
                writer.add_scalar("train/mse", loss_mse.item(), step)
                writer.add_scalar("train/weighted_mse", loss_parts["wmse"].item(), step)
                writer.add_scalar("train/smooth_l1", loss_parts["smooth_l1"].item(), step)
                writer.add_scalar("train/projection", loss_parts["projection"].item(), step)
                writer.add_scalar("train/dice", loss_parts["dice"].item(), step)
                if "axial" in loss_parts:
                    writer.add_scalar("train/axial", loss_parts["axial"].item(), step)
                if "physics" in loss_parts:
                    writer.add_scalar("train/physics", loss_parts["physics"].item(), step)
                if "spherical" in loss_parts:
                    writer.add_scalar("train/spherical", loss_parts["spherical"].item(), step)
                if "sparsity" in loss_parts:
                    writer.add_scalar("train/sparsity", loss_parts["sparsity"].item(), step)
                if "tv" in loss_parts:
                    writer.add_scalar("train/tv", loss_parts["tv"].item(), step)
                if "grad_match" in loss_parts:
                    writer.add_scalar("train/grad_match", loss_parts["grad_match"].item(), step)
                writer.add_scalar("train/psnr_db", psnr.item(), step)
                writer.add_scalar("train/psnr_db_ema", ema_psnr, step)
            if step % image_every == 0:
                # TensorBoard is 2D-only, so expose the 3D stack through
                # slice montages, orthogonal planes, and maximum projections.
                viz.log_scalar_volume_views(writer, "img/input_distorted_3d", x[0], step)
                viz.log_scalar_volume_views(writer, "img/pred_corrected_3d", pred[0], step)
                viz.log_scalar_volume_views(writer, "img/target_corrected_3d", y[0], step)

            pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr.item():.2f}")
            step += 1

        if scheduler is not None:
            scheduler.step()

        is_last_epoch = epoch == start_epoch + epochs - 1
        if (epoch + 1) % save_every == 0 or is_last_epoch:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "cfg": cfg,
                    "model_name": name,
                    "epoch": epoch,
                    "step": step,
                },
                out_root / f"ckpt_epoch{epoch:05d}.pt",
            )

    writer.close()
    print(f"[ok] finished -> {out_root}")
    return out_root


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="configs/lsfm_beads.yaml")
    p.add_argument("--model", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--resume", default=None, help="checkpoint path to resume fine-tuning")
    args = p.parse_args()
    train(args.cfg, model_name=args.model, out_dir=args.out, resume=args.resume)
