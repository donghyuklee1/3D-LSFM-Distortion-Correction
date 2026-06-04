"""Evaluate LSFM bead distortion-correction methods.

Outputs:
  - metrics.csv: per-sample metrics for each method
  - summary.md : mean ± std table suitable for a paper draft

Example:
    python scripts/evaluate_restoration.py --cfg configs/lsfm_beads.yaml \
        --cache cached/lsfm_beads --out runs/lsfm_beads/eval \
        --checkpoints ours:runs/lsfm_beads/skip_ae3d/ckpt_epoch039.pt
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import CachedAOStackDataset, LSFMDistortionBeadsDataset  # noqa: E402
from models.restoration import build_restoration_model  # noqa: E402


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.mse_loss(pred, target).item()


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    e = F.mse_loss(pred, target).clamp_min(1e-12)
    return float(20.0 * math.log10(data_range) - 10.0 * torch.log10(e).item())


def nrmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    rmse = torch.sqrt(F.mse_loss(pred, target))
    denom = (target.amax() - target.amin()).clamp_min(1e-8)
    return (rmse / denom).item()


def ssim_global(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Global SSIM over a volume; deterministic and dependency-free."""
    x = pred.float().reshape(-1)
    y = target.float().reshape(-1)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mux = x.mean()
    muy = y.mean()
    vx = x.var(unbiased=False)
    vy = y.var(unbiased=False)
    cov = ((x - mux) * (y - muy)).mean()
    val = ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux**2 + muy**2 + c1) * (vx + vy + c2))
    return val.clamp(-1.0, 1.0).item()


def axial_correct_global(x: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """Classical single-factor z-compression baseline.

    This is the correction-factor baseline discussed in the paper's related
    work: apply one global axial factor to the whole stack. It cannot handle
    non-uniform bead-wise distortion but provides an important non-DL baseline.
    """
    b, c, z, y, xw = x.shape
    device = x.device
    zz = torch.linspace(-1, 1, z, device=device)
    yy = torch.linspace(-1, 1, y, device=device)
    xx = torch.linspace(-1, 1, xw, device=device)
    Z, Y, X = torch.meshgrid(zz, yy, xx, indexing="ij")
    grids = []
    for i in range(b):
        f = factor[i].clamp_min(1.0)
        Zi = (Z * f).clamp(-1.0, 1.0)
        grids.append(torch.stack([X, Y, Zi], dim=-1))
    grid = torch.stack(grids, dim=0)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def bead_axial_ratio_error(vol: torch.Tensor, centers: torch.Tensor, radii: torch.Tensor) -> float:
    """Mean |std_z / mean(std_y,std_x) - 1| around known bead centers."""
    v = vol.detach().float()
    if v.dim() == 5:
        v = v[0, 0]
    elif v.dim() == 4:
        v = v[0]
    Nz, Ny, Nx = v.shape
    errors = []
    for center, radius in zip(centers, radii):
        cz, cy, cx = [float(a) for a in center]
        r = max(2.0, float(radius) * 5.0)
        z0, z1 = max(0, int(cz - r)), min(Nz, int(cz + r + 1))
        y0, y1 = max(0, int(cy - r)), min(Ny, int(cy + r + 1))
        x0, x1 = max(0, int(cx - r)), min(Nx, int(cx + r + 1))
        crop = v[z0:z1, y0:y1, x0:x1].clamp_min(0)
        if crop.sum() <= 1e-8:
            continue
        zz, yy, xx = torch.meshgrid(
            torch.arange(z0, z1, dtype=torch.float32),
            torch.arange(y0, y1, dtype=torch.float32),
            torch.arange(x0, x1, dtype=torch.float32),
            indexing="ij",
        )
        w = crop / crop.sum().clamp_min(1e-8)
        mz, my, mx = (w * zz).sum(), (w * yy).sum(), (w * xx).sum()
        sz = torch.sqrt((w * (zz - mz) ** 2).sum().clamp_min(1e-8))
        sy = torch.sqrt((w * (yy - my) ** 2).sum().clamp_min(1e-8))
        sx = torch.sqrt((w * (xx - mx) ** 2).sum().clamp_min(1e-8))
        ratio = sz / ((sy + sx) * 0.5).clamp_min(1e-8)
        errors.append(abs(float(ratio.item()) - 1.0))
    return float(sum(errors) / len(errors)) if errors else float("nan")


def _load_dataset(cfg: dict, cache: str | None, n: int | None):
    if cache:
        ds = CachedAOStackDataset(cache)
        if n is not None:
            ds.files = ds.files[:n]
        return ds
    else:
        return LSFMDistortionBeadsDataset(
            cfg,
            length=n or 128,
            seed=cfg.get("train", {}).get("seed", 0),
        )


def _parse_checkpoints(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        if ":" not in item:
            raise ValueError("--checkpoints entries must be name:path")
        name, path = item.split(":", 1)
        out[name] = path
    return out


def _parse_labels(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        if ":" not in item:
            raise ValueError("--method-labels entries must be key:Label")
        key, label = item.split(":", 1)
        out[key] = label
    return out


PAPER_METRIC_INFO = {
    "psnr": {"header": "PSNR (dB)", "direction": "up", "decimals": 2},
    "ssim": {"header": "SSIM", "direction": "up", "decimals": 4},
    "axial_ratio_error": {"header": "Axial ratio err.", "direction": "down", "decimals": 4},
    "mse": {"header": "MSE", "direction": "down", "decimals": 6},
    "nrmse": {"header": "NRMSE", "direction": "down", "decimals": 4},
}


def _format_metric(mean: float, std: float, metric: str) -> str:
    decimals = PAPER_METRIC_INFO.get(metric, {}).get("decimals", 4)
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def _write_summary(
    summary: dict[str, list[dict]],
    out_dir: Path,
    metric_names: list[str],
    method_order: list[str] | None,
    labels: dict[str, str],
    paper_table: bool,
) -> None:
    ordered_methods = method_order or list(summary.keys())
    for key in summary:
        if key not in ordered_methods:
            ordered_methods.append(key)

    lines = [
        "# LSFM Bead Distortion-Correction Evaluation",
        "",
    ]
    if paper_table:
        headers = [PAPER_METRIC_INFO.get(m, {}).get("header", m) for m in metric_names]
        arrow = lambda m: "↑" if PAPER_METRIC_INFO.get(m, {}).get("direction") == "up" else "↓"
        lines.append("| Method | " + " | ".join(f"{h} {arrow(m)}" for h, m in zip(headers, metric_names)) + " |")
    else:
        headers = [m.upper() for m in metric_names]
        lines.append("| Method | " + " | ".join(headers) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(metric_names)) + "|")

    table_rows: list[tuple[str, list[str]]] = []
    for method in ordered_methods:
        vals = summary.get(method)
        if not vals:
            continue
        parts = []
        for metric in metric_names:
            t = torch.tensor([float(v[metric]) for v in vals if math.isfinite(float(v[metric]))])
            mean = t.mean().item()
            std = t.std(unbiased=False).item()
            parts.append(_format_metric(mean, std, metric))
        label = labels.get(method, method)
        lines.append(f"| {label} | " + " | ".join(parts) + " |")
        table_rows.append((label, parts))

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    if paper_table:
        tex_lines = [
            "% Auto-generated by scripts/evaluate_restoration.py",
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Quantitative comparison on the synthetic LSFM bead dataset "
            f"({len(summary[ordered_methods[0]]) if ordered_methods and ordered_methods[0] in summary else '?'} volumes). "
            "Mean $\\pm$ std over all test samples.}",
            "\\label{tab:restoration_comparison}",
            "\\begin{tabular}{l" + "c" * len(metric_names) + "}",
            "\\toprule",
            "Method & " + " & ".join(
                PAPER_METRIC_INFO.get(m, {}).get("header", m) for m in metric_names
            ) + " \\\\",
            "\\midrule",
        ]
        for label, parts in table_rows:
            tex_lines.append(label.replace("_", "\\_") + " & " + " & ".join(parts) + " \\\\")
        tex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])
        (out_dir / "summary.tex").write_text("\n".join(tex_lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="configs/lsfm_beads.yaml")
    p.add_argument("--cache", default=None)
    p.add_argument("--out", default="runs/lsfm_beads/eval")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--checkpoints", nargs="*", default=[],
                   help="trained models as name:path")
    p.add_argument("--classical", nargs="*", default=["identity", "axial_factor"],
                   help="non-learned baselines to include")
    p.add_argument("--metrics", nargs="*", default=None,
                   help="metrics for summary table (default: all)")
    p.add_argument("--method-order", nargs="*", default=None,
                   help="row order in summary outputs")
    p.add_argument("--method-labels", nargs="*", default=[],
                   help="display labels as key:Label")
    p.add_argument("--paper-table", action="store_true",
                   help="write paper-focused summary.md and summary.tex")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    device = _device()
    ds = _load_dataset(cfg, args.cache or cfg.get("train", {}).get("cache_dir"), args.max_samples)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods: dict[str, object] = {}
    for classical in args.classical:
        if classical == "identity":
            methods["identity"] = None
        elif classical == "axial_factor":
            methods["axial_factor"] = "axial_factor"
        else:
            raise ValueError(f"unknown classical baseline: {classical}")
    for name, path in _parse_checkpoints(args.checkpoints).items():
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model_name = ckpt.get("model_name", name)
        ckpt_cfg = ckpt.get("cfg", cfg)
        model_cfg = ckpt_cfg.get("model", {})
        bead_cfg = ckpt_cfg.get("bead_dataset", cfg.get("bead_dataset", {}))
        model_kwargs = {k: v for k, v in model_cfg.items() if k not in {"name", "base_channels"}}
        model = build_restoration_model(
            model_name,
            base_channels=int(model_cfg.get("base_channels", 16)),
            Nz=int(bead_cfg.get("Nz", 64)),
            Ny=int(bead_cfg.get("Ny", 64)),
            Nx=int(bead_cfg.get("Nx", 64)),
            **model_kwargs,
        )
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()
        methods[name] = model

    rows = []
    max_samples = args.max_samples or len(ds)
    for idx in tqdm(range(min(max_samples, len(ds))), desc="evaluating"):
        sample = ds[idx]
        x = sample["stack_distorted"].unsqueeze(0).to(device)
        y = sample["stack_corrected"].unsqueeze(0).to(device)
        factor = sample["distortion_factor"].view(1).to(device)
        centers = sample["bead_centers"]
        radii = sample["bead_radii"]

        for name, method in methods.items():
            with torch.no_grad():
                if method is None:
                    pred = x
                elif method == "axial_factor":
                    pred = axial_correct_global(x, factor)
                else:
                    pred = method(x)
            rows.append({
                "sample": idx,
                "method": name,
                "mse": mse(pred, y),
                "psnr": psnr(pred, y),
                "ssim": ssim_global(pred, y),
                "nrmse": nrmse(pred, y),
                "axial_ratio_error": bead_axial_ratio_error(pred.cpu(), centers, radii),
                "distortion_factor": float(factor.item()),
            })

    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for row in rows:
        summary.setdefault(row["method"], []).append(row)
    metric_names = args.metrics or ["psnr", "ssim", "mse", "nrmse", "axial_ratio_error"]
    labels = _parse_labels(args.method_labels)
    _write_summary(
        summary,
        out_dir,
        metric_names,
        args.method_order,
        labels,
        args.paper_table,
    )
    extra = f", {out_dir / 'summary.tex'}" if args.paper_table else ""
    print(f"[ok] wrote {csv_path}, {out_dir / 'summary.md'}{extra}")


if __name__ == "__main__":
    main()
