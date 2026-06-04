"""Differentiable LSFM bead imaging physics for restoration priors.

The synthetic bead dataset degrades corrected spherical beads by:
    corrected -> axial z-stretch -> light-sheet profile -> Gaussian blur -> noise

This module exposes the deterministic part of that chain as a physics prior so
restoration training can penalize predictions that violate the forward model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from data.lsfm_beads_dataset import BeadDistortionConfig, _cfg_from_mapping, _separable_gaussian_blur3d


def _meshgrid_zyx(Nz: int, Ny: int, Nx: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = torch.arange(Nz, dtype=torch.float32, device=device)
    y = torch.arange(Ny, dtype=torch.float32, device=device)
    x = torch.arange(Nx, dtype=torch.float32, device=device)
    return torch.meshgrid(z, y, x, indexing="ij")


def light_profile_zyx(cfg: BeadDistortionConfig, device: torch.device) -> torch.Tensor:
    zz, yy, _ = _meshgrid_zyx(cfg.Nz, cfg.Ny, cfg.Nx, device)
    cy = (cfg.Ny - 1) / 2.0
    cz = (cfg.Nz - 1) / 2.0
    sig_y = max(1.0, cfg.light_sigma_y_frac * cfg.Ny)
    sig_z = max(1.0, cfg.light_sigma_z_frac * cfg.Nz)
    profile = torch.exp(-((yy - cy) ** 2) / (2.0 * sig_y**2)) * torch.exp(
        -((zz - cz) ** 2) / (2.0 * sig_z**2)
    )
    return 0.35 + 0.65 * profile


def normalise01_per_volume(x: torch.Tensor) -> torch.Tensor:
    """Normalize each (B,1,Z,Y,X) volume to [0, 1]."""
    if x.dim() != 5:
        raise ValueError(f"expected (B,1,Z,Y,X), got {tuple(x.shape)}")
    b = x.shape[0]
    flat = x.reshape(b, -1)
    lo = flat.min(dim=1, keepdim=True).values.view(b, 1, 1, 1, 1)
    hi = flat.max(dim=1, keepdim=True).values.view(b, 1, 1, 1, 1)
    return ((x - lo) / (hi - lo).clamp_min(1e-8)).clamp_min(0.0)


def axial_stretch_3d(vol: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """Elongate a compact bead volume along z (inverse of z-compression correction)."""
    b, c, z, y, xw = vol.shape
    device = vol.device
    zz = torch.linspace(-1, 1, z, device=device)
    yy = torch.linspace(-1, 1, y, device=device)
    xx = torch.linspace(-1, 1, xw, device=device)
    z_grid, y_grid, x_grid = torch.meshgrid(zz, yy, xx, indexing="ij")
    grids = []
    for i in range(b):
        f = factor[i].clamp_min(1.0)
        z_in = (z_grid / f).clamp(-1.0, 1.0)
        grids.append(torch.stack([x_grid, y_grid, z_in], dim=-1))
    grid = torch.stack(grids, dim=0)
    return F.grid_sample(vol, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


@dataclass
class LSFMBeadPhysicsPrior:
    cfg: BeadDistortionConfig
    include_background: bool = True

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any], **kwargs) -> "LSFMBeadPhysicsPrior":
        return cls(_cfg_from_mapping(cfg), **kwargs)

    def forward_distort(
        self,
        corrected: torch.Tensor,
        factor: torch.Tensor,
        *,
        renormalize: bool = True,
    ) -> torch.Tensor:
        """Apply deterministic LSFM degradation without Poisson/Gaussian noise."""
        device = corrected.device
        stretched = axial_stretch_3d(corrected, factor)
        profile = light_profile_zyx(self.cfg, device)
        out = stretched * profile.view(1, 1, *profile.shape)
        blurred = []
        for i in range(out.shape[0]):
            blurred.append(
                _separable_gaussian_blur3d(
                    out[i, 0],
                    sigma_z=self.cfg.blur_sigma_z,
                    sigma_xy=self.cfg.blur_sigma_xy,
                )
            )
        out = torch.stack(blurred, dim=0).unsqueeze(1)
        if renormalize:
            out = normalise01_per_volume(out)
        if self.include_background and self.cfg.background > 0:
            out = (out + self.cfg.background).clamp(0.0, 1.0)
        return out

    def consistency_loss(
        self,
        pred_corrected: torch.Tensor,
        distorted: torch.Tensor,
        factor: torch.Tensor,
    ) -> torch.Tensor:
        """Scale-matched forward consistency to avoid conflicting per-volume renorm."""
        sim = self.forward_distort(pred_corrected, factor, renormalize=False)
        w = distorted / distorted.amax(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-8)
        w = (0.15 + 0.85 * w.clamp(0.0, 1.0))
        num = (sim * w).sum(dim=(-3, -2, -1), keepdim=True)
        den = (distorted * w).sum(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-8)
        sim = sim * (den / num.clamp_min(1e-8))
        diff = F.smooth_l1_loss(sim, distorted, reduction="none")
        return (diff * w).sum() / w.sum().clamp_min(1e-8)


def bead_spherical_prior_loss(
    pred: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    valid: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Differentiable per-bead |sigma_z / mean(sigma_y,sigma_x) - 1| prior."""
    if pred.dim() != 5:
        raise ValueError(f"expected pred (B,1,Z,Y,X), got {tuple(pred.shape)}")
    b, _, nz, ny, nx = pred.shape
    losses = []
    for bi in range(b):
        for j in range(centers.shape[1]):
            if not valid[bi, j]:
                continue
            cz, cy, cx = [float(v) for v in centers[bi, j]]
            r = max(2.0, float(radii[bi, j].clamp_min(0.5)) * 5.0)
            z0, z1 = max(0, int(cz - r)), min(nz, int(cz + r + 1))
            y0, y1 = max(0, int(cy - r)), min(ny, int(cy + r + 1))
            x0, x1 = max(0, int(cx - r)), min(nx, int(cx + r + 1))
            if z1 - z0 < 2 or y1 - y0 < 2 or x1 - x0 < 2:
                continue
            crop = pred[bi, 0, z0:z1, y0:y1, x0:x1].clamp_min(0.0)
            mass = crop.sum()
            if mass <= eps:
                continue
            zz = torch.arange(z0, z1, device=pred.device, dtype=pred.dtype)
            yy = torch.arange(y0, y1, device=pred.device, dtype=pred.dtype)
            xx = torch.arange(x0, x1, device=pred.device, dtype=pred.dtype)
            zg, yg, xg = torch.meshgrid(zz, yy, xx, indexing="ij")
            w = crop / mass
            mz = (w * zg).sum()
            my = (w * yg).sum()
            mx = (w * xg).sum()
            sz = torch.sqrt((w * (zg - mz).pow(2)).sum().clamp_min(eps))
            sy = torch.sqrt((w * (yg - my).pow(2)).sum().clamp_min(eps))
            sx = torch.sqrt((w * (xg - mx).pow(2)).sum().clamp_min(eps))
            ratio = sz / ((sy + sx) * 0.5).clamp_min(eps)
            losses.append((ratio - 1.0).abs())
    if not losses:
        return pred.new_zeros(())
    return torch.stack(losses).mean()


def _huber1d(x: torch.Tensor, delta: float) -> torch.Tensor:
    ax = x.abs()
    quad = torch.clamp(ax, max=delta)
    lin = ax - quad
    return 0.5 * quad.pow(2) / delta + lin


def foreground_anisotropic_tv_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    wz: float = 3.0,
    wy: float = 1.0,
    wx: float = 1.0,
    huber_delta: float = 0.025,
    gamma: float = 8.0,
) -> torch.Tensor:
    """Edge-preserving smoothness inside beads; stronger along z to kill axial streaks."""
    fg = (target / target.amax(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-8)).clamp(0.0, 1.0)
    fg = (0.2 + 0.8 * fg).pow(0.75)
    dz = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
    dy = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
    dx = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
    mz = fg[:, :, 1:, :, :] * fg[:, :, :-1, :, :]
    my = fg[:, :, :, 1:, :] * fg[:, :, :, :-1, :]
    mx = fg[:, :, :, :, 1:] * fg[:, :, :, :, :-1]
    loss_z = (_huber1d(dz, huber_delta) * mz).sum() / mz.sum().clamp_min(1e-8)
    loss_y = (_huber1d(dy, huber_delta) * my).sum() / my.sum().clamp_min(1e-8)
    loss_x = (_huber1d(dx, huber_delta) * mx).sum() / mx.sum().clamp_min(1e-8)
    return wz * loss_z + wy * loss_y + wx * loss_x


def gradient_match_loss(pred: torch.Tensor, target: torch.Tensor, gamma: float = 8.0) -> torch.Tensor:
    """Match pred edge structure to the smooth spherical GT inside bead support."""
    fg = (target / target.amax(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-8)).clamp(0.0, 1.0)
    fg = 0.15 + 0.85 * fg
    terms = []
    for dim, axis in [(-3, "z"), (-2, "y"), (-1, "x")]:
        dp = pred.diff(dim=dim)
        dt = target.diff(dim=dim)
        w = fg.narrow(dim, 0, fg.shape[dim] - 1) * fg.narrow(dim, 1, fg.shape[dim] - 1)
        diff = F.smooth_l1_loss(dp, dt, reduction="none")
        terms.append((diff * w).sum() / w.sum().clamp_min(1e-8))
    return sum(terms) / len(terms)


def background_sparsity_loss(pred: torch.Tensor, target: torch.Tensor, gamma: float = 8.0) -> torch.Tensor:
    """Suppress predicted intensity outside the bead support defined by target."""
    fg = (target / target.amax(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-8)).clamp(0.0, 1.0)
    bg = (1.0 - fg).clamp(0.0, 1.0)
    w = 1.0 + gamma * bg
    return (pred.clamp_min(0.0) * w).sum() / w.sum().clamp_min(1e-8)
