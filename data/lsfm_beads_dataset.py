"""Paper-style LSFM 3D bead distortion dataset.

This dataset follows the calibration-data logic from
"Distortion Correction and Denoising of Light Sheet Fluorescence Images"
(Sensors 2024):

1. Build a corrected bead stack made of spherical fluorescent beads.
2. Build the corresponding microscope input by elongating beads along z,
   applying a Gaussian illumination profile, blur, background, and mixed
   Poisson-Gaussian noise.
3. Train restoration models on distorted input -> corrected target.

The original paper creates targets from real bead calibration stacks by
segmenting beads slice-by-slice, estimating their centers, and reconstructing
spherical beads at those positions. In this project we do not have the raw
calibration stack, so this class provides a reproducible synthetic equivalent
with the same supervised signal and measurable axial distortion factor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class BeadDistortionConfig:
    Nz: int = 64
    Ny: int = 64
    Nx: int = 64
    min_beads: int = 3
    max_beads: int = 8
    bead_radius_px: float = 2.0
    radius_jitter: float = 0.25
    margin_px: int = 8
    distortion_min: float = 1.8
    distortion_max: float = 10.0
    spatially_varying_distortion: bool = True
    asymmetric_ri: bool = True
    asymmetric_ri_strength: float = 0.30
    lateral_shear_px: float = 2.0
    axial_shift_px: float = 3.0
    light_sigma_y_frac: float = 0.45
    light_sigma_z_frac: float = 0.80
    blur_sigma_xy: float = 0.8
    blur_sigma_z: float = 1.2
    photons: float = 150.0
    gaussian_noise_sigma: float = 0.015
    background: float = 0.02


def _cfg_from_mapping(cfg: Mapping[str, Any]) -> BeadDistortionConfig:
    d = cfg.get("bead_dataset", cfg)
    optics = cfg.get("optics", {})
    return BeadDistortionConfig(
        Nz=int(d.get("Nz", optics.get("Nz", 64))),
        Ny=int(d.get("Ny", optics.get("Ny", 64))),
        Nx=int(d.get("Nx", optics.get("Nx", 64))),
        min_beads=int(d.get("min_beads", 3)),
        max_beads=int(d.get("max_beads", 8)),
        bead_radius_px=float(d.get("bead_radius_px", 2.0)),
        radius_jitter=float(d.get("radius_jitter", 0.25)),
        margin_px=int(d.get("margin_px", 8)),
        distortion_min=float(d.get("distortion_min", 1.8)),
        distortion_max=float(d.get("distortion_max", 10.0)),
        spatially_varying_distortion=bool(d.get("spatially_varying_distortion", True)),
        asymmetric_ri=bool(d.get("asymmetric_ri", True)),
        asymmetric_ri_strength=float(d.get("asymmetric_ri_strength", 0.30)),
        lateral_shear_px=float(d.get("lateral_shear_px", 2.0)),
        axial_shift_px=float(d.get("axial_shift_px", 3.0)),
        light_sigma_y_frac=float(d.get("light_sigma_y_frac", 0.45)),
        light_sigma_z_frac=float(d.get("light_sigma_z_frac", 0.80)),
        blur_sigma_xy=float(d.get("blur_sigma_xy", 0.8)),
        blur_sigma_z=float(d.get("blur_sigma_z", 1.2)),
        photons=float(d.get("photons", 150.0)),
        gaussian_noise_sigma=float(d.get("gaussian_noise_sigma", 0.015)),
        background=float(d.get("background", 0.02)),
    )


def _meshgrid_zyx(Nz: int, Ny: int, Nx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = torch.arange(Nz, dtype=torch.float32)
    y = torch.arange(Ny, dtype=torch.float32)
    x = torch.arange(Nx, dtype=torch.float32)
    return torch.meshgrid(z, y, x, indexing="ij")


def _gaussian_kernel1d(sigma: float, device: torch.device) -> torch.Tensor:
    if sigma <= 0:
        return torch.ones(1, device=device)
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(x**2) / (2.0 * sigma**2))
    return k / k.sum().clamp_min(1e-8)


def _separable_gaussian_blur3d(vol: torch.Tensor, sigma_z: float, sigma_xy: float) -> torch.Tensor:
    """Blur a single ``(Z, Y, X)`` volume with separable 3D Gaussian kernels."""
    x = vol[None, None]
    device = vol.device

    kz = _gaussian_kernel1d(sigma_z, device).view(1, 1, -1, 1, 1)
    ky = _gaussian_kernel1d(sigma_xy, device).view(1, 1, 1, -1, 1)
    kx = _gaussian_kernel1d(sigma_xy, device).view(1, 1, 1, 1, -1)

    x = F.conv3d(x, kz, padding=(kz.shape[2] // 2, 0, 0))
    x = F.conv3d(x, ky, padding=(0, ky.shape[3] // 2, 0))
    x = F.conv3d(x, kx, padding=(0, 0, kx.shape[4] // 2))
    return x[0, 0]


def _normalise01(x: torch.Tensor) -> torch.Tensor:
    x = x - x.amin()
    return x / x.amax().clamp_min(1e-8)


class LSFMDistortionBeadsDataset(Dataset):
    """Synthetic 3D bead calibration stacks for axial distortion correction."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        length: int = 1024,
        seed: int | None = None,
        return_legacy_keys: bool = True,
    ):
        super().__init__()
        self.cfg = _cfg_from_mapping(cfg)
        self.length = int(length)
        self.seed = seed
        self.return_legacy_keys = return_legacy_keys

    def __len__(self) -> int:
        return self.length

    def _generator(self, idx: int) -> torch.Generator:
        g = torch.Generator(device="cpu")
        if self.seed is not None:
            g.manual_seed(int(self.seed) + idx * 9973)
        return g

    def _sample_beads(self, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.cfg
        n = int(torch.randint(c.min_beads, c.max_beads + 1, (1,), generator=g).item())
        margin = c.margin_px
        z_margin = max(2, min(margin, c.Nz // 4))
        y_margin = max(2, min(margin, c.Ny // 4))
        x_margin = max(2, min(margin, c.Nx // 4))

        centers = torch.empty(n, 3, dtype=torch.float32)
        centers[:, 0] = torch.empty(n).uniform_(z_margin, c.Nz - z_margin, generator=g)
        centers[:, 1] = torch.empty(n).uniform_(y_margin, c.Ny - y_margin, generator=g)
        centers[:, 2] = torch.empty(n).uniform_(x_margin, c.Nx - x_margin, generator=g)

        jitter = torch.empty(n).uniform_(1.0 - c.radius_jitter, 1.0 + c.radius_jitter, generator=g)
        radii = c.bead_radius_px * jitter
        amps = torch.empty(n).uniform_(0.7, 1.0, generator=g)
        return centers, radii, amps

    def _render_beads(
        self,
        centers: torch.Tensor,
        radii: torch.Tensor,
        amps: torch.Tensor,
        z_stretch: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        c = self.cfg
        zz, yy, xx = _meshgrid_zyx(c.Nz, c.Ny, c.Nx)
        vol = torch.zeros(c.Nz, c.Ny, c.Nx, dtype=torch.float32)
        if isinstance(z_stretch, float):
            z_stretch = torch.full((centers.shape[0],), z_stretch)

        for i in range(centers.shape[0]):
            cz, cy, cx = centers[i]
            r = radii[i].clamp_min(0.5)
            rz = r * z_stretch[i].clamp_min(1.0)
            # Smooth binary bead: sigmoid of signed ellipsoid distance.
            dist = torch.sqrt(((zz - cz) / rz) ** 2 + ((yy - cy) / r) ** 2 + ((xx - cx) / r) ** 2) - 1.0
            bead = torch.sigmoid(-dist * 8.0)
            vol = vol + amps[i] * bead
        return _normalise01(vol)

    def _light_profile(self) -> torch.Tensor:
        c = self.cfg
        zz, yy, xx = _meshgrid_zyx(c.Nz, c.Ny, c.Nx)
        cy = (c.Ny - 1) / 2.0
        cz = (c.Nz - 1) / 2.0
        sig_y = max(1.0, c.light_sigma_y_frac * c.Ny)
        sig_z = max(1.0, c.light_sigma_z_frac * c.Nz)
        profile = torch.exp(-((yy - cy) ** 2) / (2.0 * sig_y**2)) * torch.exp(
            -((zz - cz) ** 2) / (2.0 * sig_z**2)
        )
        return 0.35 + 0.65 * profile

    def _degrade(self, vol: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        c = self.cfg
        x = vol * self._light_profile()
        x = _separable_gaussian_blur3d(x, sigma_z=c.blur_sigma_z, sigma_xy=c.blur_sigma_xy)
        x = _normalise01(x)
        if c.background > 0:
            x = x + c.background
        if c.photons > 0:
            # Mixed Poisson-Gaussian microscopy noise model.
            x = torch.poisson((x.clamp_min(0.0) * c.photons), generator=g) / c.photons
        if c.gaussian_noise_sigma > 0:
            x = x + c.gaussian_noise_sigma * torch.randn(x.shape, generator=g)
        return _normalise01(x.clamp_min(0.0))

    def _apply_asymmetric_ri(
        self,
        centers: torch.Tensor,
        factors: torch.Tensor,
        g: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Approximate asymmetric RI mismatch as a non-uniform distortion field.

        A symmetric immersion mismatch is well approximated by one global axial
        elongation factor. Real LSFM data are often worse: RI can vary across
        the cleared sample or chamber, so beads on one side of the FOV stretch
        and shift differently from beads elsewhere. We model that calibration
        artifact with a low-order field over bead centers:

            stretch(x,y,z) = base * (1 + ax*x + ay*y + az*z + axy*x*y)

        plus small lateral/depth shears. Following the paper's bead-calibration
        protocol, the corrected target is reconstructed at the bead centers
        detected in the distorted stack. Therefore, asymmetric RI may move the
        observed bead centroid, but the target uses that observed centroid and
        removes only the elongation/blur/noise. This keeps the inverse problem
        identifiable from a single sparse bead stack.
        """
        c = self.cfg
        if not c.asymmetric_ri:
            return centers, factors, torch.zeros(4)

        s = c.asymmetric_ri_strength
        coeffs = torch.empty(4).uniform_(-s, s, generator=g)  # ax, ay, az, axy
        ax, ay, az, axy = coeffs

        z_norm = 2.0 * centers[:, 0] / max(c.Nz - 1, 1) - 1.0
        y_norm = 2.0 * centers[:, 1] / max(c.Ny - 1, 1) - 1.0
        x_norm = 2.0 * centers[:, 2] / max(c.Nx - 1, 1) - 1.0

        field = 1.0 + ax * x_norm + ay * y_norm + az * z_norm + axy * x_norm * y_norm
        factors = (factors * field.clamp(0.55, 1.65)).clamp(c.distortion_min, c.distortion_max)

        shifted = centers.clone()
        shifted[:, 0] = shifted[:, 0] + c.axial_shift_px * (ax * x_norm + ay * y_norm)
        shifted[:, 1] = shifted[:, 1] + c.lateral_shear_px * ay * z_norm
        shifted[:, 2] = shifted[:, 2] + c.lateral_shear_px * ax * z_norm

        shifted[:, 0] = shifted[:, 0].clamp(1.0, c.Nz - 2.0)
        shifted[:, 1] = shifted[:, 1].clamp(1.0, c.Ny - 2.0)
        shifted[:, 2] = shifted[:, 2].clamp(1.0, c.Nx - 2.0)
        return shifted, factors, coeffs

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        g = self._generator(idx)
        c = self.cfg
        centers, radii, amps = self._sample_beads(g)

        base_factor = float(torch.empty(1).uniform_(c.distortion_min, c.distortion_max, generator=g).item())
        if c.spatially_varying_distortion:
            jitter = torch.empty(centers.shape[0]).uniform_(0.85, 1.15, generator=g)
            factors = (base_factor * jitter).clamp(c.distortion_min, c.distortion_max)
        else:
            factors = torch.full((centers.shape[0],), base_factor)

        distorted_centers, factors, asym_coeffs = self._apply_asymmetric_ri(centers, factors, g)
        corrected = self._render_beads(distorted_centers, radii, amps, z_stretch=1.0)
        elongated = self._render_beads(distorted_centers, radii, amps, z_stretch=factors)
        distorted = self._degrade(elongated, g)

        sample: dict[str, torch.Tensor] = {
            "stack_distorted": distorted.unsqueeze(0).float(),  # (C=1, Z, Y, X)
            "stack_corrected": corrected.unsqueeze(0).float(),  # (C=1, Z, Y, X)
            "bead_centers": distorted_centers.float(),           # detected/observed centers
            "undistorted_bead_centers": centers.float(),         # latent sampled centers
            "bead_radii": radii.float(),
            "bead_amplitudes": amps.float(),
            "distortion_factor": torch.tensor(base_factor, dtype=torch.float32),
            "distortion_factors_per_bead": factors.float(),
            "asymmetric_ri_coeffs": asym_coeffs.float(),
        }

        if self.return_legacy_keys:
            # Keep cached samples loadable by existing visualisation/training code.
            sample["stack_emission"] = sample["stack_distorted"].squeeze(0)
            sample["stack_clean_emission"] = sample["stack_corrected"].squeeze(0)
            sample["rho_gt"] = sample["stack_corrected"].squeeze(0)
            sample["ri_gt"] = torch.full((c.Nz, c.Ny, c.Nx), 1.33)
            sample["zernike_gt"] = torch.zeros(10)
        return sample
