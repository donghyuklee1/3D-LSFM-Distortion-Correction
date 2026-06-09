from __future__ import annotations

import torch

def make_voxel_grid(
    Nz: int, Ny: int, Nx: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    z = torch.linspace(-1.0, 1.0, Nz, device=device)
    y = torch.linspace(-1.0, 1.0, Ny, device=device)
    x = torch.linspace(-1.0, 1.0, Nx, device=device)
    Z, Y, X = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack([Z, Y, X], dim=-1).reshape(-1, 3)

def sample_voxel_coords(
    n_samples: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    return torch.empty(n_samples, 3, device=device).uniform_(-1.0, 1.0)
