from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeltaNBoundedHead(nn.Module):

    def __init__(self, delta_n_max: float, scale: float = 3.0):
        super().__init__()
        self.delta_n_max = delta_n_max
        self.scale = scale

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.delta_n_max * torch.tanh(z / self.scale)

class FourierFeatures(nn.Module):

    def __init__(self, in_dim: int = 3, num_features: int = 64, sigma: float = 2.0):
        super().__init__()
        B = torch.randn(in_dim, num_features) * sigma
        self.register_buffer("B", B, persistent=True)
        freq = B.norm(dim=0)
        self.register_buffer(
            "freq_rank", freq / freq.max().clamp_min(1e-8), persistent=True,
        )
        self.out_dim = 2 * num_features
        self._band_limit: float = 1.0

    def set_band_limit(self, ratio: float) -> None:
        self._band_limit = float(max(0.0, min(1.0, ratio)))

    def _band_weights(self, device: torch.device) -> torch.Tensor:
        if self._band_limit >= 1.0 - 1e-6:
            return torch.ones(self.B.shape[1], device=device)
        edge = 0.08
        w = (self._band_limit + edge - self.freq_rank.to(device)) / edge
        return w.clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * x @ self.B
        w = self._band_weights(x.device)
        return torch.cat([torch.sin(proj) * w, torch.cos(proj) * w], dim=-1)

class FiLMLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, cond_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.film = nn.Linear(cond_dim, 2 * out_dim)
        with torch.no_grad():
            self.film.weight.mul_(4.0)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        gb = self.film(z).unsqueeze(1)
        gamma, beta = gb.chunk(2, dim=-1)
        h = (1.0 + gamma) * h + beta
        return F.gelu(h)
