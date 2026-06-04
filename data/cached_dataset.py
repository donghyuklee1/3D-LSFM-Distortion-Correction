"""Cached version of the synthetic dataset.

`prepare_dataset.py` writes one .pt file per sample to `root/`. This class just
loads them. It is significantly faster than online generation because the BPM
forward simulation is run only *once per sample* during preparation, not every
epoch.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class CachedAOStackDataset(Dataset):
    def __init__(self, root: str | Path):
        super().__init__()
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"cache directory not found: {self.root}")
        self.files = sorted(self.root.glob("sample_*.pt"))
        if not self.files:
            raise FileNotFoundError(
                f"no sample_*.pt under {self.root}; "
                f"run scripts/prepare_dataset.py first"
            )
        meta_path = self.root / "meta.pt"
        if meta_path.exists():
            self.meta = torch.load(meta_path, map_location="cpu",
                                   weights_only=False)
        else:
            self.meta = {}

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return torch.load(
            self.files[idx], map_location="cpu", weights_only=False
        )
