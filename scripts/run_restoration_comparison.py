"""Train restoration models and run one paper-style quantitative comparison.

This is the reproducibility entry point for the LSFM bead distortion experiment.
It trains the requested neural baselines, evaluates them alongside non-learned
baselines (`identity`, `axial_factor`), and writes `metrics.csv` + `summary.md`.

Example:
    python scripts/run_restoration_comparison.py --cfg configs/lsfm_beads.yaml \
        --cache cached/lsfm_beads --models paper_ae skip_autoencoder_3d unet3d
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import train  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="configs/lsfm_beads.yaml")
    p.add_argument("--cache", default="cached/lsfm_beads")
    p.add_argument("--out-root", default="runs/lsfm_beads/comparison")
    p.add_argument("--models", nargs="+", default=["paper_ae", "skip_autoencoder_3d", "unet3d"])
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    ckpt_args = []
    for model_name in args.models:
        run_dir = out_root / model_name
        trained_dir = train(args.cfg, model_name=model_name, out_dir=run_dir.as_posix())
        ckpts = sorted(Path(trained_dir).glob("ckpt_epoch*.pt"))
        if not ckpts:
            raise RuntimeError(f"no checkpoint produced for {model_name}")
        ckpt_args.append(f"{model_name}:{ckpts[-1].as_posix()}")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_restoration.py"),
        "--cfg", args.cfg,
        "--cache", args.cache,
        "--out", str(out_root / "eval"),
        "--checkpoints",
        *ckpt_args,
    ]
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
