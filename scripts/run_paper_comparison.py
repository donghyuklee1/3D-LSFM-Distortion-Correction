"""Run a paper-ready quantitative comparison for LSFM bead restoration.

Compares the proposed ViT+INR model against two learned baselines and one
classical baseline, then writes markdown + LaTeX summary tables with PSNR,
SSIM, and bead axial-ratio error.

Example:
    python scripts/run_paper_comparison.py --cfg configs/paper_comparison.yaml

Train missing neural baselines first (same 8,192-sample cache as ours):
    python scripts/run_paper_comparison.py --cfg configs/paper_comparison.yaml \\
        --train-baselines
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import train  # noqa: E402


def _latest_ckpt(run_dir: Path) -> Path | None:
    ckpts = sorted(run_dir.glob("ckpt_epoch*.pt"))
    return ckpts[-1] if ckpts else None


def _resolve_checkpoint(path_str: str, root: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = root / path
    return path


def _train_missing_baselines(comp_cfg: dict, root: Path) -> None:
    tb = comp_cfg.get("train_baselines", {})
    train_cfg = tb.get("cfg", "configs/paper_baselines.yaml")
    out_root = root / tb.get("out_root", "runs/lsfm_beads/paper_baselines")
    out_root.mkdir(parents=True, exist_ok=True)

    for model_name in tb.get("models", []):
        run_dir = out_root / model_name
        if _latest_ckpt(run_dir) is not None:
            print(f"[skip] baseline already trained: {model_name}")
            continue
        print(f"[train] baseline {model_name} -> {run_dir}")
        train(train_cfg, model_name=model_name, out_dir=run_dir.as_posix())


def _update_checkpoints_from_disk(comp_cfg: dict, root: Path) -> dict:
    tb = comp_cfg.get("train_baselines", {})
    out_root = root / tb.get("out_root", "runs/lsfm_beads/paper_baselines")
    alias = {
        "paper_ae": "paper_ae",
        "skip_ae3d": "skip_autoencoder_3d",
        "skip_autoencoder_3d": "skip_autoencoder_3d",
    }
    for method in comp_cfg.get("methods", []):
        if method.get("type") != "checkpoint":
            continue
        key = method["key"]
        ckpt = _resolve_checkpoint(method["checkpoint"], root)
        if ckpt.exists():
            continue
        model_dir = alias.get(key, key)
        latest = _latest_ckpt(out_root / model_dir)
        if latest is None and ckpt.parent.exists():
            latest = _latest_ckpt(ckpt.parent)
        if latest is not None:
            method["checkpoint"] = latest.relative_to(root).as_posix()
            print(f"[resolve] {key} -> {method['checkpoint']}")
    return comp_cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="configs/paper_comparison.yaml")
    p.add_argument("--train-baselines", action="store_true",
                   help="train neural baselines on the shared cache if missing")
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    comp_cfg = yaml.safe_load(open(args.cfg))
    if args.train_baselines:
        _train_missing_baselines(comp_cfg, ROOT)
    comp_cfg = _update_checkpoints_from_disk(comp_cfg, ROOT)

    dataset_cfg = comp_cfg.get("dataset_cfg", "configs/lsfm_beads.yaml")
    cache = comp_cfg.get("cache")
    out = comp_cfg.get("out", "runs/lsfm_beads/paper_comparison")
    metrics = comp_cfg.get("metrics", ["psnr", "ssim", "axial_ratio_error"])

    ckpt_args: list[str] = []
    classical_args: list[str] = []
    label_args: list[str] = []
    order_args: list[str] = []

    for method in comp_cfg.get("methods", []):
        key = method["key"]
        label = method.get("label", key)
        order_args.append(key)
        label_args.append(f"{key}:{label}")
        mtype = method.get("type", "checkpoint")
        if mtype == "classical":
            classical_args.append(key)
        elif mtype == "checkpoint":
            ckpt_path = _resolve_checkpoint(method["checkpoint"], ROOT)
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"checkpoint missing for {key}: {ckpt_path}\n"
                    "Run with --train-baselines or update configs/paper_comparison.yaml"
                )
            ckpt_args.append(f"{key}:{ckpt_path.as_posix()}")
        else:
            raise ValueError(f"unknown method type: {mtype}")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_restoration.py"),
        "--cfg", dataset_cfg,
        "--cache", cache,
        "--out", out,
        "--metrics", *metrics,
        "--method-order", *order_args,
        "--method-labels", *label_args,
        "--classical", *classical_args,
        "--checkpoints", *ckpt_args,
        "--paper-table",
    ]
    max_samples = args.max_samples if args.max_samples is not None else comp_cfg.get("max_samples")
    if max_samples is not None:
        cmd.extend(["--max-samples", str(max_samples)])

    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
