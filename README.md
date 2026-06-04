### KAIST 26 Spring **EE.49904 Computational Imaging**
# Implicit Neural Representations for 3D LSFM Distortion Correction via Axial Vision Transformers

Author: Donghyuk Lee (Adviser: Iksung Kang)

Amortized **Vision Transformer (ViT) + Multi-Head INR** framework for correcting axial elongation and mixed Poisson–Gaussian degradation in synthetic LSFM bead calibration volumes. The model learns global $z$-axis distortion context from an axial slice sequence and reconstructs a continuous corrected fluorescence density field via FiLM-conditioned coordinate MLPs, supervised by foreground-weighted losses and a differentiable LSFM forward physics prior.

---

## Overview

| Component | Description |
|-----------|-------------|
| **Input** | Distorted 3D bead stack $x \in \mathbb{R}^{1 \times Z \times Y \times X}$ (64³ voxels) |
| **Output** | Corrected spherical bead volume $\hat{y}$ |
| **Encoder** | Axial ViT — treats $Z$ slices as a token sequence with a class token |
| **Decoder** | Multi-head INR — Fourier features + local 3D evidence + FiLM MLP |
| **Training** | 8,192 cached synthetic volumes; physics-informed multi-term loss |

### Key ideas

- **Global context via ViT latent space:** The encoder captures volume-wide $z$-elongation patterns, enabling amortized inference (one forward pass per volume) instead of per-scene coordinate optimization.
- **Multi-head disentangled INR:** Latent $z$ splits into auxiliary ($z_{\text{aux}}$) and density ($z_\rho$) subspaces; local coordinate evidence resolves spatial ambiguity for sparse beads.
- **Physics-informed supervision:** Differentiable re-distortion operator $D(\hat{y}, f)$ enforces forward consistency with the observed stack.

---

## Repository structure

```
configs/
  lsfm_beads.yaml           # Synthetic dataset parameters
  train_vit_inr.yaml        # Proposed ViT+INR training (1,200 epochs)
  paper_baselines.yaml      # 2D AE / 3D skip-AE baselines (2,400 epochs, matched steps)
  paper_comparison.yaml     # Evaluation table configuration
  smoke_lsfm_beads.yaml     # Fast local smoke test

data/
  lsfm_beads_dataset.py     # Synthetic bead forward model + target generation
  cached_dataset.py         # Disk cache loader

models/
  restoration.py            # ViT+INR and baseline architectures
  inr_primitives.py         # Fourier features, FiLM layers

losses/
  lsfm_bead_physics.py      # Forward consistency, spherical prior, TV, etc.

scripts/
  prepare_dataset.py        # Pre-generate cached training set
  evaluate_restoration.py   # Per-method metrics (PSNR, SSIM, axial ratio)
  run_paper_comparison.py   # Full paper comparison pipeline
  setup_server.sh           # Environment bootstrap

train.py                    # Main training entry point
```

Large artifacts (`cached/`, `runs/`, `*.pt`) are gitignored — regenerate locally (see below).

---

## Setup

```bash
bash scripts/setup_server.sh
source .venv/bin/activate
```

**Requirements:** Python 3.10+, PyTorch 2.2+, NumPy, PyYAML, tqdm, TensorBoard, Matplotlib.

---

## Quick start

### 1. Generate dataset (8,192 volumes)

Synthetic pipeline inspired by Julia et al. (2024): random spherical beads, axial elongation ($f \in [1.8, 10]$), optional asymmetric RI field, light-sheet illumination, 3D Gaussian PSF blur, mixed Poisson–Gaussian noise. Set `dataset.n_samples` in `configs/lsfm_beads.yaml` (default **8,192**, 8× the original short-run cache).

```bash
python scripts/prepare_dataset.py \
  --cfg configs/lsfm_beads.yaml \
  --out cached/lsfm_beads \
  --seed 0
```

Each sample stores `stack_distorted`, `stack_corrected`, bead centers/radii, and distortion metadata.

### 2. Train proposed model (1,200 epochs)

```bash
python train.py \
  --cfg configs/train_vit_inr.yaml \
  --out runs/lsfm_beads/vit_multihead_inr
```

Monitor with TensorBoard:

```bash
tensorboard --logdir runs/lsfm_beads/vit_multihead_inr
```

Checkpoints are saved every 100 epochs as `ckpt_epoch00000.pt` … `ckpt_epoch01199.pt` (final epoch always kept).

Train baselines separately (2,400 epochs, matched optimizer steps):

```bash
python train.py --cfg configs/paper_baselines.yaml --model paper_ae \
  --out runs/lsfm_beads/paper_baselines/paper_ae
python train.py --cfg configs/paper_baselines.yaml --model skip_autoencoder_3d \
  --out runs/lsfm_beads/paper_baselines/skip_autoencoder_3d
```

### 3. Evaluate

```bash
python scripts/evaluate_restoration.py \
  --cfg configs/train_vit_inr.yaml \
  --cache cached/lsfm_beads \
  --checkpoint runs/lsfm_beads/vit_multihead_inr/ckpt_epoch01199.pt \
  --methods vit_multihead_inr axial_factor \
  --out runs/lsfm_beads/eval
```

### 4. Paper comparison table

Trains missing baselines (if needed) and evaluates all methods on the full 8,192-sample cache:

```bash
python scripts/run_paper_comparison.py \
  --cfg configs/paper_comparison.yaml --train-baselines
```

Outputs: `runs/lsfm_beads/paper_comparison/summary.md` and `summary.tex`.

---

## Model architecture

<img width="1299" height="467" alt="image" src="https://github.com/user-attachments/assets/22cf9025-2a1e-4ddf-ae98-75bf4598610b" />

**Supported model names** (`model.name` in config):

| Name | Role |
|------|------|
| `vit_multihead_inr` | **Proposed** — ViT + multi-head INR |
| `paper_ae` | 2D slice-wise autoencoder baseline |
| `skip_autoencoder_3d` | 3D U-Net-style skip AE baseline |
| `unet3d` | Compact 3D U-Net |
| `identity` | No-op (eval only) |

**Capacity (current configs):**

| Component | Proposed (`train_vit_inr.yaml`) | Baselines (`paper_baselines.yaml`) |
|-----------|--------------------------------|-------------------------------------|
| Encoder | ViT: 384-d, 6 blocks, 6 heads, patch 8 | — |
| Latent | 256-d global code | — |
| Decoder | Multi-head INR: 128×6 MLP, 32 Fourier features | — |
| Local path | 16-ch 3D conv stem + intensity | — |
| CNN baselines | — | 2D slice AE / 3D skip AE, `base_channels=32` (~1.4M params) |
| Total params | ~13M | ~1.4M (skip AE) |

---

## Loss functions

Core supervised terms (Section 4 of the report):

| Symbol | Description |
|--------|-------------|
| $L_{\text{wmse}}$ | Foreground-weighted MSE for sparse beads |
| $L_{\text{proj}}$ | MIP consistency along Z / Y / X (Smooth L1) |
| $L_{\text{phys}}$ | Forward re-distortion consistency $\|s \cdot D(\hat{y}, f) - x\|_{1,w}$ |

Additional regularizers: axial compactness, soft Dice, spherical bead prior, background sparsity, anisotropic TV, gradient matching. Weights are set in `configs/train_vit_inr.yaml` (`lambda_*`).

---

## Reported results (prior 1,024-volume short run)

*Figures below are from an earlier 1,024-sample / short-epoch checkpoint. Re-generate the 8,192-sample cache, re-train, and re-evaluate with `ckpt_epoch01199.pt` for numbers aligned with the current config.*

<img width="815" height="199" alt="image" src="https://github.com/user-attachments/assets/8bbef38f-6012-4111-8585-3d0bfeb694c1" />

<img width="1080" height="358" alt="image" src="https://github.com/user-attachments/assets/9d2c314d-10fb-44ee-8250-992326a97ab9" />

<img width="602" height="374" alt="image" src="https://github.com/user-attachments/assets/714dc067-6220-4071-8235-3728226b0327" />

The 3D skip AE achieves the highest pixel-level PSNR/SSIM but introduces geometric blur (high axial ratio error). The 2D AE cannot correct macroscopic $z$-elongation. **Our framework achieves the lowest axial ratio error** while maintaining competitive global restoration, recovering isotropic spherical bead morphology under severe ($8.3\times$) distortion.

Representative qualitative figures in the report: MIP/slice restoration (`result1.png`), 3D isosurface render (`result2.png`), axial intensity profiles with FWHM compression 19→5 layers (`result3.png`), architecture diagram (`result4.png`).

---

## Training configuration

Long-run amortized training on a fixed **8,192-sample** synthetic cache (scaled 8× to match 1,200-epoch training). Baselines run for **2,400 epochs** so total Adam steps match the proposed **1,200-epoch** run (batch 4 vs 2). Both use AdamW, linear LR warmup, and cosine decay.

| Setting | Proposed (`train_vit_inr.yaml`) | Baselines (`paper_baselines.yaml`) |
|---------|--------------------------------|-------------------------------------|
| Training samples | 8,192 | 8,192 |
| Volume size | 64³ | 64³ |
| Model capacity | ViT 384-d × 6; INR 128×6; latent 256 | Skip / 2D AE, `base_channels=32` |
| Optimizer | AdamW | AdamW |
| Batch size | 2 | 4 |
| Steps / epoch | 4,096 | 2,048 |
| **Epochs** | **1,200** | **2,400** |
| **Total optimizer steps** | **~4.9M** | **~4.9M** |
| Peak learning rate | 1.5×10⁻⁵ | 3×10⁻⁴ |
| LR warmup | 50 epochs | 40 epochs |
| LR schedule | cosine → 1×10⁻⁷ | cosine → 3×10⁻⁶ |
| Weight decay | 5×10⁻⁵ | 1×10⁻⁵ |
| Grad clip (max norm) | 1.0 | 1.0 |
| Physics-prior ramp | 120 epochs (10%) | — |
| Checkpoint interval | every 100 epochs | every 200 epochs |
| Distortion range | 1.8× – 10× | same |

The proposed model uses a lower peak LR, stronger weight decay, and a 10% physics-prior ramp so INR decoding stabilizes before forward-consistency terms reach full weight. `local_feature_dropout=0.05` provides mild regularization on the expanded local stem.

---

## Smoke test

```bash
python scripts/prepare_dataset.py --cfg configs/smoke_lsfm_beads.yaml \
  --out cached/smoke_lsfm_beads --n 4 --seed 1
python train.py --cfg configs/smoke_lsfm_beads.yaml \
  --out runs/smoke_lsfm_beads
```

---

## References

- Xue et al. (2022) — BRIEF: 3D bi-functional RI and fluorescence microscopy.
- Feng et al. (2023) — NeuWS: Neural wavefront shaping with continuous representations.
- Julia et al. (2024) — Distortion correction and denoising of LSFM images (calibration protocol inspiration).

---

## License

Academic project for EE.49904 Computational Imaging, KAIST.
