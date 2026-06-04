"""TensorBoard visualisation helpers for the AO pipeline.

These functions are *pure*: each takes tensors + a `SummaryWriter` and a
global step, and writes images / histograms / figures.

Image conventions for tensorboard:
    * use add_image with dataformats="HW" (single-channel) or "CHW"/"NCHW"
    * volumetric views show every Z-slice side-by-side so the user can
      visually confirm the 3D structure of pred / GT.
"""
from __future__ import annotations

import io
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


# --------------------------------------------------------------------------- #
def _nrm(x: torch.Tensor) -> torch.Tensor:
    """Per-frame min-max normalisation in [0, 1]."""
    x = x.detach().float()
    lo = x.amin()
    hi = x.amax()
    return (x - lo) / (hi - lo + 1e-8)


def _nrm_shared(x: torch.Tensor, vmax: float) -> torch.Tensor:
    """Signed normalisation around 0 with a SHARED dynamic range `vmax`.

    Maps [-vmax, +vmax] -> [0, 1] (clamped).  Critical for pred-vs-GT
    comparisons of signed fields like delta_n: if each frame is normalised
    independently, a near-zero pred would look 'as bright' as a non-trivial
    GT and the user can't tell whether the model actually recovered the
    structure.  With a shared scale, an all-zero pred shows as flat 0.5
    (mid-grey) and a matching pred shows the same contrast as GT.
    """
    x = x.detach().float()
    return (x / max(vmax, 1e-8) * 0.5 + 0.5).clamp(0.0, 1.0)


def _grid_2d(frames: torch.Tensor, nrow: int = 4) -> torch.Tensor:
    """Stitch (K, H, W) frames into a single (H', W') image."""
    K, H, W = frames.shape
    ncol = (K + nrow - 1) // nrow
    pad = nrow * ncol - K
    if pad:
        frames = torch.cat([frames, torch.zeros(pad, H, W,
                                                device=frames.device)], dim=0)
    g = frames.view(ncol, nrow, H, W).permute(0, 2, 1, 3)
    return g.reshape(ncol * H, nrow * W)


def log_image_stack(
    writer: "SummaryWriter",
    tag: str,
    stack: torch.Tensor,                # (K, H, W) or (B, K, H, W)
    step: int,
    nrow: int = 4,
):
    if stack.dim() == 4:
        stack = stack[0]
    img = _grid_2d(_nrm(stack), nrow=nrow)
    writer.add_image(tag, img, global_step=step, dataformats="HW")


def log_stack_pair_shared_scale(
    writer: "SummaryWriter",
    tag_input: str,
    tag_clean: str,
    input_stack: torch.Tensor,
    clean_stack: torch.Tensor,
    step: int,
    nrow: int = 4,
):
    """Log measurement vs clean GT with a *shared* intensity scale per frame.

    Independent min-max normalisation makes weak aberrations invisible
    (blurred and sharp spots both stretch to full contrast).  Sharing the
    per-frame vmax preserves the visible difference between aberrated input
    and ideal clean reference.
    """
    if input_stack.dim() == 4:
        input_stack = input_stack[0]
    if clean_stack.dim() == 4:
        clean_stack = clean_stack[0]
    inp = input_stack.detach().float()
    cln = clean_stack.detach().float()
    vmax = torch.maximum(inp.amax(dim=(-2, -1)), cln.amax(dim=(-2, -1)))
    vmax = vmax.clamp_min(1e-8)
    inp_n = (inp / vmax[:, None, None]).clamp(0.0, 1.0)
    cln_n = (cln / vmax[:, None, None]).clamp(0.0, 1.0)
    writer.add_image(
        tag_input, _grid_2d(inp_n, nrow=nrow), global_step=step, dataformats="HW",
    )
    writer.add_image(
        tag_clean, _grid_2d(cln_n, nrow=nrow), global_step=step, dataformats="HW",
    )


def log_volume_slices(
    writer: "SummaryWriter",
    tag: str,
    vol: torch.Tensor,                  # (Z, Y, X) or (B, Z, Y, X)
    step: int,
    n_slices: int | None = None,
):
    if vol.dim() == 4:
        vol = vol[0]
    Z = vol.shape[0]
    if n_slices is None or n_slices >= Z:
        sl = vol
        nrow = Z
    else:
        idx = torch.linspace(0, Z - 1, n_slices).round().long()
        sl = vol[idx]
        nrow = n_slices
    img = _grid_2d(_nrm(sl), nrow=nrow)
    writer.add_image(tag, img, global_step=step, dataformats="HW")


def log_scalar_volume_views(
    writer: "SummaryWriter",
    tag: str,
    vol: torch.Tensor,                  # (1, Z, Y, X), (Z, Y, X), or batched
    step: int,
    max_slices: int = 16,
):
    """Log 3D scalar volume as montage, orthogonal slices, and MIPs.

    TensorBoard does not have a native 3D volume widget. Logging just one
    central z-slice makes a real 3D dataset look like a 2D image, so this
    helper records complementary 2D projections:

    * ``{tag}/z_montage``: up to ``max_slices`` XY slices across z.
    * ``{tag}/orthogonal``: central XY, YZ, and XZ slices in one panel.
    * ``{tag}/mip``: maximum-intensity projections along z, y, and x.
    """
    v = vol.detach().float()
    if v.dim() == 5:      # (B, C, Z, Y, X)
        v = v[0, 0]
    elif v.dim() == 4:
        # (C, Z, Y, X) or (B, Z, Y, X); both use first leading element.
        v = v[0]
    assert v.dim() == 3, f"expected 3D volume after squeeze, got {tuple(v.shape)}"

    Z, Y, X = v.shape
    n = min(max_slices, Z)
    idx = torch.linspace(0, Z - 1, n).round().long()
    z_slices = _nrm(v[idx])
    writer.add_image(
        f"{tag}/z_montage",
        _grid_2d(z_slices, nrow=max(1, min(4, n))),
        global_step=step,
        dataformats="HW",
    )

    zc, yc, xc = Z // 2, Y // 2, X // 2
    xy = _nrm(v[zc])
    yz = _nrm(v[:, :, xc]).transpose(0, 1)  # (Y, Z)
    xz = _nrm(v[:, yc, :])                  # (Z, X)
    yz = torch.nn.functional.interpolate(
        yz[None, None], size=(Y, X), mode="bilinear", align_corners=False,
    )[0, 0]
    xz = torch.nn.functional.interpolate(
        xz[None, None], size=(Y, X), mode="bilinear", align_corners=False,
    )[0, 0]
    writer.add_image(
        f"{tag}/orthogonal",
        torch.cat([xy, yz, xz], dim=1),
        global_step=step,
        dataformats="HW",
    )

    mip_z = _nrm(v.amax(dim=0))
    mip_y = _nrm(v.amax(dim=1))
    mip_x = _nrm(v.amax(dim=2))
    mip_y = torch.nn.functional.interpolate(
        mip_y[None, None], size=(Y, X), mode="bilinear", align_corners=False,
    )[0, 0]
    mip_x = torch.nn.functional.interpolate(
        mip_x[None, None], size=(Y, X), mode="bilinear", align_corners=False,
    )[0, 0]
    writer.add_image(
        f"{tag}/mip",
        torch.cat([mip_z, mip_y, mip_x], dim=1),
        global_step=step,
        dataformats="HW",
    )


def log_volume_pred_vs_gt(
    writer: "SummaryWriter",
    tag: str,
    pred: torch.Tensor,                 # (Z, Y, X) or (B, Z, Y, X) -- SIGNED
    gt:   torch.Tensor,                 # same shape
    step: int,
):
    """Show ALL Z-slices of pred / GT / |pred-GT| with a SHARED dynamic range.

    Layout (single greyscale image, height = 3*Y, width = Z * X):

        row 1 :  pred[z=0] | pred[z=1] | ... | pred[z=Nz-1]
        row 2 :  gt[z=0]   | gt[z=1]   | ... | gt[z=Nz-1]
        row 3 :  |pred-gt|[z=0] | ... | |pred-gt|[z=Nz-1]

    pred and gt are normalised to a shared scale based on GT's |max|, so a
    near-zero pred looks flat mid-grey rather than amplified-noise.  This
    is the right way to visually inspect whether the model actually
    recovered 3D structure, not just a single mid-slice.
    """
    if pred.dim() == 4: pred = pred[0]
    if gt.dim()   == 4: gt   = gt[0]

    vmax = gt.detach().abs().max().clamp_min(1e-8).item()
    pred_n = _nrm_shared(pred, vmax)
    gt_n   = _nrm_shared(gt,   vmax)

    diff = (pred - gt).detach().abs()
    err_max = diff.max().clamp_min(1e-8).item()
    err_n = (diff / err_max).clamp(0.0, 1.0)

    Nz = pred.shape[0]
    pred_row = _grid_2d(pred_n, nrow=Nz)        # 1 row of Z slices
    gt_row   = _grid_2d(gt_n,   nrow=Nz)
    err_row  = _grid_2d(err_n,  nrow=Nz)

    img = torch.cat([pred_row, gt_row, err_row], dim=0)
    writer.add_image(tag, img, global_step=step, dataformats="HW")


def log_pred_vs_gt(
    writer: "SummaryWriter",
    tag: str,
    pred: torch.Tensor,
    gt: torch.Tensor,
    step: int,
):
    """Backwards-compatible wrapper: now delegates to the volumetric viz."""
    log_volume_pred_vs_gt(writer, tag, pred, gt, step)


def log_zernike_bar(
    writer: "SummaryWriter",
    tag: str,
    pred: torch.Tensor,
    gt: torch.Tensor | None,
    step: int,
):
    """Render predicted vs GT Zernike coefficients as a bar plot via matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    pred = pred.detach().float().cpu()
    if pred.dim() == 2:
        pred = pred[0]
    fig, ax = plt.subplots(figsize=(6, 2.5))
    j = torch.arange(pred.numel())
    width = 0.4
    ax.bar(j - width / 2, pred.numpy(), width=width, label="pred")
    if gt is not None:
        gt = gt.detach().float().cpu()
        if gt.dim() == 2:
            gt = gt[0]
        ax.bar(j + width / 2, gt.numpy(), width=width, label="gt", alpha=0.7)
    ax.set_xlabel("Zernike index"); ax.set_ylabel("coefficient")
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    writer.add_figure(tag, fig, global_step=step)
    plt.close(fig)


def log_histograms(
    writer: "SummaryWriter",
    step: int,
    **named_tensors: torch.Tensor,
):
    for name, t in named_tensors.items():
        if t is None:
            continue
        writer.add_histogram(name, t.detach().float().cpu(), step)


def log_grad_norm(
    writer: "SummaryWriter",
    parameters,
    step: int,
    tag: str = "train/grad_norm_total",
):
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    writer.add_scalar(tag, total ** 0.5, step)
