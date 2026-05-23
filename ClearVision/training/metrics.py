"""Validation metrics: PSNR, SSIM, LPIPS (numpy / torch)."""

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def tensor_to_numpy_img(t: torch.Tensor) -> np.ndarray:
    """CHW float [0,1] -> HWC float [0,1]."""
    return t.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)


@torch.no_grad()
def compute_batch_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    lpips_model=None,
) -> dict[str, float]:
    """Mean PSNR / SSIM / LPIPS over a batch (B, C, H, W)."""
    psnr_vals = []
    ssim_vals = []
    for i in range(pred.size(0)):
        p = tensor_to_numpy_img(pred[i])
        t = tensor_to_numpy_img(target[i])
        psnr_vals.append(peak_signal_noise_ratio(t, p, data_range=1.0))
        ssim_vals.append(
            structural_similarity(t, p, channel_axis=2, data_range=1.0)
        )

    out = {
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
    }

    if lpips_model is not None:
        lp = lpips_model(pred * 2 - 1, target * 2 - 1).mean().item()
        out["lpips"] = float(lp)
    return out
