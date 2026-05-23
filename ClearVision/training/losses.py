"""Combined loss for image restoration training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)


def ssim_map(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    c1: float = 0.01**2,
    c2: float = 0.03**2,
) -> torch.Tensor:
    """Differentiable SSIM map averaged over spatial dims (per batch item)."""
    channels = pred.size(1)
    window = _gaussian_window(window_size, sigma, pred.device, pred.dtype)
    window = window.expand(channels, 1, window_size, window_size)

    mu_x = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu_y = F.conv2d(target, window, padding=window_size // 2, groups=channels)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, padding=window_size // 2, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu_xy

    ssim = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return ssim.mean(dim=(1, 2, 3))


class RestorationLoss(nn.Module):
    """L1 + (1 - SSIM) + optional LPIPS on restored vs clean targets."""

    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.25,
        lpips_weight: float = 0.1,
        use_lpips: bool = True,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.lpips_weight = lpips_weight
        self.use_lpips = use_lpips
        self._lpips = None

        if use_lpips:
            try:
                import lpips

                self._lpips = lpips.LPIPS(net="alex")
                for p in self._lpips.parameters():
                    p.requires_grad = False
            except ImportError:
                self.use_lpips = False

    def to(self, device):
        module = super().to(device)
        if self._lpips is not None:
            self._lpips = self._lpips.to(device)
        return module

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        l1 = F.l1_loss(pred, target)
        ssim_val = ssim_map(pred, target).mean()
        ssim_loss = 1.0 - ssim_val

        total = self.l1_weight * l1 + self.ssim_weight * ssim_loss
        parts = {"l1": l1.item(), "ssim": ssim_val.item()}

        if self.use_lpips and self._lpips is not None:
            # LPIPS expects inputs in [-1, 1]
            lp = self._lpips(pred * 2 - 1, target * 2 - 1).mean()
            total = total + self.lpips_weight * lp
            parts["lpips"] = lp.item()

        parts["total"] = total.item()
        return total, parts
