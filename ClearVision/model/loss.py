import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# VGG perceptual feature extractor
# ─────────────────────────────────────────────────────────────────────────────

class VGGPerceptualLoss(nn.Module):

    # relu1_2, relu2_2, relu3_3 indices in VGG-16 features
    _LAYER_IDS = [4, 9, 16]

    def __init__(self):
        super().__init__()
        try:
            import torchvision.models as models
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        except Exception as e:
            raise ImportError(
                "torchvision is required for perceptual loss. "
                "Install it with: pip install torchvision\n"
                f"Original error: {e}"
            )

        # Slice the feature extractor at each target layer
        features = vgg.features
        self.slice1 = nn.Sequential(*list(features.children())[:self._LAYER_IDS[0] + 1])
        self.slice2 = nn.Sequential(*list(features.children())[:self._LAYER_IDS[1] + 1])
        self.slice3 = nn.Sequential(*list(features.children())[:self._LAYER_IDS[2] + 1])

        # Freeze — VGG is a fixed feature extractor
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalisation constants
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        """Expects x in [0, 1]; returns ImageNet-normalised tensor."""
        return (x - self.mean) / self.std

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args
        ────
        recon  : (B, 3, H, W) in [0, 1]
        target : (B, 3, H, W) in [0, 1]

        Returns
        ───────
        Scalar perceptual loss (mean L1 across the three feature maps).
        """
        recon_n  = self._normalise(recon)
        target_n = self._normalise(target)

        loss = 0.0
        for slc in (self.slice1, self.slice2, self.slice3):
            feat_r = slc(recon_n)
            feat_t = slc(target_n)
            loss = loss + F.l1_loss(feat_r, feat_t)

        return loss / 3.0   # average across slices


# ─────────────────────────────────────────────────────────────────────────────
# Main combined loss
# ─────────────────────────────────────────────────────────────────────────────

class ClearVisionLoss(nn.Module):

    def __init__(
        self,
        lambda_l1:   float = 1.0,
        lambda_l2:   float = 0.1,
        lambda_perc: float = 0.1,
    ):
        super().__init__()
        self.lambda_l1   = lambda_l1
        self.lambda_l2   = lambda_l2
        self.lambda_perc = lambda_perc

        # Try to build the perceptual loss; gracefully fall back if torchvision
        # is absent or VGG weights can't be downloaded.
        self.perceptual: Optional[VGGPerceptualLoss] = None
        if lambda_perc > 0.0:
            try:
                self.perceptual = VGGPerceptualLoss()
                print("[ClearVisionLoss] VGG perceptual loss enabled.")
            except ImportError as e:
                print(f"[ClearVisionLoss] Perceptual loss disabled: {e}")
                self.lambda_perc = 0.0

    # ─────────────────────────────────────────
    def forward(
        self,
        recon:    torch.Tensor,
        target:   torch.Tensor,
        vq_loss:  Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        # ── Pixel losses ──────────────────────────────────────────────────
        l1_loss = F.l1_loss(recon, target)
        l2_loss = F.mse_loss(recon, target)

        # ── Perceptual loss ───────────────────────────────────────────────
        perc_loss = torch.tensor(0.0, device=recon.device)
        if self.perceptual is not None and self.lambda_perc > 0.0:
            perc_loss = self.perceptual(recon, target)

        # ── VQ commitment loss ────────────────────────────────────────────
        vq_loss_val = torch.tensor(0.0, device=recon.device)
        if vq_loss is not None:
            vq_loss_val = vq_loss

        # ── Combine ───────────────────────────────────────────────────────
        total = (
            self.lambda_l1   * l1_loss
            + self.lambda_l2   * l2_loss
            + self.lambda_perc * perc_loss
            + vq_loss_val                    # already weighted by β inside quantizer
        )

        breakdown = {
            "loss_total" : total.item(),
            "loss_l1"    : l1_loss.item(),
            "loss_l2"    : l2_loss.item(),
            "loss_perc"  : perc_loss.item(),
            "loss_vq"    : vq_loss_val.item(),
        }

        return total, breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity-check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, C, H, W = 2, 3, 128, 128

    recon  = torch.rand(B, C, H, W)
    target = torch.rand(B, C, H, W)
    vq_loss = torch.tensor(0.042)   # dummy commitment loss

    criterion = ClearVisionLoss(lambda_l1=1.0, lambda_l2=0.1, lambda_perc=0.1)
    loss, breakdown = criterion(recon, target, vq_loss)

    print("Loss breakdown:")
    for k, v in breakdown.items():
        print(f"  {k:15s}: {v:.6f}")

    loss.backward()
    print("\nBackward pass OK.")
