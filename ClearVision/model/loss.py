"""
Clear_Vision — loss.py
~~~~~~~~~~~~~~~~~~~~~~~~
Combined loss for the VQ-VAE image restoration task.

Total loss breakdown
─────────────────────
  L_total = λ_l1   · L_recon_l1          ← pixel fidelity  (L1)
           + λ_l2   · L_recon_l2          ← pixel fidelity  (L2 / MSE)
           + λ_perc · L_perceptual        ← VGG feature similarity
           + L_vq                         ← commitment loss from quantizer

Notes
──────
  • L1  →  sharp edges, good for restoration.  The dominant recon term.
  • L2  →  smoother gradients early in training.  Small weight by default.
  • Perceptual  →  compares VGG-16 intermediate features between recon
    and target.  Catches structural/texture quality L1/L2 miss.
    Disabled automatically if torchvision is not installed.
  • VQ loss  →  commitment loss returned by VectorQuantizerEMA.forward().
    The codebook update is handled by EMA so there is NO separate
    codebook loss term here.
  • No KL divergence — VQ-VAE does not use it.

Usage
──────
  criterion = ClearVisionLoss(lambda_l1=1.0, lambda_l2=0.1, lambda_perc=0.1)

  recon, vq_loss, perplexity = model(corrupted)
  loss, breakdown = criterion(recon, clean, vq_loss)

  loss.backward()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# VGG perceptual feature extractor
# ─────────────────────────────────────────────────────────────────────────────

class VGGPerceptualLoss(nn.Module):
    """
    Computes L1 distance between VGG-16 intermediate feature maps of
    the reconstruction and the clean target.

    We use the outputs of relu1_2, relu2_2, relu3_3 — early-to-mid layers
    that capture low-level texture and mid-level structure respectively.
    Deeper layers would capture semantics which matter less for restoration.

    The VGG weights are frozen (eval mode, no grad) — they are a fixed
    feature extractor, not learned.

    ImageNet normalisation is applied internally so the model always
    receives properly normalised input regardless of what the rest of the
    pipeline does.
    """

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
    """
    Combined VQ-VAE restoration loss.

    Args
    ────
    lambda_l1   : weight for L1 reconstruction loss          (default 1.0)
    lambda_l2   : weight for L2 / MSE reconstruction loss    (default 0.1)
    lambda_perc : weight for VGG perceptual loss             (default 0.1)
                  Set to 0.0 to skip perceptual loss entirely
                  (e.g. if torchvision isn't available).
    """

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
        """
        Args
        ────
        recon   : (B, C, H, W)  model reconstruction in [0, 1]
        target  : (B, C, H, W)  clean ground-truth image in [0, 1]
        vq_loss : scalar commitment loss from VectorQuantizerEMA
                  (pass None for plain U-Net ablation runs)

        Returns
        ───────
        total_loss : scalar tensor — call .backward() on this
        breakdown  : dict of float values for logging:
                     {
                       "loss_total"  : float,
                       "loss_l1"     : float,
                       "loss_l2"     : float,
                       "loss_perc"   : float,
                       "loss_vq"     : float,
                     }
        """

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