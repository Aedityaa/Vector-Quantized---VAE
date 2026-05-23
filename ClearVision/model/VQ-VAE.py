"""
Clear_Vision — UNetRestoration.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
U-Net restoration backbone (corrupted → clean, 128×128) with:
  • VQ-VAE quantizer hook at the bottleneck (down4 output)
  • Skip connections ONLY at the top-2 decoder levels (up3, up4)
    — x3/x4 skips are intentionally dropped so the VQ bottleneck
      is forced to encode all coarse/semantic information.
  • UpNoSkip for the two deep decoder stages (up1, up2)
  • UpWithSkip (original Up logic) for the two shallow stages (up3, up4)

Wire-up order
─────────────
Encoder : inc → down1 → down2 → down3 → down4 → [VQ hook] → z_q
Decoder : z_q → up1 → up2 → up3(+x2) → up4(+x1) → outc → sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ─────────────────────────────────────────────
# Shared building blocks
# ─────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Two consecutive Conv→BN→ReLU blocks (the base feature unit)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool2d ×2 followed by DoubleConv — one encoder step."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


# ─────────────────────────────────────────────
# Decoder blocks — two flavours
# ─────────────────────────────────────────────

class UpNoSkip(nn.Module):
    """
    Decoder step WITHOUT a skip connection.
    Used for up1 and up2 (deep levels) so that the VQ bottleneck
    is the sole source of coarse/semantic information.

    ConvTranspose2d doubles spatial size, then DoubleConv refines.
    in_ch  : channels coming from the previous decoder stage
    out_ch : desired output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.conv(x)


class UpWithSkip(nn.Module):
    """
    Decoder step WITH a skip connection from the encoder.
    Used for up3 (+x2) and up4 (+x1) — the top-2 high-res levels.

    ConvTranspose2d halves channels and doubles spatial size, then the
    skip is concatenated (channel-wise) before the DoubleConv.

    in_ch  : channels coming from the previous decoder stage
    out_ch : desired output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # Upsample and halve channels so concatenation gives in_ch total
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        # After cat: (in_ch // 2) from upsample + (in_ch // 2) from skip = in_ch
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Guard against off-by-one spatial mismatches
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)   # (B, in_ch, H, W)
        return self.conv(x)


# ─────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────

class UNetRestoration(nn.Module):
    """
    Clear_Vision encoder-decoder with:
      • VQ-VAE quantizer hook at the bottleneck
      • Skip connections only at the top-2 decoder levels

    Args
    ────
    in_channels  : input image channels (default 3 — RGB)
    out_channels : output image channels (default 3 — RGB)
    base         : base feature width; doubles each Down (default 64)
    quantizer    : optional VectorQuantizer module.  When provided,
                   forward() returns (reconstruction, vq_loss, perplexity).
                   When None, returns reconstruction only — useful for
                   plain U-Net ablation runs.
    """

    def __init__(
        self,
        in_channels:  int = 3,
        out_channels: int = 3,
        base:         int = 64,
        quantizer:    Optional[nn.Module] = None,
    ):
        super().__init__()

        # ── Encoder ──────────────────────────────
        self.inc   = DoubleConv(in_channels, base)          # 128 → 128,  3   → 64
        self.down1 = Down(base,      base * 2)              # 128 →  64,  64  → 128
        self.down2 = Down(base * 2,  base * 4)              #  64 →  32,  128 → 256
        self.down3 = Down(base * 4,  base * 8)              #  32 →  16,  256 → 512
        self.down4 = Down(base * 8,  base * 16)             #  16 →   8,  512 → 1024  ← bottleneck

        # ── Optional VQ quantizer ─────────────────
        # Attach any module with signature: (z_e) → (z_q, vq_loss, perplexity)
        self.quantizer = quantizer

        # ── Decoder ──────────────────────────────
        # up1, up2 — deep, NO skips (VQ bottleneck is the only information source)
        self.up1 = UpNoSkip(base * 16, base * 8)           #   8 →  16, 1024 → 512
        self.up2 = UpNoSkip(base * 8,  base * 4)           #  16 →  32,  512 → 256

        # up3, up4 — shallow, WITH skips from encoder x2 / x1
        self.up3 = UpWithSkip(base * 4,  base * 2)         #  32 →  64,  256 → 128  (+x2: 256ch)
        self.up4 = UpWithSkip(base * 2,  base)             #  64 → 128,  128 → 64   (+x1:  64ch)

        # ── Output head ──────────────────────────
        self.outc = nn.Conv2d(base, out_channels, kernel_size=1)

    # ─────────────────────────────────────────
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args
        ────
        x : (B, C, H, W) corrupted image tensor, values in [0, 1]

        Returns
        ───────
        recon       : (B, out_ch, H, W) restored image in [0, 1]
        vq_loss     : scalar VQ loss (commitment + codebook) — None if no quantizer
        perplexity  : codebook perplexity scalar — None if no quantizer
        """

        # ── Encoder forward ──────────────────────
        x1 = self.inc(x)        # (B,  64, 128, 128)  ← skip for up4
        x2 = self.down1(x1)     # (B, 128,  64,  64)  ← skip for up3
        x3 = self.down2(x2)     # (B, 256,  32,  32)  — NO skip (dropped)
        x4 = self.down3(x3)     # (B, 512,  16,  16)  — NO skip (dropped)
        x5 = self.down4(x4)     # (B,1024,   8,   8)  — bottleneck z_e

        # ── VQ bottleneck (optional) ──────────────
        vq_loss    = None
        perplexity = None

        if self.quantizer is not None:
            x5, vq_loss, perplexity = self.quantizer(x5)
            # x5 is now z_q — discrete, straight-through gradient applied inside quantizer

        # ── Decoder forward ──────────────────────
        d = self.up1(x5)        # (B, 512,  16,  16)  — no skip
        d = self.up2(d)         # (B, 256,  32,  32)  — no skip
        d = self.up3(d, x2)     # (B, 128,  64,  64)  — skip from x2 ✓
        d = self.up4(d, x1)     # (B,  64, 128, 128)  — skip from x1 ✓

        recon = torch.sigmoid(self.outc(d))   # (B, 3, 128, 128)  in [0,1]

        return recon, vq_loss, perplexity