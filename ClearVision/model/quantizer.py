"""
Clear_Vision — quantizer.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
EMA-based Vector Quantizer for the VQ-VAE bottleneck.

How it fits in the pipeline
────────────────────────────
  Encoder output  →  z_e  (B, D, H, W)   continuous
        ↓
  VectorQuantizerEMA                      ← this file
        ↓
  z_q  (B, D, H, W)   discrete (straight-through gradient)
        ↓
  Decoder

Straight-through estimator
───────────────────────────
  argmin is not differentiable, so during the backward pass we pretend
  z_q == z_e by doing:
      z_q = z_e + (z_q - z_e).detach()
  Gradients flow into the encoder through z_e; the codebook is updated
  via EMA statistics instead of gradients.

EMA update rules  (DeepMind VQ-VAE-2 style)
─────────────────────────────────────────────
  For each codebook entry i that was chosen n_i times in a batch:

      N_i  ←  γ · N_i  +  (1 - γ) · n_i          # usage count (EMA)
      m_i  ←  γ · m_i  +  (1 - γ) · Σ z_e         # sum of assigned vectors
      e_i  ←  m_i / N_i                            # updated embedding

  γ (decay) ≈ 0.99 keeps updates smooth.
  Laplace smoothing on N_i avoids division-by-zero for rarely-used codes.

Codebook collapse prevention
─────────────────────────────
  If a code hasn't been used for a while its N_i → 0.
  We add ε (1e-5) Laplace smoothing to N_i before dividing, which
  keeps dead codes from producing NaN and lets them eventually recover
  if the encoder starts sending vectors their way again.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VectorQuantizerEMA(nn.Module):
    """
    EMA-updated Vector Quantizer.

    Args
    ────
    num_embeddings  : K — codebook size (number of discrete codes)
    embedding_dim   : D — dimension of each code vector.
                      Must equal the channel count at the U-Net bottleneck
                      (base * 16 = 1024 by default).
    commitment_beta : β — weight of the commitment loss term (default 0.25).
                      Scales how hard the encoder is pushed to stay close
                      to its chosen code.
    decay           : γ — EMA decay factor (default 0.99).
    eps             : Laplace smoothing constant (default 1e-5).
    """

    def __init__(
        self,
        num_embeddings:  int   = 512,
        embedding_dim:   int   = 1024,
        commitment_beta: float = 0.25,
        decay:           float = 0.99,
        eps:             float = 1e-5,
    ):
        super().__init__()

        self.K    = num_embeddings
        self.D    = embedding_dim
        self.beta = commitment_beta
        self.gamma = decay
        self.eps   = eps

        # ── Codebook ─────────────────────────────────────────────────────
        # embedding weight — shape (K, D)
        # Initialized with a unit normal; EMA will move it to match the
        # encoder's output distribution quickly in the first few steps.
        embedding = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embedding", embedding)           # not a parameter

        # ── EMA statistics ────────────────────────────────────────────────
        # N : (K,)  — smoothed usage count per code
        # m : (K, D) — smoothed sum of encoder vectors assigned to each code
        self.register_buffer("ema_count",  torch.ones(num_embeddings))
        self.register_buffer("ema_weight", embedding.clone())

    # ─────────────────────────────────────────────────────────────────────
    def forward(
        self, z_e: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args
        ────
        z_e : (B, D, H, W)  encoder output (continuous)

        Returns
        ───────
        z_q        : (B, D, H, W)  quantized tensor (straight-through)
        vq_loss    : scalar — commitment loss  β·||z_e - sg[e]||²
                     (codebook loss is handled by EMA, not backprop)
        perplexity : scalar — measures codebook utilisation.
                     exp(H(p)) where p is the average code usage prob.
                     Maximum value = K (every code used equally).
        """
        B, D, H, W = z_e.shape
        assert D == self.D, (
            f"Encoder output has {D} channels but quantizer expects {self.D}. "
            f"Set embedding_dim=base*16 ({D}) in VectorQuantizerEMA."
        )

        # ── 1. Flatten spatial dims → (B·H·W, D) ─────────────────────────
        # Easier to compute pairwise distances against the (K, D) codebook.
        z_e_flat = z_e.permute(0, 2, 3, 1).contiguous()   # (B, H, W, D)
        z_e_flat = z_e_flat.view(-1, D)                    # (N, D),  N = B·H·W

        # ── 2. Find nearest codebook entry for every vector ───────────────
        # ||z_e - e_k||² = ||z_e||² + ||e_k||² - 2·z_e·e_k^T
        # shape: (N, K)
        distances = (
            z_e_flat.pow(2).sum(dim=1, keepdim=True)       # (N, 1)
            + self.embedding.pow(2).sum(dim=1)              # (K,)
            - 2.0 * z_e_flat @ self.embedding.t()           # (N, K)
        )

        # indices of the nearest code for each spatial position
        encoding_indices = distances.argmin(dim=1)          # (N,)

        # one-hot encodings — (N, K) — used for EMA statistics
        encodings = F.one_hot(encoding_indices, self.K).float()   # (N, K)

        # ── 3. Look up quantized vectors ──────────────────────────────────
        z_q_flat = self.embedding[encoding_indices]         # (N, D)
        z_q = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()  # (B,D,H,W)

        # ── 4. EMA codebook update (only during training) ─────────────────
        if self.training:
            with torch.no_grad():
                # n_i : how many vectors were assigned to each code this batch
                n = encodings.sum(dim=0)                    # (K,)

                # sum of encoder vectors assigned to each code
                # encodings.t() : (K, N) ;  z_e_flat : (N, D)  →  (K, D)
                sum_z_e = encodings.t() @ z_e_flat          # (K, D)

                # EMA updates
                self.ema_count  = self.gamma * self.ema_count  + (1 - self.gamma) * n
                self.ema_weight = self.gamma * self.ema_weight + (1 - self.gamma) * sum_z_e

                # Laplace-smoothed normalisation → new codebook vectors
                n_smooth = (
                    (self.ema_count + self.eps)
                    / (self.ema_count.sum() + self.K * self.eps)
                    * self.ema_count.sum()
                )
                self.embedding = self.ema_weight / n_smooth.unsqueeze(1)

        # ── 5. Straight-through estimator ─────────────────────────────────
        # Gradients pass straight through to z_e; z_q carries no gradient
        # back to the encoder on its own.
        z_q_st = z_e + (z_q - z_e).detach()

        # ── 6. Commitment loss ────────────────────────────────────────────
        # Pushes the encoder output toward the chosen codebook vector.
        # sg[e] means the codebook entry is treated as a constant here.
        commitment_loss = self.beta * F.mse_loss(z_e, z_q.detach())

        # ── 7. Perplexity ─────────────────────────────────────────────────
        # Average code usage probability across the batch
        avg_probs   = encodings.mean(dim=0)                 # (K,)
        perplexity  = torch.exp(
            -(avg_probs * (avg_probs + 1e-10).log()).sum()
        )
        # Interpretation:
        #   perplexity ≈ K  →  all codes equally used  (healthy)
        #   perplexity ≈ 1  →  only one code used       (collapsed codebook)

        return z_q_st, commitment_loss, perplexity