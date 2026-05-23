# ClearVision — VQ-VAE Image Restoration

> A blind image restoration system built from scratch using a Vector-Quantized Variational Autoencoder with a U-Net decoder. Developed as part of Summer Projects 2025, Coding Club, IIT Guwahati.

---

## Overview

Image degradation — blur, compression artifacts, noise, partial occlusion — is unavoidable in real-world visual data. Classical approaches rely on handcrafted filters or strong assumptions about the noise type. ClearVision takes a different approach: train a deep generative model to learn the mapping from corrupted to clean images directly from data, with no explicit prior on the corruption distribution.

The full pipeline is built from scratch:
- A **Selenium-based scraper** to collect clean training data from Unsplash
- A **realistic degradation module** to synthetically corrupt images
- A **VQ-VAE + U-Net restoration model** trained end-to-end
- A **multi-metric evaluation suite** (PSNR, SSIM, LPIPS)

---

## Architecture

```
Corrupted Image (128×128)
        ↓
   Encoder (U-Net backbone)
   inc → down1 → down2 → down3 → down4
        ↓
   VQ Bottleneck (EMA Vector Quantizer)
   z_e → nearest codebook entry → z_q (straight-through gradient)
        ↓
   Decoder (selective skip connections)
   up1 (no skip) → up2 (no skip) → up3 (+x2) → up4 (+x1)
        ↓
   Restored Image (128×128)
```

### Key Design Choices

**VQ Bottleneck instead of standard VAE:**
Standard VAEs produce blurry reconstructions because sampling from a continuous distribution forces the decoder to average across nearby latent points. VQ-VAE snaps each encoder output to the nearest entry in a discrete codebook — giving the decoder a sharp, precise code every time.

**Selective Skip Connections:**
Only the top-2 decoder levels (up3, up4) receive skip connections from the encoder. The deep levels (up1, up2) receive nothing — forcing the VQ bottleneck to encode all coarse and semantic information, while skips only assist with fine-grained high-frequency detail at the end.

**EMA Codebook Updates:**
The codebook is not updated via backpropagation (the argmin lookup is non-differentiable). Instead, EMA (Exponential Moving Average) statistics track which codes get used and adjust embeddings smoothly. The straight-through estimator allows gradients to flow through to the encoder unchanged.

**Combined Loss:**
```
L_total = λ_l1 · L1 + λ_l2 · MSE + λ_perc · VGG_perceptual + L_vq_commitment
```
L1 preserves sharp edges, MSE stabilizes early training, perceptual loss (VGG-16 relu1_2/relu2_2/relu3_3) catches texture and structural quality that pixel losses miss.

---

## Data Pipeline

### 1. Scraping
- Selenium scraper targeting [Unsplash](https://unsplash.com)
- Collected **5,000+ high-quality images** across categories: nature, street, people, architecture, products
- Images stored as-is in `Data/Clean/`

### 2. Degradation
Inspired by [Zhang et al., ICCV 2021](https://arxiv.org/abs/2103.14006), the degradation pipeline applies a randomized sequence of:

| Corruption | Details |
|---|---|
| Downsampling | Random scale factor, bicubic/bilinear/nearest |
| Gaussian blur | Random kernel size and sigma |
| JPEG compression | Random quality factor (10–95) |
| Additive noise | Gaussian noise at varying sigma |

This creates blind restoration pairs — the model never sees a fixed noise type during evaluation, making it robust to diverse real-world corruptions. Corrupted images are stored in `Data/Corrupted/` with matching filenames to `Data/Clean/`.

---

## Results

Trained for 100 epochs on a T4 GPU (Google Colab) with batch size 16, image size 128×128.

| Metric | Achieved | Target |
|---|---|---|
| PSNR (dB) | 27.63 | 31.59 |
| SSIM | 0.877 | 0.8415 ✅ |
| LPIPS | 0.159 | 0.113 |

SSIM target was exceeded comfortably. PSNR and LPIPS gaps are primarily attributed to early-stage codebook collapse (perplexity < 10 for the first ~20 epochs due to high EMA decay γ=0.99), which limited how much information the VQ bottleneck could encode during the critical early training phase.

A follow-up run with `ema_decay=0.95` and `num_embeddings=512` is planned for 100 epochs.

---

## Training Configuration

```python
CFG = dict(
    base            = 32,      # bottleneck channels = base*16 = 512
    num_embeddings  = 256,     # codebook size K
    commitment_beta = 0.25,    # β — commitment loss weight
    ema_decay       = 0.99,    # γ — EMA decay (0.95 recommended for next run)
    lambda_l1       = 1.0,
    lambda_l2       = 0.1,
    lambda_perc     = 0.1,
    epochs          = 100,
    batch_size      = 16,
    lr              = 2e-4,
    weight_decay    = 1e-5,
    val_fraction    = 0.1,
)
```

Optimizer: AdamW | Scheduler: CosineAnnealingLR | AMP: enabled

---

## Directory Structure

```
Clear_Vision/
├── ClearVision/
│   ├── model/
│   │   ├── VQ-VAE.py          # UNetRestoration — encoder, decoder, skip logic
│   │   ├── quantizer.py       # VectorQuantizerEMA — EMA codebook, straight-through
│   │   ├── VAE.py             # Standard VAE baseline
│   │   └── loss.py            # ClearVisionLoss — L1 + L2 + perceptual + VQ
│   ├── Degrador/
│   │   └── PairImages.py      # Dataset class for Clean/Corrupted pairs
│   ├── Scraper/               # Selenium Unsplash scraper
│   └── training/              # Training loop, metrics
├── Data/
│   ├── Clean/                 # Original high-quality images
│   └── Corrupted/             # Synthetically degraded pairs
├── checkpoints/
│   ├── best.pt                # Best validation PSNR checkpoint
│   └── last.pt                # Last epoch checkpoint
├── logs/
│   ├── history.json           # Per-epoch metrics log
│   ├── training_curves.png    # Loss, PSNR, SSIM, LPIPS, perplexity plots
│   └── visual_results.png     # Side-by-side restoration samples
└── final.ipynb                # Full training notebook (Colab-ready)
```

---

## Setup & Running

### Requirements
```bash
pip install torch torchvision pillow tqdm lpips
```

### Training (Google Colab recommended)
```python
# Mount Drive, unzip project, then:
CD_TO = "/content/Clear_Vision"
# Run all cells in final.ipynb
```

### Inference (local CPU)
```python
import torch, pathlib, importlib.util
pathlib.PosixPath = pathlib.WindowsPath

# Load model from checkpoint and run on any image
# See infer.py for full inference script
```

---

## Limitations & Future Work

- **Resolution**: Currently limited to 128×128. Future work: cascade with SwinIR or RCAN for joint restoration + super-resolution.
- **Codebook collapse**: High EMA decay (γ=0.99) caused late codebook activation. Next run uses γ=0.95 and K=512.
- **Domain generalization**: Model is trained on stock photography. Performance degrades on out-of-domain inputs (e.g. personal photos, medical images).
- **No frontend**: A Streamlit interface for drag-and-drop restoration is planned.

---

## References

1. Zhang et al., [Designing a Practical Degradation Model for Deep Blind Image Super-Resolution](https://arxiv.org/abs/2103.14006), ICCV 2021
2. Kingma & Welling, [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114), ICLR 2014
3. Van den Oord et al., [Neural Discrete Representation Learning (VQ-VAE)](https://arxiv.org/abs/1711.00937), NeurIPS 2017
4. Ronneberger et al., [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), MICCAI 2015

---

**Author:** Aditya Parate | ECE, IIT Guwahati | Summer Projects 2025, Coding Club IITG
