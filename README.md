# ClearVision: A VAE-Based Framework for Blind Image Restoration

**ClearVision** is a deep generative image restoration system developed as part of the **Summer Projects 2025** under the Coding Club of IIT Guwahati. The project addresses the challenge of restoring high-quality images from visually degraded inputs using a custom corruption pipeline and a VAE-based restoration model.

---

## Problem Overview

Image degradation due to blur, compression, noise, or partial occlusion is common in real-world visual data. Traditional image enhancement techniques often rely on handcrafted filters or explicit assumptions about noise. This project explores the use of **deep generative models** to **learn a mapping from corrupted to clean images**, without strong priors on the corruption type or distribution.

The system is built from scratch and includes:
- a custom image scraper,
- a realistic degradation pipeline based on literature,
- and a variational autoencoder with a vector quantization bottleneck and U-Net decoder.

---

##  Key References

- Zhang et al., [Designing a Practical Degradation Model for Deep Blind Image Super-Resolution](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Designing_a_Practical_Degradation_Model_for_Deep_Blind_Image_Super-Resolution_ICCV_2021_paper.html), *ICCV 2021*.
- Kingma and Welling, [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114), *ICLR 2014*.
- Van den Oord et al., [Neural Discrete Representation Learning (VQ-VAE)](https://arxiv.org/abs/1711.00937), *NeurIPS 2017*.
- Ronneberger et al., [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), *MICCAI 2015*.
---

##  Project Components

### 1. Image Scraper
- Built using **Selenium** to scrape high-quality images from [Unsplash](https://unsplash.com).
- Collected a diverse dataset of **5,000+ images** across multiple categories (nature, street, people, etc.).

### 2. Degradation Module
- Inspired by Zhang et al. (ICCV 2021), simulates realistic corruptions using:
  - **Downsampling**
  - **Gaussian blur**
  - **JPEG compression**
  - **Additive noise**
- Enables blind restoration training (model never sees clean-degraded pairs during evaluation).

### 3. Restoration Model
- A **VAE-based architecture** with:
  - **Encoder** to compress degraded images into latent codes.
  - **EMA-based Vector Quantizer** (from VQ-VAE).
  - **U-Net decoder** to reconstruct high-fidelity outputs.
- Trained on 128×128 image patches for resource efficiency.

### 4. Evaluation Metrics
- Implemented the following image restoration quality metrics:
  - **PSNR** (Peak Signal-to-Noise Ratio)
  - **SSIM** (Structural Similarity Index)
  - **LPIPS** (Learned Perceptual Image Patch Similarity)

---

##  Results

| Metric | Value   |
|--------|---------|
| PSNR   | 31.59   |
| SSIM   | 0.8415  |
| LPIPS  | 0.113   |

- Output resolution: **128×128**  
- Results indicate **structural fidelity** and **perceptual closeness** under aggressive corruption conditions.

---

## 📁 Directory Structure

<pre>
ClearVision/
│
├── scraper/             # Selenium-based image scraping scripts
├── Degrador/            # Custom image degradation pipeline
├── data/                # Raw and corrupted datasets
│   ├── Clean/      # Original high-quality images
│   └── Corrupted/        # Processed low-quality pairs  
clearvisionnb.ipynb  # Model training notebook
README.md            # Project documentation
</pre>


## Limitations and Future Work

- 🔲 **Upscaling capability** is limited to 128×128. Future versions may incorporate:
  - **SRCNN**, **RCAN**, or **SwinIR** to upscale restored outputs.
  - Cascaded pipeline for joint restoration + super-resolution.
- 🔲 **Interface not included**. A basic **Streamlit** or **Flutter** frontend could enable:
  - Uploading degraded images
  - Visualizing restorations in real time
 
Project - Aditya Parate
