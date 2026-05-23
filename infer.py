import torch
import pathlib
pathlib.PosixPath = pathlib.WindowsPath

import importlib.util
import sys
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# ── Load modules ───────────────────────────────────────
model_dir = Path(r"D:\IIMA_Show\Clear_Vision_backup\ClearVision\model")

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_vqvae = load_module("vqvae_model", model_dir / "VQ-VAE.py")
_quant = load_module("quantizer_model", model_dir / "quantizer.py")

UNetRestoration = _vqvae.UNetRestoration
VectorQuantizerEMA = _quant.VectorQuantizerEMA

# ── Load checkpoint ─────────────────────────────────────
ckpt = torch.load(r"D:\IIMA_Show\Clear_Vision_backup\checkpoints\best.pt",
                  map_location="cpu", weights_only=False)

cfg = ckpt['cfg']
print("CFG:", cfg)

import torch
import pathlib
pathlib.PosixPath = pathlib.WindowsPath

import importlib.util
import sys
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# ── Load modules ───────────────────────────────────────
model_dir = Path(r"D:\IIMA_Show\Clear_Vision_backup\ClearVision\model")

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_vqvae = load_module("vqvae_model", model_dir / "VQ-VAE.py")
_quant = load_module("quantizer_model", model_dir / "quantizer.py")

UNetRestoration = _vqvae.UNetRestoration
VectorQuantizerEMA = _quant.VectorQuantizerEMA

# ── Load checkpoint ─────────────────────────────────────
ckpt = torch.load(r"D:\IIMA_Show\Clear_Vision_backup\checkpoints\best.pt",
                  map_location="cpu", weights_only=False)
cfg = ckpt['cfg']

# ── Build model ─────────────────────────────────────────
quantizer = VectorQuantizerEMA(
    num_embeddings  = cfg['num_embeddings'],
    embedding_dim   = cfg['base'] * 16,
    commitment_beta = cfg['commitment_beta'],
    decay           = cfg['ema_decay'],
)
model = UNetRestoration(
    in_channels  = 3,
    out_channels = 3,
    base         = cfg['base'],
    quantizer    = quantizer,
)
model.load_state_dict(ckpt['model'])
model.eval()
print("Model loaded!")

# ── Inference function ──────────────────────────────────
transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
])

def restore(image_path: str, output_path: str):
    img = Image.open(image_path).convert("RGB")
    original_size = img.size  # save for resizing back
    
    x = transform(img).unsqueeze(0)  # (1, 3, 128, 128)
    
    with torch.no_grad():
        recon, _, _ = model(x)
    
    recon = recon.squeeze(0)  # (3, 128, 128)
    recon_img = T.ToPILImage()(recon)
    recon_img = recon_img.resize(original_size, Image.LANCZOS)  # resize back to original
    recon_img.save(output_path)
    print(f"Saved → {output_path}")

# ── Run on your image ───────────────────────────────────
restore(
    r"D:\IIMA_Show\Clear_Vision_backup\Tanishq_1.jpeg",
    r"D:\IIMA_Show\Clear_Vision_backup\Tanishq_1_restored.jpg"
)