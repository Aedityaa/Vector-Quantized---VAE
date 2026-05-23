import torch
from pathlib import Path
import torch
import pathlib

# Temporarily patch PosixPath to WindowsPath
pathlib.PosixPath = pathlib.WindowsPath

ckpt = torch.load(r"D:\IIMA_Show\Clear_Vision_backup\checkpoints\best.pt", map_location="cpu",weights_only=False)
print(ckpt.keys())
print("Saved at epoch:", ckpt['epoch'])
print("Best PSNR:", ckpt['best_psnr'])