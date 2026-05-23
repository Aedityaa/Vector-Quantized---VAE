"""
Train U-Net restoration on paired Clean / Corrupted folders.

Run from Clear_Vision/:
    python ClearVision/train_restoration.py --epochs 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Allow imports when run as a script from Clear_Vision/
_CLEARVISION_DIR = Path(__file__).resolve().parent
if str(_CLEARVISION_DIR) not in sys.path:
    sys.path.insert(0, str(_CLEARVISION_DIR))

from Degrador.PairImages import PairImages
from model.UNetRestoration import UNetRestoration
from training.losses import RestorationLoss
from training.metrics import compute_batch_metrics


def parse_args():
    root = _CLEARVISION_DIR.parent
    p = argparse.ArgumentParser(description="Train ClearVision restoration U-Net")
    p.add_argument("--clean-dir", type=Path, default=root / "Data" / "Clean")
    p.add_argument("--corrupt-dir", type=Path, default=root / "Data" / "Corrupted")
    p.add_argument("--checkpoint-dir", type=Path, default=root / "checkpoints")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-lpips-loss", action="store_true", help="Disable LPIPS in training loss")
    p.add_argument("--resume", type=Path, default=None, help="Checkpoint .pt to resume")
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running = {"total": 0.0, "l1": 0.0, "ssim": 0.0}
    n = 0
    for degraded, clean in tqdm(loader, desc="train", leave=False):
        degraded = degraded.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            pred = model(degraded)
            loss, parts = criterion(pred, clean)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = degraded.size(0)
        n += bs
        for k in running:
            if k in parts:
                running[k] += parts[k] * bs
    return {k: v / max(n, 1) for k, v in running.items()}


@torch.no_grad()
def validate(model, loader, criterion, device, lpips_eval):
    model.eval()
    running_loss = 0.0
    n = 0
    psnr_sum = ssim_sum = lpips_sum = 0.0
    lpips_count = 0

    for degraded, clean in tqdm(loader, desc="val", leave=False):
        degraded = degraded.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        pred = model(degraded)
        loss, _ = criterion(pred, clean)
        bs = degraded.size(0)
        running_loss += loss.item() * bs
        n += bs

        m = compute_batch_metrics(pred, clean, lpips_model=lpips_eval)
        psnr_sum += m["psnr"] * bs
        ssim_sum += m["ssim"] * bs
        if "lpips" in m:
            lpips_sum += m["lpips"] * bs
            lpips_count += bs

    out = {
        "loss": running_loss / max(n, 1),
        "psnr": psnr_sum / max(n, 1),
        "ssim": ssim_sum / max(n, 1),
    }
    if lpips_count:
        out["lpips"] = lpips_sum / lpips_count
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Clean: {args.clean_dir}")
    print(f"Corrupt: {args.corrupt_dir}")

    dataset = PairImages(str(args.clean_dir), str(args.corrupt_dir))
    print(f"Paired images: {len(dataset)}")

    n_val = max(1, int(len(dataset) * args.val_fraction))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = UNetRestoration().to(device)
    criterion = RestorationLoss(use_lpips=not args.no_lpips_loss).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    lpips_eval = None
    if not args.no_lpips_loss and criterion._lpips is not None:
        lpips_eval = criterion._lpips

    start_epoch = 1
    best_psnr = -1.0
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.checkpoint_dir / "history.json"
    history: list[dict] = []

    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", -1.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_stats = validate(model, val_loader, criterion, device, lpips_eval)
        scheduler.step()

        row = {"epoch": epoch, "train": train_stats, "val": val_stats, "lr": scheduler.get_last_lr()[0]}
        history.append(row)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        lpips_str = f"  LPIPS {val_stats['lpips']:.4f}" if "lpips" in val_stats else ""
        print(
            f"Epoch {epoch}/{args.epochs}  "
            f"train_loss {train_stats['total']:.4f}  "
            f"val_loss {val_stats['loss']:.4f}  "
            f"PSNR {val_stats['psnr']:.2f}  SSIM {val_stats['ssim']:.4f}{lpips_str}"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_psnr": best_psnr,
            "val": val_stats,
        }
        torch.save(ckpt, args.checkpoint_dir / "last.pt")

        if val_stats["psnr"] > best_psnr:
            best_psnr = val_stats["psnr"]
            ckpt["best_psnr"] = best_psnr
            torch.save(ckpt, args.checkpoint_dir / "best.pt")
            print(f"  -> new best PSNR {best_psnr:.2f}, saved best.pt")

    print("\nDone.")
    print(f"Best validation PSNR: {best_psnr:.2f} dB")
    if history:
        last = history[-1]["val"]
        print(
            f"Final val — PSNR {last['psnr']:.2f}, SSIM {last['ssim']:.4f}"
            + (f", LPIPS {last['lpips']:.4f}" if "lpips" in last else "")
        )
    print(f"Checkpoints: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
