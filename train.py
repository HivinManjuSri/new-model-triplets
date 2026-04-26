"""
train.py
--------
Full training loop for the triplet face verification model.

Usage:
    python train.py
    python train.py --epochs 30 --batch_size 32 --lr 1e-4

Saves:
    checkpoints/best_model.pt   ← best model by val loss
    checkpoints/last_model.pt   ← latest epoch checkpoint
    checkpoints/training_log.json
"""

import os
import json
import argparse
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TripletFaceDataset
from model   import build_model
from loss    import TripletLoss


# ── Helpers ─────────────────────────────────────────────────────────────────

def auto_batch_size() -> int:
    """Pick a sensible default batch size based on available GPU VRAM."""
    if not torch.cuda.is_available():
        return 8    # CPU: keep small
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb >= 20:
        return 64
    elif vram_gb >= 10:
        return 32
    elif vram_gb >= 6:
        return 16
    elif vram_gb >= 4:
        return 8
    else:
        return 4    # very low VRAM (≤ 2 GB) — e.g. MX350


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM)")
    else:
        dev = torch.device("cpu")
        print("[Device] No GPU found, using CPU.")
    return dev


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


# ── Training / Validation passes ────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    total_loss      = 0.0
    total_pos_dist  = 0.0
    total_neg_dist  = 0.0
    total_active    = 0.0
    n_batches       = 0

    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for batch_idx, (anchor, positive, negative) in enumerate(loader):
            anchor   = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            ea = model(anchor)
            ep = model(positive)
            en = model(negative)

            loss, stats = criterion(ea, ep, en)

            if training:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping prevents exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss     += stats["loss"]
            total_pos_dist += stats["pos_dist_mean"]
            total_neg_dist += stats["neg_dist_mean"]
            total_active   += stats["fraction_active"]
            n_batches += 1

            if training and batch_idx % 50 == 0:
                print(f"  batch {batch_idx:4d}/{len(loader)}  "
                      f"loss={stats['loss']:.4f}  "
                      f"d+={stats['pos_dist_mean']:.4f}  "
                      f"d-={stats['neg_dist_mean']:.4f}  "
                      f"active={stats['fraction_active']:.2%}")

    return {
        "loss":            total_loss     / n_batches,
        "pos_dist_mean":   total_pos_dist / n_batches,
        "neg_dist_mean":   total_neg_dist / n_batches,
        "fraction_active": total_active   / n_batches,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    bs = auto_batch_size()
    p = argparse.ArgumentParser(description="Train face verification triplet model")
    p.add_argument("--epochs",      type=int,   default=30,   help="Number of training epochs")
    p.add_argument("--batch_size",  type=int,   default=bs,   help=f"Batch size (auto={bs})")
    p.add_argument("--lr",          type=float, default=1e-4, help="Learning rate")
    p.add_argument("--margin",      type=float, default=0.2,  help="Triplet loss margin")
    p.add_argument("--mining",      type=str,   default="semihard", choices=["semihard", "hard", "random"], help="Triplet mining strategy")
    p.add_argument("--embedding",   type=int,   default=128,  help="Embedding dimension")
    p.add_argument("--workers",     type=int,   default=4,    help="DataLoader worker count")
    p.add_argument("--no_pretrain", action="store_true",      help="Train ResNet-50 from scratch")
    p.add_argument("--resume",      type=str,   default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


def main():
    args   = parse_args()
    device = get_device()

    print("\n=== Configuration ===")
    for k, v in vars(args).items():
        print(f"  {k:15s}: {v}")
    print()

    # ── Datasets & Loaders ────────────────────────────────────────────────
    train_ds = TripletFaceDataset(split="train", augment=True)
    val_ds   = TripletFaceDataset(split="val",   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True,  num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(
        embedding_dim=args.embedding,
        pretrained=not args.no_pretrain,
    ).to(device)

    # ── Loss & Optimiser ──────────────────────────────────────────────────
    criterion = TripletLoss(margin=args.margin, mining=args.mining)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Cosine annealing: LR decays from lr → 0 over all epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    start_epoch = 1
    best_val_loss = float("inf")
    log = []

    # ── Resume from checkpoint ────────────────────────────────────────────
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        log           = ckpt.get("log", [])
        print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # ── Training Loop ─────────────────────────────────────────────────────
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        current_lr  = scheduler.get_last_lr()[0]
        print(f"\n[Epoch {epoch}/{args.epochs}]  lr={current_lr:.2e}")

        train_stats = run_epoch(model, train_loader, criterion,
                                optimizer, device, training=True)
        val_stats   = run_epoch(model, val_loader,   criterion,
                                None,      device, training=False)
        scheduler.step()

        elapsed = time.time() - epoch_start
        print(f"  [Train] loss={train_stats['loss']:.4f}  "
              f"d+={train_stats['pos_dist_mean']:.4f}  "
              f"d-={train_stats['neg_dist_mean']:.4f}  "
              f"active={train_stats['fraction_active']:.2%}")
        print(f"  [Val  ] loss={val_stats['loss']:.4f}  "
              f"d+={val_stats['pos_dist_mean']:.4f}  "
              f"d-={val_stats['neg_dist_mean']:.4f}  "
              f"active={val_stats['fraction_active']:.2%}")
        print(f"  Epoch time: {elapsed:.1f}s")

        # Checkpoint state
        state = {
            "epoch":         epoch,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "args":          vars(args),
            "log":           log,
        }

        # Save last
        save_checkpoint(state, os.path.join(ckpt_dir, "last_model.pt"))

        # Save best
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            state["best_val_loss"] = best_val_loss
            save_checkpoint(state, os.path.join(ckpt_dir, "best_model.pt"))
            print(f"  *** New best val loss: {best_val_loss:.4f} — checkpoint saved ***")

        # Log entry
        log.append({
            "epoch":   epoch,
            "lr":      current_lr,
            "train":   train_stats,
            "val":     val_stats,
            "elapsed": elapsed,
        })
        with open(os.path.join(ckpt_dir, "training_log.json"), "w") as f:
            json.dump(log, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {os.path.join(ckpt_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()
