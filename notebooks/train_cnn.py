"""
train_cnn.py - CNN trainer for ForensiX AI
==========================================
Trains a binary forgery-detection CNN and saves the model to
modules/image/model.pt, which detector.py loads for cnn_score.

Label convention (saved in checkpoint): 0 = AUTHENTIC, 1 = FORGED

Dataset layouts supported:
  CIFAKE  ->  root/train/{REAL,FAKE}  +  root/test/{REAL,FAKE}
  CASIA   ->  root/{Au,Tp}            (80/20 random split)

Models:
  fast (default) - compact 4-block CNN, trains in ~10-15 min on CPU
  resnet18        - ResNet-18, ImageNet-pretrained, 224px (recommended)
  efficientnet_b0 - EfficientNet-B0, ImageNet-pretrained, 224px

Usage:
  python notebooks/train_cnn.py --data data/cifake --epochs 20 --batch 32
  python notebooks/train_cnn.py --data data/cifake,data/casia --epochs 20 --batch 32
  python notebooks/train_cnn.py --arch resnet18 --img-size 224 --subsample 0 --data data/cifake,data/casia

Requires:
  pip install torch torchvision tqdm
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, ConcatDataset, Subset
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
    import torchvision.models as models
    from tqdm import tqdm
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Install with: pip install torch torchvision tqdm")
    sys.exit(1)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Transforms ────────────────────────────────────────────────────────────────

def make_transforms(img_size: int):
    train_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(p=0.1),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])
    val_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


# ── Models ────────────────────────────────────────────────────────────────────

class FastCNN(nn.Module):
    """
    Compact CNN designed for 32-64px images (e.g. CIFAKE).
    ~300k parameters, trains in minutes on CPU.
    """
    def __init__(self, img_size: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 — 32->16
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2 — 16->8
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3 — 8->4
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 4 — 4->2
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        spatial = img_size // 16  # after 4 poolings
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * spatial * spatial, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_efficientnet() -> nn.Module:
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 2),
    )
    return m


def build_resnet18() -> nn.Module:
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Replace final FC with Dropout + Linear to reduce overfitting
    m.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(512, 2),
    )
    return m


# Architectures that expect 224x224 ImageNet-style input
BIG_ARCHS = {"resnet18", "efficientnet_b0", "efficientnet"}


def build_model(model_name: str, img_size: int) -> nn.Module:
    if model_name == "fast":
        return FastCNN(img_size)
    if model_name in ("efficientnet", "efficientnet_b0"):
        return build_efficientnet()
    if model_name == "resnet18":
        return build_resnet18()
    raise ValueError(
        f"Unknown model: {model_name}. "
        f"Choose 'fast', 'resnet18', or 'efficientnet_b0'."
    )


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _remap_cifake(ds: ImageFolder) -> ImageFolder:
    """ImageFolder sorts alphabetically: FAKE->0, REAL->1. Flip so FORGED=1."""
    if ("FAKE" in ds.class_to_idx and ds.class_to_idx["FAKE"] == 0
            and "REAL" in ds.class_to_idx):
        ds.targets = [1 - t for t in ds.targets]
        ds.samples = [(p, 1 - l) for p, l in ds.samples]
    return ds


def _has_images(folder: Path) -> bool:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    return any(f.suffix.lower() in exts for f in folder.rglob("*") if f.is_file())


def _subsample(ds, max_per_class: int):
    """Cap each class to max_per_class samples, chosen at random (seed=42)."""
    if max_per_class <= 0:
        return ds
    g = torch.Generator().manual_seed(42)
    from collections import defaultdict
    buckets = defaultdict(list)
    # Resolve per-sample targets, supporting Subset wrappers
    if hasattr(ds, "targets"):
        targets = ds.targets
    elif hasattr(ds, "samples"):
        targets = [s[1] for s in ds.samples]
    elif isinstance(ds, Subset):
        base = ds.dataset
        base_targets = (base.targets if hasattr(base, "targets")
                        else [s[1] for s in base.samples])
        targets = [base_targets[i] for i in ds.indices]
    else:
        targets = [ds[i][1] for i in range(len(ds))]
    for i, t in enumerate(targets):
        buckets[t].append(i)
    kept = []
    for cls_idx in sorted(buckets):
        idxs = buckets[cls_idx]
        perm = torch.randperm(len(idxs), generator=g).tolist()
        kept += [idxs[p] for p in perm[:max_per_class]]
    return Subset(ds, kept)


def load_dataset(root: Path, train_tf, val_tf, max_per_class: int = 0):
    """Return (train_ds, val_ds) or None if root has no usable images."""
    train_dir = root / "train"
    test_dir  = root / "test"

    if train_dir.is_dir() and test_dir.is_dir():
        try:
            train_ds = _remap_cifake(ImageFolder(str(train_dir), transform=train_tf))
            val_ds   = _remap_cifake(ImageFolder(str(test_dir),  transform=val_tf))
        except Exception as e:
            print(f"  [{root.name}] WARNING: {e} — skipping")
            return None
        if len(train_ds) == 0:
            print(f"  [{root.name}] WARNING: no images found — skipping")
            return None
        if max_per_class > 0:
            train_ds = _subsample(train_ds, max_per_class)
            val_ds   = _subsample(val_ds,   max_per_class // 5)
        classes = (train_ds.dataset.classes
                   if hasattr(train_ds, "dataset") else train_ds.classes)
        print(f"  [{root.name}]  train {len(train_ds):,}  val {len(val_ds):,}"
              f"  (pre-split,  classes: {classes})")
        return train_ds, val_ds

    # Flat layout (CASIA: Au/, Tp/)
    class_dirs = [d for d in root.iterdir() if d.is_dir()] if root.is_dir() else []
    if not class_dirs or not any(_has_images(d) for d in class_dirs):
        print(f"  [{root.name}] WARNING: no images — skipping"
              f"  (place images in {root}/Au/ and {root}/Tp/)")
        return None

    try:
        base_t = ImageFolder(str(root), transform=train_tf)
        base_v = ImageFolder(str(root), transform=val_tf)
    except Exception as e:
        print(f"  [{root.name}] WARNING: {e} — skipping")
        return None

    n = len(base_t)
    g = torch.Generator().manual_seed(42)
    idx = torch.randperm(n, generator=g).tolist()
    n_train = int(0.8 * n)
    train_ds = Subset(base_t, idx[:n_train])
    val_ds   = Subset(base_v, idx[n_train:])
    if max_per_class > 0:
        train_ds = _subsample(train_ds, max_per_class)
        val_ds   = _subsample(val_ds,   max_per_class // 5)
    print(f"  [{root.name}]  train {len(train_ds):,}  val {len(val_ds):,}"
          f"  (80/20 split,  classes: {base_t.classes})")
    return train_ds, val_ds


# ── Train / eval ──────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    desc = "  train" if training else "  val  "
    with ctx:
        for imgs, labels in tqdm(loader, desc=desc, leave=False, ncols=80):
            imgs, labels = imgs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
    return total_loss / total, correct / total


# ── Checkpoint / metrics helpers ──────────────────────────────────────────────

def save_full_checkpoint(path: Path, epoch: int, model, optimizer, scheduler,
                         best_val_acc: float, best_epoch: int,
                         history: List[Dict], arch: str, img_size: int) -> None:
    """Full state for resuming (model + optimizer + scheduler + bookkeeping)."""
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_acc":         best_val_acc,
        "best_epoch":           best_epoch,
        "history":              history,
        "architecture":         arch,
        "img_size":             img_size,
        "label_convention":     {"0": "AUTHENTIC", "1": "FORGED"},
        "imagenet_norm":        {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    }, path)


def save_best_checkpoint(path: Path, epoch: int, model, val_acc: float,
                         arch: str, img_size: int) -> None:
    """Slim inference checkpoint that detector.py loads (best val_acc only)."""
    torch.save({
        "epoch":            epoch,
        "model_state_dict": model.state_dict(),
        "val_acc":          val_acc,
        "architecture":     arch,
        "img_size":         img_size,
        "label_convention": {"0": "AUTHENTIC", "1": "FORGED"},
        "imagenet_norm":    {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    }, path)


def write_metrics(metrics_path: Path, arch: str, img_size: int,
                  best_epoch: int, best_val_acc: float,
                  history: List[Dict]) -> None:
    """Always-current training metrics, written every epoch and on crash."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({
            "arch":         arch,
            "img_size":     img_size,
            "best_epoch":   best_epoch,
            "best_val_acc": best_val_acc,
            "history":      history,
        }, f, indent=2)


def find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """Newest resumable checkpoint (model_last.pt or model_epoch_*.pt) by mtime."""
    candidates = (list(ckpt_dir.glob("model_last.pt")) +
                  list(ckpt_dir.glob("model_epoch_*.pt")))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 0 if sys.platform == "win32" else 2

    # Auto-resolve input size: 224 for ImageNet-style backbones, else 32.
    if args.img_size is None:
        args.img_size = 224 if args.arch in BIG_ARCHS else 32
    train_tf, val_tf = make_transforms(args.img_size)

    subsample_note = f"{args.subsample}/class" if args.subsample > 0 else "all (no cap)"
    print(f"\nForensiX CNN Trainer")
    print(f"  Device:      {device}")
    print(f"  Arch:        {args.arch}")
    print(f"  Image size:  {args.img_size}x{args.img_size}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch}")
    print(f"  LR:          {args.lr}")
    print(f"  Subsample:   {subsample_note}")
    print(f"  Resume:      {args.resume}")
    print(f"  Output:      {args.out}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("\nLoading datasets:")
    train_sets, val_sets = [], []
    for path_str in args.data.split(","):
        root = (ROOT / path_str.strip()).resolve()
        if not root.is_dir():
            print(f"  [{path_str.strip()}] WARNING: directory not found — skipping")
            continue
        result = load_dataset(root, train_tf, val_tf, args.subsample)
        if result:
            train_sets.append(result[0])
            val_sets.append(result[1])

    if not train_sets:
        print("\n[ERROR] No usable datasets found.")
        sys.exit(1)

    train_ds = ConcatDataset(train_sets)
    val_ds   = ConcatDataset(val_sets)
    print(f"\n  Total  train {len(train_ds):,}  |  val {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=num_workers,
                              pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=num_workers,
                              pin_memory=(device.type == "cuda"))

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nBuilding model ({args.arch})...")
    model    = build_model(args.arch, args.img_size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    out_path     = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_path    = out_path.parent / "model_last.pt"
    metrics_path = (ROOT / "outputs" / "training_metrics.json").resolve()

    best_val_acc = 0.0
    best_epoch   = 0
    history: List[Dict] = []
    start_epoch  = 1

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume:
        ckpt_path = find_latest_checkpoint(out_path.parent)
        if ckpt_path is not None:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model_state_dict"])
            if "optimizer_state_dict" in ck:
                optimizer.load_state_dict(ck["optimizer_state_dict"])
            if "scheduler_state_dict" in ck:
                scheduler.load_state_dict(ck["scheduler_state_dict"])
            best_val_acc = ck.get("best_val_acc", ck.get("val_acc", 0.0))
            best_epoch   = ck.get("best_epoch", ck.get("epoch", 0))
            history      = ck.get("history", [])
            start_epoch  = ck.get("epoch", 0) + 1
            print(f"\nResumed from {ckpt_path.name} "
                  f"-> continuing at epoch {start_epoch} "
                  f"(best so far {best_val_acc:.2%} @ epoch {best_epoch})")
        else:
            print("\n[--resume] No checkpoint found — starting fresh.")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining  (best checkpoint -> {out_path})\n")
    print(f"  {'Ep':>3}  {'Tr Loss':>8}  {'Tr Acc':>7}  "
          f"{'Va Loss':>8}  {'Va Acc':>7}  {'LR':>8}")
    print(f"  {'-'*3}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*8}")

    epoch = start_epoch
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer,
                                        device, training=True)
            va_loss, va_acc = run_epoch(model, val_loader,   criterion, optimizer,
                                        device, training=False)
            scheduler.step()

            lr      = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            is_best = va_acc > best_val_acc
            marker  = " *" if is_best else ""

            print(f"  {epoch:>3}  {tr_loss:>8.4f}  {tr_acc:>6.2%}  "
                  f"{va_loss:>8.4f}  {va_acc:>6.2%}  {lr:>8.2e}  ({elapsed:.0f}s){marker}",
                  flush=True)

            history.append({
                "epoch":      epoch,
                "train_loss": round(tr_loss, 6),
                "train_acc":  round(tr_acc, 6),
                "val_loss":   round(va_loss, 6),
                "val_acc":    round(va_acc, 6),
                "lr":         lr,
            })

            if is_best:
                best_val_acc = va_acc
                best_epoch   = epoch
                save_best_checkpoint(out_path, epoch, model, va_acc,
                                     args.arch, args.img_size)

            # Rolling full-state checkpoint + metrics (every epoch)
            save_full_checkpoint(last_path, epoch, model, optimizer, scheduler,
                                 best_val_acc, best_epoch, history,
                                 args.arch, args.img_size)
            write_metrics(metrics_path, args.arch, args.img_size,
                          best_epoch, best_val_acc, history)

    except RuntimeError as e:
        if "CUDA" not in str(e):
            raise
        # CUDA crash: save resume checkpoint and exit cleanly
        crash_path = out_path.parent / f"model_epoch_{epoch}.pt"
        save_full_checkpoint(crash_path, max(epoch - 1, 0), model, optimizer,
                             scheduler, best_val_acc, best_epoch, history,
                             args.arch, args.img_size)
        write_metrics(metrics_path, args.arch, args.img_size,
                      best_epoch, best_val_acc, history)
        print(f"\n[CUDA ERROR] Training crashed during epoch {epoch}.")
        print(f"  Error:            {e}")
        print(f"  Checkpoint saved: {crash_path}")
        print(f"  Metrics saved:    {metrics_path}")
        print(f"  Best so far:      {best_val_acc:.2%} (epoch {best_epoch}) -> {out_path}")
        print(f"\n  Resume with:")
        print(f"    python notebooks/train_cnn.py --resume --arch {args.arch} "
              f"--img-size {args.img_size} --epochs {args.epochs} "
              f"--batch {args.batch} --lr {args.lr} --subsample {args.subsample} "
              f"--data {args.data}")
        sys.exit(1)

    write_metrics(metrics_path, args.arch, args.img_size,
                  best_epoch, best_val_acc, history)
    print(f"\nBest val acc: {best_val_acc:.2%} (epoch {best_epoch})")
    print(f"Model saved:  {out_path}")
    print(f"Metrics saved: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ForensiX CNN trainer")
    parser.add_argument("--data", required=True,
                        help="Comma-separated data roots (e.g. data/cifake,data/casia)")
    parser.add_argument("--epochs",   type=int,   default=20)
    parser.add_argument("--batch",    type=int,   default=32)
    parser.add_argument("--arch", "--model", type=str, default="fast", dest="arch",
                        choices=["fast", "resnet18", "efficientnet_b0", "efficientnet"],
                        help="'fast' = compact CPU-friendly CNN; "
                             "'resnet18'/'efficientnet_b0' = ImageNet-pretrained "
                             "backbones (224px, GPU recommended). "
                             "'--model' is accepted as an alias of '--arch'.")
    parser.add_argument("--img-size", type=int,   default=None, dest="img_size",
                        help="Input image size in pixels. Default: 224 for "
                             "resnet18/efficientnet_b0, 32 otherwise.")
    parser.add_argument("--lr",       type=float, default=1e-4,
                        help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--subsample", type=int,  default=5000,
                        help="Max images per class for training (0 = use all). "
                             "Default 5000/class. Use 0 for full dataset.")
    parser.add_argument("--resume",   action="store_true",
                        help="Resume from the latest checkpoint "
                             "(model_last.pt / model_epoch_*.pt) and continue.")
    parser.add_argument("--out",      type=str,   default="modules/image/model.pt",
                        help="Output path for best (inference) checkpoint")
    main(parser.parse_args())