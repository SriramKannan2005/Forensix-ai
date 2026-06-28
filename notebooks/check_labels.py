"""
check_labels.py — Diagnose label inversion in trained model.pt
==============================================================
Run this to confirm whether the model has inverted labels.

Usage: python notebooks/check_labels.py

It tests the model on 10 REAL and 10 FAKE images from the test set
and prints the raw scores. If REAL images score ~0.99 and FAKE score ~0.01,
the labels are inverted and cnn_classifier.py needs the inversion fix.
"""

import sys
from pathlib import Path

sys.path.insert(0, '.')

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = Path("modules/image/model.pt")
CIFAKE_TEST  = Path("data/cifake/test")
IMG_SIZE     = 224
N_SAMPLES    = 10   # how many images to test per class
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def load_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 1),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model


def score_folder(model, folder: Path, n: int) -> list:
    exts = {".jpg", ".jpeg", ".png"}
    files = [f for f in folder.rglob("*") if f.suffix.lower() in exts][:n]
    scores = []
    with torch.no_grad():
        for f in files:
            img    = Image.open(f).convert("RGB")
            tensor = tf(img).unsqueeze(0).to(DEVICE)
            logit  = model(tensor)
            prob   = torch.sigmoid(logit).item()
            scores.append(round(prob, 4))
    return scores


def main():
    print(f"\n{'='*55}")
    print("  CNN Label Diagnosis")
    print(f"{'='*55}")

    if not MODEL_PATH.exists():
        print(f"ERROR: {MODEL_PATH} not found.")
        sys.exit(1)

    # Find REAL and FAKE test folders — handle case variations
    real_dir = fake_dir = None
    if CIFAKE_TEST.exists():
        for d in CIFAKE_TEST.iterdir():
            if d.is_dir():
                name = d.name.upper()
                if "REAL" in name:
                    real_dir = d
                elif "FAKE" in name:
                    fake_dir = d

    if not real_dir or not fake_dir:
        print(f"ERROR: Cannot find REAL/FAKE subdirs in {CIFAKE_TEST}")
        print(f"  Found: {[d.name for d in CIFAKE_TEST.iterdir() if d.is_dir()]}")
        sys.exit(1)

    print(f"\n  REAL folder: {real_dir}")
    print(f"  FAKE folder: {fake_dir}")

    model = load_model()
    print(f"\n  Scoring {N_SAMPLES} REAL images (expected: score near 0.0 if labels correct):")
    real_scores = score_folder(model, real_dir, N_SAMPLES)
    real_mean   = sum(real_scores) / len(real_scores)
    print(f"  Scores: {real_scores}")
    print(f"  Mean:   {real_mean:.4f}")

    print(f"\n  Scoring {N_SAMPLES} FAKE images (expected: score near 1.0 if labels correct):")
    fake_scores = score_folder(model, fake_dir, N_SAMPLES)
    fake_mean   = sum(fake_scores) / len(fake_scores)
    print(f"  Scores: {fake_scores}")
    print(f"  Mean:   {fake_mean:.4f}")

    print(f"\n{'='*55}")
    if fake_mean > real_mean:
        print("  LABELS ARE CORRECT ✓")
        print(f"  FAKE mean ({fake_mean:.4f}) > REAL mean ({real_mean:.4f})")
        print("  No fix needed.")
    else:
        print("  LABELS ARE INVERTED ✗")
        print(f"  REAL mean ({real_mean:.4f}) > FAKE mean ({fake_mean:.4f})")
        print()
        print("  Fix: cnn_classifier.py line 112 — change:")
        print("    prob = torch.sigmoid(logit).item()")
        print("  to:")
        print("    prob = 1.0 - torch.sigmoid(logit).item()")
        print()
        print("  This flips the score without retraining.")
        print("  Running fix automatically...")
        apply_inversion_fix()

    print(f"{'='*55}\n")


def apply_inversion_fix():
    """Patch cnn_classifier.py to flip the sigmoid output."""
    path = Path("modules/image/cnn_classifier.py")
    src  = path.read_text()

    old = "prob  = torch.sigmoid(logit).item()   # probability of FAKE"
    new = "prob  = 1.0 - torch.sigmoid(logit).item()   # inverted: model trained with flipped labels"

    if old in src:
        path.write_text(src.replace(old, new))
        print("  ✓ Fix applied to modules/image/cnn_classifier.py")
        print("  Verify with:")
        print("    python -c \"import sys; sys.path.insert(0,'.'); from modules.image.cnn_classifier import cnn_predict; print(cnn_predict('path/to/ai_image.png'))\"")
    elif new in src:
        print("  Fix already applied — cnn_classifier.py already has inverted output.")
    else:
        print("  WARNING: Could not find exact line to patch.")
        print("  Manually change line 112 in cnn_classifier.py:")
        print("    FROM: prob = torch.sigmoid(logit).item()")
        print("    TO:   prob = 1.0 - torch.sigmoid(logit).item()")


if __name__ == "__main__":
    main()
