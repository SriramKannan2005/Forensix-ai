"""
cnn_classifier.py — Deep-Learning Forgery Classifier for ForensiX
------------------------------------------------------------------
Loads the trained CNN weights from modules/image/model.pt and runs a single
image through it to produce a probability that the image is FORGED / AI-generated.

IMPORTANT — architecture is read from the checkpoint, not assumed
----------------------------------------------------------------
The checkpoint written by notebooks/train_cnn.py is a dict:
    {
        "epoch", "val_acc",
        "architecture":     "fast" | "efficientnet",
        "img_size":         32 (fast) | 224 (efficientnet),
        "label_convention": {"0": "AUTHENTIC", "1": "FORGED"},
        "imagenet_norm":    {"mean": [...], "std": [...]},
        "model_state_dict": <weights>,
    }

The current model.pt on disk is the compact "fast" architecture (FastCNN,
32x32 input, 2-class softmax). This module therefore rebuilds the matching
architecture from the checkpoint metadata so the weights load cleanly, and it
also supports the EfficientNet-B0 variant for when the model is retrained with
`train_cnn.py --model efficientnet`.

Output (cnn_predict returns a dict):
  cnn_score         — float in [0, 1] = P(forged/AI).  ~1.0 = fake, ~0.0 = real.
                      None if the model could not be loaded or inference failed.
  cnn_model_loaded  — bool: True if model.pt loaded successfully.
  cnn_note          — human-readable status string.
  cnn_flag          — "AI_CNN_DETECTED" if cnn_score > CNN_FLAG_THRESHOLD, else None.

Threshold:
  cnn_score > 0.65  → AI_CNN_DETECTED
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
from torchvision import models

MODEL_PATH = Path(__file__).parent / "model.pt"

# Probability threshold above which the CNN raises the AI_CNN_DETECTED flag.
CNN_FLAG_THRESHOLD = 0.65

# Fallback ImageNet normalisation (used only if checkpoint omits imagenet_norm).
_DEFAULT_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_STD  = [0.229, 0.224, 0.225]

# Inference device.
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Module-level cache so the weights are only loaded from disk once.
_MODEL = None          # nn.Module (eval mode)
_TRANSFORM = None      # torchvision transform pipeline
# Whether the checkpoint's label convention is inverted relative to our
# "P(fake) = P(class index 1)" assumption. label_convention {0: AUTHENTIC,
# 1: FORGED} => FAKE is index 1 => NOT inverted. If FAKE were index 0, we would
# need prob = 1 - prob.
_label_inverted = False
_LOAD_ERROR: Optional[str] = None


# ── Architectures ───────────────────────────────────────────────────────────────

class FastCNN(nn.Module):
    """Compact 4-block CNN for small images (must match train_cnn.FastCNN)."""

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


def _build_efficientnet() -> nn.Module:
    """EfficientNet-B0 with a 2-class head (matches train_cnn.build_efficientnet)."""
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    return m


def _build_model(architecture: str, img_size: int) -> nn.Module:
    if architecture == "efficientnet":
        return _build_efficientnet()
    # default / "fast"
    return FastCNN(img_size)


# ── Loading ─────────────────────────────────────────────────────────────────────

def _load() -> bool:
    """Load model.pt into the module cache. Returns True on success."""
    global _MODEL, _TRANSFORM, _label_inverted, _LOAD_ERROR

    if _MODEL is not None:
        return True
    if _LOAD_ERROR is not None:
        return False  # previous attempt failed; don't retry every call

    if not MODEL_PATH.exists():
        _LOAD_ERROR = f"model.pt not found at {MODEL_PATH}"
        return False

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=_device)

        # Support both a wrapped checkpoint dict and a bare state_dict.
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )

        if isinstance(checkpoint, dict):
            architecture = checkpoint.get("architecture", "fast")
            img_size     = int(checkpoint.get("img_size", 32))
            norm         = checkpoint.get("imagenet_norm") or {}
            label_conv   = checkpoint.get("label_convention") or {}
        else:
            architecture, img_size, norm, label_conv = "fast", 32, {}, {}

        mean = norm.get("mean", _DEFAULT_MEAN)
        std  = norm.get("std",  _DEFAULT_STD)

        # Determine label orientation. We treat class index 1 as "fake"; if the
        # checkpoint says FORGED/FAKE is index 0, the convention is inverted and
        # cnn_predict must flip the probability.
        _label_inverted = False
        for idx, name in label_conv.items():
            if str(name).upper() in ("FORGED", "FAKE", "AI"):
                _label_inverted = (int(idx) == 0)
                break

        model = _build_model(architecture, img_size)
        model.load_state_dict(state_dict)
        model.to(_device)
        model.eval()

        _MODEL = model
        _TRANSFORM = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        return True

    except Exception as e:  # noqa: BLE001 — surface any load failure as a note
        _LOAD_ERROR = f"Failed to load model.pt: {e}"
        _MODEL = None
        return False


# ── Public API ──────────────────────────────────────────────────────────────────

def cnn_predict(image_path: str) -> dict:
    """
    Run the trained CNN on a single image.

    Args:
        image_path: path to the image file.

    Returns:
        dict with keys: cnn_score, cnn_model_loaded, cnn_note, cnn_flag.
    """
    if not _load():
        return {
            "cnn_score":        None,
            "cnn_model_loaded": False,
            "cnn_note": (
                f"CNN unavailable — {_LOAD_ERROR}. Train and export a model to "
                f"{MODEL_PATH} (see notebooks/train_cnn.py)."
            ),
            "cnn_flag":         None,
        }

    path = Path(image_path)
    if not path.exists():
        return {
            "cnn_score":        None,
            "cnn_model_loaded": True,
            "cnn_note":         f"Image not found: {image_path}",
            "cnn_flag":         None,
        }

    try:
        img    = Image.open(path).convert("RGB")
        tensor = _TRANSFORM(img).unsqueeze(0).to(_device)  # add batch dim

        with torch.no_grad():
            logits = _MODEL(tensor)
            # The model has a 2-class head (AUTHENTIC=0, FORGED=1). The forged-vs-
            # authentic logit is (logit_1 - logit_0); its sigmoid equals the
            # softmax probability of the FORGED class. This keeps a single-logit
            # sigmoid interpretation while matching the 2-class architecture.
            logit = (logits[0, 1] - logits[0, 0])
            prob  = torch.sigmoid(logit).item()

        if _label_inverted:
            prob = 1.0 - prob

        score = prob
        flag = "AI_CNN_DETECTED" if score > CNN_FLAG_THRESHOLD else None
        note = (
            f"CNN P(forged)={score:.3f} "
            f"({'flagged' if flag else 'below'} threshold {CNN_FLAG_THRESHOLD})."
        )
        return {
            "cnn_score":        round(score, 4),
            "cnn_model_loaded": True,
            "cnn_note":         note,
            "cnn_flag":         flag,
        }

    except Exception as e:  # noqa: BLE001
        return {
            "cnn_score":        None,
            "cnn_model_loaded": True,
            "cnn_note":         f"CNN inference failed: {e}",
            "cnn_flag":         None,
        }
