"""
model_loader.py — Singleton CNN model loader for ForensiX
----------------------------------------------------------
Loads the trained forgery-classification model from modules/image/model.pt
exactly once and exposes a single predict() entry point used by the detector,
the GradCAM module, and the PDF report.

The architecture is reconstructed from checkpoint["architecture"] so the weights
load cleanly regardless of which backbone was trained:

  "resnet18":              torchvision resnet18, fc -> Dropout(0.4) + Linear(512, 2)
  "fast":                  FastCNN (copied from notebooks/train_cnn.py)
  "efficientnet[_b0]":     efficientnet_b0, classifier -> Dropout(0.4) + Linear(in, 2)

Label convention (from checkpoint): {0: AUTHENTIC, 1: FORGED}, so
P(FORGED) = softmax(logits)[1].

Public API:
    predict(image_input)  -> dict (see below)
    get_model()           -> (model, device, meta)   # used by gradcam.py
    is_loaded()           -> bool

predict() return schema:
    {
      "cnn_score":      float,        # P(FORGED) 0.0-1.0
      "cnn_label":      str,          # "AUTHENTIC" | "FORGED" | "UNKNOWN"
      "cnn_confidence": float,        # max(softmax)
      "cnn_arch":       str,          # e.g. "resnet18"
      "cnn_val_acc":    float,        # e.g. 0.9589
      "cnn_error":      Optional[str] # None on success
    }
"""

from pathlib import Path
from typing import Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT / "modules" / "image" / "model.pt"

_DEFAULT_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_STD = [0.229, 0.224, 0.225]

# ── Module-level singletons ──────────────────────────────────────────────────
_MODEL: Optional[nn.Module] = None
_DEVICE: Optional[torch.device] = None
_META: Optional[dict] = None
_TRANSFORM: Optional[T.Compose] = None
_LOAD_ERROR: Optional[str] = None


# ── Architectures ────────────────────────────────────────────────────────────

class FastCNN(nn.Module):
    """Compact 4-block CNN — must match notebooks/train_cnn.py FastCNN exactly."""

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


def _build_architecture(architecture: str, img_size: int) -> nn.Module:
    """Reconstruct the model architecture so the checkpoint state_dict loads strictly."""
    arch = (architecture or "resnet18").lower()

    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, 2))
        return m

    if arch in ("efficientnet", "efficientnet_b0"):
        m = models.efficientnet_b0(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, 2))
        return m

    if arch == "fast":
        return FastCNN(img_size)

    raise ValueError(f"Unknown architecture in checkpoint: {architecture!r}")


# ── Loader ───────────────────────────────────────────────────────────────────

def _load() -> bool:
    """Load model.pt into the module singletons. Returns True on success."""
    global _MODEL, _DEVICE, _META, _TRANSFORM, _LOAD_ERROR

    if _MODEL is not None:
        return True
    if _LOAD_ERROR is not None:
        return False  # previous attempt failed — don't retry on every call

    if not MODEL_PATH.exists():
        _LOAD_ERROR = f"model.pt not found at {MODEL_PATH}"
        return False

    try:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(MODEL_PATH, map_location=_DEVICE, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            architecture = checkpoint.get("architecture", "resnet18")
            img_size = int(checkpoint.get("img_size", 224))
            val_acc = float(checkpoint.get("val_acc", checkpoint.get("best_val_acc", 0.0)))
            norm = checkpoint.get("imagenet_norm") or {}
            label_conv = checkpoint.get("label_convention") or {"0": "AUTHENTIC", "1": "FORGED"}
        else:
            # Bare state_dict fallback
            state_dict = checkpoint
            architecture, img_size, val_acc = "resnet18", 224, 0.0
            norm, label_conv = {}, {"0": "AUTHENTIC", "1": "FORGED"}

        mean = norm.get("mean", _DEFAULT_MEAN)
        std = norm.get("std", _DEFAULT_STD)

        model = _build_architecture(architecture, img_size)
        model.load_state_dict(state_dict, strict=True)
        model.to(_DEVICE)
        model.eval()

        _MODEL = model
        _META = {
            "architecture": architecture,
            "img_size": img_size,
            "val_acc": val_acc,
            "mean": mean,
            "std": std,
            "label_convention": {str(k): str(v) for k, v in label_conv.items()},
        }
        _TRANSFORM = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        return True

    except Exception as e:  # noqa: BLE001 — surface any load failure to caller
        _LOAD_ERROR = f"Failed to load model.pt: {e}"
        _MODEL = None
        return False


def _to_pil(image_input: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
    """Coerce any supported input into an RGB PIL.Image."""
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, np.ndarray):
        return Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
    if isinstance(image_input, (str, Path)):
        return Image.open(str(image_input)).convert("RGB")
    raise TypeError(f"Unsupported image_input type: {type(image_input)}")


# ── Public API ───────────────────────────────────────────────────────────────

def is_loaded() -> bool:
    return _load()


def get_model() -> Tuple[nn.Module, torch.device, dict]:
    """
    Return (model, device, meta) for downstream use (e.g. GradCAM).
    Raises RuntimeError if the model could not be loaded.
    """
    if not _load():
        raise RuntimeError(_LOAD_ERROR or "model not loaded")
    return _MODEL, _DEVICE, _META


def predict(image_input: Union[str, Path, Image.Image, np.ndarray]) -> dict:
    """
    Run the trained CNN on a single image.

    Args:
        image_input: PIL.Image, numpy ndarray, or path (str / Path) to an image.

    Returns:
        dict with cnn_score, cnn_label, cnn_confidence, cnn_arch, cnn_val_acc,
        cnn_error (see module docstring). On any failure returns a safe dict with
        cnn_score=0.5, cnn_label="UNKNOWN", cnn_confidence=0.0 and cnn_error set.
    """
    if not _load():
        return {
            "cnn_score": 0.5,
            "cnn_label": "UNKNOWN",
            "cnn_confidence": 0.0,
            "cnn_arch": "unknown",
            "cnn_val_acc": 0.0,
            "cnn_error": _LOAD_ERROR,
        }

    try:
        img = _to_pil(image_input)
        tensor = _TRANSFORM(img).unsqueeze(0).to(_DEVICE)

        with torch.no_grad():
            logits = _MODEL(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        # Label convention: index 1 == FORGED, index 0 == AUTHENTIC
        forged_prob = float(probs[1].item())
        confidence = float(probs.max().item())
        pred_idx = int(probs.argmax().item())
        label = _META["label_convention"].get(str(pred_idx), "UNKNOWN")

        return {
            "cnn_score": round(forged_prob, 4),
            "cnn_label": label,
            "cnn_confidence": round(confidence, 4),
            "cnn_arch": _META["architecture"],
            "cnn_val_acc": round(_META["val_acc"], 4),
            "cnn_error": None,
        }

    except Exception as e:  # noqa: BLE001
        return {
            "cnn_score": 0.5,
            "cnn_label": "UNKNOWN",
            "cnn_confidence": 0.0,
            "cnn_arch": _META["architecture"] if _META else "unknown",
            "cnn_val_acc": _META["val_acc"] if _META else 0.0,
            "cnn_error": str(e),
        }
