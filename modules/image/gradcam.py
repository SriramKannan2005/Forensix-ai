"""
gradcam.py — Grad-CAM visualisation for the ForensiX CNN
---------------------------------------------------------
Pure-PyTorch Grad-CAM (no grad-cam / captum dependency). Produces a class-
activation heatmap showing which regions drove the CNN's FORGED prediction,
overlaid on the original image with a JET colormap.

Only uses: torch, torchvision (via model_loader), numpy, PIL, cv2.

Public API:
    generate_gradcam(image_input, output_path=None, layer_name="layer4") -> dict

Returns:
    {
      "heatmap":         np.ndarray,        # H x W float32 in [0, 1]
      "overlay":         PIL.Image.Image,   # original + heatmap overlay
      "predicted_class": str,               # "AUTHENTIC" | "FORGED"
      "predicted_score": float,             # P(FORGED) 0-1
      "output_path":     Optional[str],     # PNG path if saved
      "error":           Optional[str],     # None on success
    }
"""

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from modules.image.model_loader import get_model, _to_pil

# FORGED is class index 1 (label_convention {0: AUTHENTIC, 1: FORGED}).
_FORGED_IDX = 1


def _safe_overlay(error: str) -> dict:
    return {
        "heatmap": None,
        "overlay": None,
        "predicted_class": "UNKNOWN",
        "predicted_score": 0.0,
        "output_path": None,
        "error": error,
    }


def generate_gradcam(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    output_path: Optional[str] = None,
    layer_name: str = "layer4",
) -> dict:
    """Generate a Grad-CAM heatmap for the FORGED class of the trained CNN."""
    try:
        model, device, meta = get_model()
    except Exception as e:  # noqa: BLE001 — model unavailable
        return _safe_overlay(f"Model unavailable: {e}")

    target_layer = getattr(model, layer_name, None)
    if target_layer is None:
        return _safe_overlay(
            f"Layer '{layer_name}' not found on architecture "
            f"'{meta.get('architecture')}'. Grad-CAM supports CNN backbones "
            f"with a '{layer_name}' module (e.g. resnet18)."
        )

    try:
        original = _to_pil(image_input)
        orig_w, orig_h = original.size

        transform = T.Compose([
            T.Resize((meta["img_size"], meta["img_size"])),
            T.ToTensor(),
            T.Normalize(meta["mean"], meta["std"]),
        ])
        tensor = transform(original).unsqueeze(0).to(device)

        # ── Hooks to capture activations + gradients of the target layer ───────
        activations = {}
        gradients = {}

        def fwd_hook(_module, _inp, output):
            activations["value"] = output.detach()

        def bwd_hook(_module, _grad_in, grad_out):
            gradients["value"] = grad_out[0].detach()

        h1 = target_layer.register_forward_hook(fwd_hook)
        # full_backward_hook is the supported API on modern torch
        h2 = target_layer.register_full_backward_hook(bwd_hook)

        try:
            model.zero_grad()
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            forged_score = float(probs[_FORGED_IDX].item())
            pred_idx = int(probs.argmax().item())
            pred_class = meta["label_convention"].get(str(pred_idx), "UNKNOWN")

            # Backprop the FORGED-class score
            forged_logit = logits[0, _FORGED_IDX]
            forged_logit.backward()
        finally:
            h1.remove()
            h2.remove()

        acts = activations["value"][0]      # (C, h, w)
        grads = gradients["value"][0]       # (C, h, w)

        # Grad-CAM: global-average-pool gradients -> channel weights
        weights = grads.mean(dim=(1, 2))                       # (C,)
        cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))  # (h, w)

        cam_np = cam.cpu().numpy().astype(np.float32)
        # Normalise to [0, 1]
        cam_min, cam_max = float(cam_np.min()), float(cam_np.max())
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        # Resize heatmap to the original image size via PIL. PIL.Image.resize
        # takes (width, height) — passing (orig_w, orig_h) keeps the heatmap
        # aligned with the image instead of confining it to one corner.
        cam_uint8 = (np.clip(cam_np, 0.0, 1.0) * 255).astype(np.uint8)   # (h, w)
        cam_pil = Image.fromarray(cam_uint8, mode="L")
        cam_resized = cam_pil.resize((orig_w, orig_h), Image.BILINEAR)
        heatmap = np.asarray(cam_resized, dtype=np.float32) / 255.0      # (orig_h, orig_w)

        # Build JET colormap overlay (alpha=0.4). Result shape: (orig_h, orig_w, 3)
        heatmap_u8 = (heatmap * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

        orig_rgb = np.array(original.convert("RGB"))   # (orig_h, orig_w, 3)
        overlay_np = np.uint8(0.6 * orig_rgb + 0.4 * colored)
        overlay = Image.fromarray(overlay_np)

        saved_path = None
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            overlay.save(str(out))
            saved_path = str(out)

        return {
            "heatmap": heatmap,
            "overlay": overlay,
            "predicted_class": pred_class,
            "predicted_score": round(forged_score, 4),
            "output_path": saved_path,
            "error": None,
        }

    except Exception as e:  # noqa: BLE001
        return _safe_overlay(str(e))
