"""
ela.py — Error Level Analysis for ForensiX
-------------------------------------------
Computes ELA by re-saving the image at a lower JPEG quality and
measuring the per-pixel difference from the original.

Key design principle (Bug 2 fix):
  - compute_ela_raw()   → returns the RAW difference array (for scoring)
  - compute_ela_display() → returns the amplified array (for the heatmap)
  - ela_score()         → ALWAYS receives the RAW array, never the amplified one
  - compute_ela()       → kept for backward-compat; returns display array only

Scoring is done on the raw diff so the mean/std values are real pixel-difference
magnitudes (0–255 range un-stretched), making ELA_MEAN_TIERS calibration
in the aggregator directly meaningful.
"""

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io


# ─────────────────────────────────────────────────────────────────────────────
# Core ELA computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_ela_raw(image_path: str, quality: int = 90) -> np.ndarray:
    """
    Return the RAW (un-amplified) ELA difference array.
    Shape: (H, W, 3), dtype uint8, values in 0–255.
    Use this for computing forensic scores.
    """
    original = Image.open(image_path).convert("RGB")

    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    ela_image = ImageChops.difference(original, recompressed)
    return np.array(ela_image)


def compute_ela_display(image_path: str, quality: int = 90, scale: int = 15) -> np.ndarray:
    """
    Return the amplified ELA array for visual heatmap rendering.
    Shape: (H, W, 3), dtype uint8.
    NOT suitable for scoring — amplification distorts the magnitude.
    """
    original = Image.open(image_path).convert("RGB")

    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    ela_image = ImageChops.difference(original, recompressed)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale * 255.0 / max_diff)

    return np.array(ela_image)


def compute_ela(image_path: str, quality: int = 90, scale: int = 15) -> np.ndarray:
    """
    Backward-compatible wrapper — returns the amplified display array.
    Use compute_ela_raw() for scoring.
    """
    return compute_ela_display(image_path, quality=quality, scale=scale)


# ─────────────────────────────────────────────────────────────────────────────
# Signal extraction from RAW array
# ─────────────────────────────────────────────────────────────────────────────

def ela_score(raw_array: np.ndarray) -> float:
    """
    Compute a normalized suspicion score from the RAW ELA difference array.

    Formula: (mean + std) / 255.0
    Range:   0.0 (no difference = authentic) → 1.0 (extreme difference = tampered)

    IMPORTANT: always pass compute_ela_raw() output, never compute_ela_display().
    """
    mean_val = float(np.mean(raw_array))
    std_val  = float(np.std(raw_array))
    raw_score = (mean_val + std_val) / 255.0
    return float(np.clip(raw_score, 0.0, 1.0))


def ela_extract_signals(raw_array: np.ndarray) -> dict:
    """
    Extract the full set of ELA signals required by the aggregator schema.

    Returns dict with keys:
        ela_suspicion           — overall suspicion score (0–1)
        ela_mean_diff           — mean pixel difference across image
        ela_std_diff            — std dev of pixel difference
        ela_regional_variance   — variance of block-mean values (tampering = local hotspots)
        ela_high_energy_ratio   — fraction of pixels above anomaly threshold
    """
    # Work on grayscale for per-pixel magnitude
    if raw_array.ndim == 3:
        gray = raw_array.mean(axis=2)          # (H, W) float
    else:
        gray = raw_array.astype(float)

    ela_mean = float(np.mean(gray))
    ela_std  = float(np.std(gray))

    # Regional variance: split image into 4×4 grid, compute variance of block means.
    # Authentic images have uniform compression → low variance across blocks.
    # Tampered images have localized hotspots → high variance.
    h, w = gray.shape
    grid_h, grid_w = max(1, h // 4), max(1, w // 4)
    block_means = []
    for i in range(4):
        for j in range(4):
            block = gray[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w]
            if block.size > 0:
                block_means.append(float(np.mean(block)))
    ela_regional_variance = float(np.var(block_means)) if block_means else 0.0

    # High energy ratio: fraction of pixels with difference > 30 (out of 255).
    # Tampered regions tend to have dense clusters of high-energy pixels.
    ela_high_energy_ratio = float(np.mean(gray > 30))

    ela_suspicion_score = float(np.clip((ela_mean + ela_std) / 255.0, 0.0, 1.0))

    return {
        "ela_suspicion":         round(ela_suspicion_score, 4),
        "ela_mean_diff":         round(ela_mean, 4),
        "ela_std_diff":          round(ela_std, 4),
        "ela_regional_variance": round(ela_regional_variance, 4),
        "ela_high_energy_ratio": round(ela_high_energy_ratio, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap saver
# ─────────────────────────────────────────────────────────────────────────────

def save_ela_heatmap(ela_display_array: np.ndarray, output_path: str) -> str:
    """
    Save a false-colour JET heatmap from the display (amplified) ELA array.
    Pass compute_ela_display() output, not the raw array.
    """
    heatmap = cv2.applyColorMap(
        cv2.cvtColor(ela_display_array, cv2.COLOR_RGB2GRAY),
        cv2.COLORMAP_JET
    )
    cv2.imwrite(output_path, heatmap)
    return output_path