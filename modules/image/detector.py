"""
detector.py — Image Forensics Detection Module for ForensiX
------------------------------------------------------------
Runs ELA signal extraction, metadata parsing, and noise/frequency
analysis on an image. Outputs a ForensixResult dict conforming to
the standard schema.

Signal keys produced (must match aggregator REQUIRED_SIGNALS):
  ela_suspicion           — overall ELA suspicion score (0–1)
  ela_mean_diff           — mean raw pixel difference
  ela_std_diff            — std dev of raw pixel difference
  ela_regional_variance   — variance across 4×4 grid block means
  ela_high_energy_ratio   — fraction of pixels above anomaly threshold
  noise_score             — noise inconsistency score (0–1)
  noise_block_cv          — coefficient of variation of block noise levels
  freq_high_ratio         — fraction of DCT energy in high-frequency bands
  ai_smooth_flag          — bool: image is suspiciously smooth (AI tell)
  metadata                — full metadata dict from metadata_extractor
  heatmap_path            — path to saved ELA heatmap (if save_heatmap=True)
  cnn_score               — None (placeholder until model is trained)
  cnn_note                — human-readable CNN status string

Flag thresholds:
  ela_suspicion > 0.50    → HIGH_ELA_SUSPICION
  ela_suspicion > 0.35    → MODERATE_ELA_SUSPICION
  ai_smooth_flag = True   → AI_SMOOTH_DETECTED
  NO_CAMERA_DATA in meta  → NO_CAMERA_DATA (propagated from metadata)
  MISSING_EXIF_ON_PNG     → MISSING_EXIF_ON_PNG (propagated from metadata)
  MISSING_EXIF_ON_JPEG    → MISSING_EXIF_ON_JPEG (propagated from metadata)
  AI_GENERATOR_SIGNATURE  → AI_GENERATOR_SIGNATURE (propagated from metadata)
  freq_high_ratio < 0.12  → LOW_FREQ_ENERGY (AI-smooth spectral signature)
"""

import time
from pathlib import Path
from typing import Optional

from schema import ForensixResult, make_error_result
from modules.image.ela import (
    compute_ela_raw,
    compute_ela_display,
    ela_extract_signals,
    save_ela_heatmap,
)
from modules.image.metadata_extractor import extract_metadata
from modules.image.noise_analyzer import analyze_noise

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Metadata flags that should be propagated up to the top-level flags list
# so the aggregator and LLM prompts see them directly
PROPAGATE_META_FLAGS = {
    "MISSING_EXIF_ON_JPEG",
    "MISSING_EXIF_ON_PNG",
    "NO_CAMERA_DATA",
    "AI_GENERATOR_SIGNATURE",
    "EDITING_SOFTWARE_DETECTED",
    "TIMESTAMP_MISMATCH",
}


def analyze_image(image_path: str, save_heatmap: bool = True) -> dict:
    """
    Run full image forensics pipeline on a single image file.

    Args:
        image_path:   Absolute or relative path to the image.
        save_heatmap: Whether to save the ELA heatmap to outputs/heatmaps/.

    Returns:
        ForensixResult dict — ready for aggregator.aggregate().
    """
    start = time.time()
    path  = Path(image_path)

    if not path.exists():
        return make_error_result("image", path.name, f"File not found: {image_path}")

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        return make_error_result("image", path.name, f"Unsupported format: {path.suffix}")

    try:
        flags   = []
        signals = {}

        # ── 1. ELA — raw array for scoring, display array for heatmap ─────────
        raw_array   = compute_ela_raw(str(path))
        ela_signals = ela_extract_signals(raw_array)
        signals.update(ela_signals)

        ela_suspicion = ela_signals["ela_suspicion"]
        if ela_suspicion > 0.50:
            flags.append("HIGH_ELA_SUSPICION")
        elif ela_suspicion > 0.35:
            flags.append("MODERATE_ELA_SUSPICION")

        # ── 2. Save heatmap (amplified display array) ─────────────────────────
        if save_heatmap:
            display_array = compute_ela_display(str(path))
            heatmap_dir   = Path("outputs/heatmaps")
            heatmap_dir.mkdir(parents=True, exist_ok=True)
            heatmap_path  = str(heatmap_dir / f"ela_{path.stem}.jpg")
            save_ela_heatmap(display_array, heatmap_path)
            signals["heatmap_path"] = heatmap_path

        # ── 3. Metadata extraction ─────────────────────────────────────────────
        meta = extract_metadata(str(path))
        signals["metadata"] = meta

        # Propagate forensically significant metadata flags to top-level flags
        for mf in meta.get("metadata_flags", []):
            if mf in PROPAGATE_META_FLAGS:
                if mf not in flags:
                    flags.append(mf)

        # ── 4. Noise & frequency analysis ─────────────────────────────────────
        noise_sigs = analyze_noise(str(path))
        # Merge noise signals directly (not nested)
        for k, v in noise_sigs.items():
            if k != "noise_error":
                signals[k] = v
        if noise_sigs.get("noise_error"):
            signals["noise_error"] = noise_sigs["noise_error"]

        # AI-smooth flag: unnaturally low noise → AI generation tell
        if noise_sigs.get("ai_smooth_flag"):
            flags.append("AI_SMOOTH_DETECTED")

        # Low frequency energy flag: spectral signature of AI-generated images
        freq_hr = noise_sigs.get("freq_high_ratio")
        if freq_hr is not None and freq_hr < 0.12:
            flags.append("LOW_FREQ_ENERGY")

        # ── 5. CNN placeholder ─────────────────────────────────────────────────
        signals["cnn_score"]        = None
        signals["cnn_model_loaded"] = False
        signals["cnn_note"]         = (
            "CNN model not loaded — train and export model to "
            "modules/image/model.pt to enable deep-learning classification."
        )

        # ── 6. Authenticity score ──────────────────────────────────────────────
        # Weighted blend of available signals:
        #   ELA is the primary signal (0.50 weight)
        #   Noise score contributes when available (0.30 weight)
        #   Metadata origin contributes (0.20 weight)
        #
        # Each component: 0.0 = definitely tampered, 1.0 = definitely authentic
        ela_component   = 1.0 - ela_suspicion

        noise_val = noise_sigs.get("noise_score")
        if noise_val is not None:
            noise_component = 1.0 - noise_val
            # Penalise AI-smooth images regardless of absolute noise_score
            if noise_sigs.get("ai_smooth_flag"):
                noise_component = min(noise_component, 0.40)
        else:
            noise_component = None

        # Metadata origin component
        meta_flags = meta.get("metadata_flags", [])
        meta_penalty = 0.0
        if "AI_GENERATOR_SIGNATURE" in meta_flags:
            meta_penalty = 0.70   # strong signal
        elif "MISSING_EXIF_ON_PNG" in meta_flags and "NO_CAMERA_DATA" in meta_flags:
            meta_penalty = 0.35   # moderate: PNG with no camera data
        elif "MISSING_EXIF_ON_JPEG" in meta_flags:
            meta_penalty = 0.25
        elif "NO_CAMERA_DATA" in meta_flags:
            meta_penalty = 0.15
        meta_component = 1.0 - meta_penalty

        # Blend
        if noise_component is not None:
            authenticity_score = (
                0.50 * ela_component +
                0.30 * noise_component +
                0.20 * meta_component
            )
        else:
            # Noise unavailable — reweight to ELA + metadata
            authenticity_score = (
                0.65 * ela_component +
                0.35 * meta_component
            )

        # Hard floor: if LOW_FREQ_ENERGY or AI_SMOOTH_DETECTED, cap score at 0.50
        if "LOW_FREQ_ENERGY" in flags or "AI_SMOOTH_DETECTED" in flags:
            authenticity_score = min(authenticity_score, 0.50)

        authenticity_score = round(float(authenticity_score), 4)
        processing_time    = round(time.time() - start, 3)

        return ForensixResult(
            module="image",
            file=path.name,
            processing_time=processing_time,
            authenticity_score=authenticity_score,
            signals=signals,
            flags=flags,
            error=None
        ).to_dict()

    except Exception as e:
        return make_error_result("image", path.name, str(e))