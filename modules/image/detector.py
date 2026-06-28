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
  cnn_score               — P(FORGED) 0–1 from the trained CNN (model_loader)
  cnn_label               — "AUTHENTIC" | "FORGED" | "UNKNOWN"
  cnn_confidence          — max softmax probability
  cnn_arch                — model architecture (e.g. resnet18)
  cnn_val_acc             — model validation accuracy
  cnn_model_loaded        — bool: True if the CNN ran successfully
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
    "POSSIBLE_WHATSAPP_RECOMPRESSION",
}


def _determine_verdict(flags: list, cnn_score: Optional[float],
                       cnn_error: Optional[str], authenticity_score: float) -> str:
    """
    Determine the overall verdict, with a dedicated AI-generation path.

    The CNN is trained for splice/tamper detection, not AI generation, so a
    confidently "not spliced" CNN score (low cnn_score) combined with AI-origin
    signal flags (flat texture + no EXIF + no camera data) indicates a synthetic
    image rather than an authentic photo.

    Returns one of:
      "AI_GENERATED" | "SUSPICIOUS_AI" | "AUTHENTIC" | "SUSPICIOUS" | "FORGED"
    """
    ai_smooth    = "AI_SMOOTH_DETECTED" in flags
    missing_exif = "MISSING_EXIF_ON_PNG" in flags or "NO_EXIF_DATA" in flags
    no_camera    = "NO_CAMERA_DATA" in flags

    # When the CNN is unavailable, treat its score as "not informative" (1.0) so
    # the AI rules (which require a LOW cnn_score) do not fire spuriously.
    cnn_ok = cnn_error is None and cnn_score is not None
    cs = float(cnn_score) if cnn_ok else 1.0
    ai_flag_count = sum([ai_smooth, missing_exif, no_camera])

    # ── AI-generation rules (checked in priority order) ───────────────────────
    if ai_smooth and missing_exif and no_camera and cs < 0.5:
        return "AI_GENERATED"
    if ai_flag_count >= 2 and cs < 0.3:
        return "AI_GENERATED"
    if ai_smooth and cs < 0.3:
        return "SUSPICIOUS_AI"

    # ── Standard authenticity thresholds ──────────────────────────────────────
    if authenticity_score >= 0.70:
        return "AUTHENTIC"
    if authenticity_score >= 0.40:
        return "SUSPICIOUS"
    return "FORGED"


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

        # ── 5. CNN deep-learning classifier ────────────────────────────────────
        from modules.image.model_loader import predict as _cnn_predict
        _cnn_result = _cnn_predict(str(path))
        cnn_score      = _cnn_result["cnn_score"]
        cnn_label      = _cnn_result["cnn_label"]
        cnn_confidence = _cnn_result["cnn_confidence"]
        cnn_error      = _cnn_result["cnn_error"]

        signals["cnn_score"]        = cnn_score
        signals["cnn_label"]        = cnn_label
        signals["cnn_confidence"]   = cnn_confidence
        signals["cnn_arch"]         = _cnn_result["cnn_arch"]
        signals["cnn_val_acc"]      = _cnn_result["cnn_val_acc"]
        signals["cnn_model_loaded"] = cnn_error is None
        if cnn_error is None:
            signals["cnn_note"] = (
                f"CNN ({_cnn_result['cnn_arch']}, val_acc "
                f"{_cnn_result['cnn_val_acc']:.2%}) P(forged)={cnn_score:.3f} "
                f"-> {cnn_label} (confidence {cnn_confidence:.1%})."
            )
        else:
            signals["cnn_note"] = (
                f"CNN unavailable — {cnn_error}. Train and export a model to "
                "modules/image/model.pt to enable deep-learning classification."
            )

        # CNN-based flags
        if cnn_error is None:
            if cnn_score > 0.8 and cnn_confidence > 0.85:
                flags.append("CNN_HIGH_CONFIDENCE_FORGED")
            elif cnn_score > 0.6:
                flags.append("CNN_FORGED_PROBABILITY")

        # ── 6. Authenticity score ──────────────────────────────────────────────
        # Weighted blend of available signals. Each component is oriented so that
        #   0.0 = definitely tampered/forged, 1.0 = definitely authentic.
        #
        # With CNN available:   CNN 0.40 + ELA 0.25 + Noise 0.20 + Metadata 0.15
        # Without CNN the remaining weights are re-normalised to sum to 1.0.
        #
        # NOTE: no hard floor is applied here — AI-generation is now surfaced via
        # the dedicated AI_GENERATED verdict (see _determine_verdict), so capping
        # the authenticity score (which previously pinned it at 0.50 and hid the
        # CNN's confident contribution) is no longer needed.

        def _clamp01(x):
            return max(0.0, min(1.0, float(x)))

        # CNN component: cnn_score is P(FORGED); authenticity = 1 - P(FORGED).
        # Only used when the model loaded cleanly (cnn_error is None).
        if cnn_error is None and cnn_score is not None:
            cnn_component = 1.0 - _clamp01(cnn_score)
        else:
            cnn_component = None

        # ELA component — ela_suspicion is already normalised to [0, 1].
        ela_component = 1.0 - _clamp01(ela_suspicion)

        # Noise component — noise_score is already normalised to [0, 1].
        noise_val = noise_sigs.get("noise_score")
        noise_component = (1.0 - _clamp01(noise_val)) if noise_val is not None else None

        # Metadata score: 1.0 = no flags, 0.5 = warning flags, 0.0 = critical flags.
        # Social-media (WhatsApp) recompression softens the penalty to 0.6 because
        # the missing EXIF is a benign artefact of sharing, not tampering.
        meta_flags = meta.get("metadata_flags", [])
        CRITICAL_META_FLAGS = {"AI_GENERATOR_SIGNATURE"}
        if any(f in CRITICAL_META_FLAGS for f in meta_flags):
            metadata_score = 0.0
        elif "POSSIBLE_WHATSAPP_RECOMPRESSION" in meta_flags:
            metadata_score = 0.6
        elif meta_flags:
            metadata_score = 0.5
        else:
            metadata_score = 1.0

        # Blend — re-normalise weights over whichever components are available so
        # the score always stays in [0, 1] even when noise or CNN is missing.
        weighted = []   # (weight, component) pairs
        if cnn_component is not None:
            weighted.append((0.40, cnn_component))
        weighted.append((0.25, ela_component))
        if noise_component is not None:
            weighted.append((0.20, noise_component))
        weighted.append((0.15, metadata_score))

        total_w = sum(w for w, _ in weighted)
        authenticity_score = sum(w * c for w, c in weighted) / total_w
        authenticity_score = round(_clamp01(authenticity_score), 4)

        # ── 6b. Verdict (AI-generation aware) ──────────────────────────────────
        verdict = _determine_verdict(flags, cnn_score, cnn_error, authenticity_score)
        signals["detection_verdict"] = verdict

        processing_time    = round(time.time() - start, 3)

        result = ForensixResult(
            module="image",
            file=path.name,
            processing_time=processing_time,
            authenticity_score=authenticity_score,
            signals=signals,
            flags=flags,
            error=None
        ).to_dict()
        # Surface the signal-based verdict at the top level for the aggregator/UI.
        result["verdict"] = verdict
        return result

    except Exception as e:
        return make_error_result("image", path.name, str(e))