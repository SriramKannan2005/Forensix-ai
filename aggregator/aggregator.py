"""
aggregator.py — Signal Aggregator for ForensiX (Image Proto)
--------------------------------------------------------------
Sits between the image detection module and the LLM reasoning layer.

Responsibilities:
  1. Validate the ForensixResult conforms to schema
  2. Normalize and interpret all raw signal values into human-readable tiers
  3. Compute a unified risk profile (per-signal severity levels)
  4. Build a clean, enriched context dict that the LLM prompts consume
  5. Flag any signals that are missing or unreliable

Flow:
    detector.analyze_image()
           │
           ▼
    aggregator.aggregate()
           │
           ▼
    llm.reasoning_engine.run_llm_reasoning()
"""

from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Signal interpretation tables
# Converts raw float values → labeled severity tiers for LLM context
# ─────────────────────────────────────────────────────────────────────────────

def _tier(value: float, thresholds: list[tuple]) -> str:
    """
    Map a float value to a severity label using threshold list.
    thresholds: [(upper_bound, label), ...] in ascending order
    """
    for bound, label in thresholds:
        if value <= bound:
            return label
    return thresholds[-1][1]


ELA_SUSPICION_TIERS = [
    (0.20, "CLEAN"),
    (0.35, "LOW_SUSPICION"),
    (0.50, "MODERATE_SUSPICION"),
    (0.70, "HIGH_SUSPICION"),
    (1.00, "CRITICAL"),
]

ELA_MEAN_TIERS = [
    (8.0,  "NORMAL"),       # authentic JPEG equilibrium
    (15.0, "ELEVATED"),
    (25.0, "HIGH"),
    (40.0, "VERY_HIGH"),
    (999,  "EXTREME"),
]

ELA_VARIANCE_TIERS = [
    (4.0,  "UNIFORM"),       # consistent compression = authentic
    (8.0,  "SLIGHTLY_UNEVEN"),
    (12.0, "UNEVEN"),        # tampered regions cluster
    (999,  "HIGHLY_UNEVEN"),
]

NOISE_TIERS = [
    (0.30, "CONSISTENT"),
    (0.50, "SLIGHTLY_INCONSISTENT"),
    (0.65, "INCONSISTENT"),
    (1.00, "HIGHLY_INCONSISTENT"),
]

AUTH_SCORE_TIERS = [
    (0.30, "LIKELY_FORGED"),
    (0.50, "SUSPICIOUS"),
    (0.70, "UNCERTAIN"),
    (0.85, "LIKELY_AUTHENTIC"),
    (1.00, "AUTHENTIC"),
]

# Frequency domain: high-freq energy ratio
# Low = AI-smooth (missing sensor noise); high = normal camera texture
FREQ_HIGH_RATIO_TIERS = [
    (0.08,  "AI_SMOOTH"),
    (0.12,  "SUSPICIOUSLY_SMOOTH"),
    (0.20,  "SLIGHTLY_SMOOTH"),
    (0.45,  "NORMAL"),
    (1.00,  "HIGH_FREQ_HEAVY"),
]

# CNN forged-probability tiers (P(FORGED) from the trained classifier)
CNN_SCORE_TIERS = [
    (0.20, "LIKELY_AUTHENTIC"),
    (0.40, "PROBABLY_AUTHENTIC"),
    (0.60, "UNCERTAIN"),
    (0.80, "PROBABLY_FORGED"),
    (1.00, "LIKELY_FORGED"),
]

# Noise block CV: how variable is local noise across the image?
NOISE_CV_TIERS = [
    (0.08, "AI_SMOOTH"),
    (0.15, "VERY_CONSISTENT"),
    (0.55, "CONSISTENT"),
    (0.70, "SLIGHTLY_INCONSISTENT"),
    (1.00, "HIGHLY_INCONSISTENT"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Required schema fields — validation
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_FIELDS = ["module", "file", "processing_time", "authenticity_score",
                   "signals", "flags"]

REQUIRED_SIGNALS = ["ela_suspicion", "ela_mean_diff", "ela_std_diff",
                    "ela_regional_variance", "ela_high_energy_ratio"]


def _validate(result: dict) -> list[str]:
    """Return list of validation errors. Empty = valid."""
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in result:
            errors.append(f"Missing top-level field: '{field}'")
    if "signals" in result:
        for sig in REQUIRED_SIGNALS:
            if sig not in result["signals"]:
                errors.append(f"Missing signal: '{sig}'")
    if result.get("error"):
        errors.append(f"Detection module reported error: {result['error']}")
    return errors


# ─────────────────────────────────────────────────────────────────────────────
# Risk profile builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_risk_profile(signals: dict, flags: list) -> dict:
    """
    Interpret each signal into a severity tier and build a risk summary.
    This is the enriched context that goes into LLM prompts.

    Signal set (v2): ELA + noise texture + frequency domain + metadata flags
    """
    ela_s    = signals.get("ela_suspicion", 0.0)
    ela_m    = signals.get("ela_mean_diff", 0.0)
    ela_v    = signals.get("ela_regional_variance", 0.0)
    ela_e    = signals.get("ela_high_energy_ratio", 0.0)
    noise_s  = signals.get("noise_score")
    noise_cv = signals.get("noise_block_cv")
    freq_hr  = signals.get("freq_high_ratio")
    ai_smooth = signals.get("ai_smooth_flag", False)

    # CNN signals (trained deep-learning classifier). cnn_score is P(FORGED).
    # cnn_model_loaded gates whether the score is trustworthy; if the model is
    # unavailable we skip CNN-based signals gracefully.
    cnn_loaded = signals.get("cnn_model_loaded", False)
    cnn_score  = signals.get("cnn_score") if cnn_loaded else None
    cnn_conf   = signals.get("cnn_confidence", 0.0) if cnn_loaded else 0.0
    cnn_forged_probability      = cnn_score is not None and cnn_score > 0.6
    cnn_high_confidence_forged  = (cnn_score is not None and cnn_score > 0.8
                                   and cnn_conf > 0.85)

    # Metadata-derived risk indicators
    meta_flags = signals.get("metadata", {}).get("metadata_flags", [])
    has_ai_sig   = "AI_GENERATOR_SIGNATURE" in meta_flags
    missing_exif = ("MISSING_EXIF_ON_JPEG" in meta_flags
                    or "MISSING_EXIF_ON_PNG" in meta_flags)
    no_camera    = "NO_CAMERA_DATA" in meta_flags

    risk = {
        # ELA tiers
        "ela_suspicion_tier":    _tier(ela_s, ELA_SUSPICION_TIERS),
        "ela_mean_tier":         _tier(ela_m, ELA_MEAN_TIERS),
        "ela_variance_tier":     _tier(ela_v, ELA_VARIANCE_TIERS),
        # Noise tiers
        "noise_tier":            _tier(noise_s, NOISE_TIERS) if noise_s is not None else "UNAVAILABLE",
        "noise_cv_tier":         _tier(noise_cv, NOISE_CV_TIERS) if noise_cv is not None else "UNAVAILABLE",
        # Frequency tier
        "freq_tier":             _tier(freq_hr, FREQ_HIGH_RATIO_TIERS) if freq_hr is not None else "UNAVAILABLE",
        # AI-generation specific
        "ai_smooth_detected":    ai_smooth,
        "ai_generator_signature": has_ai_sig,

        # CNN (trained classifier) interpretation
        "cnn_score":                 cnn_score,
        "cnn_loaded":                cnn_loaded,
        "cnn_tier":                  (_tier(cnn_score, CNN_SCORE_TIERS)
                                      if cnn_score is not None else "UNAVAILABLE"),
        "cnn_forged_probability":    cnn_forged_probability,
        "cnn_high_confidence_forged": cnn_high_confidence_forged,

        # ── Warning signals (any single indicator that something is off) ───────
        # Expanded to include the trained CNN's forged-probability signal.
        "signals_in_warning": sum([
            ela_s > 0.35,                              # ELA
            ela_m > 12.0,
            ela_v > 5.0,
            ela_e > 0.08,
            (noise_s or 0) > 0.50,                    # noise inconsistency
            noise_cv is not None and noise_cv < 0.10, # AI-smooth noise
            freq_hr is not None and freq_hr < 0.15,   # low-freq spectrum
            missing_exif,                              # metadata
            no_camera,
            cnn_forged_probability,                    # CNN P(forged) > 0.6
        ]),
        "signals_in_critical": sum([
            ela_s > 0.55,
            ela_m > 25.0,
            ela_v > 10.0,
            ela_e > 0.20,
            (noise_s or 0) > 0.65,
            ai_smooth,                                 # AI-smooth is critical
            has_ai_sig,                                # explicit AI sig is critical
            freq_hr is not None and freq_hr < 0.09,   # extreme smoothness
            cnn_high_confidence_forged,                # CNN high-confidence forged
        ]),
        "tamper_flags_active": [f for f in flags if "ERROR" not in f],
    }

    # ── Overall risk level ────────────────────────────────────────────────────
    if risk["signals_in_critical"] >= 2:
        risk["overall_risk"] = "CRITICAL"
    elif risk["signals_in_critical"] >= 1 or risk["signals_in_warning"] >= 3:
        risk["overall_risk"] = "HIGH"
    elif risk["signals_in_warning"] >= 2:
        risk["overall_risk"] = "MEDIUM"
    elif risk["signals_in_warning"] >= 1:
        risk["overall_risk"] = "LOW"
    else:
        risk["overall_risk"] = "NONE"

    return risk


# ─────────────────────────────────────────────────────────────────────────────
# Metadata summary builder
# ─────────────────────────────────────────────────────────────────────────────

def _summarize_metadata(signals: dict) -> dict:
    """Extract and interpret metadata signals into a clean summary."""
    meta = signals.get("metadata", {})
    if not meta:
        return {"available": False}

    has_exif     = meta.get("has_exif", False)
    software     = meta.get("software_tag")
    camera_make  = meta.get("camera_make")
    camera_model = meta.get("camera_model")
    meta_flags   = meta.get("metadata_flags", [])

    origin = "UNKNOWN"
    whatsapp = "POSSIBLE_WHATSAPP_RECOMPRESSION" in meta_flags
    if camera_make or camera_model:
        origin = "CAMERA_CAPTURED"
    elif whatsapp:
        origin = "SOCIAL_MEDIA_RECOMPRESSED"  # benign: WhatsApp/social re-encode
    elif not has_exif:
        origin = "POSSIBLY_GENERATED_OR_EDITED"  # no EXIF = suspicious
    elif software:
        origin = "COMPUTER_PROCESSED"

    # Metadata score mirrors detector.py: 1.0 = clean, 0.6 = WhatsApp recompress
    # (partial penalty), 0.5 = other warning flags, 0.0 = critical (AI signature).
    if "AI_GENERATOR_SIGNATURE" in meta_flags:
        metadata_score = 0.0
    elif whatsapp:
        metadata_score = 0.6
    elif meta_flags:
        metadata_score = 0.5
    else:
        metadata_score = 1.0

    return {
        "available":      True,
        "has_exif":       has_exif,
        "origin_guess":   origin,
        "software_found": software,
        "camera_make":    camera_make,
        "camera_model":   camera_model,
        "meta_flags":     meta_flags,
        "metadata_score": metadata_score,
        "whatsapp_recompression": whatsapp,
        "editing_software_detected": "EDITING_SOFTWARE_DETECTED" in meta_flags,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(detection_result: dict) -> dict:
    """
    Aggregate and enrich a raw ForensixResult from the image detection module.

    Args:
        detection_result: Raw dict output from detector.analyze_image()

    Returns:
        Enriched dict ready for LLM reasoning layer. Includes:
          - Original detection result (passthrough)
          - validation_errors: list of schema violations
          - risk_profile: per-signal severity tiers + overall risk level
          - metadata_summary: interpreted EXIF / origin summary
          - auth_score_tier: human-readable authenticity label
          - aggregator_ready: bool — True if safe to send to LLM
    """
    signals = detection_result.get("signals", {})
    flags   = detection_result.get("flags", [])
    auth    = detection_result.get("authenticity_score", -1)

    # ── Validation ────────────────────────────────────────────────────────────
    validation_errors = _validate(detection_result)

    # ── Risk profile ──────────────────────────────────────────────────────────
    risk_profile = _build_risk_profile(signals, flags)

    # ── Metadata summary ──────────────────────────────────────────────────────
    metadata_summary = _summarize_metadata(signals)

    # ── Auth score tier ───────────────────────────────────────────────────────
    auth_tier = _tier(auth, AUTH_SCORE_TIERS) if auth >= 0 else "UNKNOWN"

    # ── Ready to send to LLM? ─────────────────────────────────────────────────
    # Block if: detection errored, or critical schema fields missing
    fatal_errors = [e for e in validation_errors if "Missing top-level" in e or "error" in e.lower()]
    aggregator_ready = len(fatal_errors) == 0

    # ── Build enriched result ─────────────────────────────────────────────────
    enriched = {
        **detection_result,                    # full passthrough of original
        "aggregator": {
            "validation_errors":  validation_errors,
            "aggregator_ready":   aggregator_ready,
            "risk_profile":       risk_profile,
            "metadata_summary":   metadata_summary,
            "auth_score_tier":    auth_tier,
            "signal_verdict":     detection_result.get("verdict", "UNKNOWN"),
            "signal_count":       len(signals),
            "active_flag_count":  len(risk_profile["tamper_flags_active"]),
        }
    }

    return enriched


def aggregator_report(enriched: dict) -> str:
    """
    Pretty-print summary of aggregator output for CLI display.
    """
    agg  = enriched.get("aggregator", {})
    risk = agg.get("risk_profile", {})
    meta = agg.get("metadata_summary", {})

    lines = [
        f"\n{'─'*52}",
        "  SIGNAL AGGREGATOR REPORT",
        f"{'─'*52}",
        f"  Auth Score:       {enriched.get('authenticity_score', -1):.4f}  [{agg.get('auth_score_tier', 'N/A')}]",
        f"  Overall Risk:     {risk.get('overall_risk', 'N/A')}",
        f"  Signals Warning:  {risk.get('signals_in_warning', 0)} / 9",
        f"  Signals Critical: {risk.get('signals_in_critical', 0)} / 8",
        f"  Active Flags:     {', '.join(risk.get('tamper_flags_active', [])) or 'NONE'}",
        f"",
        f"  ELA Suspicion:    {enriched.get('signals',{}).get('ela_suspicion','N/A')}  [{risk.get('ela_suspicion_tier','N/A')}]",
        f"  ELA Mean Diff:    {enriched.get('signals',{}).get('ela_mean_diff','N/A')}  [{risk.get('ela_mean_tier','N/A')}]",
        f"  ELA Variance:     {enriched.get('signals',{}).get('ela_regional_variance','N/A')}  [{risk.get('ela_variance_tier','N/A')}]",
        f"  Noise Score:      {enriched.get('signals',{}).get('noise_score','N/A')}  [{risk.get('noise_tier','N/A')}]",
        f"  Noise CV:         {enriched.get('signals',{}).get('noise_block_cv','N/A')}  [{risk.get('noise_cv_tier','N/A')}]",
        f"  Freq High Ratio:  {enriched.get('signals',{}).get('freq_high_ratio','N/A')}  [{risk.get('freq_tier','N/A')}]",
        f"  AI Smooth:        {risk.get('ai_smooth_detected', False)}",
        f"  AI Generator Sig: {risk.get('ai_generator_signature', False)}",
        f"",
        f"  Metadata Origin:  {meta.get('origin_guess','N/A')}",
        f"  Software Found:   {meta.get('software_found') or 'NONE'}",
        f"  Camera:           {meta.get('camera_make','?')} {meta.get('camera_model','')}".strip(),
        f"",
        f"  LLM Ready:        {'YES' if agg.get('aggregator_ready') else 'NO — ' + str(agg.get('validation_errors'))}",
        f"{'─'*52}",
    ]
    return "\n".join(lines)