"""
prompts.py — Specialist Forensic Prompts for ForensiX
-------------------------------------------------------
Three LLMs, three roles:

  ROLE A — Visual Anomaly Analyst (llava / Claude)
    Looks at ELA heatmap + pixel signals to find WHERE tampering occurred.
    Detects: splicing, cloning, object insertion/removal, background replacement.

  ROLE B — Metadata & Timeline Analyst (Mistral / GPT-4o)
    Reads EXIF, software tags, timestamps to determine WHEN and HOW tampering happened.
    Detects: timestamp inconsistencies, editing software fingerprints, missing metadata.

  ROLE C — AI-Generation Pattern Analyst (Gemma / Gemini)
    Looks for GAN/diffusion model artifacts in the signal profile.
    Detects: AI-generated faces, inpainting artifacts, Stable Diffusion patterns.

v2 changes:
  - All three prompts now receive noise texture + frequency domain signals
  - Metadata prompt receives full metadata_flags list including MISSING_EXIF_ON_PNG,
    NO_CAMERA_DATA, AI_GENERATOR_SIGNATURE
  - AI pattern prompt receives ai_smooth_flag, freq_high_ratio, noise_block_cv
    which are the primary signals for ChatGPT/Stable Diffusion image detection
  - Signal counts updated to 9 warning / 8 critical
"""

import json
from typing import Optional


def build_visual_analyst_prompt(forensix_result: dict) -> str:
    """
    Prompt for Model A: Visual Anomaly Analyst.
    Focus: WHERE is the tampering? What visually changed?
    """
    signals    = forensix_result.get("signals", {})
    flags      = forensix_result.get("flags", [])
    filename   = forensix_result.get("file", "unknown")
    auth_score = forensix_result.get("authenticity_score", -1)

    ela_suspicion    = signals.get("ela_suspicion", "N/A")
    ela_mean_diff    = signals.get("ela_mean_diff", "N/A")
    ela_std_diff     = signals.get("ela_std_diff", "N/A")
    ela_variance     = signals.get("ela_regional_variance", "N/A")
    ela_energy_ratio = signals.get("ela_high_energy_ratio", "N/A")
    noise_score      = signals.get("noise_score", "N/A")
    noise_cv         = signals.get("noise_block_cv", "N/A")
    freq_high_ratio  = signals.get("freq_high_ratio", "N/A")
    ai_smooth        = signals.get("ai_smooth_flag", False)
    heatmap_path     = signals.get("heatmap_path", None)
    cnn_score        = signals.get("cnn_score")
    cnn_confidence   = signals.get("cnn_confidence", 0.0) or 0.0
    cnn_label        = signals.get("cnn_label", "UNKNOWN")

    # Aggregator enrichment
    agg          = forensix_result.get("aggregator", {})
    risk         = agg.get("risk_profile", {})
    overall_risk = risk.get("overall_risk", "UNKNOWN")
    ela_tier     = risk.get("ela_suspicion_tier", "UNKNOWN")
    noise_tier   = risk.get("noise_tier", "UNKNOWN")
    freq_tier    = risk.get("freq_tier", "UNKNOWN")
    noise_cv_tier = risk.get("noise_cv_tier", "UNKNOWN")
    warn_count   = risk.get("signals_in_warning", "N/A")
    crit_count   = risk.get("signals_in_critical", "N/A")

    return f"""You are a digital image forensics expert specializing in visual tampering detection.
Your job is to analyze pixel-level forensic signals and determine WHERE tampering occurred in this image.

═══ IMAGE UNDER ANALYSIS ═══
File: {filename}
Overall Authenticity Score: {auth_score} (0.0 = definitely forged, 1.0 = definitely authentic)
Flags raised by detection system: {', '.join(flags) if flags else 'NONE'}

═══ ERROR LEVEL ANALYSIS (ELA) SIGNALS ═══
ELA Suspicion Score:      {ela_suspicion}  (>0.35 = suspicious, >0.55 = highly suspicious)
ELA Mean Pixel Diff:      {ela_mean_diff}  px  (authentic JPEG: 2–8 px, tampered: 15–40 px)
ELA Std Dev of Diff:      {ela_std_diff}   px
ELA Regional Variance:    {ela_variance}   (authentic: 1–4, tampered: 8–20)
High-Energy Block Ratio:  {ela_energy_ratio}  (fraction of blocks with anomalous energy)

IMPORTANT NOTE ON ELA: ELA is most reliable on JPEG images. PNG files (especially
AI-generated images) typically show low ELA scores because they have no prior JPEG
compression history. Low ELA on a PNG does NOT mean the image is authentic.

═══ NOISE TEXTURE SIGNALS ═══
Noise Inconsistency Score: {noise_score}   (>0.65 = inconsistent texture, possible splice)
Block Noise CV:            {noise_cv}      [{noise_cv_tier}]
  (authentic cameras: 0.15–0.55; AI-generated: <0.10 = unnaturally smooth)
AI Smooth Flag:            {ai_smooth}     (True = image lacks natural camera sensor noise)

═══ FREQUENCY DOMAIN (DCT) SIGNALS ═══
High-Frequency Energy Ratio: {freq_high_ratio}  [{freq_tier}]
  (authentic camera photos: 0.20–0.45; AI-generated: often <0.12 = too smooth)
  Low values indicate missing sensor noise — a hallmark of AI image generation.

═══ AGGREGATED RISK PROFILE ═══
Overall Risk Level:    {overall_risk}
ELA Suspicion Tier:    {ela_tier}
Noise Texture Tier:    {noise_tier}
Frequency Tier:        {freq_tier}
Signals in Warning:    {warn_count} / 10
Signals in Critical:   {crit_count} / 9

═══ CNN FORGERY CLASSIFIER ═══
CNN Forgery Classifier Score: {f"{cnn_score:.3f}" if cnn_score is not None else "UNAVAILABLE"} (0=authentic, 1=forged)
CNN Confidence: {cnn_confidence:.3f} | CNN Label: {cnn_label}
Trained on 110k images (CIFAKE + CASIA v2), val_acc 95.89%.
Weight this heavily. If cnn_score > 0.8, lean toward FORGED unless other signals
strongly contradict.

{'═══ ELA HEATMAP ═══' if heatmap_path else ''}
{'An ELA heatmap has been attached. Bright/red regions = high compression inconsistency = likely tampered zones.' if heatmap_path else ''}

═══ YOUR TASK ═══
Based ONLY on the forensic signals above:

1. Determine if this image has been tampered with.
   IMPORTANT: If AI_SMOOTH_DETECTED or LOW_FREQ_ENERGY flags are present, or if
   noise_block_cv < 0.10, this is strong evidence of AI generation — even if ELA is clean.
2. Identify WHERE in the image the tampering likely occurred (region: face, background, etc.)
3. Identify WHAT was changed: AI generation, face swap, background replacement, etc.
4. Assess how sophisticated the tampering is.

Respond ONLY in valid JSON. No preamble, no markdown fences:
{{
  "role": "visual_analyst",
  "verdict": "AUTHENTIC" | "SUSPICIOUS" | "TAMPERED",
  "confidence": "LOW" | "MEDIUM" | "HIGH",
  "tampered_regions": [
    {{
      "region": "describe where (e.g. entire image, face area, background)",
      "anomaly_type": "what kind of tampering (e.g. AI generation, splicing, inpainting)",
      "evidence": "which signal supports this finding"
    }}
  ],
  "what_was_changed": "Plain-English description of the visual change made to the image",
  "tampering_sophistication": "LOW | MEDIUM | HIGH",
  "reasoning": "Step-by-step explanation of how you reached this verdict from the signals",
  "limitations": "What this visual analysis cannot determine"
}}"""


def build_metadata_analyst_prompt(forensix_result: dict) -> str:
    """
    Prompt for Model B: Metadata & Timeline Analyst.
    Focus: WHEN was it tampered? What software was used?
    """
    signals  = forensix_result.get("signals", {})
    flags    = forensix_result.get("flags", [])
    filename = forensix_result.get("file", "unknown")
    metadata = signals.get("metadata", {})

    has_exif      = metadata.get("has_exif", False)
    file_format   = metadata.get("file_format", "UNKNOWN")
    software_tag  = metadata.get("software_tag", None)
    camera_make   = metadata.get("camera_make", None)
    camera_model  = metadata.get("camera_model", None)
    date_original = metadata.get("date_time_original", None)
    date_digitized = metadata.get("date_time_digitized", None)
    image_width   = metadata.get("image_width", None)
    image_height  = metadata.get("image_height", None)
    raw_exif_count = metadata.get("raw_exif_count", 0)
    meta_flags    = metadata.get("metadata_flags", [])
    meta_error    = metadata.get("metadata_error", None)
    cnn_score      = signals.get("cnn_score")
    cnn_confidence = signals.get("cnn_confidence", 0.0) or 0.0

    # Aggregator metadata summary
    agg          = forensix_result.get("aggregator", {})
    meta_summary = agg.get("metadata_summary", {})
    origin_guess = meta_summary.get("origin_guess", "UNKNOWN")
    overall_risk = agg.get("risk_profile", {}).get("overall_risk", "UNKNOWN")

    # Build flag-specific context strings
    missing_exif_jpeg   = "MISSING_EXIF_ON_JPEG" in flags or "MISSING_EXIF_ON_JPEG" in meta_flags
    missing_exif_png    = "MISSING_EXIF_ON_PNG" in flags or "MISSING_EXIF_ON_PNG" in meta_flags
    no_camera_data      = "NO_CAMERA_DATA" in flags or "NO_CAMERA_DATA" in meta_flags
    ai_generator_sig    = "AI_GENERATOR_SIGNATURE" in flags or "AI_GENERATOR_SIGNATURE" in meta_flags
    editing_software    = "EDITING_SOFTWARE_DETECTED" in flags or "EDITING_SOFTWARE_DETECTED" in meta_flags
    timestamp_mismatch  = "TIMESTAMP_MISMATCH" in flags or "TIMESTAMP_MISMATCH" in meta_flags

    return f"""You are a digital forensics metadata specialist. Your job is to analyze EXIF metadata
and timeline signals to determine WHEN an image was tampered with and WHAT software was used.

═══ IMAGE UNDER ANALYSIS ═══
File: {filename}
File Format: {file_format}
Image Dimensions: {image_width}×{image_height} px
Flags raised: {', '.join(flags) if flags else 'NONE'}

═══ EXIF METADATA ═══
EXIF Present:         {has_exif}  (EXIF fields found: {raw_exif_count})
Software Tag:         {software_tag if software_tag else 'NOT FOUND'}
Camera Make:          {camera_make if camera_make else 'NOT FOUND'}
Camera Model:         {camera_model if camera_model else 'NOT FOUND'}
DateTimeOriginal:     {date_original if date_original else 'NOT FOUND'}
DateTimeDigitized:    {date_digitized if date_digitized else 'NOT FOUND'}
All Metadata Flags:   {', '.join(meta_flags) if meta_flags else 'NONE'}
Metadata Error:       {meta_error if meta_error else 'NONE'}

═══ FORENSIC METADATA FLAG ANALYSIS ═══
MISSING_EXIF_ON_JPEG:    {'⚠ YES — re-saved JPEG usually strips EXIF; indicates post-processing' if missing_exif_jpeg else 'NO'}
MISSING_EXIF_ON_PNG:     {'⚠ YES — PNG with no EXIF or metadata; consistent with AI-generated images (ChatGPT, Midjourney, DALL-E always produce clean PNGs)' if missing_exif_png else 'NO'}
NO_CAMERA_DATA:          {'⚠ YES — no Make/Model EXIF; inconsistent with a real camera photo. AI generators and screenshots never have camera hardware data.' if no_camera_data else 'NO'}
AI_GENERATOR_SIGNATURE:  {'🔴 YES — Software EXIF tag matches known AI image generator' if ai_generator_sig else 'NO'}
EDITING_SOFTWARE_DETECTED: {'⚠ YES — explicit editing software found in EXIF' if editing_software else 'NO'}
TIMESTAMP_MISMATCH:      {'⚠ YES — DateTimeOriginal ≠ DateTimeDigitized; indicates editing after capture' if timestamp_mismatch else 'NO'}

═══ AGGREGATOR INTERPRETATION ═══
Aggregated Origin Guess: {origin_guess}
Overall Risk Level:      {overall_risk}

═══ CNN FORGERY CLASSIFIER ═══
CNN classifier score: {f"{cnn_score:.3f}" if cnn_score is not None else "UNAVAILABLE"} (confidence: {cnn_confidence:.3f})
Treat as strong corroborating or conflicting evidence.

═══ CRITICAL FORENSIC CONTEXT ═══
A PNG file with no EXIF, no camera data, and no software tag is the standard output
of AI image generation tools (ChatGPT/DALL-E, Midjourney, Stable Diffusion, Firefly).
Real photographs saved as PNG would typically retain EXIF if the camera supports it,
or would show camera hardware metadata. The combination of PNG format + no EXIF
+ no camera data is strong forensic evidence of AI generation, not just editing.

═══ YOUR TASK ═══
Based ONLY on the metadata signals above:

1. Determine the TIMELINE: was this captured by a camera, generated by AI, or edited on a computer?
2. Identify WHEN the tampering likely occurred based on timestamp and software evidence.
3. Identify WHAT SOFTWARE was likely used (Photoshop, GIMP, ChatGPT, Midjourney, DALL-E, etc.)
4. Flag all metadata inconsistencies that indicate post-processing or AI generation.

Respond ONLY in valid JSON. No preamble, no markdown fences:
{{
  "role": "metadata_analyst",
  "verdict": "AUTHENTIC" | "SUSPICIOUS" | "TAMPERED",
  "confidence": "LOW" | "MEDIUM" | "HIGH",
  "timeline": {{
    "origin": "CAMERA_CAPTURED | COMPUTER_GENERATED | AI_GENERATED | UNKNOWN",
    "tampering_time": "When tampering likely occurred (e.g. 'at file creation by AI tool', 'after original capture', 'unknown')",
    "time_evidence": "Which metadata field or absence supports the timeline claim"
  }},
  "software_used": {{
    "detected_software": "Name of editing/generation software or 'NONE DETECTED' or 'UNKNOWN'",
    "software_confidence": "LOW | MEDIUM | HIGH",
    "software_evidence": "Which EXIF field or flag revealed this"
  }},
  "metadata_inconsistencies": [
    "List each inconsistency found"
  ],
  "when_was_it_changed": "Plain-English answer to when and how this image was created or modified",
  "reasoning": "Step-by-step explanation of your metadata analysis",
  "limitations": "What metadata analysis cannot determine without more data"
}}"""


def build_ai_pattern_analyst_prompt(forensix_result: dict) -> str:
    """
    Prompt for Model C: AI-Generation Pattern Analyst.
    Focus: Was this generated or tampered using AI (GAN, diffusion, ChatGPT)?
    This is the primary model for detecting AI-generated images like ChatGPT outputs.
    """
    signals    = forensix_result.get("signals", {})
    flags      = forensix_result.get("flags", [])
    filename   = forensix_result.get("file", "unknown")
    auth_score = forensix_result.get("authenticity_score", -1)

    ela_suspicion   = signals.get("ela_suspicion", "N/A")
    ela_variance    = signals.get("ela_regional_variance", "N/A")
    energy_ratio    = signals.get("ela_high_energy_ratio", "N/A")
    noise_score     = signals.get("noise_score", "N/A")
    noise_cv        = signals.get("noise_block_cv", "N/A")
    freq_high_ratio = signals.get("freq_high_ratio", "N/A")
    freq_uniformity = signals.get("freq_uniformity", "N/A")
    ai_smooth       = signals.get("ai_smooth_flag", False)
    software_tag    = signals.get("metadata", {}).get("software_tag", "NOT FOUND")
    meta_flags      = signals.get("metadata", {}).get("metadata_flags", [])
    cnn_note        = signals.get("cnn_note", "CNN not loaded")
    cnn_score       = signals.get("cnn_score")
    cnn_confidence  = signals.get("cnn_confidence", 0.0) or 0.0

    # Aggregator risk
    agg      = forensix_result.get("aggregator", {})
    risk     = agg.get("risk_profile", {})
    freq_tier = risk.get("freq_tier", "UNKNOWN")
    noise_cv_tier = risk.get("noise_cv_tier", "UNKNOWN")
    overall_risk = risk.get("overall_risk", "UNKNOWN")
    ai_smooth_flag = risk.get("ai_smooth_detected", False)
    ai_gen_sig     = risk.get("ai_generator_signature", False)

    # Specific metadata flags
    missing_exif_png = "MISSING_EXIF_ON_PNG" in meta_flags or "MISSING_EXIF_ON_PNG" in flags
    no_camera        = "NO_CAMERA_DATA" in meta_flags or "NO_CAMERA_DATA" in flags
    ai_sig           = "AI_GENERATOR_SIGNATURE" in meta_flags or "AI_GENERATOR_SIGNATURE" in flags

    return f"""You are an AI-generated media detection specialist with expertise in identifying
GAN artifacts, diffusion model outputs, and images created by tools like ChatGPT (DALL-E),
Midjourney, Stable Diffusion, Adobe Firefly, and similar AI image generators.

Your job is to determine if this image was FULLY AI-GENERATED or AI-assisted tampered.

═══ IMAGE UNDER ANALYSIS ═══
File: {filename}
Overall Authenticity Score: {auth_score}
Active flags: {', '.join(flags) if flags else 'NONE'}
Overall Risk Level: {overall_risk}

═══ PRIMARY AI-DETECTION SIGNALS ═══
These are the MOST IMPORTANT signals for detecting AI-generated images:

AI Smooth Flag:              {ai_smooth}  ← True = lacks natural camera sensor noise
Block Noise CV:              {noise_cv}   [{noise_cv_tier}]
  Authentic cameras: 0.15–0.55 | AI-generated: <0.10 (unnaturally uniform)
High-Frequency Energy Ratio: {freq_high_ratio}  [{freq_tier}]
  Authentic cameras: 0.20–0.45 | AI-generated: often <0.12 (missing sensor noise)
Frequency Uniformity:        {freq_uniformity}
  High uniformity = very flat spectrum = AI over-smoothing
Noise Inconsistency Score:   {noise_score}
AI Generator Signature:      {ai_sig}    ← True = software EXIF matches AI tool name

═══ METADATA AI INDICATORS ═══
MISSING_EXIF_ON_PNG:  {'🔴 YES — PNG with no EXIF is the standard output of ChatGPT/DALL-E, Midjourney, Stable Diffusion' if missing_exif_png else 'NO'}
NO_CAMERA_DATA:       {'🔴 YES — No camera Make/Model. Real photographs have this; AI images never do.' if no_camera else 'NO'}
AI_GENERATOR_SIGNATURE: {'🔴 YES — Explicit AI generator name found in metadata' if ai_sig else 'NO'}
Detected Software Tag:  {software_tag}

═══ ELA SIGNALS (SECONDARY — less reliable on AI-generated PNG) ═══
ELA Suspicion Score:   {ela_suspicion}  (NOTE: AI-generated PNGs often score low on ELA)
ELA Regional Variance: {ela_variance}
High-Energy Ratio:     {energy_ratio}
CNN Model Status:      {cnn_note}

═══ CNN AI-DETECTION SCORE ═══
CNN AI-detection score: {f"{cnn_score:.3f}" if cnn_score is not None else "UNAVAILABLE"} (confidence: {cnn_confidence:.3f})
CNN trained on CIFAKE (60k AI-generated vs real).
If cnn_score > 0.75, treat as strong evidence of AI generation.

═══ AI TAMPERING PATTERNS REFERENCE ═══
• ChatGPT/DALL-E full generation: MISSING_EXIF_ON_PNG + NO_CAMERA_DATA + very low noise_cv + low freq_high_ratio
• Stable Diffusion inpainting: ELA hotspots at fill boundaries, noise mismatch at edges
• GAN face swap (DeepFake): High ELA around face-hairline, noise mismatch in face region
• AI background replacement: Sharp energy difference between foreground/background
• AI upscaling: Uniformly low noise_cv across entire image, artificially sharp edges
• Midjourney/full generation: Absence of all camera metadata + smooth frequency spectrum

═══ DECISION GUIDE ═══
If ai_smooth_flag=True AND freq_high_ratio < 0.12 AND NO_CAMERA_DATA:
  → Strong evidence of AI generation (ChatGPT, Midjourney, SD)
If noise_cv < 0.10 AND missing_exif_png=True:
  → Consistent with AI-generated image output
If ELA is low but all metadata flags fire:
  → Trust metadata + frequency signals over ELA (ELA is unreliable on PNG)

═══ YOUR TASK ═══
Based on the forensic signals above, determine:
1. Whether this image was FULLY AI-GENERATED or AI-tampered
2. Which specific AI technique/tool was most likely used
3. Confidence level and supporting evidence

Respond ONLY in valid JSON. No preamble, no markdown fences:
{{
  "role": "ai_pattern_analyst",
  "verdict": "AUTHENTIC" | "AI_ASSISTED_TAMPER" | "FULLY_AI_GENERATED" | "SUSPICIOUS" | "HUMAN_EDITED",
  "confidence": "LOW" | "MEDIUM" | "HIGH",
  "ai_technique_detected": {{
    "technique": "e.g. Full AI generation, Diffusion inpainting, GAN face swap, AI upscaling, or NONE",
    "tool_likely_used": "e.g. ChatGPT/DALL-E, Midjourney, Stable Diffusion, DeepFaceLab, or UNKNOWN",
    "signal_evidence": "Which specific signal pattern points to this AI technique"
  }},
  "ai_artifacts_found": [
    "List specific AI artifact patterns found in the signals"
  ],
  "human_vs_ai_edit": "Was this generated by AI, edited by human, AI-assisted human edit, or authentic?",
  "reasoning": "Step-by-step explanation linking signal values to AI detection conclusions",
  "recommended_next_analysis": "What deeper analysis would confirm or deny AI usage",
  "limitations": "What this analysis cannot determine"
}}"""