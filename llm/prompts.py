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
"""

import json
from typing import Optional


def build_visual_analyst_prompt(forensix_result: dict) -> str:
    """
    Prompt for Model A: Visual Anomaly Analyst.
    Focus: WHERE is the tampering? What visually changed?
    """
    signals = forensix_result.get("signals", {})
    flags = forensix_result.get("flags", [])
    filename = forensix_result.get("file", "unknown")
    auth_score = forensix_result.get("authenticity_score", -1)

    ela_suspicion     = signals.get("ela_suspicion", "N/A")
    ela_mean_diff     = signals.get("ela_mean_diff", "N/A")
    ela_std_diff      = signals.get("ela_std_diff", "N/A")
    ela_variance      = signals.get("ela_regional_variance", "N/A")
    ela_energy_ratio  = signals.get("ela_high_energy_ratio", "N/A")
    noise_score       = signals.get("noise_score", "N/A")
    noise_cv          = signals.get("noise_block_cv", "N/A")
    heatmap_path      = signals.get("heatmap_path", None)

    # Aggregator enrichment
    agg          = forensix_result.get("aggregator", {})
    risk         = agg.get("risk_profile", {})
    overall_risk = risk.get("overall_risk", "UNKNOWN")
    ela_tier     = risk.get("ela_suspicion_tier", "UNKNOWN")
    noise_tier   = risk.get("noise_tier", "UNKNOWN")
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
ELA Mean Pixel Diff:      {ela_mean_diff}  px  (authentic: 2–8, tampered: 15–40)
ELA Std Dev of Diff:      {ela_std_diff}   px
ELA Regional Variance:    {ela_variance}   (authentic: 1–4, tampered: 8–20)
High-Energy Block Ratio:  {ela_energy_ratio}  (fraction of blocks with anomalous energy)

═══ NOISE TEXTURE SIGNALS ═══
Noise Inconsistency Score: {noise_score}  (>0.65 = inconsistent texture, possible splice)
Block Noise CV:            {noise_cv}

═══ AGGREGATED RISK PROFILE ═══
Overall Risk Level:    {overall_risk}
ELA Suspicion Tier:    {ela_tier}
Noise Texture Tier:    {noise_tier}
Signals in Warning:    {warn_count} / 5
Signals in Critical:   {crit_count} / 5

{'═══ ELA HEATMAP ═══' if heatmap_path else ''}
{'An ELA heatmap has been attached. Bright/red regions = high compression inconsistency = likely tampered zones.' if heatmap_path else ''}

═══ YOUR TASK ═══
Based ONLY on the forensic signals above:

1. Determine if this image has been tampered with.
2. Identify WHERE in the image the tampering likely occurred (describe region: face, background, object, edges, etc.)
3. Identify WHAT was changed: object insertion, face swap, background replacement, text editing, cloning, airbrushing, etc.
4. Assess how sophisticated the tampering is.

Respond ONLY in valid JSON. No preamble, no markdown fences:
{{
  "role": "visual_analyst",
  "verdict": "AUTHENTIC" | "SUSPICIOUS" | "TAMPERED",
  "confidence": "LOW" | "MEDIUM" | "HIGH",
  "tampered_regions": [
    {{
      "region": "describe where (e.g. top-left background, face area, text overlay)",
      "anomaly_type": "what kind of tampering (e.g. splicing, cloning, inpainting)",
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
    signals = forensix_result.get("signals", {})
    flags = forensix_result.get("flags", [])
    filename = forensix_result.get("file", "unknown")
    metadata = signals.get("metadata", {})

    has_exif      = metadata.get("has_exif", False)
    software_tag  = metadata.get("software_tag", None)
    camera_make   = metadata.get("camera_make", None)
    camera_model  = metadata.get("camera_model", None)
    meta_flags    = metadata.get("metadata_flags", [])
    meta_error    = metadata.get("metadata_error", None)

    # Aggregator metadata summary
    agg          = forensix_result.get("aggregator", {})
    meta_summary = agg.get("metadata_summary", {})
    origin_guess = meta_summary.get("origin_guess", "UNKNOWN")
    overall_risk = agg.get("risk_profile", {}).get("overall_risk", "UNKNOWN")

    return f"""You are a digital forensics metadata specialist. Your job is to analyze EXIF metadata
and timeline signals to determine WHEN an image was tampered with and WHAT software was used.

═══ IMAGE UNDER ANALYSIS ═══
File: {filename}
Flags raised: {', '.join(flags) if flags else 'NONE'}

═══ EXIF METADATA ═══
EXIF Present:    {has_exif}
Software Tag:    {software_tag if software_tag else 'NOT FOUND'}
Camera Make:     {camera_make if camera_make else 'NOT FOUND'}
Camera Model:    {camera_model if camera_model else 'NOT FOUND'}
Metadata Flags:  {', '.join(meta_flags) if meta_flags else 'NONE'}
Metadata Error:  {meta_error if meta_error else 'NONE'}

═══ AGGREGATOR INTERPRETATION ═══
Aggregated Origin Guess: {origin_guess}
Overall Risk Level:      {overall_risk}

═══ CONTEXTUAL FLAG ANALYSIS ═══
MISSING_EXIF_ON_JPEG: {'YES — JPEG without EXIF is unusual for camera photos, common in edited files' if 'MISSING_EXIF_ON_JPEG' in flags else 'NO'}
EDITING_SOFTWARE_DETECTED: {'YES — explicit editing software found in EXIF Software tag' if 'EDITING_SOFTWARE_DETECTED' in flags else 'NO'}

═══ YOUR TASK ═══
Based ONLY on the metadata signals above:

1. Determine the TIMELINE of this image: was it originally captured by a camera, or was it generated/edited on a computer?
2. Identify WHEN the tampering likely occurred based on available timestamp and software evidence.
3. Identify WHAT SOFTWARE was likely used to make changes (Photoshop, GIMP, AI tools, etc.)
4. Flag any metadata inconsistencies that indicate post-processing.

Respond ONLY in valid JSON. No preamble, no markdown fences:
{{
  "role": "metadata_analyst",
  "verdict": "AUTHENTIC" | "SUSPICIOUS" | "TAMPERED",
  "confidence": "LOW" | "MEDIUM" | "HIGH",
  "timeline": {{
    "origin": "CAMERA_CAPTURED | COMPUTER_GENERATED | UNKNOWN",
    "tampering_time": "When tampering likely occurred (e.g. 'after original capture', 'at file creation', 'unknown')",
    "time_evidence": "Which metadata field supports the timeline claim"
  }},
  "software_used": {{
    "detected_software": "Name of editing software or 'NONE DETECTED' or 'UNKNOWN'",
    "software_confidence": "LOW | MEDIUM | HIGH",
    "software_evidence": "Which EXIF field revealed this"
  }},
  "metadata_inconsistencies": [
    "List each inconsistency found, e.g. 'JPEG missing EXIF suggests file was re-saved after editing'"
  ],
  "when_was_it_changed": "Plain-English answer: when was this image likely modified and what does the metadata tell us",
  "reasoning": "Step-by-step explanation of your metadata analysis",
  "limitations": "What metadata analysis cannot determine without more data"
}}"""


def build_ai_pattern_analyst_prompt(forensix_result: dict) -> str:
    """
    Prompt for Model C: AI-Generation Pattern Analyst.
    Focus: Was this tampered using AI (GAN, diffusion, inpainting)?
    """
    signals = forensix_result.get("signals", {})
    flags = forensix_result.get("flags", [])
    filename = forensix_result.get("file", "unknown")
    auth_score = forensix_result.get("authenticity_score", -1)

    ela_suspicion    = signals.get("ela_suspicion", "N/A")
    ela_variance     = signals.get("ela_regional_variance", "N/A")
    energy_ratio     = signals.get("ela_high_energy_ratio", "N/A")
    noise_score      = signals.get("noise_score", "N/A")
    software_tag     = signals.get("metadata", {}).get("software_tag", "NOT FOUND")
    cnn_note         = signals.get("cnn_note", "CNN not loaded")

    return f"""You are an AI-generated media detection specialist with expertise in identifying
GAN artifacts, diffusion model outputs, and AI-assisted image manipulation.

Your job is to determine if this image was tampered using AI tools (e.g. Stable Diffusion inpainting,
DALL-E edits, Midjourney compositing, DeepFake face swaps, GANs, or AI object removal/insertion).

═══ IMAGE UNDER ANALYSIS ═══
File: {filename}
Overall Authenticity Score: {auth_score}
Active flags: {', '.join(flags) if flags else 'NONE'}

═══ FORENSIC SIGNALS ═══
ELA Suspicion Score:       {ela_suspicion}
ELA Regional Variance:     {ela_variance}   — AI inpainting creates sharp variance boundaries
High-Energy Block Ratio:   {energy_ratio}   — AI fills often show distinct energy signatures
Noise Inconsistency Score: {noise_score}    — AI-generated regions have different noise profiles
Detected Software:         {software_tag}
CNN Model Status:          {cnn_note}

═══ AI TAMPERING PATTERNS TO CONSIDER ═══
• GAN/Diffusion inpainting: Creates ELA hotspots at fill boundaries with smooth interiors
• Face swapping (DeepFake): High ELA around face-hairline boundary, noise mismatch
• AI background replacement: Sharp energy cliff between foreground/background
• AI object insertion: Regional variance spike at object edges, noise inconsistency
• AI upscaling artifacts: Uniform but unnaturally low noise across image
• Prompt-based generation: Missing EXIF entirely, suspiciously clean metadata

═══ YOUR TASK ═══
Based on the forensic signals above, determine:
1. Whether AI tools were used to tamper with this image
2. Which specific AI technique was likely used
3. How certain you are, and what additional analysis would confirm it

Respond ONLY in valid JSON. No preamble, no markdown fences:
{{
  "role": "ai_pattern_analyst",
  "verdict": "AUTHENTIC" | "AI_ASSISTED_TAMPER" | "FULLY_AI_GENERATED" | "SUSPICIOUS" | "HUMAN_EDITED",
  "confidence": "LOW" | "MEDIUM" | "HIGH",
  "ai_technique_detected": {{
    "technique": "e.g. Diffusion inpainting, GAN face swap, background replacement, AI upscaling, or NONE",
    "tool_likely_used": "e.g. Stable Diffusion, Midjourney, DALL-E, FaceSwap, DeepFaceLab, or UNKNOWN",
    "signal_evidence": "Which specific signal pattern points to this AI technique"
  }},
  "ai_artifacts_found": [
    "List specific AI artifact patterns found in the signals"
  ],
  "human_vs_ai_edit": "Was this edited by a human (Photoshop brush), by AI (inpainting/generation), or both?",
  "reasoning": "Step-by-step explanation linking signal values to AI detection conclusions",
  "recommended_next_analysis": "What deeper analysis would confirm or deny AI usage (e.g. run CLIP, check frequency domain)",
  "limitations": "What this analysis cannot determine"
}}"""
