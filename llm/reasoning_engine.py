"""
reasoning_engine.py — Multi-LLM Forensic Reasoning Engine for ForensiX
------------------------------------------------------------------------
Runs three specialist LLMs in parallel threads, each analyzing a different
forensic dimension, then synthesizes their outputs into a final consensus verdict.

Pipeline:
    ForensixResult (from detection modules)
         │
         ├──► Model A: Visual Anomaly Analyst   → WHERE was it changed?
         ├──► Model B: Metadata Timeline Analyst → WHEN was it changed?
         └──► Model C: AI Pattern Analyst        → Was AI used to tamper?
                        │
                        ▼
              Consensus Synthesizer
                        │
                        ▼
              FinalVerdict (structured JSON)
"""

import json
import time
import threading
from typing import Optional
from pathlib import Path

import requests

from llm.provider import (
    call_llm, get_active_models, ENV, OLLAMA_BASE, OLLAMA_MODELS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Ollama warm-up / health check
# ─────────────────────────────────────────────────────────────────────────────

def _ping_model(model_name: str, base_url: str, timeout: int = 10) -> bool:
    """Lightweight check that an Ollama model is reachable and responds."""
    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={"model": model_name, "prompt": "hi", "stream": False},
            timeout=timeout,
        )
        return resp.status_code == 200
    except Exception:
        return False


def _warm_up_ollama() -> None:
    """
    In dev (Ollama) mode, ping each specialist model so a failure is logged
    early. Never raises — a cold model may load slower than the ping timeout, so
    inference is still attempted regardless of the ping result.
    """
    if ENV != "dev":
        return
    for role, model in OLLAMA_MODELS.items():
        if not _ping_model(model, OLLAMA_BASE):
            print(f"[reasoning_engine] WARNING: Ollama model '{model}' "
                  f"({role}) did not respond to warm-up ping — inference will "
                  f"still be attempted (cold start may take longer).")
from llm.prompts import (
    build_visual_analyst_prompt,
    build_metadata_analyst_prompt,
    build_ai_pattern_analyst_prompt,
)


# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing helper
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json_response(raw: str, role: str) -> dict:
    """
    Safely parse a JSON response from an LLM.
    Strips markdown fences, handles partial JSON.
    """
    clean = raw.strip()
    # Strip markdown fences
    for fence in ("```json", "```"):
        if fence in clean:
            parts = clean.split(fence)
            # Take the content between first pair of fences
            if len(parts) >= 3:
                clean = parts[1].strip()
            elif len(parts) == 2:
                clean = parts[1].strip()
            break

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Fallback: return error structure so pipeline doesn't break
        return {
            "role": role,
            "verdict": "PARSE_ERROR",
            "confidence": "NONE",
            "reasoning": raw[:500],  # keep first 500 chars for debug
            "parse_error": True,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Individual LLM callers (run in threads)
# ─────────────────────────────────────────────────────────────────────────────

def _run_visual_analyst(forensix_result: dict, results: dict, errors: dict):
    """Thread target: runs visual analyst LLM."""
    role = "visual_analyst"
    try:
        prompt = build_visual_analyst_prompt(forensix_result)
        heatmap = forensix_result.get("signals", {}).get("heatmap_path", None)
        raw = call_llm(role, prompt, image_path=heatmap)
        results[role] = _parse_json_response(raw, role)
        results[role]["raw_response"] = raw
    except Exception as e:
        errors[role] = str(e)
        results[role] = {
            "role": role,
            "verdict": "ERROR",
            "confidence": "NONE",
            "reasoning": str(e),
            "error": True,
        }


def _run_metadata_analyst(forensix_result: dict, results: dict, errors: dict):
    """Thread target: runs metadata timeline analyst LLM."""
    role = "metadata_analyst"
    try:
        prompt = build_metadata_analyst_prompt(forensix_result)
        raw = call_llm(role, prompt)
        results[role] = _parse_json_response(raw, role)
        results[role]["raw_response"] = raw
    except Exception as e:
        errors[role] = str(e)
        results[role] = {
            "role": role,
            "verdict": "ERROR",
            "confidence": "NONE",
            "reasoning": str(e),
            "error": True,
        }


def _run_ai_pattern_analyst(forensix_result: dict, results: dict, errors: dict):
    """Thread target: runs AI generation pattern analyst LLM."""
    role = "ai_pattern_analyst"
    try:
        prompt = build_ai_pattern_analyst_prompt(forensix_result)
        raw = call_llm(role, prompt)
        results[role] = _parse_json_response(raw, role)
        results[role]["raw_response"] = raw
    except Exception as e:
        errors[role] = str(e)
        results[role] = {
            "role": role,
            "verdict": "ERROR",
            "confidence": "NONE",
            "reasoning": str(e),
            "error": True,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Consensus synthesizer
# ─────────────────────────────────────────────────────────────────────────────

# Verdict severity ranking (higher = more severe finding)
VERDICT_RANK = {
    "AUTHENTIC":           0,
    "SUSPICIOUS":          1,
    "HUMAN_EDITED":        2,
    "AI_ASSISTED_TAMPER":  3,
    "TAMPERED":            3,
    "FULLY_AI_GENERATED":  4,
    "UNKNOWN":             1,
    "PARSE_ERROR":         1,
    "ERROR":               1,
}

CONFIDENCE_WEIGHT = {
    "HIGH":   1.0,
    "MEDIUM": 0.6,
    "LOW":    0.3,
    "NONE":   0.0,
}


def _synthesize_consensus(
    visual:   dict,
    metadata: dict,
    ai_pat:   dict,
    forensix_result: dict,
) -> dict:
    """
    Combine three LLM verdicts into a single consensus verdict.

    Strategy:
    - Weighted vote: each model's verdict is weighted by its confidence
    - If any model is HIGH confidence tampered → flag escalates
    - Final verdict is the weighted-highest severity finding
    - Build a unified timeline: WHAT changed + WHEN + AI or human tool
    """

    models = [visual, metadata, ai_pat]

    # ── Weighted severity scoring ────────────────────────────────────────────
    weighted_severity = 0.0
    total_weight = 0.0
    verdict_votes = {}

    for m in models:
        v = m.get("verdict", "UNKNOWN")
        c = m.get("confidence", "NONE")
        severity = VERDICT_RANK.get(v, 1)
        weight   = CONFIDENCE_WEIGHT.get(c, 0.0)

        weighted_severity += severity * weight
        total_weight += weight
        verdict_votes[v] = verdict_votes.get(v, 0) + 1

    avg_severity = weighted_severity / total_weight if total_weight > 0 else 0

    # ── Map average severity back to final verdict ──────────────────────────
    if avg_severity >= 3.5:
        final_verdict = "FULLY_AI_GENERATED"
    elif avg_severity >= 2.5:
        final_verdict = "AI_ASSISTED_TAMPER"
    elif avg_severity >= 1.8:
        final_verdict = "TAMPERED"
    elif avg_severity >= 0.8:
        final_verdict = "SUSPICIOUS"
    else:
        final_verdict = "AUTHENTIC"

    # Override: if 2+ models agree on TAMPERED/AI_ASSISTED → escalate
    tamper_verdicts = sum(1 for m in models
                          if m.get("verdict") in ("TAMPERED", "AI_ASSISTED_TAMPER",
                                                   "FULLY_AI_GENERATED"))
    if tamper_verdicts >= 2 and final_verdict == "SUSPICIOUS":
        final_verdict = "TAMPERED"

    # ── Overall confidence ───────────────────────────────────────────────────
    high_conf_count  = sum(1 for m in models if m.get("confidence") == "HIGH")
    low_error_count  = sum(1 for m in models if m.get("error") or m.get("parse_error"))

    if high_conf_count >= 2 and low_error_count == 0:
        overall_confidence = "HIGH"
    elif high_conf_count >= 1 and low_error_count <= 1:
        overall_confidence = "MEDIUM"
    else:
        overall_confidence = "LOW"

    # ── Extract cross-model insights ─────────────────────────────────────────

    # WHAT was changed (from visual analyst)
    what_changed = visual.get("what_was_changed", "Could not determine")
    tampered_regions = visual.get("tampered_regions", [])
    tampering_sophistication = visual.get("tampering_sophistication", "UNKNOWN")

    # WHEN it was changed (from metadata analyst)
    timeline = metadata.get("timeline", {})
    when_changed = metadata.get("when_was_it_changed", "Could not determine")
    software_used = metadata.get("software_used", {})
    metadata_inconsistencies = metadata.get("metadata_inconsistencies", [])

    # HOW (AI or human tool) — from AI pattern analyst
    ai_technique = ai_pat.get("ai_technique_detected", {})
    human_vs_ai = ai_pat.get("human_vs_ai_edit", "Could not determine")
    ai_artifacts = ai_pat.get("ai_artifacts_found", [])
    recommended_next = ai_pat.get("recommended_next_analysis", "")

    # ── Build unified tampering report ───────────────────────────────────────
    tampering_report = None
    if final_verdict not in ("AUTHENTIC",):
        tampering_report = {
            "what_was_changed":    what_changed,
            "where_was_changed":   tampered_regions,
            "when_was_changed":    when_changed,
            "how_was_changed":     human_vs_ai,
            "tool_or_technique":   ai_technique.get("technique", "UNKNOWN"),
            "likely_tool_used":    (
                ai_technique.get("tool_likely_used")
                or software_used.get("detected_software")
                or "UNKNOWN"
            ),
            "ai_artifacts":        ai_artifacts,
            "metadata_clues":      metadata_inconsistencies,
            "tampering_sophistication": tampering_sophistication,
        }

    # ── Per-model summary (compact, for report) ──────────────────────────────
    model_summaries = {
        "visual_analyst": {
            "verdict":    visual.get("verdict"),
            "confidence": visual.get("confidence"),
            "key_finding": what_changed,
            "error": visual.get("error", False),
        },
        "metadata_analyst": {
            "verdict":    metadata.get("verdict"),
            "confidence": metadata.get("confidence"),
            "key_finding": when_changed,
            "error": metadata.get("error", False),
        },
        "ai_pattern_analyst": {
            "verdict":    ai_pat.get("verdict"),
            "confidence": ai_pat.get("confidence"),
            "key_finding": human_vs_ai,
            "error": ai_pat.get("error", False),
        },
    }

    return {
        # ── Top-level verdict ─────────────────────────────────────
        "final_verdict":         final_verdict,
        "overall_confidence":    overall_confidence,
        "authenticity_score":    forensix_result.get("authenticity_score", -1),
        "verdict_votes":         verdict_votes,
        "weighted_severity":     round(avg_severity, 3),

        # ── Core forensic answer ──────────────────────────────────
        "tampering_report":      tampering_report,

        # ── Per-model breakdown ───────────────────────────────────
        "model_analyses": {
            "visual_analyst":     visual,
            "metadata_analyst":   metadata,
            "ai_pattern_analyst": ai_pat,
        },
        "model_summaries":       model_summaries,

        # ── Recommended next steps ────────────────────────────────
        "recommended_next_analysis": recommended_next or (
            "Run GradCAM visualization and frequency domain (DCT) analysis"
            if final_verdict not in ("AUTHENTIC",) else
            "No further analysis required"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_llm_reasoning(forensix_result: dict) -> dict:
    """
    Main entry point. Accepts a ForensixResult dict from any detection module.
    Runs three LLMs in parallel, synthesizes a final verdict.

    Args:
        forensix_result: Output dict from image/video/pdf detection module

    Returns:
        Full reasoning report with final verdict + tampering breakdown
    """
    start = time.time()
    active_models = get_active_models()

    # Warm-up / health check (dev mode only; logs warnings, never raises)
    _warm_up_ollama()

    results = {}
    errors  = {}

    # ── Launch three LLMs in parallel threads ────────────────────────────────
    threads = [
        threading.Thread(target=_run_visual_analyst,
                         args=(forensix_result, results, errors), daemon=True),
        threading.Thread(target=_run_metadata_analyst,
                         args=(forensix_result, results, errors), daemon=True),
        threading.Thread(target=_run_ai_pattern_analyst,
                         args=(forensix_result, results, errors), daemon=True),
    ]

    for t in threads:
        t.start()

    # Wait for all to finish (max 3 minutes per model)
    for t in threads:
        t.join(timeout=200)

    # ── Synthesize consensus ─────────────────────────────────────────────────
    visual   = results.get("visual_analyst",     {"role": "visual_analyst",    "verdict": "ERROR", "confidence": "NONE", "error": True})
    metadata = results.get("metadata_analyst",   {"role": "metadata_analyst",  "verdict": "ERROR", "confidence": "NONE", "error": True})
    ai_pat   = results.get("ai_pattern_analyst", {"role": "ai_pattern_analyst","verdict": "ERROR", "confidence": "NONE", "error": True})

    consensus = _synthesize_consensus(visual, metadata, ai_pat, forensix_result)

    # ── Wrap final output ────────────────────────────────────────────────────
    return {
        "llm_reasoning": {
            **consensus,
            "models_used":        active_models,
            "llm_processing_time": round(time.time() - start, 2),
            "errors":             errors if errors else None,
        }
    }