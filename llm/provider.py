"""
provider.py — LLM Provider Abstraction for ForensiX
------------------------------------------------------
DEV  mode: calls Ollama running locally at localhost:11434
PROD mode: calls Claude / GPT-4 / Gemini APIs

Switch via environment variable:
    FORENSIX_ENV=dev   → Ollama
    FORENSIX_ENV=prod  → API keys

Three specialist models, each with a dedicated forensic role:
    Model A (llava / claude-sonnet)   → Visual anomaly detector
    Model B (mistral / gpt-4o)        → Metadata & timeline analyst
    Model C (gemma / gemini-pro)      → AI-generation pattern detector
"""

import os
import json
import requests
import base64
from pathlib import Path
from typing import Optional

# ─── Environment ──────────────────────────────────────────────────────────────
ENV = os.getenv("FORENSIX_ENV", "dev").lower()  # "dev" or "prod"

# ─── Ollama config (DEV) ──────────────────────────────────────────────────────
OLLAMA_BASE = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Three Ollama models — each plays a different forensic specialist role
OLLAMA_MODELS = {
    "visual_analyst":    os.getenv("OLLAMA_MODEL_VISUAL",   "llava:13b"),
    "metadata_analyst":  os.getenv("OLLAMA_MODEL_META",     "mistral:7b"),
    "ai_pattern_analyst": os.getenv("OLLAMA_MODEL_AI",      "gemma:7b"),
}

# ─── API config (PROD) ────────────────────────────────────────────────────────
PROD_MODELS = {
    "visual_analyst":     "claude-sonnet-4-6",          # Anthropic
    "metadata_analyst":   "gpt-4o",                     # OpenAI
    "ai_pattern_analyst": "gemini-1.5-pro",             # Google
}

API_KEYS = {
    "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
    "openai":    os.getenv("OPENAI_API_KEY", ""),
    "google":    os.getenv("GOOGLE_API_KEY", ""),
}


def _encode_image(image_path: str) -> str:
    """Base64-encode an image file."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# DEV: Ollama calls
# ─────────────────────────────────────────────────────────────────────────────

def _call_ollama(role: str, prompt: str, image_path: Optional[str] = None) -> str:
    """
    Call an Ollama model by role. Supports multimodal (llava) for visual role.
    Returns raw text response.
    """
    model = OLLAMA_MODELS[role]
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,   # low temp → deterministic forensic output
            "num_predict": 1200,
        }
    }

    # llava supports image input
    if image_path and role == "visual_analyst" and Path(image_path).exists():
        payload["images"] = [_encode_image(image_path)]

    try:
        response = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json=payload,
            timeout=(30, 180)   # (connect timeout, read timeout)
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Ollama not running. Start it with: ollama serve\n"
            f"Then pull models: ollama pull {model}"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Ollama timed out for model {model}")
    except Exception as e:
        raise RuntimeError(f"Ollama error ({model}): {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PROD: API calls
# ─────────────────────────────────────────────────────────────────────────────

def _call_claude(prompt: str, image_path: Optional[str] = None) -> str:
    """Call Claude API (Anthropic) — visual analyst in production."""
    headers = {
        "x-api-key": API_KEYS["anthropic"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    content = []
    if image_path and Path(image_path).exists():
        ext = Path(image_path).suffix.lower().lstrip(".")
        media_type = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": _encode_image(image_path),
            }
        })
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": PROD_MODELS["visual_analyst"],
        "max_tokens": 1200,
        "messages": [{"role": "user", "content": content}],
    }
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers, json=payload, timeout=60
    )
    r.raise_for_status()
    return r.json()["content"][0]["text"].strip()


def _call_openai(prompt: str) -> str:
    """Call GPT-4o API (OpenAI) — metadata analyst in production."""
    headers = {
        "Authorization": f"Bearer {API_KEYS['openai']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": PROD_MODELS["metadata_analyst"],
        "temperature": 0.1,
        "max_tokens": 1200,
        "messages": [{"role": "user", "content": prompt}],
    }
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers, json=payload, timeout=60
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _call_gemini(prompt: str) -> str:
    """Call Gemini Pro API (Google) — AI pattern analyst in production."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{PROD_MODELS['ai_pattern_analyst']}:generateContent"
        f"?key={API_KEYS['google']}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1200},
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Unified public interface
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(role: str, prompt: str, image_path: Optional[str] = None) -> str:
    """
    Route to Ollama (dev) or API (prod) based on FORENSIX_ENV.

    Args:
        role: "visual_analyst" | "metadata_analyst" | "ai_pattern_analyst"
        prompt: The forensic prompt
        image_path: Optional path to image (used by visual_analyst)

    Returns:
        Raw text response from the LLM
    """
    if ENV == "prod":
        if role == "visual_analyst":
            return _call_claude(prompt, image_path)
        elif role == "metadata_analyst":
            return _call_openai(prompt)
        elif role == "ai_pattern_analyst":
            return _call_gemini(prompt)
        else:
            raise ValueError(f"Unknown role: {role}")
    else:
        return _call_ollama(role, prompt, image_path)


def get_active_models() -> dict:
    """Return which models are active in current environment."""
    if ENV == "prod":
        return {
            "env": "production",
            "visual_analyst":     PROD_MODELS["visual_analyst"],
            "metadata_analyst":   PROD_MODELS["metadata_analyst"],
            "ai_pattern_analyst": PROD_MODELS["ai_pattern_analyst"],
        }
    return {
        "env": "development",
        "visual_analyst":     OLLAMA_MODELS["visual_analyst"],
        "metadata_analyst":   OLLAMA_MODELS["metadata_analyst"],
        "ai_pattern_analyst": OLLAMA_MODELS["ai_pattern_analyst"],
    }
