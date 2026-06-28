"""
metadata_extractor.py — EXIF & Image Origin Metadata Analysis for ForensiX
---------------------------------------------------------------------------
Extracts EXIF/metadata signals that indicate image origin and editing history.

Key signals produced:
  has_exif            — bool: EXIF block present in file
  software_tag        — editing software string if present (Photoshop, GIMP, etc.)
  camera_make         — camera manufacturer if present
  camera_model        — camera model if present
  date_time_original  — original capture timestamp
  date_time_digitized — digitization timestamp
  file_format         — PNG / JPEG / etc.
  metadata_flags      — list of forensic flag strings raised

Forensic flag logic:
  MISSING_EXIF_ON_JPEG    → JPEG with no EXIF (likely re-saved/edited)
  MISSING_EXIF_ON_PNG     → PNG with no EXIF (common for AI-generated images)
  EDITING_SOFTWARE_DETECTED → explicit editor name in Software EXIF tag
  NO_CAMERA_DATA          → no Make/Model (consistent with AI generation or screenshot)
  TIMESTAMP_MISMATCH      → DateTimeOriginal != DateTimeDigitized
  AI_GENERATOR_SIGNATURE  → software tag matches known AI tool names
"""

from pathlib import Path
from typing import Optional

# PIL / Pillow — use getexif() on JPEG, info dict on PNG
from PIL import Image
from PIL.ExifTags import TAGS

# ─── Known AI generation software signatures ─────────────────────────────────
AI_SOFTWARE_KEYWORDS = {
    "stable diffusion", "midjourney", "dall-e", "dall·e",
    "firefly", "imagen", "openai", "chatgpt", "comfyui",
    "automatic1111", "invokeai", "leonardo", "runway",
    "bing image creator", "adobe firefly",
}

# ─── Known editing software keywords ─────────────────────────────────────────
EDITING_SOFTWARE_KEYWORDS = {
    "photoshop", "gimp", "lightroom", "affinity photo",
    "paint.net", "canva", "snapseed", "pixelmator",
    "capture one", "darktable", "rawtherapee",
}


def extract_metadata(image_path: str) -> dict:
    """
    Extract EXIF and image-format metadata from any supported image file.

    Returns a flat dict ready to be stored as signals["metadata"].
    All fields are present; missing values are None or False.
    """
    path = Path(image_path)
    result = {
        "has_exif":            False,
        "file_format":         path.suffix.upper().lstrip("."),
        "software_tag":        None,
        "camera_make":         None,
        "camera_model":        None,
        "date_time_original":  None,
        "date_time_digitized": None,
        "image_width":         None,
        "image_height":        None,
        "color_space":         None,
        "file_size_bytes":     None,
        "file_size_kb":        None,
        "metadata_flags":      [],
        "metadata_error":      None,
        "raw_exif_count":      0,
    }

    # File size — used to detect social-media recompression (WhatsApp etc.)
    try:
        size_bytes = path.stat().st_size
        result["file_size_bytes"] = size_bytes
        result["file_size_kb"] = round(size_bytes / 1024.0, 1)
    except Exception:
        size_bytes = None

    try:
        img = Image.open(image_path)
        result["image_width"]  = img.width
        result["image_height"] = img.height
        result["color_space"]  = img.mode

        # ── Try EXIF (works on JPEG, TIFF, some PNG with exif chunk) ──────────
        exif_data = {}
        try:
            raw_exif = img.getexif()
            if raw_exif:
                exif_data = {TAGS.get(k, str(k)): v for k, v in raw_exif.items()}
                result["has_exif"]       = True
                result["raw_exif_count"] = len(exif_data)
        except Exception:
            pass

        # ── Also check PNG info dict (may carry metadata even without EXIF) ──
        png_info = {}
        if hasattr(img, "info") and img.info:
            png_info = img.info

        # ── Extract key fields ────────────────────────────────────────────────
        result["software_tag"]        = (exif_data.get("Software")
                                          or png_info.get("Software")
                                          or png_info.get("software"))
        result["camera_make"]         = exif_data.get("Make")
        result["camera_model"]        = exif_data.get("Model")
        result["date_time_original"]  = exif_data.get("DateTimeOriginal")
        result["date_time_digitized"] = exif_data.get("DateTimeDigitized")

        # Clean up string fields
        for field in ("software_tag", "camera_make", "camera_model",
                      "date_time_original", "date_time_digitized"):
            v = result[field]
            if isinstance(v, bytes):
                result[field] = v.decode("utf-8", errors="replace").strip()
            elif isinstance(v, str):
                result[field] = v.strip() or None

        # ── Forensic flag logic ───────────────────────────────────────────────
        flags = []
        fmt = result["file_format"]
        sw  = (result["software_tag"] or "").lower()

        # EXIF absence flags
        if not result["has_exif"]:
            if fmt in ("JPG", "JPEG"):
                flags.append("MISSING_EXIF_ON_JPEG")
            elif fmt == "PNG":
                flags.append("MISSING_EXIF_ON_PNG")

        # No camera hardware data — consistent with AI generation or screengrab
        if not result["camera_make"] and not result["camera_model"]:
            flags.append("NO_CAMERA_DATA")

        # AI generation software signature
        if sw and any(kw in sw for kw in AI_SOFTWARE_KEYWORDS):
            flags.append("AI_GENERATOR_SIGNATURE")

        # Editing software signature
        if sw and any(kw in sw for kw in EDITING_SOFTWARE_KEYWORDS):
            flags.append("EDITING_SOFTWARE_DETECTED")

        # Timestamp mismatch
        dto = result["date_time_original"]
        dtd = result["date_time_digitized"]
        if dto and dtd and dto != dtd:
            flags.append("TIMESTAMP_MISMATCH")

        # ── Social-media (WhatsApp) recompression ─────────────────────────────
        # WhatsApp re-encodes shared photos as JPEG, stripping EXIF down to
        # nothing (or a handful of orientation/size tags) and compressing the
        # file well under ~500KB. A JPEG with little/no EXIF AND a small file
        # size is consistent with this benign social-media pipeline rather than
        # malicious tampering, so we flag it to soften the metadata penalty.
        WHATSAPP_MAX_BYTES = 500 * 1024   # 500 KB
        WHATSAPP_MAX_EXIF  = 4            # "minimal EXIF" threshold
        minimal_exif = (not result["has_exif"]) or result["raw_exif_count"] <= WHATSAPP_MAX_EXIF
        if (fmt in ("JPG", "JPEG")
                and minimal_exif
                and size_bytes is not None
                and size_bytes < WHATSAPP_MAX_BYTES):
            flags.append("POSSIBLE_WHATSAPP_RECOMPRESSION")

        result["metadata_flags"] = flags

    except Exception as e:
        result["metadata_error"] = str(e)

    return result

