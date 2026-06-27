"""
noise_analyzer.py — Noise Texture & Frequency Analysis for ForensiX
--------------------------------------------------------------------
Detects AI-generated images (ChatGPT/DALL-E, Midjourney, SD) using signals
that remain effective even when ELA returns near-zero (which it always does
on AI-generated PNGs that have no prior JPEG compression history).

Signals produced:
  noise_score           — composite 0–1 suspicion score (0=clean, 1=AI/tampered)
  noise_block_cv        — robust noise coefficient of variation (MAD/median)
  noise_mean            — mean local noise across image
  noise_std_of_stds     — std dev of block noise levels
  freq_high_ratio       — fraction of DCT energy in high-frequency bands
  freq_uniformity       — flatness of frequency spectrum
  bg_min_corner_std     — minimum corner patch std (AI backgrounds are perfectly flat)
  bg_mean_corner_std    — mean of all four corner patch stds
  ai_smooth_flag        — True if image is suspiciously smooth

Key calibrated thresholds (derived from the ChatGPT vs real photo comparison):
  bg_min_corner_std < 3.0  → AI-generated (perfectly rendered background)
  bg_min_corner_std < 1.0  → near-certain AI generation (pure white/solid bg)
  noise_block_cv > 1.0     → highly heterogeneous (many flat + some high blocks)
"""

import numpy as np
from PIL import Image


# ─── Thresholds ───────────────────────────────────────────────────────────────
AI_BG_STD_CRITICAL  = 1.0   # min corner std below this = near-certain AI
AI_BG_STD_WARNING   = 3.0   # min corner std below this = suspicious
FREQ_LOW_THRESHOLD  = 0.12  # freq_high_ratio below this = suspiciously smooth


def _to_gray(image_path: str) -> np.ndarray:
    return np.array(Image.open(image_path).convert("L"), dtype=np.float32)


def _to_rgb(image_path: str) -> np.ndarray:
    return np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)


def analyze_noise(image_path: str, block_size: int = 8) -> dict:
    """
    Compute noise texture and frequency signals from an image.

    Returns:
        dict of noise/frequency signals ready to merge into detector signals dict.
    """
    try:
        gray = _to_gray(image_path)
        arr  = _to_rgb(image_path)
        h, w = gray.shape

        # ── 1. Block noise (local std per block) ─────────────────────────────
        block_stds = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i + block_size, j:j + block_size]
                block_stds.append(float(np.std(block)))
        block_stds = np.array(block_stds)

        noise_mean     = float(np.mean(block_stds))
        noise_std_stds = float(np.std(block_stds))

        # Robust CV: median absolute deviation / median
        # More stable than std/mean when many blocks are near-zero (flat bg)
        med_bs = np.median(block_stds)
        mad_bs = np.median(np.abs(block_stds - med_bs))
        noise_block_cv = float(mad_bs / (med_bs + 1e-9))

        # ── 2. Background uniformity — KEY AI-detection signal ────────────────
        # Sample the four corners (most likely to be background in portraits)
        # AI-generated images: perfectly rendered white/solid bg → std < 1.0
        # Real photographs:    real-world texture/noise → std typically > 5.0
        patch_size = max(32, min(h, w) // 8)
        patches = [
            arr[:patch_size, :patch_size],            # top-left
            arr[:patch_size, -patch_size:],            # top-right
            arr[-patch_size:, :patch_size],            # bottom-left
            arr[-patch_size:, -patch_size:],           # bottom-right
        ]
        patch_stds = [float(np.std(p)) for p in patches]
        bg_min_corner_std  = float(min(patch_stds))
        bg_mean_corner_std = float(np.mean(patch_stds))

        # ── 3. Frequency domain (FFT magnitude spectrum) ──────────────────────
        crop_size = min(256, h, w)
        cy, cx = h // 2, w // 2
        half   = crop_size // 2
        crop   = gray[cy - half:cy + half, cx - half:cx + half]

        fft     = np.fft.fft2(crop)
        fft_mag = np.abs(np.fft.fftshift(fft))
        total_e = float(np.sum(fft_mag))

        if total_e < 1e-6:
            freq_high_ratio = 0.0
            freq_uniformity = 1.0
        else:
            fh, fw = fft_mag.shape
            cy2, cx2 = fh // 2, fw // 2
            yy, xx   = np.ogrid[:fh, :fw]
            dist     = np.sqrt((yy - cy2) ** 2 + (xx - cx2) ** 2)
            inner_r  = 0.30 * min(fh, fw)
            high_e   = float(np.sum(fft_mag[dist > inner_r]))
            freq_high_ratio = high_e / total_e

            # Angular uniformity (8 sectors)
            sector_energies = []
            for s in range(8):
                a0 = s * (2 * np.pi / 8)
                a1 = (s + 1) * (2 * np.pi / 8)
                angle = np.arctan2(yy - cy2, xx - cx2) % (2 * np.pi)
                mask  = (angle >= a0) & (angle < a1)
                sector_energies.append(float(np.sum(fft_mag[mask])))
            se  = np.array(sector_energies)
            mse = float(np.mean(se))
            freq_uniformity = float(
                np.clip(1.0 - (np.std(se) / (mse + 1e-9)), 0.0, 1.0)
            )

        # ── 4. AI-smooth flag ─────────────────────────────────────────────────
        # Fires when background is suspiciously flat (strongest AI tell)
        ai_smooth_flag = bg_min_corner_std < AI_BG_STD_WARNING

        # ── 5. Composite noise_score ──────────────────────────────────────────
        # Weighted combination of the two most reliable signals:
        #   bg_min_corner_std: 0.0 (perfectly flat) → 1.0 (very noisy background)
        #   freq_high_ratio:   already 0–1 (higher = more high-freq content)
        #
        # For noise_score: 0 = authentic, 1 = suspicious
        #
        # Background component: penalise suspiciously flat backgrounds
        if bg_min_corner_std < AI_BG_STD_CRITICAL:
            bg_component = 0.9    # near-certain AI
        elif bg_min_corner_std < AI_BG_STD_WARNING:
            # Scale linearly between critical and warning threshold
            bg_component = 0.4 + 0.5 * (1.0 - (bg_min_corner_std - AI_BG_STD_CRITICAL)
                                              / (AI_BG_STD_WARNING - AI_BG_STD_CRITICAL))
        else:
            bg_component = 0.0

        # Freq component: very low high-freq = suspicious (but only mildly)
        freq_component = max(0.0, (FREQ_LOW_THRESHOLD - freq_high_ratio) / FREQ_LOW_THRESHOLD) \
                         if freq_high_ratio < FREQ_LOW_THRESHOLD else 0.0

        noise_score = float(np.clip(0.70 * bg_component + 0.30 * freq_component, 0.0, 1.0))

        return {
            "noise_score":         round(noise_score, 4),
            "noise_block_cv":      round(noise_block_cv, 4),
            "noise_mean":          round(noise_mean, 4),
            "noise_std_of_stds":   round(noise_std_stds, 4),
            "freq_high_ratio":     round(freq_high_ratio, 4),
            "freq_uniformity":     round(freq_uniformity, 4),
            "bg_min_corner_std":   round(bg_min_corner_std, 4),
            "bg_mean_corner_std":  round(bg_mean_corner_std, 4),
            "ai_smooth_flag":      ai_smooth_flag,
            "noise_error":         None,
        }

    except Exception as e:
        return {
            "noise_score":        None,
            "noise_block_cv":     None,
            "noise_mean":         None,
            "noise_std_of_stds":  None,
            "freq_high_ratio":    None,
            "freq_uniformity":    None,
            "bg_min_corner_std":  None,
            "bg_mean_corner_std": None,
            "ai_smooth_flag":     False,
            "noise_error":        str(e),
        }
