import time
from pathlib import Path
from typing import Optional

from schema import ForensixResult, make_error_result
from modules.image.ela import compute_ela, ela_score, save_ela_heatmap

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def analyze_image(image_path: str, save_heatmap: bool = True) -> dict:
    start = time.time()
    path = Path(image_path)

    if not path.exists():
        return make_error_result("image", path.name, f"File not found: {image_path}")

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        return make_error_result("image", path.name, f"Unsupported format: {path.suffix}")

    try:
        flags = []
        signals = {}

        # ELA Analysis
        ela_array = compute_ela(str(path))
        ela_suspicion = ela_score(ela_array)
        signals["ela_score"] = round(ela_suspicion, 4)

        if ela_suspicion > 0.15:
            flags.append("HIGH_ELA_SUSPICION")

        # Save heatmap
        if save_heatmap:
            heatmap_dir = Path("outputs/heatmaps")
            heatmap_dir.mkdir(parents=True, exist_ok=True)
            heatmap_path = str(heatmap_dir / f"ela_{path.stem}.jpg")
            save_ela_heatmap(ela_array, heatmap_path)
            signals["heatmap_path"] = heatmap_path

        # CNN placeholder
        signals["cnn_score"] = None
        signals["model_loaded"] = False

        # Final score
        authenticity_score = round(1.0 - ela_suspicion, 4)
        processing_time = round(time.time() - start, 3)

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