import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io


def compute_ela(image_path: str, quality: int = 90, scale: int = 15) -> np.ndarray:
    original = Image.open(image_path).convert("RGB")

    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    ela_image = ImageChops.difference(original, recompressed)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale * 255.0 / max_diff)

    return np.array(ela_image)


def ela_score(ela_array: np.ndarray) -> float:
    mean_val = np.mean(ela_array)
    std_val = np.std(ela_array)
    raw_score = (mean_val + std_val) / 255.0
    return float(np.clip(raw_score, 0.0, 1.0))


def save_ela_heatmap(ela_array: np.ndarray, output_path: str) -> str:
    heatmap = cv2.applyColorMap(
        cv2.cvtColor(ela_array, cv2.COLOR_RGB2GRAY),
        cv2.COLORMAP_JET
    )
    cv2.imwrite(output_path, heatmap)
    return output_path