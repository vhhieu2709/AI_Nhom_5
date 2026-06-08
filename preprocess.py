import logging
from pathlib import Path

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)


def read_image_bgr(image_path: str | Path) -> np.ndarray | None:
    """Read images through imdecode because cv2.imread often fails on Unicode Windows paths."""
    path = Path(image_path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            LOGGER.warning("Could not decode image: %s", path)
        return image
    except OSError as exc:
        LOGGER.warning("Could not read image %s: %s", path, exc)
        return None


def resize_with_padding(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Keep geometry stable; landmark models are sensitive to distorted finger proportions."""
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    padded = np.zeros((size, size, 3), dtype=resized.dtype)
    top = (size - new_height) // 2
    left = (size - new_width) // 2
    padded[top : top + new_height, left : left + new_width] = resized
    return padded


def apply_clahe_lab(image_bgr: np.ndarray) -> np.ndarray:
    """CLAHE is applied only on luminance so hand color is not shifted unnecessarily."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def preprocess_bgr(image_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    image_bgr = resize_with_padding(image_bgr, size=size)
    image_bgr = cv2.GaussianBlur(image_bgr, (3, 3), 0)
    return apply_clahe_lab(image_bgr)


def preprocess_for_mediapipe(image_path: str | Path, size: int = 224) -> np.ndarray | None:
    image_bgr = read_image_bgr(image_path)
    if image_bgr is None:
        return None
    image_bgr = preprocess_bgr(image_bgr, size=size)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
