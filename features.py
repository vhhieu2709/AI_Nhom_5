import logging
import shutil
import tempfile
import urllib.request
from pathlib import Path

import mediapipe as mp
import numpy as np


LOGGER = logging.getLogger(__name__)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def ensure_hand_landmarker_model(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "hand_landmarker.task"
    if not model_path.exists():
        LOGGER.info("Downloading MediaPipe hand landmarker model to %s", model_path)
        urllib.request.urlretrieve(MODEL_URL, model_path)

    # MediaPipe's C layer can fail on non-ASCII Windows paths, so keep an ASCII temp copy.
    temp_model = Path(tempfile.gettempdir()) / "asl_hand_landmarker.task"
    if not temp_model.exists() or temp_model.stat().st_size != model_path.stat().st_size:
        shutil.copyfile(model_path, temp_model)
    return temp_model


class HandDetector:
    def __init__(self, output_dir: str | Path, min_detection_confidence: float = 0.7):
        self.output_dir = Path(output_dir)
        self.min_detection_confidence = min_detection_confidence
        self._mode = None
        self._detector = None

    def __enter__(self):
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self._mode = "solutions"
            self._detector = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=self.min_detection_confidence,
            )
            LOGGER.info("Using MediaPipe solutions Hands API")
            return self

        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import hand_landmarker

        model_path = ensure_hand_landmarker_model(self.output_dir)
        options = hand_landmarker.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            num_hands=1,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_detection_confidence,
        )
        self._mode = "tasks"
        self._detector = hand_landmarker.HandLandmarker.create_from_options(options)
        LOGGER.info("Using MediaPipe Tasks HandLandmarker API")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._detector is not None:
            self._detector.close()

    def detect(self, image_rgb: np.ndarray) -> np.ndarray | None:
        if self._mode == "solutions":
            result = self._detector.process(image_rgb)
            if not result.multi_hand_landmarks:
                return None
            landmarks = result.multi_hand_landmarks[0].landmark
            return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

        result = self._detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb))
        if not result.hand_landmarks:
            return None
        return np.array([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]], dtype=np.float32)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray | None:
    """Anchor to wrist and scale by max wrist distance to remove hand position and size."""
    centered = landmarks.astype(np.float32) - landmarks[0].astype(np.float32)
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale < 1e-8:
        return None
    return centered / scale


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-8:
        return 0.0
    cos_value = float(np.dot(ba, bc) / denom)
    return float(np.arccos(np.clip(cos_value, -1.0, 1.0)))


def angle_features(normalized_landmarks: np.ndarray) -> np.ndarray:
    pts = normalized_landmarks.reshape(21, 3)
    pip_triplets = [(5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19)]
    mcp_triplets = [(0, 5, 6), (0, 9, 10), (0, 13, 14), (0, 17, 18)]
    values = [_angle(pts[a], pts[b], pts[c]) for a, b, c in pip_triplets]
    values.extend(_angle(pts[a], pts[b], pts[c]) for a, b, c in mcp_triplets)
    values.append(_angle(pts[4], pts[0], pts[8]))
    return np.array(values, dtype=np.float32)


def build_feature_vector(landmarks: np.ndarray, use_angles: bool = False) -> np.ndarray | None:
    normalized = normalize_landmarks(landmarks)
    if normalized is None:
        return None
    coords = normalized.reshape(-1).astype(np.float32)
    if not use_angles:
        return coords
    return np.concatenate([coords, angle_features(normalized)], axis=0).astype(np.float32)
