import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np


DEFAULT_MODEL_PATH = "model_baseline.pkl"
DEFAULT_HAND_TASK_PATH = "hand_landmarker.task"


def read_image_bgr(image_path):
    data = np.fromfile(str(image_path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def resize_with_padding(image, size=224):
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


def apply_clahe_lab(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def preprocess_image_bgr(image_bgr):
    image_bgr = resize_with_padding(image_bgr, size=224)
    image_bgr = cv2.GaussianBlur(image_bgr, (3, 3), 0)
    image_bgr = apply_clahe_lab(image_bgr)
    return image_bgr


def preprocess_for_mediapipe(image_bgr):
    image_bgr = preprocess_image_bgr(image_bgr)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def normalize_landmarks(landmarks):
    centered = landmarks.astype(np.float32) - landmarks[0].astype(np.float32)
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale < 1e-8:
        return None
    return centered / scale


def build_feature_vector(landmarks):
    normalized = normalize_landmarks(landmarks)
    if normalized is None:
        return None
    return normalized.reshape(-1).astype(np.float32)


def safe_hand_task_path(hand_task_path):
    # MediaPipe C++ can fail on Windows paths containing Vietnamese characters.
    # Copying the task file to temp keeps local and Colab behavior consistent.
    hand_task_path = Path(hand_task_path)
    safe_path = Path(tempfile.gettempdir()) / "asl_hand_landmarker.task"
    if not safe_path.exists() or safe_path.stat().st_size != hand_task_path.stat().st_size:
        shutil.copyfile(hand_task_path, safe_path)
    return safe_path


class HandDetector:
    def __init__(self, hand_task_path=DEFAULT_HAND_TASK_PATH, min_detection_confidence=0.7):
        self.hand_task_path = Path(hand_task_path).resolve()
        self.min_detection_confidence = min_detection_confidence
        self.detector = None

    def __enter__(self):
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import hand_landmarker

        self.previous_cwd = Path.cwd()
        safe_runtime_dir = Path(tempfile.gettempdir()) / "asl_runtime"
        safe_runtime_dir.mkdir(exist_ok=True)
        os.chdir(safe_runtime_dir)

        try:
            options = hand_landmarker.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(safe_hand_task_path(self.hand_task_path))),
                num_hands=1,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_detection_confidence,
            )
            self.detector = hand_landmarker.HandLandmarker.create_from_options(options)
        except Exception:
            os.chdir(self.previous_cwd)
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.detector is not None:
            self.detector.close()
        if hasattr(self, "previous_cwd"):
            os.chdir(self.previous_cwd)

    def detect(self, image_rgb):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.detector.detect(mp_image)

        if not result.hand_landmarks:
            return None

        landmarks = result.hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


class ASLRecognizer:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, hand_task_path=DEFAULT_HAND_TASK_PATH):
        self.artifact = joblib.load(Path(model_path).resolve())
        self.hand_task_path = Path(hand_task_path).resolve()

    def predict_image(self, image_path):
        image_bgr = read_image_bgr(image_path)
        if image_bgr is None:
            return "Nothing", 1.0

        image_rgb = preprocess_for_mediapipe(image_bgr)
        with HandDetector(self.hand_task_path) as detector:
            landmarks = detector.detect(image_rgb)

        if landmarks is None:
            return "Nothing", 1.0

        features = build_feature_vector(landmarks)
        if features is None:
            return "Nothing", 1.0

        probabilities = self.artifact["model"].predict_proba(features.reshape(1, -1))[0]
        index = int(np.argmax(probabilities))
        label = self.artifact["label_encoder"].inverse_transform([index])[0]
        return str(label), float(probabilities[index])

    def predict_webcam(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam {camera_index}")

        with HandDetector(self.hand_task_path) as detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                image_rgb = preprocess_for_mediapipe(frame)
                landmarks = detector.detect(image_rgb)

                label, confidence = "Nothing", 1.0
                if landmarks is not None:
                    features = build_feature_vector(landmarks)
                    if features is not None:
                        probabilities = self.artifact["model"].predict_proba(features.reshape(1, -1))[0]
                        index = int(np.argmax(probabilities))
                        label = str(self.artifact["label_encoder"].inverse_transform([index])[0])
                        confidence = float(probabilities[index])

                    height, width = frame.shape[:2]
                    for x, y, _ in landmarks:
                        cv2.circle(frame, (int(x * width), int(y * height)), 3, (0, 255, 0), -1)

                cv2.putText(
                    frame,
                    f"{label} {confidence:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("ASL Recognition", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="ASL Recognition with MediaPipe + XGBoost")
    parser.add_argument("--image", help="Path to image for single-image prediction")
    parser.add_argument("--webcam", action="store_true", help="Run webcam demo")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--hand-task", default=DEFAULT_HAND_TASK_PATH)
    args = parser.parse_args()

    recognizer = ASLRecognizer(model_path=args.model, hand_task_path=args.hand_task)

    if args.image:
        label, confidence = recognizer.predict_image(args.image)
        print(json.dumps({"label": label, "confidence": confidence}, ensure_ascii=False))
    elif args.webcam:
        recognizer.predict_webcam(camera_index=args.camera_index)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
