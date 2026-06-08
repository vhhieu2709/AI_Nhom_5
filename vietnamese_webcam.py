import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from asl_model import ASLRecognizer, HandDetector, build_feature_vector, preprocess_for_mediapipe
from vietnamese_telex import VietnameseTelexComposer


HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


class StableSignReader:
    def __init__(self, min_confidence=0.82, stable_frames=8, repeat_seconds=1.25):
        self.min_confidence = min_confidence
        self.stable_frames = stable_frames
        self.repeat_seconds = repeat_seconds
        self.candidate = None
        self.count = 0
        self.last_committed = None
        self.last_commit_time = 0.0

    def update(self, label, confidence):
        if label == "Nothing" or confidence < self.min_confidence:
            self.candidate = None
            self.count = 0
            self.last_committed = None
            return None

        if label == self.candidate:
            self.count += 1
        else:
            self.candidate = label
            self.count = 1

        if self.count < self.stable_frames:
            return None

        now = time.monotonic()
        if label != self.last_committed or now - self.last_commit_time >= self.repeat_seconds:
            self.last_committed = label
            self.last_commit_time = now
            return label
        return None


def load_font(size):
    paths = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in paths:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_unicode_panel(frame, lines):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image, "RGBA")
    font_big = load_font(34)
    font_small = load_font(24)

    panel_height = 168
    draw.rectangle((0, 0, image.width, panel_height), fill=(0, 0, 0, 120))
    y = 14
    for index, (text, color) in enumerate(lines):
        font = font_big if index == 0 else font_small
        draw.text((18, y), text, font=font, fill=color)
        y += 44 if index == 0 else 34

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def draw_landmarks(frame, landmarks):
    height, width = frame.shape[:2]
    points = [(int(x * width), int(y * height)) for x, y, _ in landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 4, (0, 255, 0), -1, cv2.LINE_AA)


def predict_from_frame(frame, detector, artifact):
    image_rgb = preprocess_for_mediapipe(frame)
    landmarks = detector.detect(image_rgb)
    if landmarks is None:
        return "Nothing", 1.0, None

    features = build_feature_vector(landmarks)
    if features is None:
        return "Nothing", 1.0, landmarks

    probabilities = artifact["model"].predict_proba(features.reshape(1, -1))[0]
    index = int(np.argmax(probabilities))
    label = str(artifact["label_encoder"].inverse_transform([index])[0])
    confidence = float(probabilities[index])
    return label, confidence, landmarks


def run(args):
    recognizer = ASLRecognizer(model_path=args.model, hand_task_path=args.hand_task)
    composer = VietnameseTelexComposer(uppercase_output=args.uppercase_output)
    reader = StableSignReader(
        min_confidence=args.min_confidence,
        stable_frames=args.stable_frames,
        repeat_seconds=args.repeat_seconds,
    )

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam {args.camera_index}")

    last_committed = ""
    with HandDetector(args.hand_task, min_detection_confidence=args.min_confidence) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            label, confidence, landmarks = predict_from_frame(frame, detector, recognizer.artifact)
            committed = reader.update(label, confidence)
            if committed is not None:
                before = composer.text
                composer.feed_label(committed)
                last_committed = f"{committed} -> {composer.text[len(before):] or composer.text[-1:]}"

            if landmarks is not None:
                draw_landmarks(frame, landmarks)

            text_preview = composer.text[-48:] if composer.text else "(empty)"
            lines = [
                (f"{label} {confidence:.2f}", (0, 255, 0, 255)),
                (f"Text: {text_preview}", (255, 255, 255, 255)),
                (f"Last: {last_committed or '-'}", (200, 245, 220, 255)),
                ("q quit | space space | b backspace | c clear | n newline", (220, 220, 220, 255)),
            ]
            frame = draw_unicode_panel(frame, lines)

            cv2.imshow("ASL Vietnamese realtime", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("b"):
                composer.backspace()
                last_committed = "keyboard backspace"
            elif key == ord("c"):
                composer.clear()
                last_committed = "keyboard clear"
            elif key == ord("n"):
                composer.text += "\n"
                last_committed = "keyboard newline"
            elif key == ord(" "):
                composer.text += " "
                last_committed = "keyboard space"

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Local webcam ASL to Vietnamese Telex text")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--model", default="model_baseline.pkl")
    parser.add_argument("--hand-task", default="hand_landmarker.task")
    parser.add_argument("--min-confidence", type=float, default=0.82)
    parser.add_argument("--stable-frames", type=int, default=8)
    parser.add_argument("--repeat-seconds", type=float, default=1.25)
    parser.add_argument("--uppercase-output", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
