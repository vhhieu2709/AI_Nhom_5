import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import cv2
import joblib
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from features import HandDetector, build_feature_vector
from preprocess import preprocess_for_mediapipe, read_image_bgr


CODE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("ASL_DATA_ROOT", os.environ.get("ASL_ROOT", CODE_ROOT))).resolve()
OUTPUT_DIR = CODE_ROOT / "output"
LOG_PATH = OUTPUT_DIR / "pipeline.log"
TARGET_AUG_CLASSES = ["N", "P", "Space", "M", "D"]
CRITICAL_CLASSES = ["N", "M", "P", "Space", "D"]
CONFUSION_PAIRS = [("M", "N"), ("R", "U"), ("U", "V")]


def setup_logging() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
    )


def class_dirs(root: Path) -> list[Path]:
    dirs = []
    for split in ["train", "val", "test"]:
        split_dir = root / split
        if split_dir.exists():
            dirs.extend([p for p in split_dir.iterdir() if p.is_dir()])
    return dirs


def collect_images(root: Path, include_augmented: bool = False) -> list[tuple[Path, str]]:
    images = []
    for cls_dir in class_dirs(root):
        label = cls_dir.name
        for path in cls_dir.glob("*.jpg"):
            images.append((path, label))

    if include_augmented:
        aug_root = OUTPUT_DIR / "augmented"
        if aug_root.exists():
            for cls_dir in sorted([p for p in aug_root.iterdir() if p.is_dir()]):
                for path in cls_dir.glob("*.jpg"):
                    images.append((path, cls_dir.name))
    return images


def extract_features(use_angles: bool, include_augmented: bool, prefix: str) -> dict:
    logger = logging.getLogger("extract")
    images = collect_images(DATA_ROOT, include_augmented=include_augmented)
    x_values, y_values, paths = [], [], []
    skipped = Counter()
    detected = Counter()
    totals = Counter(label for _, label in images)

    with HandDetector(OUTPUT_DIR, min_detection_confidence=0.7) as detector:
        for idx, (path, label) in enumerate(images, start=1):
            if label == "Nothing":
                skipped["nothing_rule"] += 1
                continue
            image_rgb = preprocess_for_mediapipe(path, size=224)
            if image_rgb is None:
                skipped["read_error"] += 1
                continue
            landmarks = detector.detect(image_rgb)
            if landmarks is None:
                skipped[f"no_hand:{label}"] += 1
                continue
            vector = build_feature_vector(landmarks, use_angles=use_angles)
            if vector is None:
                skipped["bad_landmarks"] += 1
                continue
            x_values.append(vector)
            y_values.append(label)
            paths.append(str(path))
            detected[label] += 1
            if idx % 1000 == 0:
                logger.info("Feature extraction progress: %s/%s", idx, len(images))

    x_array = np.asarray(x_values, dtype=np.float32)
    y_array = np.asarray(y_values)
    np.save(OUTPUT_DIR / f"{prefix}_features.npy", x_array)
    np.save(OUTPUT_DIR / f"{prefix}_labels.npy", y_array)
    if prefix == "baseline":
        np.save(OUTPUT_DIR / "features.npy", x_array)
        np.save(OUTPUT_DIR / "labels.npy", y_array)

    metadata = {
        "feature_dim": int(x_array.shape[1]) if x_array.size else 0,
        "samples": int(len(y_array)),
        "use_angles": use_angles,
        "include_augmented": include_augmented,
        "totals": dict(sorted(totals.items())),
        "detected": dict(sorted(detected.items())),
        "skipped": dict(sorted(skipped.items())),
        "paths": paths,
    }
    (OUTPUT_DIR / f"{prefix}_features_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved %s feature matrix: %s", prefix, x_array.shape)
    return {"X": x_array, "y": y_array, "metadata": metadata}


def train_xgb(X_train: np.ndarray, y_train: np.ndarray, label_encoder: LabelEncoder) -> XGBClassifier:
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        n_estimators=260,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def split_features(X: np.ndarray, y: np.ndarray):
    labels = np.asarray(y)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, labels, test_size=0.30, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def top_confusions(cm: np.ndarray, labels: list[str], limit: int = 5) -> list[dict]:
    pairs = []
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            if i != j and cm[i, j] > 0:
                pairs.append({"actual": actual, "predicted": predicted, "count": int(cm[i, j])})
    return sorted(pairs, key=lambda item: item["count"], reverse=True)[:limit]


def evaluate_model(model, encoder, X_test, y_test, prefix: str) -> dict:
    y_test_encoded = encoder.transform(y_test)
    y_pred_encoded = model.predict(X_test)
    y_pred = encoder.inverse_transform(y_pred_encoded)
    labels = list(encoder.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    text_report = classification_report(y_test, y_pred, labels=labels, zero_division=0)
    result = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "f1_by_class": {label: float(report[label]["f1-score"]) for label in labels},
        "top_confusions": top_confusions(cm, labels, limit=10),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "classification_report_text": text_report,
    }
    (OUTPUT_DIR / f"{prefix}_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / f"{prefix}_classification_report.txt").write_text(text_report, encoding="utf-8")
    np.save(OUTPUT_DIR / f"{prefix}_confusion_matrix.npy", cm)
    return result


def run_training(prefix: str, use_angles: bool = False, include_augmented: bool = False) -> dict:
    logger = logging.getLogger(prefix)
    data = extract_features(use_angles=use_angles, include_augmented=include_augmented, prefix=prefix)
    X_train, X_val, X_test, y_train, y_val, y_test = split_features(data["X"], data["y"])

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    model = train_xgb(X_train, y_train_encoded, encoder)
    metrics = evaluate_model(model, encoder, X_test, y_test, prefix=prefix)

    artifact = {
        "model": model,
        "label_encoder": encoder,
        "use_angles": use_angles,
        "include_augmented": include_augmented,
        "feature_dim": int(data["X"].shape[1]),
        "metrics": metrics,
    }
    joblib.dump(artifact, OUTPUT_DIR / f"model_{prefix}.pkl")
    if prefix == "baseline":
        joblib.dump(artifact, OUTPUT_DIR / "model_baseline.pkl")
    logger.info("Saved model_%s.pkl with accuracy %.4f", prefix, metrics["accuracy"])
    return {"metrics": metrics, "metadata": data["metadata"]}


def augment_image(image_bgr: np.ndarray, angle: float | None = None, brightness: float | None = None) -> np.ndarray:
    output = image_bgr.copy()
    if angle is not None:
        h, w = output.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        output = cv2.warpAffine(output, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if brightness is not None:
        output = cv2.convertScaleAbs(output, alpha=1.0 + brightness, beta=0)
    return output


def write_image(path: Path, image_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = cv2.imencode(".jpg", image_bgr)[1]
    encoded.tofile(str(path))


def run_augmentation_until_target(target: int = 380) -> dict:
    logger = logging.getLogger("augment")
    aug_root = OUTPUT_DIR / "augmented"
    aug_root.mkdir(parents=True, exist_ok=True)

    before = json.loads((OUTPUT_DIR / "baseline_features_metadata.json").read_text(encoding="utf-8"))
    detected = Counter(before["detected"])
    created = Counter()
    kept = Counter()

    transforms = [("rot_m10", -10, None), ("rot_p10", 10, None), ("rot_m15", -15, None), ("rot_p15", 15, None), ("b_dark", None, -0.2), ("b_bright", None, 0.2)]
    with HandDetector(OUTPUT_DIR, min_detection_confidence=0.7) as detector:
        for label in TARGET_AUG_CLASSES:
            if detected[label] >= target:
                logger.info("%s already has %s detected samples", label, detected[label])
                continue
            originals = []
            for split in ["train", "val", "test"]:
                originals.extend(sorted((DATA_ROOT / split / label).glob("*.jpg")))
            for image_path in originals:
                image_bgr = read_image_bgr(image_path)
                if image_bgr is None:
                    continue
                for name, angle, brightness in transforms:
                    if detected[label] + kept[label] >= target:
                        break
                    augmented = augment_image(image_bgr, angle=angle, brightness=brightness)
                    out_path = aug_root / label / f"{image_path.stem}_{name}.jpg"
                    write_image(out_path, augmented)
                    created[label] += 1
                    image_rgb = preprocess_for_mediapipe(out_path, size=224)
                    landmarks = detector.detect(image_rgb) if image_rgb is not None else None
                    if landmarks is None:
                        out_path.unlink(missing_ok=True)
                    else:
                        kept[label] += 1
                if detected[label] + kept[label] >= target:
                    break
            logger.info("%s augmentation created=%s kept=%s total_detected_target=%s", label, created[label], kept[label], detected[label] + kept[label])

    summary = {"target": target, "created": dict(created), "kept": dict(kept), "baseline_detected": dict(detected)}
    (OUTPUT_DIR / "augmentation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def compare_metrics(before: dict, after: dict, classes: list[str]) -> dict:
    return {
        label: {
            "before": before["f1_by_class"].get(label, 0.0),
            "after": after["f1_by_class"].get(label, 0.0),
            "delta": after["f1_by_class"].get(label, 0.0) - before["f1_by_class"].get(label, 0.0),
        }
        for label in classes
    }


def decide_next_phase(metrics: dict) -> dict:
    low_f1 = [label for label in ["N", "M", "P", "Space"] if metrics["f1_by_class"].get(label, 0.0) < 0.70]
    high_confusion = []
    for actual, predicted in CONFUSION_PAIRS:
        total = sum(1 for item in metrics["top_confusions"] if {item["actual"], item["predicted"]} == {actual, predicted})
        if total:
            high_confusion.append(f"{actual}/{predicted}")
    if metrics["accuracy"] >= 0.88:
        return {"next": "phase3", "reason": "overall accuracy >= 88%"}
    if low_f1 and high_confusion:
        return {"next": "phase2a_then_2b", "reason": f"low F1 {low_f1}; high confusion {high_confusion}"}
    if low_f1:
        return {"next": "phase2a", "reason": f"low F1 {low_f1}"}
    return {"next": "phase2b", "reason": "accuracy below threshold and confusion/shape pairs need feature engineering"}


def git_commit(message: str) -> None:
    import subprocess

    if not (CODE_ROOT / ".git").exists():
        subprocess.run(["git", "init"], cwd=CODE_ROOT, check=True)
        subprocess.run(["git", "config", "user.email", "codex@example.local"], cwd=CODE_ROOT, check=True)
        subprocess.run(["git", "config", "user.name", "Codex"], cwd=CODE_ROOT, check=True)
    subprocess.run(["git", "add", ".gitignore", "*.py", "PROGRESS.md", "output"], cwd=CODE_ROOT, check=True)
    status = subprocess.run(["git", "status", "--short"], cwd=CODE_ROOT, check=True, capture_output=True, text=True)
    if status.stdout.strip():
        subprocess.run(["git", "commit", "-m", message], cwd=CODE_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["auto", "baseline", "augment", "angle"], default="auto")
    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger("main")
    logger.info("Code root: %s", CODE_ROOT)
    logger.info("Dataset root: %s", DATA_ROOT)

    baseline = run_training("baseline", use_angles=False, include_augmented=False)
    decision = decide_next_phase(baseline["metrics"])
    phase_report = {"phase1": baseline, "decision_after_phase1": decision}
    (OUTPUT_DIR / "phase_decisions.json").write_text(json.dumps(phase_report, ensure_ascii=False, indent=2), encoding="utf-8")
    git_commit("Phase 1 baseline MediaPipe XGBoost")

    best_prefix = "baseline"
    best_metrics = baseline["metrics"]

    if decision["next"] in {"phase2a", "phase2a_then_2b"}:
        run_augmentation_until_target(target=380)
        aug = run_training("aug", use_angles=False, include_augmented=True)
        aug_compare = compare_metrics(baseline["metrics"], aug["metrics"], CRITICAL_CLASSES)
        phase_report["phase2a"] = {"result": aug, "f1_comparison": aug_compare}
        (OUTPUT_DIR / "phase_decisions.json").write_text(json.dumps(phase_report, ensure_ascii=False, indent=2), encoding="utf-8")
        git_commit("Phase 2A targeted augmentation")
        if aug["metrics"]["accuracy"] > best_metrics["accuracy"]:
            best_prefix, best_metrics = "aug", aug["metrics"]

    if decision["next"] in {"phase2b", "phase2a_then_2b"}:
        include_aug = decision["next"] == "phase2a_then_2b"
        base_for_compare = phase_report.get("phase2a", {}).get("result", baseline)
        angle = run_training("angle", use_angles=True, include_augmented=include_aug)
        angle_compare = compare_metrics(base_for_compare["metrics"], angle["metrics"], ["M", "N", "R", "U", "V"])
        phase_report["phase2b"] = {"result": angle, "f1_comparison": angle_compare}
        (OUTPUT_DIR / "phase_decisions.json").write_text(json.dumps(phase_report, ensure_ascii=False, indent=2), encoding="utf-8")
        git_commit("Phase 2B angle features")
        if angle["metrics"]["accuracy"] > best_metrics["accuracy"]:
            best_prefix, best_metrics = "angle", angle["metrics"]

    phase_report["best_model"] = {"prefix": best_prefix, "accuracy": best_metrics["accuracy"]}
    (OUTPUT_DIR / "phase_decisions.json").write_text(json.dumps(phase_report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Best model: %s accuracy=%.4f", best_prefix, best_metrics["accuracy"])


if __name__ == "__main__":
    main()
