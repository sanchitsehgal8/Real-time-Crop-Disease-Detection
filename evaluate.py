from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best.pt"
DATA_DIR = ROOT / "data"
EVAL_DIR = ROOT / "runs" / "evaluation"
VAL_ARTIFACT_DIR = EVAL_DIR / "val"
TEST_ARTIFACT_DIR = EVAL_DIR / "test"
SAMPLES_DIR = EVAL_DIR / "prediction_samples"
CONFUSION_MATRIX_PATH = EVAL_DIR / "confusion_matrix.png"
PER_CLASS_CSV_PATH = EVAL_DIR / "per_class_accuracy.csv"
PER_CLASS_PLOT_PATH = EVAL_DIR / "per_class_accuracy.png"
METRICS_PLOT_PATH = EVAL_DIR / "metrics_plot.png"
METRICS_JSON_PATH = EVAL_DIR / "metrics.json"
REPORT_PATH = EVAL_DIR / "report.txt"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def detect_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def numeric_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return None
    return None


def extract_metric(metrics: Any, attr_candidates: list[str], key_candidates: list[str]) -> float | None:
    for attr in attr_candidates:
        value = numeric_or_none(getattr(metrics, attr, None))
        if value is not None:
            return value

    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        lowered = {str(k).lower(): v for k, v in results_dict.items()}
        for key in key_candidates:
            if key in results_dict:
                value = numeric_or_none(results_dict[key])
                if value is not None:
                    return value
            value = numeric_or_none(lowered.get(key.lower()))
            if value is not None:
                return value
    return None


def extract_split_metrics(metrics: Any) -> dict[str, float | None]:
    top1 = extract_metric(
        metrics,
        attr_candidates=["top1", "accuracy_top1", "acc"],
        key_candidates=["metrics/accuracy_top1", "metrics/top1", "top1", "accuracy_top1"],
    )
    top5 = extract_metric(
        metrics,
        attr_candidates=["top5", "accuracy_top5"],
        key_candidates=["metrics/accuracy_top5", "metrics/top5", "top5", "accuracy_top5"],
    )

    loss = extract_metric(
        metrics,
        attr_candidates=["loss", "val_loss"],
        key_candidates=["val/loss", "loss", "metrics/loss", "val/loss_total"],
    )

    return {"top1": top1, "top5": top5, "loss": loss}


def count_split_samples(split: str) -> int:
    split_dir = DATA_DIR / split
    if not split_dir.exists():
        return 0
    return sum(1 for p in split_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def class_names_from_model(model: YOLO) -> list[str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        ordered = [names[i] for i in sorted(names.keys())]
        return [str(x) for x in ordered]
    if isinstance(names, list):
        return [str(x) for x in names]
    return []


def evaluate_split(model: YOLO, split: str, save_name: str, device: str) -> tuple[Any, dict[str, float | None], Path]:
    metrics = model.val(
        data=str(DATA_DIR),
        split=split,
        device=device,
        project=str(EVAL_DIR),
        name=save_name,
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    save_dir = Path(getattr(metrics, "save_dir", EVAL_DIR / save_name))
    return metrics, extract_split_metrics(metrics), save_dir


def save_confusion_matrix(metrics: Any, class_names: list[str]) -> np.ndarray | None:
    matrix = None
    confusion_obj = getattr(metrics, "confusion_matrix", None)
    if confusion_obj is not None:
        matrix = getattr(confusion_obj, "matrix", None)
    if matrix is None:
        candidates = [
            TEST_ARTIFACT_DIR / "confusion_matrix.png",
            TEST_ARTIFACT_DIR / "confusion_matrix_normalized.png",
            VAL_ARTIFACT_DIR / "confusion_matrix.png",
            VAL_ARTIFACT_DIR / "confusion_matrix_normalized.png",
        ]
        for candidate in candidates:
            if candidate.exists():
                shutil.copy2(candidate, CONFUSION_MATRIX_PATH)
                return None
        return None

    cm = np.array(matrix, dtype=float)
    n = len(class_names)
    if n and cm.shape[0] >= n and cm.shape[1] >= n:
        cm = cm[:n, :n]

    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    if class_names:
        ticks = np.arange(len(class_names))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(class_names, rotation=90, fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=200)
    plt.close(fig)

    return cm


def save_per_class_accuracy(cm: np.ndarray | None, class_names: list[str]) -> dict[str, float]:
    if cm is None or cm.size == 0:
        PER_CLASS_CSV_PATH.write_text("class,accuracy\n", encoding="utf-8")
        return {}

    totals = cm.sum(axis=1)
    diagonal = np.diag(cm)
    per_class = np.divide(diagonal, totals, out=np.zeros_like(diagonal), where=totals > 0)

    if class_names and len(class_names) == len(per_class):
        labels = class_names
    else:
        labels = [f"class_{i}" for i in range(len(per_class))]

    rows = ["class,accuracy"]
    per_class_dict: dict[str, float] = {}
    for label, acc in zip(labels, per_class):
        value = float(acc)
        per_class_dict[label] = value
        rows.append(f"{label},{value:.6f}")

    PER_CLASS_CSV_PATH.write_text("\n".join(rows) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))
    ax.bar(x, per_class, color="#16a34a")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy")
    fig.tight_layout()
    fig.savefig(PER_CLASS_PLOT_PATH, dpi=200)
    plt.close(fig)

    return per_class_dict


def save_metrics_plot(val_metrics: dict[str, float | None], test_metrics: dict[str, float | None]) -> None:
    labels = ["Top-1", "Top-5", "Loss"]
    val_values = [
        float(val_metrics.get("top1") or 0.0),
        float(val_metrics.get("top5") or 0.0),
        float(val_metrics.get("loss") or 0.0),
    ]
    test_values = [
        float(test_metrics.get("top1") or 0.0),
        float(test_metrics.get("top5") or 0.0),
        float(test_metrics.get("loss") or 0.0),
    ]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, val_values, width, label="Validation", color="#2563eb")
    ax.bar(x + width / 2, test_values, width, label="Test", color="#16a34a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Evaluation Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(METRICS_PLOT_PATH, dpi=200)
    plt.close(fig)


def save_prediction_samples(model: YOLO, device: str) -> None:
    samples: list[Path] = []
    for split in ("test", "val"):
        split_dir = DATA_DIR / split
        if split_dir.exists():
            split_images = [
                p
                for p in split_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            ]
            samples.extend(split_images)
        if len(samples) >= 24:
            break

    if not samples:
        return

    selected = [str(p) for p in samples[:24]]
    model.predict(
        source=selected,
        device=device,
        project=str(EVAL_DIR),
        name=SAMPLES_DIR.name,
        exist_ok=True,
        save=True,
        verbose=False,
    )


def metrics_for_json(split_metrics: dict[str, float | None], split_samples: int) -> dict[str, float | int | None]:
    return {
        "top1_accuracy": split_metrics.get("top1"),
        "top5_accuracy": split_metrics.get("top5"),
        "loss": split_metrics.get("loss"),
        "samples": split_samples,
    }


def format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def main() -> int:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        print(f"Error: missing model weights at {MODEL_PATH}")
        return 1

    if not DATA_DIR.exists():
        print(f"Error: missing dataset directory at {DATA_DIR}")
        return 1

    device = detect_device()
    model = YOLO(str(MODEL_PATH))
    class_names = class_names_from_model(model)

    print("Evaluating validation split...")
    val_metrics_obj, val_metrics, _ = evaluate_split(model, "val", "val", device)

    print("Evaluating test split...")
    test_metrics_obj, test_metrics, _ = evaluate_split(model, "test", "test", device)

    val_samples = count_split_samples("val")
    test_samples = count_split_samples("test")
    total_samples = val_samples + test_samples

    print(f"Validation Top-1 accuracy: {format_metric(val_metrics.get('top1'))}")
    print(f"Validation Top-5 accuracy: {format_metric(val_metrics.get('top5'))}")
    print(f"Validation Loss: {format_metric(val_metrics.get('loss'))}")

    print(f"Test Top-1 accuracy: {format_metric(test_metrics.get('top1'))}")
    print(f"Test Top-5 accuracy: {format_metric(test_metrics.get('top5'))}")
    print(f"Test Loss: {format_metric(test_metrics.get('loss'))}")

    print("Saving plots...")
    cm = save_confusion_matrix(test_metrics_obj, class_names)
    if cm is None and not CONFUSION_MATRIX_PATH.exists():
        _ = save_confusion_matrix(val_metrics_obj, class_names)

    per_class_accuracy = save_per_class_accuracy(cm, class_names)
    save_metrics_plot(val_metrics, test_metrics)
    save_prediction_samples(model, device)

    metrics_json = {
        "device": device,
        "model_path": str(MODEL_PATH),
        "output_dir": str(EVAL_DIR),
        "num_classes": len(class_names),
        "total_samples_evaluated": total_samples,
        "validation": metrics_for_json(val_metrics, val_samples),
        "test": metrics_for_json(test_metrics, test_samples),
        "per_class_accuracy": per_class_accuracy,
        "artifacts": {
            "confusion_matrix": str(CONFUSION_MATRIX_PATH),
            "per_class_accuracy_csv": str(PER_CLASS_CSV_PATH),
            "per_class_accuracy_plot": str(PER_CLASS_PLOT_PATH),
            "metrics_plot": str(METRICS_PLOT_PATH),
            "prediction_samples_dir": str(SAMPLES_DIR),
            "validation_artifacts_dir": str(VAL_ARTIFACT_DIR),
            "test_artifacts_dir": str(TEST_ARTIFACT_DIR),
        },
    }
    METRICS_JSON_PATH.write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")

    print("Saving report...")
    report_lines = [
        "Crop Disease Classifier Evaluation Report",
        "",
        f"Model: {MODEL_PATH}",
        f"Device: {device}",
        f"Validation accuracy (Top-1): {format_metric(val_metrics.get('top1'))}",
        f"Validation Top-5 accuracy: {format_metric(val_metrics.get('top5'))}",
        f"Test accuracy (Top-1): {format_metric(test_metrics.get('top1'))}",
        f"Test Top-5 accuracy: {format_metric(test_metrics.get('top5'))}",
        f"Validation Loss: {format_metric(val_metrics.get('loss'))}",
        f"Test Loss: {format_metric(test_metrics.get('loss'))}",
        f"Number of classes: {len(class_names)}",
        f"Total samples evaluated: {total_samples}",
        "",
        f"Confusion matrix: {CONFUSION_MATRIX_PATH}",
        f"Metrics JSON: {METRICS_JSON_PATH}",
    ]
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Evaluation complete. Outputs saved to: {EVAL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
