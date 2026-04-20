from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO


class ModelNotLoadedError(RuntimeError):
    pass


class InferenceError(RuntimeError):
    pass


class YOLOClassifier:
    def __init__(self, model_path: Path, imgsz: int = 224) -> None:
        self.model_path = model_path
        self.imgsz = imgsz
        self.device = self._detect_device()
        self.model: YOLO | None = None

    @staticmethod
    def _detect_device() -> str:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        return "cpu"

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model = YOLO(str(self.model_path))

    def predict(self, image: Image.Image) -> tuple[str, float]:
        if self.model is None:
            raise ModelNotLoadedError("Model is not loaded")

        if image.mode != "RGB":
            image = image.convert("RGB")

        try:
            results = self.model.predict(
                source=image,
                device=self.device,
                imgsz=self.imgsz,
                verbose=False,
            )
        except Exception as exc:
            if self.device == "mps":
                try:
                    results = self.model.predict(
                        source=image,
                        device="cpu",
                        imgsz=self.imgsz,
                        verbose=False,
                    )
                except Exception as fallback_exc:
                    raise InferenceError(str(fallback_exc)) from fallback_exc
            else:
                raise InferenceError(str(exc)) from exc

        if not results:
            raise InferenceError("No inference results returned")

        result = results[0]
        probs = getattr(result, "probs", None)
        if probs is None:
            raise InferenceError("Model output does not contain classification probabilities")

        top1 = int(probs.top1)
        top1conf_raw = probs.top1conf
        top1conf = float(top1conf_raw.item() if hasattr(top1conf_raw, "item") else top1conf_raw)

        names = self.model.names
        if isinstance(names, dict):
            class_name = str(names.get(top1, top1))
        elif isinstance(names, list):
            class_name = str(names[top1]) if 0 <= top1 < len(names) else str(top1)
        else:
            class_name = str(top1)

        return class_name, top1conf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "best.pt"
_classifier = YOLOClassifier(MODEL_PATH)


def load_model() -> None:
    _classifier.load()


def predict(image: Image.Image) -> tuple[str, float]:
    return _classifier.predict(image)


def is_model_loaded() -> bool:
    return _classifier.model is not None


def get_device() -> str:
    return _classifier.device


def get_model_path() -> str:
    return str(_classifier.model_path)
