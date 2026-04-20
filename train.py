from pathlib import Path
import shutil
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

MODEL_NAME = "yolov8s-cls.pt"
EPOCHS = 50
IMG_SIZE = 224

RUN_DIR = ROOT / "runs" / "classify"
RUN_NAME = "crop_disease_cls"


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(device):
    model = YOLO(MODEL_NAME)

    model.train(
        data=str(DATA_DIR),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device=device,
        project=str(RUN_DIR),
        name=RUN_NAME,
        exist_ok=True,
    )

    return model


def find_best(model):
    trainer = getattr(model, "trainer", None)
    if trainer is None:
        return None

    best = getattr(trainer, "best", None)
    if best:
        p = Path(best)
        if p.exists():
            return p

    save_dir = getattr(trainer, "save_dir", None)
    if save_dir:
        fallback = Path(save_dir) / "weights" / "best.pt"
        if fallback.exists():
            return fallback

    return None


def copy_best(src):
    dst = ROOT / "best.pt"
    shutil.copy2(src, dst)
    return dst


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError("data directory not found")

    device = get_device()
    print("Device:", device)

    model = train(device)

    best = find_best(model)
    if best is None:
        raise FileNotFoundError("best.pt not produced")

    dst = copy_best(best)
    print("Saved:", dst)


if __name__ == "__main__":
    main()
