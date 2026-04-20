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
    if torch.cuda.is_available():
        return "cuda"
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
        pretrained=True,
        verbose=True,
    )

    return model


def locate_best_weights():
    weights_path = RUN_DIR / RUN_NAME / "weights" / "best.pt"
    if weights_path.exists():
        return weights_path
    return None


def copy_best(src):
    dst = ROOT / "best.pt"
    shutil.copy2(src, dst)
    return dst


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError("data directory not found")

    device = get_device()
    print(f"Using device: {device}")

    train(device)

    best = locate_best_weights()
    if best is None:
        raise FileNotFoundError("best.pt not found after training")

    dst = copy_best(best)
    print(f"Best model copied to: {dst}")


if __name__ == "__main__":
    main()