from pathlib import Path
import shutil
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

MODEL_NAME = "yolov8n-cls.pt"
EPOCHS = 50
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

RUN_DIR = ROOT / "runs" / "classify"
RUN_NAME = "crop_disease_cls"


def get_device():
    if torch.backends.mps.is_available():
        print("Apple Silicon GPU (MPS) detected — using GPU for training")
        return "mps"
    if torch.cuda.is_available():
        print("NVIDIA CUDA GPU detected — using GPU for training")
        return "cuda"
    print("No GPU detected — falling back to CPU (training will be slow)")
    return "cpu"


def check_data():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory not found at: {DATA_DIR}\n"
            "Make sure your data folder is inside the train/ directory."
        )

    for split in ["train", "val", "test"]:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Missing split folder: {split_dir}\n"
                f"Expected train/, val/, and test/ inside {DATA_DIR}"
            )

        image_count = len(list(split_dir.rglob("*.jpg"))) + \
                      len(list(split_dir.rglob("*.jpeg"))) + \
                      len(list(split_dir.rglob("*.png")))

        class_count = len([d for d in split_dir.iterdir() if d.is_dir()])

        print(f"  {split:>6}: {image_count:>6} images across {class_count} classes")


def train(device):
    model = YOLO(MODEL_NAME)

    model.train(
        data=str(DATA_DIR),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device,
        workers=NUM_WORKERS,
        amp=True,
        cache="ram",
        cos_lr=True,
        lr0=0.01,
        lrf=0.01,
        patience=15,
        dropout=0.3,
        label_smoothing=0.1,
        seed=42,
        project=str(RUN_DIR),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        verbose=True,
        hsv_h=0.05,
        hsv_s=0.7,
        hsv_v=0.5,
        degrees=45,
        translate=0.2,
        scale=0.6,
        fliplr=0.5,
        flipud=0.3,
        mosaic=0.0,
        erasing=0.4,
        auto_augment="randaugment",
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
    print("=" * 50)
    print("Crop Disease Detection — YOLOv8n Training")
    print("=" * 50)

    print("\nChecking dataset structure...")
    check_data()

    print("\nDetecting compute device...")
    device = get_device()

    print(f"\nStarting training — {EPOCHS} epochs, batch {BATCH_SIZE}, imgsz {IMG_SIZE}")
    print("RAM caching will take 2-3 minutes before epoch 1 starts — this is normal.\n")

    train(device)

    best = locate_best_weights()
    if best is None:
        raise FileNotFoundError(
            "best.pt not found after training.\n"
            f"Check {RUN_DIR / RUN_NAME / 'weights'} manually."
        )

    dst = copy_best(best)

    print("\n" + "=" * 50)
    print("Training complete.")
    print(f"Best model saved to: {dst}")
    print("Next step: transfer best.pt to your Jetson Nano")
    print("=" * 50)


if __name__ == "__main__":
    main()