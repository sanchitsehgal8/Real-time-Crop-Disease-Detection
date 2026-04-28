from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image
import io
import threading


MODEL_PATH = "best.pt"
FRAME_SKIP = 15

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

model = YOLO(MODEL_PATH, task="classify")
model_lock = threading.Lock()

latest_label = {"class": "unknown", "confidence": 0.0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # nothing to release at startup/shutdown since camera is managed per-stream


app = FastAPI(lifespan=lifespan)


def predict_image(image: np.ndarray) -> tuple[str, float]:
    image_resized = cv2.resize(image, (224, 224))
    with model_lock:
        results = model(image_resized, device=DEVICE)[0]
    probs = results.probs.data.tolist()
    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])
    class_name = model.names[class_id]
    return class_name, confidence


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "model_loaded": True
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image file. Please upload a valid JPEG or PNG."}
        )

    image_np = np.array(image)
    class_name, confidence = predict_image(image_np)

    return {
        "class": class_name,
        "confidence": round(confidence, 4)
    }


def generate_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open camera. Check device index or permissions.")

    frame_count = 0
    class_name = "unknown"
    confidence = 0.0

    try:
        while True:
            success, frame = camera.read()
            if not success:
                continue

            frame_count += 1
            if frame_count % FRAME_SKIP == 0:
                class_name, confidence = predict_image(frame)

            label = f"{class_name} {confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame_bytes +
                b"\r\n"
            )
    finally:
        camera.release()


@app.get("/video")
async def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )