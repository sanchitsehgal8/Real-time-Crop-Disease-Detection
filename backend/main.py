from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image
import io

app = FastAPI()

MODEL_PATH = "best.pt"

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

model = YOLO(MODEL_PATH)

camera = cv2.VideoCapture(0)


def predict_image(image):
    results = model(image, device=DEVICE)[0]
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
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    class_name, confidence = predict_image(image_np)

    return {
        "class": class_name,
        "confidence": round(confidence, 4)
    }


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

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


@app.get("/video")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )