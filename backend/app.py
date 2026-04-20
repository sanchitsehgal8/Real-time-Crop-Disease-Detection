from __future__ import annotations

from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from backend.model import (
    InferenceError,
    ModelNotLoadedError,
    get_device,
    get_model_path,
    is_model_loaded,
    load_model,
    predict,
)


ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    model_path: str


app = FastAPI(title="Crop Disease Classification API", version="1.0.0")
app.state.startup_error = None


@app.on_event("startup")
async def startup_event() -> None:
    try:
        load_model()
        app.state.startup_error = None
    except Exception as exc:
        app.state.startup_error = str(exc)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    status = "ok" if is_model_loaded() else "error"
    return HealthResponse(
        status=status,
        device=get_device(),
        model_loaded=is_model_loaded(),
        model_path=get_model_path(),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict_image(file: UploadFile = File(...)) -> PredictResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file extension")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported content type")

    if not is_model_loaded():
        detail = app.state.startup_error or "Model is not loaded"
        raise HTTPException(status_code=503, detail=detail)

    try:
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        image = Image.open(BytesIO(payload)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read image: {exc}") from exc

    try:
        class_name, confidence = predict(image)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except InferenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return PredictResponse(predicted_class=class_name, confidence=float(confidence))
