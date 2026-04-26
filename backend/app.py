from __future__ import annotations

import math
from io import BytesIO

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

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

disease_info = {
    "Apple___Apple_scab": {
        "description": "Fungal disease that causes olive-brown lesions on leaves and fruit, reducing fruit quality.",
        "treatment": "Prune infected foliage, improve airflow, and apply preventive fungicides during wet periods.",
    },
    "Apple___Black_rot": {
        "description": "Fungal infection causing dark leaf spots, cankers, and fruit rot in apples.",
        "treatment": "Remove mummified fruits and cankered wood, sanitize orchard debris, and spray recommended fungicides.",
    },
    "Apple___Cedar_apple_rust": {
        "description": "Rust disease producing orange-yellow lesions on apple leaves and fruit.",
        "treatment": "Eliminate nearby juniper hosts where possible and apply rust-targeted fungicides at early growth stages.",
    },
    "Apple___healthy": {
        "description": "Plant tissue appears healthy with no visible disease symptoms.",
        "treatment": "No treatment required",
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "description": "Fungal leaf disease causing rectangular gray lesions that reduce photosynthesis and yield.",
        "treatment": "Use resistant hybrids, rotate crops, manage residue, and apply fungicide when disease pressure is high.",
    },
    "Corn_(maize)___healthy": {
        "description": "Corn leaves show normal color and structure without disease symptoms.",
        "treatment": "No treatment required",
    },
    "Grape___Black_rot": {
        "description": "Fungal grape disease causing leaf spots and black, shriveled fruit mummies.",
        "treatment": "Remove infected clusters, prune for airflow, and follow a protective fungicide schedule.",
    },
    "Grape___Esca_(Black_Measles)": {
        "description": "Trunk disease associated with tiger-striped leaves, berry spotting, and vine decline.",
        "treatment": "Prune infected wood, protect pruning wounds, and maintain vine nutrition and irrigation balance.",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "description": "Leaf spot disease causing angular brown lesions and premature defoliation in grapevines.",
        "treatment": "Improve canopy ventilation, remove infected leaves, and apply suitable fungicides.",
    },
    "Grape___healthy": {
        "description": "Grapevine foliage appears healthy with no signs of active disease.",
        "treatment": "No treatment required",
    },
    "Peach___Bacterial_spot": {
        "description": "Bacterial disease causing small angular lesions on leaves and pitted fruit spots.",
        "treatment": "Use tolerant cultivars, avoid overhead irrigation, and apply copper-based bactericides as advised.",
    },
    "Peach___healthy": {
        "description": "Peach leaves and fruit surfaces appear healthy and disease-free.",
        "treatment": "No treatment required",
    },
    "Tomato___Early_blight": {
        "description": "Fungal disease with concentric ring lesions on older leaves and potential fruit damage.",
        "treatment": "Remove infected leaves, mulch soil, rotate crops, and use protective fungicides.",
    },
    "Tomato___Late_blight": {
        "description": "Aggressive oomycete disease causing water-soaked lesions and rapid plant collapse.",
        "treatment": "Rogue infected plants promptly, reduce leaf wetness, and apply late blight-specific fungicides.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "Viral disease causing leaf curling, yellowing, and severe growth stunting.",
        "treatment": "Control whitefly vectors, remove infected plants, and use resistant tomato varieties.",
    },
    "Tomato___healthy": {
        "description": "Tomato plant tissue appears healthy with no visual disease indicators.",
        "treatment": "No treatment required",
    },
}


class PredictResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(alias="class")
    confidence: float
    description: str
    treatment: str
    entropy: float = Field(default=0.0, description="Shannon entropy for debugging")


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    model_path: str


app = FastAPI(title="Crop Disease Classification API", version="1.0.0")
app.state.startup_error = None

# Add CORS middleware to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        class_name, confidence, prob_dist = predict(image)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except InferenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    # Calculate Shannon entropy for uncertainty quantification
    # Entropy = -sum(p * log(p)) measures how spread out the probability distribution is
    # Max entropy for 17 classes = log(17) ≈ 2.833 (uniform distribution)
    # Threshold of 1.8 flags predictions where model confidence is distributed across many classes
    # This catches ambiguous inputs that should not be trusted (e.g., non-leaf images)
    probs_numpy = np.array(prob_dist.cpu().numpy() if hasattr(prob_dist, 'cpu') else prob_dist)
    probs_numpy = np.clip(probs_numpy, 1e-10, 1.0)  # Avoid log(0)
    entropy = float(-np.sum(probs_numpy * np.log(probs_numpy)))

    # Entropy-based rejection: if model is very uncertain, reject prediction
    if entropy > 1.8:
        class_name = "Unknown___background"
        confidence = 0.0
    # If top prediction is background or confidence too low, mark as non-leaf
    elif class_name == "Unknown___background" or confidence < 0.65:
        class_name = "Unknown___background"
        confidence = 0.0

    info = disease_info.get(
        class_name,
        {
            "description": "Information not available",
            "treatment": "Consult agricultural specialist",
        },
    )

    # Override description/treatment for background class
    if class_name == "Unknown___background":
        description = "Input is not a crop leaf or is too ambiguous for diagnosis."
        treatment = "Please provide a clear image of a crop leaf."
    else:
        description = info["description"]
        treatment = info["treatment"]

    return PredictResponse(
        class_name="Not a crop leaf" if class_name == "Unknown___background" else class_name,
        confidence=round(float(confidence), 4),
        description=description,
        treatment=treatment,
        entropy=round(entropy, 4),
    )
