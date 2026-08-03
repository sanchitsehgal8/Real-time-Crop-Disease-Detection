# Real-time Crop Disease Detection — Full Project Summary

**Product name (UI):** AgroVision
**Repo:** `Real-time-Crop-Disease-Detection`
**Branch:** `main`
**License:** MIT (see [LICENSE](LICENSE))
**Summary generated:** 2026-08-03

---

## 1. What this project is

An end-to-end crop leaf disease classification system. A YOLOv8n classification model is trained on a
PlantVillage-style leaf image dataset (17 classes across 5 crops), served through a FastAPI backend, and
consumed by a Next.js web app where a user drops a leaf photo and gets back a disease label, confidence
score, plain-language description, and a treatment recommendation.

Three distinct surfaces exist in the repo:

| Surface | Purpose | Entry point |
|---|---|---|
| **ML pipeline** | Split, train, evaluate the classifier | [split.py](split.py), [train.py](train.py), [evaluate.py](evaluate.py) |
| **Inference API** | Serve predictions over HTTP | [backend/app.py](backend/app.py) (primary), [backend/main.py](backend/main.py) (alternate/webcam) |
| **Web UI** | Upload + diagnostic report | [frontend/app/page.tsx](frontend/app/page.tsx) |

The training script's closing message ("Next step: transfer best.pt to your Jetson Nano") indicates the
original deployment target was an **NVIDIA Jetson Nano** for real-time edge inference, though nothing in the
current repo implements that deployment.

---

## 2. Technology stack

### Machine learning
- **Ultralytics YOLOv8** `8.3.39` — `yolov8n-cls.pt` backbone, classification task
- **PyTorch** `>=2.2.0` + **torchvision** `>=0.17.0`
- **Device auto-detection**: Apple Silicon MPS → CUDA → CPU
- **matplotlib** / **numpy** for evaluation plots and metrics
- **OpenCV** (`cv2`) for the webcam streaming path

### Backend
- **FastAPI** `0.115.2`
- **Uvicorn** `0.30.6` (standard extras)
- **Pydantic v2** response models
- **python-multipart** `0.0.9` for file uploads
- **Pillow** `10.4.0` for image decoding
- CORS middleware allowlisting `localhost:3000`, `:5173`, `:8000` (and `127.0.0.1` equivalents)

### Frontend
- **Next.js** `16.2.0` (App Router, RSC enabled)
- **React** `19.2.4`
- **TypeScript** `5.7.3` (strict mode on, but build errors ignored — see §10)
- **Tailwind CSS v4** via `@tailwindcss/postcss`
- **shadcn/ui** (`new-york` style, `neutral` base, ~50 generated primitives in `frontend/components/ui/`)
- **Radix UI** primitives (accordion, dialog, dropdown, select, toast, tooltip, …)
- **lucide-react** icons, **react-dropzone** upload, **next-themes**, **recharts**, **sonner**, **zod**, **react-hook-form**
- **@vercel/analytics** (production only)
- Fonts: Geist, Geist Mono, Playfair Display (Google Fonts via `next/font`)
- Package manager: pnpm lockfile present alongside an npm lockfile (both committed)

---

## 3. Repository layout

```
Real-time-Crop-Disease-Detection/
├── README.md                     # single-line stub
├── LICENSE                       # MIT
├── requirements.txt              # Python deps (has duplicates, see §10)
├── data.yaml                     # dataset descriptor (NOT used by training, see §10)
├── split.py                      # 70/15/15 stratified split
├── check balance.py              # prints per-class counts per split
├── train.py                      # YOLOv8n-cls training driver
├── evaluate.py                   # val + test evaluation, plots, metrics.json, report.txt
├── best.pt                       # trained weights (2.9 MB, ~1.46 M params, 17 classes)
├── yolov8n-cls.pt                # pretrained nano backbone (5.3 MB)
├── yolov8s-cls.pt                # pretrained small backbone (12.3 MB, unused)
├── backend/
│   ├── __init__.py
│   ├── model.py                  # YOLOClassifier wrapper + module-level singleton
│   ├── app.py                    # PRIMARY FastAPI app (upload → diagnosis)
│   └── main.py                   # ALTERNATE FastAPI app (upload + MJPEG webcam stream)
├── frontend/
│   ├── app/{layout,page}.tsx, globals.css
│   ├── components/{header,footer,specimen-input,diagnostic-preview,theme-provider}.tsx
│   ├── components/ui/            # ~50 shadcn primitives
│   ├── hooks/{use-prediction,use-mobile,use-toast}.ts
│   ├── lib/{api,utils}.ts
│   ├── styles/globals.css        # near-duplicate of app/globals.css
│   ├── public/                   # icons + placeholder assets
│   └── {next.config.mjs, tsconfig.json, postcss.config.mjs, components.json, package.json}
├── data/                         # git-ignored, 736 MB — train/ val/ test/ + *.cache
├── runs/
│   ├── classify/                 # training run: args.yaml, results.csv, weights/, batch previews
│   └── evaluation/               # metrics.json, report.txt, confusion_matrix.png, per-class CSV/PNG
├── weights/clip/ViT-B-32.pt      # 354 MB, UNTRACKED and UNUSED by any code
└── venv/                         # git-ignored, 1.1 GB
```

`data/`, `venv/`, `__pycache__/`, `node_modules/`, `.DS_Store` are git-ignored. `weights/` is **not** ignored
and currently shows as untracked.

---

## 4. Dataset

### Classes (17)

Sixteen PlantVillage disease/healthy classes across 5 crops, plus one negative class for non-leaf inputs.

| # | Class | Crop |
|---|---|---|
| 0 | `Apple___Apple_scab` | Apple |
| 1 | `Apple___Black_rot` | Apple |
| 2 | `Apple___Cedar_apple_rust` | Apple |
| 3 | `Apple___healthy` | Apple |
| 4 | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | Corn |
| 5 | `Corn_(maize)___healthy` | Corn |
| 6 | `Grape___Black_rot` | Grape |
| 7 | `Grape___Esca_(Black_Measles)` | Grape |
| 8 | `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` | Grape |
| 9 | `Grape___healthy` | Grape |
| 10 | `Peach___Bacterial_spot` | Peach |
| 11 | `Peach___healthy` | Peach |
| 12 | `Tomato___Early_blight` | Tomato |
| 13 | `Tomato___Late_blight` | Tomato |
| 14 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | Tomato |
| 15 | `Tomato___healthy` | Tomato |
| 16 | `unknown_background` | — (negative / non-leaf) |

Class names are inferred by Ultralytics from directory names under `data/{train,val,test}/`, verified by
loading `best.pt`.

### Split sizes (file counts on disk)

| Split | Files | Unique images |
|---|---|---|
| train | 30,528 | ~15,538 |
| val | 6,544 | ~3,331 |
| test | 6,559 | ~3,339 |
| **total** | **43,631** | **~22,208** |

### Per-class distribution

| Class | train | val | test |
|---|---|---|---|
| Apple___Apple_scab | 882 | 188 | 190 |
| Apple___Black_rot | 868 | 186 | 188 |
| Apple___Cedar_apple_rust | 384 | 82 | 84 |
| Apple___healthy | 2,302 | 494 | 494 |
| Corn___Cercospora/Gray_leaf_spot | 718 | 154 | 154 |
| Corn___healthy | 1,626 | 348 | 350 |
| Grape___Black_rot | 1,652 | 354 | 354 |
| Grape___Esca_(Black_Measles) | 1,936 | 414 | 416 |
| Grape___Leaf_blight | 1,506 | 322 | 324 |
| Grape___healthy | 592 | 126 | 128 |
| Peach___Bacterial_spot | 3,214 | 690 | 690 |
| Peach___healthy | 502 | 110 | 108 |
| Tomato___Early_blight | 1,400 | 300 | 300 |
| Tomato___Late_blight | 2,672 | 572 | 574 |
| Tomato___Yellow_Leaf_Curl_Virus | 7,498 | 1,608 | 1,608 |
| Tomato___healthy | 2,226 | 478 | 478 |
| unknown_background | 550 | 118 | 119 |

**Class imbalance is severe**: `Tomato___Yellow_Leaf_Curl_Virus` (7,498 train) is ~19× larger than
`Apple___Cedar_apple_rust` (384 train). No class weighting or resampling is applied in training.

### Split integrity — verified findings

- ✅ **No cross-split leakage.** Comparing basenames (prefix-stripped) across splits: `train ∩ val = 0`,
  `train ∩ test = 0`, `val ∩ test = 0`.
- ⚠️ **Every image is duplicated *within* its own split.** Each split contains both `X.JPG` and `<split>_X.JPG`
  and they are **byte-identical** (verified by MD5). This is an artifact of [split.py](split.py) being run over a
  `data/` directory that already contained split output — the script copies with a `{split}_` filename prefix,
  so a re-run duplicates everything in place. Effect: ~2× the real dataset size, each training image seen twice
  per epoch, and val/test metrics computed on doubled samples. It does *not* create train/val leakage, but the
  "43,631 images" figure is really ~22,208 unique images.

### `data.yaml` — present but inert

[data.yaml](data.yaml) declares `nc: 17` and a `names:` list, but:
- [train.py](train.py) passes `data=str(DATA_DIR)` (the directory), not `data.yaml`. Ultralytics classification
  mode derives classes from folder names, so this file is never read.
- Its 17th entry is `Unknown_Background`, which does **not** match the actual folder/class `unknown_background`.

---

## 5. Data preparation

### [split.py](split.py)
- Reads class folders from `data/`, writes to `dataset/`
- Ratios: **70 % train / 15 % val / 15 % test**, `random.seed(42)`
- Copies files (does not move), renaming to `{split}_{original_name}`
- Note the source/destination mismatch: it writes to `dataset/` but the pipeline consumes `data/` — the current
  `data/train|val|test` layout was evidently produced by this script and then moved/renamed manually.

### [check balance.py](check%20balance.py)
Utility that walks `data/{train,val,test}` and prints per-class file counts. (Filename contains a space.)

---

## 6. Training

### [train.py](train.py) configuration

| Parameter | Value |
|---|---|
| Base model | `yolov8n-cls.pt` (pretrained, ImageNet) |
| Epochs | 50 |
| Image size | 224 × 224 |
| Batch size | 32 |
| Workers | 4 |
| Optimizer | `auto` (Ultralytics default) |
| LR schedule | `lr0=0.01`, `lrf=0.01`, cosine (`cos_lr=True`), 3 warmup epochs |
| Momentum / weight decay | 0.937 / 0.0005 |
| Regularization | `dropout=0.3`, `label_smoothing=0.1` |
| Early stopping | `patience=15` |
| Mixed precision | `amp=True` |
| Caching | `cache="ram"` |
| Seed | 42 (`deterministic=True`) |
| Device | auto: MPS → CUDA → CPU (actual run: `mps`) |
| Output | `runs/classify/crop_disease_cls` |

### Augmentation policy
`hsv_h=0.05`, `hsv_s=0.7`, `hsv_v=0.5`, `degrees=45`, `translate=0.2`, `scale=0.6`, `fliplr=0.5`,
`flipud=0.3`, `erasing=0.4`, `auto_augment="randaugment"`, `mosaic=0.0` (disabled), `mixup=0.0`.

This is an aggressive augmentation set — appropriate for the goal of robustness to real-world field photos
(rotation, lighting, occlusion) rather than clean lab-style PlantVillage images.

### Workflow
1. `check_data()` — asserts `data/{train,val,test}` exist, prints image + class counts
2. `get_device()` — device detection with human-readable log lines
3. `train(device)` — Ultralytics `model.train(...)`
4. `locate_best_weights()` → `copy_best()` — copies `runs/.../weights/best.pt` to repo root as `best.pt`

Run with:
```bash
python train.py
```

### Training results ([runs/classify/results.csv](runs/classify/results.csv))

All 50 epochs completed (early stopping never triggered). Total wall time ≈ **8,691 s (~2 h 25 min)** on MPS.

| Epoch | train/loss | val top-1 | val/loss |
|---|---|---|---|
| 1 | 0.9592 | 0.9697 | 0.1176 |
| 2 | 0.1742 | 0.9869 | 0.0391 |
| 5 | 0.0982 | 0.9908 | 0.0248 |
| 7 | 0.0679 | 0.9963 | 0.0090 |
| 19 | 0.0355 | 0.9994 | 0.0020 |
| 29 | 0.0185 | 0.9997 | 0.0009 |
| 39 | 0.0102 | **1.0000** | 0.0005 |
| 50 | 0.0070 | **1.0000** | 0.0005 |

Val top-1 saturates at 1.0000 by epoch ~39 and val loss plateaus at 0.00048 — the last ~15 epochs add nothing.

Artifacts in `runs/classify/`: `args.yaml`, `results.csv`, `results.png`, `confusion_matrix.png`,
`confusion_matrix_normalized.png`, train/val batch preview JPGs, and `weights/{best,last}.pt` (both 2.9 MB,
identical size).

### Final model
- **File:** `best.pt`, 2.9 MB
- **Parameters:** 1,460,065 (~1.46 M)
- **Task:** `classify`, 17 classes
- Small enough for real-time edge inference — consistent with the stated Jetson Nano target

---

## 7. Evaluation

### [evaluate.py](evaluate.py)
A ~370-line evaluation harness that runs `model.val()` on both `val` and `test` splits and emits:

| Artifact | Path |
|---|---|
| Metrics JSON | `runs/evaluation/metrics.json` |
| Text report | `runs/evaluation/report.txt` |
| Confusion matrix | `runs/evaluation/confusion_matrix.png` |
| Per-class accuracy CSV | `runs/evaluation/per_class_accuracy.csv` |
| Per-class accuracy plot | `runs/evaluation/per_class_accuracy.png` |
| Val/Test metric comparison | `runs/evaluation/metrics_plot.png` |
| Sample predictions (24 images) | `runs/evaluation/prediction_samples/` |

Notable implementation details:
- `extract_metric()` / `numeric_or_none()` defensively probe multiple attribute names and `results_dict` keys,
  so the script survives Ultralytics API drift across versions
- Falls back to copying Ultralytics' own confusion-matrix PNG if the raw matrix object isn't exposed
- Returns a proper exit code via `raise SystemExit(main())`

Run with:
```bash
python evaluate.py
```

### Reported metrics ([runs/evaluation/metrics.json](runs/evaluation/metrics.json))

| Metric | Validation | Test |
|---|---|---|
| Top-1 accuracy | **1.0000** | **0.99909** |
| Top-5 accuracy | 1.0000 | 1.0000 |
| Loss | not captured (`null`) | not captured (`null`) |
| Samples | 6,544 | 6,559 |

Device: `mps`. Total samples evaluated: 13,103. Classes: 17.

### Per-class test accuracy
15 of 17 classes at **1.000**, including `unknown_background`. Only two below perfect:

| Class | Accuracy |
|---|---|
| `Tomato___Early_blight` | 0.9933 |
| `Tomato___Late_blight` | 0.9931 |

Early blight ↔ late blight confusion is the single failure mode — expected, since both present as necrotic
foliar lesions on tomato.

### Reading these numbers honestly

100 % / 99.9 % is **not** an indication of field-ready performance. Contributing factors:
1. **PlantVillage is a lab dataset** — uniform backgrounds, single centered leaves, controlled lighting. It is
   well documented that models trained on it collapse on real field photography.
2. **Intra-split duplication** (§4) means each val/test image is scored twice, which doesn't bias accuracy but
   halves the effective sample diversity behind the number.
3. **No held-out real-world test set** exists in the repo.
4. `loss` is `null` in the metrics — the extraction path for classification-mode loss didn't resolve.

The entropy-based rejection logic in the API (§8) is effectively the project's compensation for factor 1.

---

## 8. Backend

Two FastAPI applications exist. They are independent and **not** wired together.

### 8a. [backend/app.py](backend/app.py) — the primary API (used by the frontend)

`FastAPI(title="Crop Disease Classification API", version="1.0.0")`

**Endpoints**

| Method | Path | Response model | Description |
|---|---|---|---|
| GET | `/health` | `HealthResponse` | `{status, device, model_loaded, model_path}` |
| POST | `/predict` | `PredictResponse` | multipart image → `{class, confidence, description, treatment, entropy}` |

**Startup:** `@app.on_event("startup")` calls `load_model()` and stores any exception in
`app.state.startup_error`, so a failed model load degrades to a `503` on `/predict` rather than crashing boot.

**Upload validation (in order)**
1. Filename required → `400`
2. Extension in `{jpg, jpeg, png}` → `400`
3. Content-type in `{image/jpeg, image/png}` → `400`
4. Model loaded → else `503` (with the startup error as detail)
5. Non-empty payload → `400`
6. `PIL.Image.open(...).convert("RGB")`; `UnidentifiedImageError` → `400`

**Error mapping:** `ModelNotLoadedError` → 503, `FileNotFoundError` → 500, `InferenceError` → 500,
anything else → 500 with the message.

**Uncertainty rejection — the notable design decision**

After inference, the full probability distribution is used to compute Shannon entropy:

```
entropy = -Σ p · log(p)     (probs clipped to [1e-10, 1.0])
max entropy for 17 classes = log(17) ≈ 2.833
```

Rejection rules:
- `entropy > 1.8` → reject as `Unknown___background`, confidence forced to `0.0`
- else if predicted class is `Unknown___background` **or** `confidence < 0.65` → reject the same way

Rejected predictions are returned as class `"Not a crop leaf"` with:
- description: *"Input is not a crop leaf or is too ambiguous for diagnosis."*
- treatment: *"Please provide a clear image of a crop leaf."*

This is a genuine out-of-distribution guard: a model trained only on 17 leaf classes will otherwise confidently
label a photo of a keyboard as tomato blight. High entropy = probability mass spread across many classes = don't
trust it. ⚠️ **There is a string-mismatch bug here** — the actual class name is `unknown_background`, not
`Unknown___background`. See §10.

**Knowledge base:** `disease_info` — a hardcoded dict of 16 entries (all classes except the background class),
each with a `description` and a `treatment`. Healthy classes return *"No treatment required"*. Unknown keys fall
back to *"Information not available"* / *"Consult agricultural specialist"*.

**Response shaping:** `PredictResponse` uses `Field(alias="class")` with `populate_by_name=True`, so the wire
format uses `class` (a Python reserved word) while the model attribute is `class_name`.

**CORS:** explicit origin allowlist for the Next.js dev server (`:3000`), Vite (`:5173`), and self (`:8000`);
`allow_credentials=True`, all methods and headers.

Run with:
```bash
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

### 8b. [backend/model.py](backend/model.py) — model wrapper

- `class YOLOClassifier` — holds `model_path`, `imgsz=224`, detected device, lazily-loaded `YOLO` instance
- `_detect_device()` — MPS if `is_available() and is_built()`, else **CPU** (note: this path does **not** check CUDA,
  unlike `train.py` and `evaluate.py`)
- `load()` — raises `FileNotFoundError` if `best.pt` is missing
- `predict(image)` — converts to RGB, runs `model.predict(source, device, imgsz, verbose=False)`, and on MPS
  failure **automatically retries on CPU** before raising `InferenceError`. Returns
  `(class_name, top1_confidence, full_probability_distribution)`. Handles `names` as dict *or* list.
- Custom exceptions: `ModelNotLoadedError`, `InferenceError`
- Module-level singleton `_classifier` with functional accessors: `load_model()`, `predict()`,
  `is_model_loaded()`, `get_device()`, `get_model_path()`
- Resolves weights as `PROJECT_ROOT / "best.pt"` (absolute, `parents[1]` from the file) — robust to cwd

### 8c. [backend/main.py](backend/main.py) — alternate app with live webcam feed

A separate, simpler FastAPI app that does **not** import `backend/model.py`. It loads `YOLO("best.pt")` directly
(relative path → cwd-dependent) and guards inference with a `threading.Lock`.

**Endpoints**

| Method | Path | Description |
|---|---|---|
| GET | `/health` | `{status, device, model_loaded: True}` (hardcoded `True`) |
| POST | `/predict` | image → `{class, confidence}` — no description/treatment, no entropy gate |
| GET | `/video` | **MJPEG stream** (`multipart/x-mixed-replace`) from `cv2.VideoCapture(0)` with the label drawn on each frame |

**Real-time loop:** `FRAME_SKIP = 15` — inference runs on every 15th frame, and the last label is overlaid on all
intervening frames via `cv2.putText` in green at `(20, 40)`. Frames are JPEG-encoded and yielded as multipart
chunks; the camera is released in a `finally` block.

Device detection here is MPS → CUDA → CPU, and it resizes to 224×224 with `cv2.resize` before inference.

This is the "real-time" in the project title, but the frontend never calls `/video`, and it captures from a camera
attached to the **server**, not the browser client.

---

## 9. Frontend

Next.js App Router app branded **AgroVision** ("The Digital Greenhouse Architecture"). Originally scaffolded with
v0 (`generator: 'v0.app'` in metadata).

### [app/layout.tsx](frontend/app/layout.tsx)
Root layout. Loads Geist / Geist Mono / Playfair Display, sets metadata (title *"AgroVision - Crop Disease
Analysis"*), light/dark favicons + Apple icon, and mounts `@vercel/analytics` only when
`NODE_ENV === 'production'`.

### [app/page.tsx](frontend/app/page.tsx) — `CropDiseaseAnalysis`
Client component holding the whole flow:
- Local state: `uploadedFile` (the `File`) and `uploadedImage` (a data URL via `FileReader`)
- Pulls `{isHealthy, isPredicting, prediction, error, isLoading, runPrediction}` from `usePrediction()`
- `BackendStatusAlert` — three states: blue "Connecting to backend…" pulse, red "Backend Unavailable" with the
  error message and a hint to start the API on `http://127.0.0.1:8000`, or nothing when healthy
- Two-column grid: `SpecimenInput` | `DiagnosticPreview`
- CTA button "RUN PREDICTION MODEL" / "ANALYZING…", disabled unless a file is staged, not already predicting,
  backend healthy, and health check settled

### [components/specimen-input.tsx](frontend/components/specimen-input.tsx)
`react-dropzone` upload zone. Accepts **JPEG, PNG, and TIFF**, `maxSize` 50 MB, single file. Visual states for
idle / drag-active / uploaded, with an inline thumbnail preview and a green "READY" badge.

### [components/diagnostic-preview.tsx](frontend/components/diagnostic-preview.tsx)
Renders, in priority order: analyzing spinner → error card → prediction result → staged-image preview → empty
state. The result card shows the thumbnail, a green `CheckCircle` if the class name contains `"healthy"` else an
amber `AlertCircle`, the class name, a confidence pill (`(confidence * 100).toFixed(1)%`), and `DESCRIPTION` /
`TREATMENT` rows.

### [components/header.tsx](frontend/components/header.tsx) / [footer.tsx](frontend/components/footer.tsx)
Hardcoded warm-neutral (`#e8e4df`) header with the italic serif *AgroVision* wordmark, a non-functional search
input, Dashboard / Support links, a settings button, a sprout avatar, and a green gradient rule. Dark green
(`#1a3a2a`) footer with "© 2024 AgroVision" and four placeholder links (Sustainability Report, API Access,
Network Status, Legal).

### [lib/api.ts](frontend/lib/api.ts) — API client
- `API_BASE_URL = "http://127.0.0.1:8000"` — **hardcoded**, no env var
- `checkHealth(): Promise<HealthResponse>` — GET `/health`
- `predictDisease(file): Promise<PredictResponse>` — POST `/predict` as `FormData` under field name `file`;
  surfaces FastAPI's `detail` field as the error message
- `handleApiError()` normalizes thrown values into `{message, status: 0}`
- Interfaces: `HealthResponse`, `PredictResponse` (`class`, `confidence`, `description`, `treatment`), `ApiError`

### [hooks/use-prediction.ts](frontend/hooks/use-prediction.ts)
Single state object `{isHealthy, isLoading, isPredicting, prediction, error, healthData}`. Runs one health check
in a mount `useEffect` (`isHealthy = status === "ok"`), and exposes `runPrediction(file)` which clears prior
state, awaits `predictDisease`, then stores result or error and re-throws.

Other hooks: `use-mobile.ts`, `use-toast.ts` (shadcn boilerplate, unused by the current page).

### Config
- [next.config.mjs](frontend/next.config.mjs) — `typescript.ignoreBuildErrors: true`, `images.unoptimized: true`
- [tsconfig.json](frontend/tsconfig.json) — strict, `@/*` path alias to project root, bundler resolution
- [components.json](frontend/components.json) — shadcn config, `new-york` / `neutral` / CSS variables
- Scripts: `dev`, `build`, `start`, `lint`

Run with:
```bash
cd frontend && pnpm install && pnpm dev   # http://localhost:3000
```

---

## 10. Known issues, bugs, and technical debt

### Correctness bugs

1. **Background-class name mismatch — the OOD guard is partly dead code.**
   [backend/app.py](frontend/../backend/app.py) compares against `"Unknown___background"`, but the model's actual
   class 16 is `"unknown_background"`. Consequences:
   - The `elif class_name == "Unknown___background"` branch can never fire on a genuine model prediction.
   - When the model *correctly and confidently* predicts a non-leaf image as `unknown_background` (entropy ≤ 1.8,
     confidence ≥ 0.65), it falls through to the `disease_info` lookup, misses, and returns class
     `"unknown_background"` with *"Information not available"* / *"Consult agricultural specialist"* instead of
     the intended *"Not a crop leaf"* message.
   - The entropy > 1.8 rule still works, so rejection isn't fully broken — just inconsistent.

2. **TIFF is accepted by the UI and rejected by the API.**
   [specimen-input.tsx](frontend/components/specimen-input.tsx) accepts `image/tiff`; [app.py](backend/app.py)
   allows only `{jpg, jpeg, png}` extensions and `{image/jpeg, image/png}` content types. A TIFF upload passes the
   dropzone and then fails with a `400`.

3. **`data.yaml` is inert and its class list is wrong.** Never read by training (see §4), and its 17th name
   `Unknown_Background` doesn't match the real class.

4. **[split.py](split.py) is not idempotent.** Re-running over an already-split `data/` duplicates every image with
   a `{split}_` prefix — which is exactly what happened (§4). It also writes to `dataset/` while the rest of the
   pipeline reads `data/`.

5. **Frontend `PredictResponse` is missing `entropy`.** The backend returns it; the TS interface omits it, so it's
   invisible to the UI even for debugging.

### Architectural / hygiene

6. **Two competing backend apps.** [app.py](backend/app.py) and [main.py](backend/main.py) both define `/health`
   and `/predict` with *different* response shapes and *different* validation. `main.py` hardcodes
   `"model_loaded": True`, uses a cwd-relative `MODEL_PATH`, skips the entropy gate entirely, and duplicates
   device detection. Only `app.py` matches what the frontend expects.

7. **The "real-time" path is server-side and unreachable from the UI.** `/video` streams from
   `cv2.VideoCapture(0)` on the *server*, and no frontend component consumes it. There is also no
   concurrency control — two simultaneous `/video` clients would contend for one camera device.

8. **`requirements.txt` is malformed.** `ultralytics`, `torch`, and `torchvision` are each listed twice with
   conflicting specifiers (`ultralytics>=8.3.0` and `ultralytics==8.3.39`). Missing entirely despite being
   imported: **opencv-python** (`cv2`), **numpy**, **matplotlib**.

9. **No environment configuration.** `API_BASE_URL` is hardcoded to `127.0.0.1:8000` in the client; CORS origins
   are hardcoded in the server. Neither is deployable without a code edit.

10. **Deprecated FastAPI API.** `@app.on_event("startup")` is deprecated in favor of a `lifespan` handler —
    which [main.py](backend/main.py) already demonstrates (though its lifespan body is a no-op).

11. **Untracked 354 MB dead weight.** `weights/clip/ViT-B-32.pt` is a CLIP ViT-B/32 checkpoint. Grep across the
    whole repo finds **zero** references to CLIP — it appears to be an abandoned experiment (likely
    zero-shot OOD detection as an alternative to the entropy gate). `weights/` is not in `.gitignore`, so it will
    be committed by an unguarded `git add .`.

12. **No tests, no CI, no containerization.** No `pytest`, no GitHub Actions, no Dockerfile, no `.env.example`.

13. **Duplicated stylesheets.** `frontend/app/globals.css` (126 lines) and `frontend/styles/globals.css`
    (125 lines) are near-identical; only the former is referenced.

14. **Both lockfiles committed.** `pnpm-lock.yaml` and `package-lock.json` coexist — a recipe for divergent installs.

15. **`typescript.ignoreBuildErrors: true`** silently ships type errors to production despite `strict: true`.

16. **Cosmetic:** `README.md` is a single heading line; `package.json` name is `"my-project"`; footer says
    "© 2024"; `check balance.py` has a space in its filename; all 10+ commits are titled some misspelling of
    "inital commit"; `frontend/.next/` build output and `data/*.cache` files sit in the working tree.

### Model / ML concerns

17. **No production security posture on the API.** No authentication, no rate limiting, no request size cap
    (the 50 MB limit is client-side only), no structured logging or request tracing.

18. **Severe class imbalance is unaddressed** (19:1 between largest and smallest train class) — no class weights,
    no oversampling, no per-class threshold tuning.

19. **Metrics almost certainly overstate field performance.** See §7. There is no real-world validation set, and
    `loss` never gets captured in `metrics.json`.

20. **Training ran ~15 epochs past convergence.** Val top-1 hit 1.0000 by epoch 39; `patience=15` never triggered
    because the metric was already saturated, not degrading.

21. **The `0.65` confidence floor and `1.8` entropy ceiling are unjustified constants** — not derived from a
    validation sweep, and not documented as tunable.

---

## 11. How to run the whole thing

```bash
# 0. Python env
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install opencv-python numpy matplotlib      # missing from requirements.txt

# 1. (optional) inspect the dataset
python "check balance.py"

# 2. (optional) retrain — ~2.5 h on Apple Silicon MPS
python train.py                                 # writes best.pt to repo root

# 3. (optional) evaluate
python evaluate.py                              # writes runs/evaluation/*

# 4. Serve the API  (from the repo root, so `backend.app` resolves)
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
#    Alternate webcam-streaming app instead:
#    uvicorn backend.main:app --reload --port 8000

# 5. Serve the UI
cd frontend && pnpm install && pnpm dev          # http://localhost:3000
```

Sanity check the API directly:

```bash
curl http://127.0.0.1:8000/health
curl -F "file=@data/test/Tomato___Early_blight/<some>.JPG" http://127.0.0.1:8000/predict
```

---

## 12. Recommended next steps, in priority order

1. **Fix the background-class string** — change `"Unknown___background"` to `"unknown_background"` in
   [backend/app.py](backend/app.py) (3 occurrences), or normalize with `.lower()` comparisons. This restores the
   intended OOD behavior.
2. **De-duplicate `data/`** — delete either the prefixed or unprefixed copy in each split, then re-run
   `evaluate.py` so the reported metrics reflect real sample counts.
3. **Delete or promote `backend/main.py`.** If server-side webcam streaming is a goal, merge `/video` into
   `app.py` (reusing `backend/model.py` and the entropy gate) and add a frontend `<img src="/video">` view. If not,
   remove it.
4. **Repair `requirements.txt`** — deduplicate, pin consistently, add `opencv-python`, `numpy`, `matplotlib`.
5. **Externalize configuration** — `NEXT_PUBLIC_API_URL` on the client, `CORS_ORIGINS` / `MODEL_PATH` env vars on
   the server.
6. **Align accepted formats** — either add TIFF handling to the backend or drop it from the dropzone.
7. **Delete or document `weights/clip/`** (354 MB unused) and add `weights/` to `.gitignore`.
8. **Build a small real-world validation set** — field photos with varied backgrounds and lighting. Until then,
   treat the 99.9 % figure as a lab-dataset ceiling, not an accuracy claim.
9. **Tune the rejection thresholds empirically** — sweep entropy and confidence on a mixed leaf/non-leaf set and
   report the precision/recall tradeoff instead of hardcoding `1.8` / `0.65`.
10. **Add the missing engineering scaffolding** — `pytest` tests for the validation branches and the entropy gate,
    a Dockerfile, CI on push, a real README, and `lifespan` in place of `on_event`.
11. **Write a real README** covering setup, endpoints, and the metrics caveat — the current one-line file is the
    biggest onboarding blocker.

---

## 13. At a glance

| | |
|---|---|
| Model | YOLOv8n-cls, 1.46 M params, 2.9 MB, 224×224 input, 17 classes |
| Dataset | ~22.2 K unique leaf images (43.6 K files due to duplication), 5 crops + background |
| Training | 50 epochs, batch 32, MPS, ~2 h 25 min, aggressive augmentation, cosine LR |
| Val / Test top-1 | 1.0000 / 0.99909 (lab dataset — see §7 caveats) |
| Weakest classes | Tomato Early blight (0.9933), Tomato Late blight (0.9931) |
| API | FastAPI, 2 endpoints, entropy-based OOD rejection (partly broken — §10.1) |
| UI | Next.js 16 + React 19 + Tailwind v4 + shadcn/ui, single-page upload → diagnosis |
| Intended target | NVIDIA Jetson Nano edge deployment (not implemented) |
| Biggest risks | Class-name mismatch bug, duplicated dataset, lab-only metrics, two competing backends |
