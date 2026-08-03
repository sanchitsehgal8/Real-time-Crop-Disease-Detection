# AgroVision — Crop Disease Detection

Upload a crop leaf photo, get back a disease classification with a confidence
score, a plain-language description, and a treatment recommendation.

A YOLOv8n classifier (1.46 M params, 17 classes across 5 crops) is served by
FastAPI and consumed by a Next.js web app.

| | |
|---|---|
| Model | YOLOv8n-cls, 224×224 input, 17 classes, 2.9 MB (`best.pt`) |
| Backend | FastAPI + Ultralytics — [backend/app.py](backend/app.py) |
| Frontend | Next.js 16 + React 19 + Tailwind v4 + shadcn/ui — [frontend/](frontend/) |
| Test top-1 | 0.99909 — a lab-dataset ceiling, **not** a field accuracy claim (see below) |

## Quickstart

Two processes. Backend first.

```bash
# 1. Backend — from the repo root
python -m venv venv && source venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.11.0+cpu torchvision==0.26.0+cpu
pip install -r requirements.txt
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

Installing torch from the CPU index first is deliberate — the default PyPI
wheels bundle CUDA and are roughly 10× larger. On a machine with a CUDA GPU,
skip that line and let `requirements.txt` resolve torch normally.

```bash
# 2. Frontend — in a second shell
cd frontend
cp .env.example .env.local
pnpm install
pnpm dev                      # http://localhost:3000
```

Sanity check the API on its own:

```bash
curl http://127.0.0.1:8000/health
curl -F "file=@data/test/Tomato___Early_blight/<some>.JPG" http://127.0.0.1:8000/predict
```

Interactive API docs are at `http://127.0.0.1:8000/docs`.

## API

| Method | Path | Returns |
|---|---|---|
| `GET` | `/health` | `{status, device, model_loaded, model_path}` |
| `POST` | `/predict` | `{class, confidence, description, treatment, entropy}` |

`POST /predict` takes a multipart upload under the field name `file`. JPEG, PNG,
and TIFF are accepted, up to `MAX_UPLOAD_BYTES` (10 MB default).

Predictions the model is not confident about are rejected rather than guessed:
if the Shannon entropy of the class distribution exceeds `ENTROPY_MAX`, or top-1
confidence is under `CONFIDENCE_MIN`, or the model picks the `unknown_background`
class, the response comes back as `"Not a crop leaf"` with confidence `0.0`. This
is what stops a photo of a keyboard from being labelled tomato blight.

Configuration is via environment variables — see [.env.example](.env.example).

## Deploying

See **[DEPLOYMENT.md](DEPLOYMENT.md)**. In short: the frontend goes to Cloudflare
Workers via `@opennextjs/cloudflare`; the backend needs a container, because
PyTorch cannot run on Workers.

## Training and evaluation

```bash
pip install -r requirements-train.txt

python "check balance.py"   # per-class counts per split
python train.py             # 50 epochs, ~2.5 h on Apple Silicon MPS
python evaluate.py          # writes runs/evaluation/
```

`train.py` copies the best checkpoint to `best.pt` in the repo root when it
finishes. `data/` is git-ignored and not distributed with the repo.

## On the accuracy number

Validation top-1 is 1.0000 and test top-1 is 0.99909. Treat both as a ceiling on
a clean lab dataset, not evidence of field readiness:

- PlantVillage images have uniform backgrounds, centered single leaves, and
  controlled lighting. Models trained on it are known to degrade sharply on real
  field photography.
- Every image is byte-duplicated within its own split (an artifact of `split.py`
  being re-run over already-split output), so ~43.6 K files are ~22.2 K unique
  images and the effective diversity behind the metric is half the sample count.
- There is no held-out real-world test set in this repo.

The only class pair the model confuses is tomato early blight ↔ late blight
(0.9933 / 0.9931), which is the expected failure mode.

## License

MIT — see [LICENSE](LICENSE).
