# Deploying AgroVision

## Render — both services, one blueprint

[render.yaml](render.yaml) defines the API (Docker) and the frontend (Node) as
two web services in this repo. Render dashboard → **New** → **Blueprint** → pick
this repo → Apply. Then:

1. Check the API service's real URL. Render derives it from the service name, but
   appends a suffix if `agrovision-api.onrender.com` was already taken. If it
   differs, fix `NEXT_PUBLIC_API_URL` in `render.yaml` and redeploy the frontend
   — that value is inlined at build time.
2. Set `CORS_ORIGINS` on **agrovision-api** to the frontend's URL
   (`https://agrovision-web.onrender.com`). It is marked `sync: false` because
   the URL isn't known until the first deploy.

Both are on the `starter` plan (512 MB / 0.5 CPU). Measured peak RSS of the
inference path is **358 MB**, so that fits. The `free` plan has the same 512 MB
but only 0.1 CPU and spins down after 15 minutes idle with a ~1 minute cold
start, which the frontend's on-mount health check will surface as "Backend
Unavailable".

You can also run the API on Render and keep the frontend on Cloudflare Workers
(free, already wired — see Part 2). That drops one paid instance.

## Fastest path — Fly.io + Cloudflare

Backend on Fly.io, frontend on Cloudflare Workers. Roughly 15 minutes, most of
it waiting on the image build. **Docker is not needed locally** — Fly builds on a
remote builder.

```bash
# ── Backend ───────────────────────────────────────────────────────────────
brew install flyctl
fly auth login                      # or `fly auth signup`

fly launch --no-deploy --copy-config --name agrovision-api   # reuses fly.toml
fly deploy                                                   # ~5 min

fly open /health                    # expect model_loaded: true
# note the hostname, e.g. https://agrovision-api.fly.dev

# ── Frontend ──────────────────────────────────────────────────────────────
cd frontend
pnpm install
pnpm exec wrangler login

NEXT_PUBLIC_API_URL=https://agrovision-api.fly.dev pnpm run deploy
# note the deployed URL, e.g. https://agrovision.<subdomain>.workers.dev

# ── Connect them ──────────────────────────────────────────────────────────
cd ..
fly secrets set CORS_ORIGINS=https://agrovision.<subdomain>.workers.dev
```

`fly secrets set` restarts the machine automatically. Then load the frontend URL —
the "Connecting to backend…" banner should clear within a second or two.

Everything below is the detail behind those commands, plus the Cloudflare
Containers alternative if you want the backend on Cloudflare too.

## Architecture

Two independently deployed pieces:

| Piece | Runs on | Why |
|---|---|---|
| `frontend/` (Next.js 16) | Cloudflare Workers, via `@opennextjs/cloudflare` | Static-ish React app; fits the Workers model. |
| `backend/app.py` (FastAPI + YOLOv8) | A **container** (Cloudflare Containers, Fly.io, Render, Railway, …) | Needs PyTorch. |

**The backend cannot run on Cloudflare Workers.** Python Workers execute on
Pyodide/WebAssembly and accept only pure-Python, PyEmscripten, or Pyodide-built
wheels. `torch`, `ultralytics`, and `opencv-python` are native C/C++ extensions
with no Pyodide builds. This is not a bundle-size problem — there is no wheel to
install. (The 3 MB free / 10 MB paid compressed Worker limit is a second wall.)

The browser calls the backend **directly**; the Worker never proxies it. Two
consequences:

- The backend must be reachable over **HTTPS**. A page served from `https://`
  cannot call an `http://` API — browsers block it as mixed content.
- `CORS_ORIGINS` on the backend must list the deployed frontend origin.

---

## Prerequisites

- Node 20+ and pnpm 11 (pinned via `packageManager` in `frontend/package.json`)
- A Cloudflare account (frontend). Workers Paid is required *only* if you host
  the backend on Cloudflare Containers.
- Docker, for building the backend image.

---

## Part 1 — Backend

### Build and run locally

```bash
# from the repo root
docker build -t agrovision-api .
docker run --rm -p 8000:8000 \
  -e CORS_ORIGINS=http://localhost:3000 \
  agrovision-api

curl http://127.0.0.1:8000/health
```

The [Dockerfile](Dockerfile) installs CPU-only torch from
`https://download.pytorch.org/whl/cpu` **before** `requirements.txt`. This is not
optional: `pip install torch` from the default PyPI index resolves to the CUDA
build and pulls the `nvidia-*` packages, landing at roughly 2 GB installed. The
pins (`torch==2.11.0+cpu`, `torchvision==0.26.0+cpu`, Python 3.11) match the
versions verified working locally.

The build ends with a `python -c "... load_model()"` step, so a missing or
corrupt `best.pt` fails the build instead of surfacing as a 503 in production.

[.dockerignore](.dockerignore) is an allow-list. The repo contains `data/`
(736 MB), `venv/` (1.1 GB), and `weights/` (354 MB); a missed deny rule would
silently balloon the build context.

### Configuration

All optional — see [.env.example](.env.example). The one you **must** set in
production is `CORS_ORIGINS`; without it the backend allows only localhost and
the browser blocks every request.

| Variable | Default | Purpose |
|---|---|---|
| `CORS_ORIGINS` | localhost dev origins | Comma-separated allowed browser origins |
| `MODEL_PATH` | `<repo root>/best.pt` | Weights location |
| `MAX_UPLOAD_BYTES` | `10485760` (10 MB) | Server-side upload cap |
| `ENTROPY_MAX` | `1.8` | Reject above this entropy |
| `CONFIDENCE_MIN` | `0.65` | Reject below this top-1 confidence |
| `PORT` | `8000` | Bind port |

### Option A — Cloudflare Containers

Keeps everything on one platform. Requires **Workers Paid**. Containers are
fronted by a Worker that routes requests to a container instance, so you need a
small Worker wrapper in addition to this image.

Scaffold it from Cloudflare's official template rather than hand-writing the
Durable Object plumbing:

```bash
npm create cloudflare@latest -- --template=cloudflare/templates/containers-template
```

Then point the generated `wrangler.jsonc` at this repo's `Dockerfile` and pick an
instance type. `standard-1` (1/2 vCPU, 4 GiB memory, 8 GB disk) is a sane start
for a 1.46 M-parameter YOLOv8n at 224×224 on CPU; the image limit is the
instance's disk size and account image storage is 50 GB.

**Cold starts are the thing to watch.** Containers scale to zero. The first
request after idle pays container boot + `import torch` + model load — on the
order of 10–20 s. `usePrediction` runs its health check on mount, so a cold
backend will likely render "Backend Unavailable" before it finishes waking. If
that matters, keep an instance warm or raise the client timeout.

### Option B — Any container host

Fly.io, Render, and Railway all deploy a `Dockerfile` directly and give you
HTTPS plus a persistent instance, which sidesteps the cold-start issue. Set
`CORS_ORIGINS` in the host's environment settings. Nothing in the image is
Cloudflare-specific.

---

## Part 2 — Frontend

Everything below runs in `frontend/`.

`next build` alone does **not** produce a deployable Worker — it emits `.next/`.
`opennextjs-cloudflare build` transforms that into `.open-next/worker.js` plus
`.open-next/assets`, which is what [wrangler.jsonc](frontend/wrangler.jsonc)
points at.

### Set the API URL first

`NEXT_PUBLIC_API_URL` is inlined into the client bundle **at build time**.
Setting it on the running Worker does nothing — you must rebuild to change it.

```bash
cp .env.example .env.local   # local dev
```

### Deploy from your machine

```bash
cd frontend
pnpm install
NEXT_PUBLIC_API_URL=https://your-backend.example.com pnpm run deploy
```

`pnpm run preview` does the same build but serves it locally in workerd first.

### Deploy from git (Workers Builds)

In the Cloudflare dashboard, Workers & Pages → your Worker → Settings → Build:

| Setting | Value |
|---|---|
| Root directory | `frontend` |
| Build command | `pnpm run cf:build` |
| Deploy command | `npx wrangler deploy` |

Set `NEXT_PUBLIC_API_URL` under **Variables and secrets** in the same Build
section, so it is present when `next build` runs.

**Root directory must be `frontend`.** Left at the repo root, Cloudflare finds
`requirements.txt`, decides the project is Python, and runs `pip install` — which
is what the earlier failing builds were doing.

### Wiring order

1. Deploy the backend, note its HTTPS URL.
2. Build and deploy the frontend with `NEXT_PUBLIC_API_URL` set to that URL.
3. Set `CORS_ORIGINS` on the backend to the frontend's origin, and restart it.

Step 3 is last because you don't know the `*.workers.dev` hostname until step 2.

---

## Verifying a deploy

```bash
# backend reachable and model loaded
curl https://your-backend.example.com/health
# => {"status":"ok","device":"cpu","model_loaded":true,"model_path":"/app/best.pt"}

# a real prediction
curl -F "file=@data/test/Tomato___Early_blight/<some>.JPG" \
     https://your-backend.example.com/predict

# CORS preflight actually allows the frontend
curl -i -X OPTIONS https://your-backend.example.com/predict \
  -H "Origin: https://agrovision.<subdomain>.workers.dev" \
  -H "Access-Control-Request-Method: POST"
# => 200 with access-control-allow-origin echoing your frontend
```

Then load the frontend. If the banner says "Backend Unavailable", it is almost
always one of: mixed content (`http://` API from an `https://` page), a missing
origin in `CORS_ORIGINS`, or a cold container.

---

## What is *not* production-hardened

Deliberately out of scope of the deployment work — flagging so nothing here is a
surprise.

- **No auth or rate limiting** on `/predict`. It is an open, CPU-expensive
  endpoint. Put Cloudflare WAF rate limiting in front of it before sharing the
  URL publicly.
- **`ENTROPY_MAX=1.8` and `CONFIDENCE_MIN=0.65` are unvalidated.** They were
  never derived from a sweep on a mixed leaf/non-leaf set. They are env-tunable
  now, but the defaults are guesses.
- **The reported 99.9 % test accuracy is a lab-dataset ceiling**, not a field
  performance claim. PlantVillage images have uniform backgrounds and centered
  leaves; every image in each split is also byte-duplicated, so the effective
  sample diversity behind that number is half of what the counts suggest.
- **`typescript.ignoreBuildErrors: true`** in `frontend/next.config.mjs` still
  ships type errors. Left on because turning it off could break the build; worth
  fixing separately.
- **`backend/main.py` is excluded from the image.** It streams from
  `cv2.VideoCapture(0)` — a camera attached to the server — which cannot work in
  a container. It is not wired to the frontend either.
- **No tests and no CI.** There is nothing gating a bad deploy.
