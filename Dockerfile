# CPU-only inference image for backend/app.py.
#
# Pins match the versions verified working locally: Python 3.11, torch 2.11.0,
# torchvision 0.26.0, ultralytics 8.3.39, numpy 1.26.4.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # ultralytics and matplotlib both need a writable config dir, and $HOME is
    # not writable for the non-root user created below.
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

# libGL and libglib are runtime shared-library deps of opencv-python, which
# ultralytics pulls in unconditionally even though app.py never imports cv2.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch, installed BEFORE requirements.txt. See the note in
# requirements.txt: the default PyPI wheels bundle CUDA and are ~10x larger.
RUN pip install \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.11.0+cpu \
      torchvision==0.26.0+cpu

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY backend/ ./backend/
COPY best.pt ./best.pt

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

# Load the model at build time so a missing or corrupt best.pt fails the build
# instead of surfacing as a 503 on the first request in production.
RUN python -c "from backend.model import load_model, get_device; load_model(); print('model loaded on', get_device())"

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import os,sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health').status==200 else 1)"

# Single worker on purpose: every worker loads its own copy of the model into
# memory. Scale with container replicas, not with --workers.
CMD ["sh", "-c", "exec uvicorn backend.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
