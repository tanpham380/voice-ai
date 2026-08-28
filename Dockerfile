# Voice service — TTS (VieNeu) + STT (sherpa-onnx, nghi-stt-v3) + AI voice flow.
#
# Build target is selected by the GPU build-arg:
#   GPU=0 (default) -> CPU image (torch-free ONNX TTS)
#   GPU=1           -> NVIDIA CUDA image (PyTorch GPU backend for VieNeu)
# docker-compose.yml passes GPU=1 + TARGET_STAGE=gpu automatically via build.args.

# Global ARGs (must be declared before any FROM to be visible in all stages).
ARG GPU=0
ARG TARGET_STAGE=cpu

# ─────────────────────────────────────────────────────────────────────────────
# CPU stage — python:3.11-slim, torch-free (ONNX Runtime TTS).
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# System deps: libsndfile (soundfile), curl (healthcheck).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache).
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the app.
COPY app ./app
COPY static ./static
COPY scripts ./scripts

# Pre-stage the VieNeu ONNX backbone + cloning artifacts into models/tts so the
# service can boot offline (the MOSS codec is still fetched from HF on first
# run and cached in the hf-cache volume). The host mounts ./models over /app/models
# at runtime, so this pre-stage is only a fallback if the host dir is empty.
RUN python scripts/download_tts_models.py || true

# Non-root user.
RUN useradd -m appuser \
    && mkdir -p /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser
USER appuser

# Models are mounted as a volume (see docker-compose.yml).
ENV STT_MODEL_DIR=models/stt \
    VAD_MODEL=models/vad/silero_vad.onnx \
    TTS_MODEL_DIR=models/tts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ─────────────────────────────────────────────────────────────────────────────
# GPU stage — NVIDIA CUDA 12.4 runtime (matches host driver 555.97 / CUDA 12.5).
# VieNeu auto-detects the CUDA torch build and switches to the PyTorch backend.
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# System deps: libsndfile (soundfile), curl (healthcheck), python3 + pip.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 curl python3.11 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3.11 -m pip install --upgrade pip

WORKDIR /app

# Install Python deps (GPU torch first, then the rest).
COPY requirements.txt .
RUN python3.11 -m pip install --upgrade pip \
    && python3.11 -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 \
    && python3.11 -m pip install "transformers==4.49.0" \
    && python3.11 -m pip install -r requirements.txt

# Copy the app.
COPY app ./app
COPY static ./static
COPY scripts ./scripts

# Pre-stage the VieNeu ONNX backbone + cloning artifacts (offline fallback).
RUN python3.11 scripts/download_tts_models.py || true

# Non-root user.
RUN useradd -m appuser \
    && mkdir -p /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser
USER appuser

ENV STT_MODEL_DIR=models/stt \
    VAD_MODEL=models/vad/silero_vad.onnx \
    TTS_MODEL_DIR=models/tts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ─────────────────────────────────────────────────────────────────────────────
# Final stage — pick cpu or gpu based on the TARGET_STAGE build-arg
# (declared globally at the top of this file).
# ─────────────────────────────────────────────────────────────────────────────
FROM ${TARGET_STAGE} AS final

