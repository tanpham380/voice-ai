"""Environment-driven configuration for the voice service."""
from __future__ import annotations

import os
from pathlib import Path

# Load .env into os.environ on import (so enable_thinking, API keys, etc. apply).
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Project root = the directory that contains `app/`.
ROOT = Path(__file__).resolve().parent.parent


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, "").strip() or default)
    except ValueError:
        return default


# ─── HTTP ───
HOST = _env("HOST", "0.0.0.0")
PORT = _env_int("PORT", 8000)

# ─── STT ───
# Vietnamese Zipformer (transducer) — the `nghi-stt-v3` model served by the
# live nghitts.app web app (fetched + extracted by scripts/download_models.py).
STT_MODEL_NAME = _env("STT_MODEL_NAME", "nghi-stt-v3")
STT_NUM_THREADS = _env_int("STT_NUM_THREADS", 4)
STT_MODEL_DIR = _env("STT_MODEL_DIR", "models/stt")
VAD_MODEL = _env("VAD_MODEL", "models/vad/silero_vad.onnx")

# Resolve model paths relative to the project root.
STT_MODEL_DIR_ABS = (ROOT / STT_MODEL_DIR).resolve()
VAD_MODEL_ABS = (ROOT / VAD_MODEL).resolve()

# ─── TTS ───
# Local directory holding the pre-staged VieNeu ONNX artifacts
# (backbone int8 graphs + denoiser.onnx + speaker_encoder.onnx).
# When this directory exists, the engine loads from it directly (no HF fetch
# for the backbone / clone files). The MOSS codec is still fetched from HF on
# first run and cached in the HuggingFace cache.
TTS_MODEL_DIR = _env("TTS_MODEL_DIR", "models/tts")
TTS_MODEL_DIR_ABS = (ROOT / TTS_MODEL_DIR).resolve()
# Default voice used when the caller doesn't specify one.
TTS_DEFAULT_VOICE = _env("TTS_DEFAULT_VOICE", "Adam")
# Pre-configured quick-pick voices: one male, one female. These are surfaced in
# the UI as one-tap buttons and returned by /health + /v1/tts/voices.
TTS_DEFAULT_VOICE_MALE = _env("TTS_DEFAULT_VOICE_MALE", "Adam")
TTS_DEFAULT_VOICE_FEMALE = _env("TTS_DEFAULT_VOICE_FEMALE", "Ngọc Huyền")
DEFAULT_VOICES = {
    "male": TTS_DEFAULT_VOICE_MALE,
    "female": TTS_DEFAULT_VOICE_FEMALE,
}

# ─── TTS (VieNeu) sampling / reading params ───
# These tune how the model "reads" the text. All overridable per-request via
# the /v1/tts/* endpoints. Defaults mirror VieNeu's own defaults.
TTS_TEMPERATURE = float(_env("TTS_TEMPERATURE", "0.8"))
TTS_TOP_P = float(_env("TTS_TOP_P", "0.95"))
TTS_TOP_K = _env_int("TTS_TOP_K", 25)
TTS_REPETITION_PENALTY = float(_env("TTS_REPETITION_PENALTY", "1.2"))
TTS_REPETITION_WINDOW = _env_int("TTS_REPETITION_WINDOW", 64)
TTS_MAX_CHARS = _env_int("TTS_MAX_CHARS", 256)
# silence_p adds pauses between sentences (0..1); crossfade_p smooths chunks.
TTS_SILENCE_P = float(_env("TTS_SILENCE_P", "0.15"))
TTS_CROSSFADE_P = float(_env("TTS_CROSSFADE_P", "0.3"))
# VieNeu output sample rate (Hz).
TTS_SAMPLE_RATE = _env_int("TTS_SAMPLE_RATE", 48000)
# Silence padding (samples) inserted between streamed TTS chunks to avoid hard
# clicks at chunk joins. At 48 kHz, ~30 ms is a natural micro-pause.
TTS_SILENCE_SAMPLES = int(round(TTS_SAMPLE_RATE * 0.030))
# Speaking rate. 1.0 = natural; >1 faster, <1 slower. Mapped to VieNeu's
# max_new_frames (base 300): max_new_frames = int(300 * TTS_SPEED).
TTS_SPEED = float(_env("TTS_SPEED", "1.35"))
TTS_MAX_NEW_FRAMES_BASE = _env_int("TTS_MAX_NEW_FRAMES_BASE", 300)
# Pitch-preserving playback speed-up. VieNeu has NO native speaking-rate
# parameter, so this time-stretches the synthesized audio (faster speech,
# same pitch). 1.0 = natural; >1 faster. Tuned in .env.
TTS_PLAYBACK_SPEED = float(_env("TTS_PLAYBACK_SPEED", "1.25"))
# Optional style/emotion cue passed to the engine (e.g. "happy", "calm").
# Empty = default neutral reading. Can also embed [cười] etc. in the text.
TTS_STYLE = _env("TTS_STYLE", "")
# Apply the VieNeu watermark to synthesized audio (provenance). 1/0.
TTS_APPLY_WATERMARK = _env_int("TTS_APPLY_WATERMARK", 1) == 1

# ─── AI (LLM) ───
# Local-first: STT + TTS run offline; only the LLM calls an OpenAI-compatible API.
# Inspired by xiaozhi-esp32-server's provider layering (core/providers/llm/openai).
AI_CHAT_URL = _env("AI_CHAT_URL", "")  # kept for backward-compat; empty => use OpenAI
AI_TIMEOUT_SECONDS = _env_int("AI_TIMEOUT_SECONDS", 60)

# OpenAI-compatible Chat Completions settings.
# Point at any OpenAI-compatible server. For a bare /v1/chat/completions URL,
# set OPENAI_BASE_URL to the part BEFORE /chat/completions (e.g. http://host:8000/v1).
OPENAI_API_KEY = _env("OPENAI_API_KEY", "EMPTY")
OPENAI_BASE_URL = _env("OPENAI_BASE_URL", "http://192.168.1.128:8000/v1")
OPENAI_MODEL = _env("OPENAI_MODEL", "Ornith-1.5-35B-A3B")
OPENAI_TEMPERATURE = float(_env("OPENAI_TEMPERATURE", "0.6"))
OPENAI_MAX_TOKENS = _env_int("OPENAI_MAX_TOKENS", 16384)
OPENAI_TOP_P = float(_env("OPENAI_TOP_P", "0.95"))
OPENAI_PRESENCE_PENALTY = float(_env("OPENAI_PRESENCE_PENALTY", "0.2"))
OPENAI_REPETITION_PENALTY = float(_env("OPENAI_REPETITION_PENALTY", "1.05"))
# Reasoning / thinking control. Ornith 1.5 is a reasoning model that emits a
# <think>…</think> block before the answer, which adds latency. Set to False to
# disable thinking (model answers immediately). vLLM reads this via extra_body.
OPENAI_ENABLE_THINKING = _env_int("OPENAI_ENABLE_THINKING", 0) == 1
# Stop sequences (JSON array in env, or comma list). Empty = none.
# NOTE: do NOT include </think> / \n</think> — Ornith 1.5 is a reasoning model
# that emits a <think>…</think> block first; stopping on </think> would cut off
# the real answer (which follows the thinking block). Reasoning is returned
# separately by vLLM in the `reasoning` field, so we never need to stop on it.
_OPENAI_STOP_RAW = _env("OPENAI_STOP", "<|im_end|>,<|endoftext|>")
OPENAI_STOP = [s for s in _OPENAI_STOP_RAW.split(",") if s.strip()] if _OPENAI_STOP_RAW else []
# System prompt for the AI consultant. {{current_time}} is replaced at runtime.
OPENAI_SYSTEM_PROMPT = _env(
    "OPENAI_SYSTEM_PROMPT",
    "Bạn là chuyên viên tư vấn tuyển sinh và cố vấn hướng nghiệp thân thiện, "
    "nói tiếng Việt tự nhiên. Trả lời ngắn gọn, ấm áp, đúng trọng tâm. "
    "Thời gian hiện tại: {{current_time}}.",
)

# ─── VAD (Silero) ───
# Voice activity detection gates the ASR (only run recognition on voiced segments).
# Model auto-downloaded by scripts/download_models.py into models/vad/.
VAD_MODEL = _env("VAD_MODEL", "models/vad/silero_vad.onnx")
VAD_MODEL_ABS = (ROOT / VAD_MODEL).resolve()
# Frames (16 kHz, 16-bit) buffered before/after voice to avoid clipping words.
VAD_PAD_FRAMES = _env_int("VAD_PAD_FRAMES", 8)
# Silence frames required to declare "voice stopped" (triggers ASR).
VAD_SILENCE_FRAMES = _env_int("VAD_SILENCE_FRAMES", 12)
