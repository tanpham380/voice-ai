"""TTS (text-to-speech) engine built on ZeroTTS (Vietnamese zero-shot, 48 kHz).

Why ZeroTTS
-----------
ZeroTTS is a Vietnamese zero-shot TTS that runs entirely on CPU via ONNX (no
PyTorch). It replaces the VieNeu engine in this service because it is markedly
better for a low-latency streaming assistant:

  • Time-to-first-audio ~70 ms (vs ~0.5 s for VieNeu) — the bot speaks almost
    as soon as the first sentence arrives, with no visible pause.
  • Near-zero dead air between sentences (~29 ms), so there is no awkward gap
    mid-conversation.
  • Higher naturalness (UTMOS ~2.91 vs ~2.35) and much lower WER (~1 % vs
    ~16 %), so Vietnamese reads sound more natural and accurate.
  • Real-time streaming: synthesizes at ~0.5× wall-clock on a typical laptop
    CPU, so it can keep up with live playback.

How it works
------------
The model generates MOSS codec codes frame-by-frame and decodes them with a
continuous streaming codec session, so audio arrives in small chunks rather
than after the whole utterance is rendered. A voice is a tiny float32 latent
array (a "voice pack"), not a reference clip — ZeroTTS cannot clone a voice
from audio (that encoder is not published); it ships with 8 preset voices.

The engine is a lazily-created singleton so the (heavy, ~900 MB) model load
happens once, on first use.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from . import config

logger = logging.getLogger("voice.tts")

_lock = threading.Lock()
_engine = None
_engine_desc = ""

# ZeroTTS outputs 48 kHz audio.
SAMPLE_RATE = 48000


def _build_engine():
    import zerotts  # lazy import so the app boots before the model is downloaded

    # If a local TTS model dir is pre-staged (scripts/download_tts_models.py),
    # point the engine at it so the weights load offline.
    local_dir = config.TTS_MODEL_DIR_ABS
    if local_dir.is_dir() and any(local_dir.glob("onnx/**/*.onnx")):
        logger.info("Using pre-staged ZeroTTS weights from %s", local_dir)
        engine = zerotts.ZeroTTS(model_dir=str(local_dir), warmup=True)
        desc = f"zerotts (local={local_dir})"
    else:
        engine = zerotts.ZeroTTS.from_pretrained("zeroweight-ai/ZeroTTS")
        desc = "zerotts (hf)"
    return engine, desc


def get_engine():
    """Return the singleton ZeroTTS engine, building it on first call."""
    global _engine, _engine_desc
    if _engine is None:
        with _lock:
            if _engine is None:
                logger.info("Loading ZeroTTS TTS engine…")
                _engine, _engine_desc = _build_engine()
                logger.info("TTS engine ready: %s", _engine_desc)
    return _engine


def _tts_kwargs(overrides: dict | None = None) -> dict:
    """Build the ZeroTTS sampling kwargs from config, with per-call overrides.

    `overrides` may carry any of: cfg_scale, text_temperature, text_topk,
    audio_temperature, audio_topk, audio_topp, audio_repetition_penalty.
    Unknown keys are ignored.
    """
    base = {
        "cfg_scale": config.TTS_CFG_SCALE,
        "text_temperature": config.TTS_TEXT_TEMPERATURE,
        "text_topk": config.TTS_TEXT_TOPK,
        "audio_temperature": config.TTS_AUDIO_TEMPERATURE,
        "audio_topk": config.TTS_AUDIO_TOPK,
        "audio_topp": config.TTS_AUDIO_TOPP,
        "audio_repetition_penalty": config.TTS_AUDIO_REPETITION_PENALTY,
        "min_frames": config.TTS_MIN_FRAMES,
        "max_frames": config.TTS_MAX_FRAMES,
    }
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                base[k] = v
    return base


def synthesize(text: str, voice: str | None = None, **kwargs) -> bytes:
    """Synthesize text to a 16-bit PCM WAV byte string (48 kHz mono).

    The text is first normalized by ``ai.format_for_tts`` so markdown / emoji
    / stray punctuation don't get read aloud. Extra kwargs tune the reading
    (see _tts_kwargs).
    """
    if not text or not text.strip():
        raise ValueError("text is empty")
    from . import ai

    clean = ai.format_for_tts(text)
    if not clean:
        raise ValueError("text rỗng sau khi format")
    engine = get_engine()
    voice = voice or config.TTS_DEFAULT_VOICE
    audio: np.ndarray = engine.synthesize(clean, voice=voice, **_tts_kwargs(kwargs))
    return _to_pcm16_bytes(audio)


def _to_pcm16_bytes(audio: np.ndarray) -> bytes:
    """Encode a float32 waveform to raw little-endian PCM16 (for WAV files)."""
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    a = np.clip(a * 32767.0, -32768, 32767)
    return a.astype(np.int16).tobytes()


def synthesize_pcm16(text: str, voice: str | None = None, **kwargs) -> bytes:
    """Synthesize text to raw PCM16 bytes (48 kHz mono) — convenience wrapper."""
    if not text or not text.strip():
        raise ValueError("text is empty")
    from . import ai

    clean = ai.format_for_tts(text)
    if not clean:
        raise ValueError("text rỗng sau khi format")
    engine = get_engine()
    voice = voice or config.TTS_DEFAULT_VOICE
    audio: np.ndarray = engine.synthesize(clean, voice=voice, **_tts_kwargs(kwargs))
    return _to_pcm16_bytes(audio)


def _pcm16(audio_f32: np.ndarray) -> bytes:
    """Encode a float32 chunk to raw little-endian PCM16 (for streaming)."""
    a = np.asarray(audio_f32, dtype=np.float32)
    a = np.clip(a * 32767.0, -32768, 32767)
    return a.astype(np.int16).tobytes()


def synthesize_stream(text: str, voice: str | None = None, **kwargs):
    """Yield raw PCM16 mono @48 kHz chunks as they are generated (low latency).

    The text is normalized by ``ai.format_for_tts`` first (same as
    ``synthesize``) so streamed sentences are clean speech. Because ZeroTTS
    has near-zero dead air and ~70 ms time-to-first-audio, the first audio
    arrives in a few hundred ms instead of after the whole utterance is
    rendered. Extra kwargs tune the reading (see _tts_kwargs).
    """
    if not text or not text.strip():
        raise ValueError("text is empty")
    from . import ai

    clean = ai.format_for_tts(text)
    if not clean:
        return
    engine = get_engine()
    voice = voice or config.TTS_DEFAULT_VOICE
    for chunk in engine.synthesize_stream(clean, voice=voice, **_tts_kwargs(kwargs)):
        if chunk is None or chunk.size == 0:
            continue
        yield _pcm16(chunk.reshape(-1))


def list_voices() -> list[dict]:
    """Return the available preset voices as [{label, voice}].

    ZeroTTS ships with 8 preset voices (see docs/VOICES.md). Each carries a
    gender tag read from its pack metadata so the UI can group male/female.
    """
    engine = get_engine()
    out = []
    for name in engine.list_voices():
        try:
            v = engine.load_voice(name)
            gender = getattr(v, "gender", "") or ""
            label = getattr(v, "display_name", None) or name
        except Exception:  # noqa: BLE001
            gender, label = "", name
        out.append({"label": str(label), "voice": str(name), "gender": str(gender)})
    return out


def status() -> dict:
    return {
        "engine": _engine_desc or "vieneu (chưa tải)",
        "default_voice": config.TTS_DEFAULT_VOICE,
        "default_voices": config.DEFAULT_VOICES,
        "sample_rate": SAMPLE_RATE,
        "ready": _engine is not None,
    }
