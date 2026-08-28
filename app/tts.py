"""TTS (text-to-speech) engine built on VieNeu (Vietnamese, 48 kHz).

VieNeu v3 Turbo capabilities used here:
  • 20 preset voices            -> synthesize(text, voice="Adam")
  • Voice cloning (3–8s clip)   -> clone(text, ref_audio_bytes)
  • Real-time streaming         -> synthesize_stream(text, voice)  (PCM16 chunks)
  • Emotion / non-verbal cues   -> just embed [cười] [thở dài] [hắng giọng]
                                   in the text (handled by the engine)
  • En–Vi code-switching        -> automatic in the text

VieNeu auto-detects the backend: if a CUDA build of torch is present it uses
PyTorch (batched, fast); otherwise it falls back to a torch-free ONNX path
(which is also the low-latency streaming path). The engine is a lazily-created
singleton so the (heavy) model load happens once, on first use.
"""
from __future__ import annotations

import io
import logging
import os
import tempfile
import threading

import numpy as np
import soundfile as sf

from . import config

logger = logging.getLogger("voice.tts")

_lock = threading.Lock()
_engine = None
_engine_desc = ""

# VieNeu v3 Turbo outputs 48 kHz audio.
SAMPLE_RATE = 48000


def _build_engine():
    import vieneu  # lazy import so the app boots before the model is downloaded

    # If a local TTS model dir is pre-staged (scripts/download_tts_models.py),
    # point the engine at it so the backbone + clone artifacts load offline.
    # The MOSS codec is still pulled from HF on first run (cached).
    local_dir = config.TTS_MODEL_DIR_ABS
    if local_dir.is_dir() and any(local_dir.glob("vieneu_*.onnx")):
        logger.info("Using pre-staged VieNeu ONNX artifacts from %s", local_dir)
        engine = vieneu.Vieneu(
            backend="onnx",                 # force CPU/ONNX path
            backbone_repo=str(local_dir),   # denoiser.onnx + speaker_encoder.onnx
            onnx_dir=str(local_dir),        # backbone int8 graphs
            precision="int8",
        )
        desc = f"vieneu (onnx, local={local_dir})"
    else:
        engine = vieneu.Vieneu()  # auto-detects CUDA (torch) vs ONNX (CPU)
        desc = "vieneu (auto)"
    return engine, desc


def get_engine():
    """Return the singleton VieNeu engine, building it on first call."""
    global _engine, _engine_desc
    if _engine is None:
        with _lock:
            if _engine is None:
                logger.info("Loading VieNeu TTS engine…")
                _engine, _engine_desc = _build_engine()
                logger.info("TTS engine ready: %s", _engine_desc)
    return _engine


def _to_wav_bytes(audio: np.ndarray) -> bytes:
    """Encode a float32 waveform to a 16-bit PCM WAV byte string (48 kHz)."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _tts_kwargs(overrides: dict | None = None) -> dict:
    """Build the VieNeu sampling kwargs from config, with per-call overrides.

    `overrides` may carry any of: temperature, top_p, top_k, repetition_penalty,
    repetition_window, max_chars, silence_p, crossfade_p, style, denoise,
    apply_watermark, speed. Unknown keys are ignored (forwarded to the engine).
    """
    # Speaking rate -> VieNeu max_new_frames (higher = faster synthesis/speech).
    speed = config.TTS_SPEED
    max_new_frames = int(config.TTS_MAX_NEW_FRAMES_BASE * speed)
    base = {
        "temperature": config.TTS_TEMPERATURE,
        "top_p": config.TTS_TOP_P,
        "top_k": config.TTS_TOP_K,
        "repetition_penalty": config.TTS_REPETITION_PENALTY,
        "repetition_window": config.TTS_REPETITION_WINDOW,
        "max_chars": config.TTS_MAX_CHARS,
        "silence_p": config.TTS_SILENCE_P,
        "crossfade_p": config.TTS_CROSSFADE_P,
        "max_new_frames": max_new_frames,
        "style": config.TTS_STYLE or None,
        "apply_watermark": config.TTS_APPLY_WATERMARK,
    }
    if overrides:
        for k, v in overrides.items():
            if v is None:
                continue
            if k == "speed":
                base["max_new_frames"] = int(config.TTS_MAX_NEW_FRAMES_BASE * float(v))
            else:
                base[k] = v
    return base


def synthesize(text: str, voice: str | None = None, **kwargs) -> bytes:
    """Synthesize text to a 16-bit PCM WAV byte string (48 kHz mono).

    The text is first normalized by ``ai.format_for_tts`` so markdown / emoji
    / stray punctuation don't get read aloud. Emotion / non-verbal cues such
    as ``[cười]`` or ``[thở dài]`` survive the cleanup and are handled by the
    engine. Extra kwargs tune the reading (see _tts_kwargs).
    """
    if not text or not text.strip():
        raise ValueError("text is empty")
    from . import ai

    clean = ai.format_for_tts(text)
    if not clean:
        raise ValueError("text rỗng sau khi format")
    engine = get_engine()
    voice = voice or config.TTS_DEFAULT_VOICE
    audio: np.ndarray = engine.infer(clean, voice=voice, **_tts_kwargs(kwargs))
    return _to_wav_bytes(audio)


def clone(text: str, ref_audio: bytes, voice: str | None = None,
          denoise: bool = True, **kwargs) -> bytes:
    """Synthesize ``text`` with a voice cloned from an uploaded reference clip.

    ``ref_audio`` is the raw bytes of a 3–8 s WAV/MP3 clip. It is written to a
    temp file (the SDK expects a path), used for one synthesis, then removed.
    ``voice`` is ignored when cloning (the reference defines the voice). Extra
    kwargs tune the reading (see _tts_kwargs).
    """
    if not text or not text.strip():
        raise ValueError("text is empty")
    if not ref_audio:
        raise ValueError("ref_audio is empty")
    engine = get_engine()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        tmp.write(ref_audio)
        tmp.close()
        audio: np.ndarray = engine.infer(
            text, ref_audio=tmp.name, denoise=denoise, **_tts_kwargs(kwargs)
        )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    return _to_wav_bytes(audio)


def _pcm16(audio_f32: np.ndarray) -> bytes:
    """Encode a float32 chunk to raw little-endian PCM16 (for streaming)."""
    return (np.asarray(audio_f32) * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


def synthesize_stream(text: str, voice: str | None = None, **kwargs):
    """Yield raw PCM16 mono @48 kHz chunks as they are generated (low latency).

    The text is normalized by ``ai.format_for_tts`` first (same as
    ``synthesize``) so streamed sentences are clean speech. Use this behind a
    streaming WAV response so the first audio arrives in a few hundred ms
    instead of after the whole utterance is rendered. Extra kwargs tune the
    reading (see _tts_kwargs).
    """
    if not text or not text.strip():
        raise ValueError("text is empty")
    from . import ai

    clean = ai.format_for_tts(text)
    if not clean:
        return
    engine = get_engine()
    voice = voice or config.TTS_DEFAULT_VOICE
    for chunk in engine.infer_stream(clean, voice=voice, **_tts_kwargs(kwargs)):
        if chunk is None or len(chunk) == 0:
            continue
        yield _pcm16(chunk)


def list_voices() -> list[dict]:
    """Return the available preset voices as [{label, voice}]."""
    engine = get_engine()
    voices = engine.list_preset_voices()
    out = []
    for item in voices:
        # list_preset_voices() yields (label, voice_id) tuples.
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            out.append({"label": str(item[0]), "voice": str(item[1])})
        else:
            out.append({"label": str(item), "voice": str(item)})
    return out


def status() -> dict:
    return {
        "engine": _engine_desc or "vieneu (chưa tải)",
        "default_voice": config.TTS_DEFAULT_VOICE,
        "default_voices": config.DEFAULT_VOICES,
        "sample_rate": SAMPLE_RATE,
        "ready": _engine is not None,
    }
