"""STT (speech-to-text) engine built on sherpa-onnx.

Uses the **nghi-stt-v3** Vietnamese Zipformer (transducer) model — the model
served by the live nghitts.app web app (fetched + extracted by
`scripts/download_models.py`). The model is a standard sherpa-onnx transducer
layout:

    models/stt/
      transducer-encoder.onnx
      transducer-decoder.onnx
      transducer-joiner.onnx
      tokens.txt

File names are auto-detected (glob), so any ONNX export of the Vietnamese
Zipformer works without code changes. Transcription is offline and private —
audio never leaves this process.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from . import config

logger = logging.getLogger("voice.stt")

_lock = threading.Lock()
_recognizer = None
_engine_desc = ""


def _detect_model_files() -> dict[str, str]:
    """Locate encoder/decoder/joiner/tokens in the model dir.

    Prefers a plain ``encoder.onnx`` over ``encoder-epoch-*.onnx`` and prefers
    int8 variants. Raises a friendly error listing what's missing.
    """
    d = config.STT_MODEL_DIR_ABS
    if not d.is_dir():
        raise RuntimeError(
            f"Thiếu thư mục model STT: {d}. "
            "Chạy `python scripts/download_models.py` hoặc đặt file model vào đây."
        )

    def pick(patterns: list[str], label: str) -> str:
        for pat in patterns:
            matches = sorted(d.glob(pat))
            if matches:
                return str(matches[0])
        raise RuntimeError(f"Thiếu file {label} trong {d} (đã tìm: {', '.join(patterns)})")

    return {
        "encoder": pick(["*encoder*.int8.onnx", "*encoder*.onnx"], "encoder"),
        "decoder": pick(["*decoder*.int8.onnx", "*decoder*.onnx"], "decoder"),
        "joiner": pick(["*joiner*.int8.onnx", "*joiner*.onnx"], "joiner"),
        "tokens": pick(["tokens.txt"], "tokens.txt"),
    }


def _missing_files() -> list[str]:
    try:
        _detect_model_files()
        if not config.VAD_MODEL_ABS.is_file():
            return [str(config.VAD_MODEL_ABS)]
        return []
    except RuntimeError as e:
        return [str(e)]


def _build_recognizer():
    import sherpa_onnx  # lazy import so the app boots before models exist

    files = _detect_model_files()
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=files["encoder"],
        decoder=files["decoder"],
        joiner=files["joiner"],
        tokens=files["tokens"],
        num_threads=config.STT_NUM_THREADS,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        debug=False,
    )
    desc = f"zipformer-vi (nghi-stt-v3, {Path(files['encoder']).name})"
    return recognizer, desc


def get_recognizer():
    """Return the singleton recognizer, building it on first call."""
    global _recognizer, _engine_desc
    if _recognizer is None:
        with _lock:
            if _recognizer is None:
                missing = _missing_files()
                if missing:
                    raise RuntimeError(
                        "Thiếu file model STT: " + "; ".join(missing)
                        + ". Chạy `python scripts/download_models.py` để tải."
                    )
                logger.info("Loading STT recognizer (nghi-stt-v3)…")
                _recognizer, _engine_desc = _build_recognizer()
                logger.info("STT recognizer ready: %s", _engine_desc)
    return _recognizer


def transcribe(samples: np.ndarray, sample_rate: int) -> str:
    """Transcribe a 1-D float32 mono buffer (range [-1, 1]) to text."""
    recognizer = get_recognizer()
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


def status() -> dict:
    missing = _missing_files()
    return {
        "engine": _engine_desc or "zipformer-vi (chưa tải)",
        "model": "nghi-stt-v3",
        "ready": _recognizer is not None and not missing,
        "missing_files": missing,
    }
