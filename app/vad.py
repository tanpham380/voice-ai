"""Silero VAD wrapper (local, ONNX).

Mirrors xiaozhi-esp32-server's VAD gating idea (core/providers/vad/silero.py):
only run ASR when voice is detected, and declare "voice stopped" after a run of
silence frames. This keeps latency low and avoids transcribing silence.

The model (models/vad/silero_vad.onnx) is fetched by scripts/download_models.py.
We run it with onnxruntime (already available via sherpa-onnx).
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from . import config

logger = logging.getLogger("voice.vad")

# Silero expects 16 kHz mono float32, 512-sample (32 ms) frames.
_SAMPLE_RATE = 16000
_FRAME_SAMPLES = 512


class VAD:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session = None
        self._sr_tensor = None
        self._state = None

    def _ensure(self):
        if self._session is not None:
            return
        import onnxruntime as ort

        if not config.VAD_MODEL_ABS.exists():
            raise RuntimeError(
                f"Thiếu model VAD: {config.VAD_MODEL_ABS}. "
                "Chạy `python scripts/download_models.py`."
            )
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(config.VAD_MODEL_ABS), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._sr_tensor = np.array(_SAMPLE_RATE, dtype=np.int64)
        # Silero state: (2,1,64) for h and c, init to zeros.
        self._state_h = np.zeros((2, 1, 64), dtype=np.float32)
        self._state_c = np.zeros((2, 1, 64), dtype=np.float32)
        logger.info("Silero VAD ready")

    def reset(self) -> None:
        with self._lock:
            self._state_h = np.zeros((2, 1, 64), dtype=np.float32)
            self._state_c = np.zeros((2, 1, 64), dtype=np.float32)

    def is_voice(self, frame16k_mono_f32: np.ndarray) -> bool:
        """Return True if the given 512-sample frame contains speech."""
        with self._lock:
            self._ensure()
            x = np.asarray(frame16k_mono_f32, dtype=np.float32).reshape(1, -1)
            prob, new_h, new_c = self._session.run(
                ["prob", "new_h", "new_c"],
                {
                    "x": x,
                    "h": self._state_h,
                    "c": self._state_c,
                },
            )
            self._state_h = new_h
            self._state_c = new_c
            score = float(np.squeeze(prob))
            return score > 0.5


# Module-level singleton (lazy).
_instance: VAD | None = None


def get_vad() -> VAD:
    global _instance
    if _instance is None:
        _instance = VAD()
    return _instance
