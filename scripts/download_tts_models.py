"""Pre-stage the ZeroTTS weights into models/tts/ for fully offline boot.

Why
---
ZeroTTS (zeroweight-ai/ZeroTTS) is a Vietnamese zero-shot TTS that runs on CPU
via ONNX (no PyTorch). By default the SDK pulls the ~900 MB weights from
Hugging Face on first use (cached in ~/.cache/huggingface). That works, but the
first request pays the download cost.

This script downloads the whole model repo into a local directory
(models/tts/) so the app can boot fully offline. app/tts.py points the engine
at this directory when it exists (it looks for ``onnx/**/*.onnx``).

Layout written here (mirrors the published repo):

    models/tts/
      config.json
      tokenizer.json
      onnx/                 <- text_encoder, prefix_step, local_frame_decode
      codec/                <- MOSS audio decoder (decode_full + decode_step)
      voices/               <- 8 preset voice packs (voice.npz + meta.json)
      null_voice_emb.npy
      silence_frame.npy

Usage:
    python scripts/download_tts_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("huggingface_hub not installed; run: pip install huggingface_hub")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
TTS_DIR = ROOT / "models" / "tts"

# ZeroTTS weights repo. ~900 MB, MIT licensed.
HF_REPO = "zeroweight-ai/ZeroTTS"


def download_tts() -> None:
    print(f"== ZeroTTS weights ({HF_REPO}) → {TTS_DIR}")
    print("   This is ~900 MB and may take a while on a slow connection.")
    files = snapshot_download(
        HF_REPO,
        local_dir=str(TTS_DIR),
        local_dir_use_symlinks=False,
    )
    _verify()


def _verify() -> None:
    required = ["config.json", "tokenizer.json", "null_voice_emb.npy"]
    missing = [f for f in required if not (TTS_DIR / f).is_file()]
    if missing:
        print(f"  ✗ missing after download: {missing}")
        sys.exit(1)
    if not any(TTS_DIR.rglob("onnx/**/*.onnx")):
        print("  ✗ no ONNX graphs found under onnx/ after download")
        sys.exit(1)
    voices = [p.parent.name for p in (TTS_DIR / "voices").glob("*/voice.npz")] if (TTS_DIR / "voices").is_dir() else []
    if not voices:
        print("  ✗ no voice packs found under voices/ after download")
        sys.exit(1)
    total = sum(p.stat().st_size for p in TTS_DIR.rglob("*") if p.is_file())
    print(f"  ✓ TTS models ready ({total / 1e6:.1f} MB total, {len(voices)} voices)")


if __name__ == "__main__":
    download_tts()
