"""Pre-stage the VieNeu-TTS v3 Turbo ONNX artifacts into models/tts/.

Why
---
On CPU the VieNeu SDK runs a torch-free ONNX engine. By default it pulls the
graphs it needs from Hugging Face on first use (cached in ~/.cache/huggingface).
That works, but the first request pays the download cost.

This script downloads the SAME files the engine needs into a FLAT local
directory (models/tts/) so the app can boot fully offline for the backbone +
cloning. The MOSS audio codec still comes from HF on first run (the public SDK
does not expose a codec_dir), but it is cached in the hf-cache volume.

Layout written here (mirrors what app/tts.py expects — flat, no subfolder):

    models/tts/
      vieneu_prefill.onnx
      vieneu_decode_step.onnx
      vieneu_acoustic_cached.onnx
      vieneu_backbone_shared.data
      vieneu_v3_heads.npz
      config.json
      tokenizer.json
      denoiser.onnx         <- cloning (repo root)
      speaker_encoder.onnx  <- cloning (repo root)

Usage:
    python scripts/download_tts_models.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TTS_DIR = ROOT / "models" / "tts"

# VieNeu v3 Turbo backbone repo (HF). The ONNX graphs live in the onnx_int8
# subfolder; we download them flat into TTS_DIR (app/tts.py points onnx_dir
# at TTS_DIR directly).
V3_REPO = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
ONNX_SUBFOLDER = "onnx_int8"

# Backbone graphs + setup (fetched from <V3_REPO>/<ONNX_SUBFOLDER>).
BACKBONE_FILES = [
    "vieneu_prefill.onnx",
    "vieneu_decode_step.onnx",
    "vieneu_acoustic_cached.onnx",
    "vieneu_backbone_shared.data",
    "vieneu_v3_heads.npz",
    "config.json",
    "tokenizer.json",
]
# Cloning artifacts (fetched from the repo root).
ROOT_FILES = [
    "denoiser.onnx",
    "speaker_encoder.onnx",
]

HF_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "*/*",
}


def _hf_url(repo: str, filename: str, subfolder: str | None = None) -> str:
    base = f"https://huggingface.co/{repo}/resolve/main"
    return f"{base}/{subfolder}/{filename}" if subfolder else f"{base}/{filename}"


def _hf_size(repo: str, filename: str, subfolder: str | None = None) -> int | None:
    """Best-effort expected size (bytes) from the HF API; None if unknown."""
    api = f"https://huggingface.co/api/models/{repo}"
    if subfolder:
        api += f"/tree/main/{subfolder}"
    try:
        req = urllib.request.Request(api, headers=HF_HEADERS)
        with urllib.request.urlopen(req, timeout=30) as r:
            import json

            data = json.loads(r.read().decode("utf-8"))
        for item in data:
            if item.get("path") == filename or item.get("rfilename") == filename:
                return int(item.get("size") or 0) or None
    except Exception:
        pass
    return None


def _download(url: str, dest: Path, expected: int | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if expected is None or dest.stat().st_size == expected:
            print(f"  ✓ already present: {dest.name}")
            return
        print(f"  ↻ size mismatch, re-downloading: {dest.name}")
    print(f"  ↓ {url}")
    req = urllib.request.Request(url, headers=HF_HEADERS)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or expected or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e6:.1f}/{total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.rename(dest)
    print(f"  ✓ saved: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")


def download_tts() -> None:
    print(f"== VieNeu-TTS v3 Turbo (ONNX, int8) → {TTS_DIR}")
    print(f"   backbone subfolder: {ONNX_SUBFOLDER}")
    for fn in BACKBONE_FILES:
        _download(_hf_url(V3_REPO, fn, ONNX_SUBFOLDER), TTS_DIR / fn, _hf_size(V3_REPO, fn, ONNX_SUBFOLDER))
    for fn in ROOT_FILES:
        _download(_hf_url(V3_REPO, fn), TTS_DIR / fn, _hf_size(V3_REPO, fn))
    _verify()


def _verify() -> None:
    missing = [f for f in BACKBONE_FILES if not (TTS_DIR / f).is_file()]
    missing += [f for f in ROOT_FILES if not (TTS_DIR / f).is_file()]
    if missing:
        print(f"  ✗ missing after download: {missing}")
        sys.exit(1)
    total = sum(p.stat().st_size for p in TTS_DIR.rglob("*") if p.is_file())
    print(f"  ✓ TTS models ready ({total / 1e6:.1f} MB total)")


if __name__ == "__main__":
    download_tts()
