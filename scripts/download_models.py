"""Download / check the models the voice service needs.

What it does:
  • Silero VAD  (silero_vad.onnx)  — auto-downloaded from the sherpa-onnx
    release assets (public).
  • nghi-stt-v3 (Vietnamese Zipformer transducer) — the model the live
    nghitts.app web app actually serves. It is packaged as an emscripten
    `.data` bundle (a single concatenated blob) plus a `.js` file that holds
    the byte-offset table for each file inside the blob. This script
    downloads both, then slices the blob into the 4 model files:

        transducer-encoder.onnx
        transducer-decoder.onnx
        transducer-joiner.onnx
        tokens.txt

    and drops them into models/stt/ (names are auto-detected by app/stt.py).

Usage:
    python scripts/download_models.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAD_DIR = ROOT / "models" / "vad"
STT_DIR = ROOT / "models" / "stt"

SILERO_VAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
)
SILERO_VAD_PATH = VAD_DIR / "silero_vad.onnx"

# The live nghitts.app web app serves the Vietnamese Zipformer as an
# emscripten bundle. The model name on the live site is `nghi-stt-v3`.
# The .js file contains:  loadPackage({"files":[{"filename","start","end"},...]})
# which is the byte-offset table into the .data blob.
ASR_MODEL = "nghi-stt-v3"
ASR_BASE = f"https://nghitts.app/api/model/asr/{ASR_MODEL}"
ASR_DATA_URL = f"{ASR_BASE}/sherpa-onnx-wasm-main-vad-asr.data"
ASR_JS_URL = f"{ASR_BASE}/sherpa-onnx-wasm-main-vad-asr.js"

# The 4 files we actually need out of the bundle (the bundle also carries
# README.md and silero_vad.onnx, which we ignore here). Names are
# auto-detected by app/stt.py via glob.
STT_FILES = [
    "transducer-encoder.onnx",
    "transducer-decoder.onnx",
    "transducer-joiner.onnx",
    "tokens.txt",
]


# nghitts.app sits behind Cloudflare, which blocks the default Python
# user-agent — send a browser-like one.
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  ✓ already present: {dest}")
        return
    print(f"  ↓ downloading {url}")
    req = urllib.request.Request(url, headers=_HEADERS)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
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
    print(f"  ✓ saved: {dest}")


def _extract_bundle(data_path: Path, js_path: Path) -> list[str]:
    """Slice the emscripten .data blob into its component files.

    Returns the list of file names written into STT_DIR.
    """
    blob = data_path.read_bytes()
    js = js_path.read_text(encoding="utf-8", errors="replace")

    idx = js.find('"files":')
    if idx < 0:
        raise RuntimeError("Could not find the 'files' offset table in the .js bundle.")
    arr_start = js.find("[", idx)
    files, _ = json.JSONDecoder().raw_decode(js[arr_start:])

    written: list[str] = []
    for f in files:
        name = f["filename"].lstrip("/")
        start, end = f["start"], f["end"]
        dest = STT_DIR / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob[start:end])
        written.append(name)
        print(f"    {name:32s} {start:9d}-{end:9d}  ({(end - start) / 1e6:7.2f} MB)")
    return written


def main() -> int:
    print("== Voice service model setup ==\n")

    # 1) Silero VAD (public, auto-download).
    print("[1/2] Silero VAD")
    try:
        _download(SILERO_VAD_URL, SILERO_VAD_PATH)
    except Exception as e:
        print(f"  ✗ failed: {e}")
        print(f"    Download manually from: {SILERO_VAD_URL}")
        print(f"    and save to: {SILERO_VAD_PATH}\n")

    # 2) Vietnamese Zipformer (nghi-stt-v3) — download the emscripten bundle
    #    and slice out the 4 model files.
    print(f"[2/2] {ASR_MODEL} (Vietnamese Zipformer — from nghitts.app)")
    STT_DIR.mkdir(parents=True, exist_ok=True)
    present = [f for f in STT_FILES if (STT_DIR / f).is_file()]
    if len(present) == len(STT_FILES):
        print(f"  ✓ all STT files present in {STT_DIR}")
    else:
        data_path = STT_DIR / "bundle.data"
        js_path = STT_DIR / "bundle.js"
        try:
            _download(ASR_DATA_URL, data_path)
            _download(ASR_JS_URL, js_path)
            print("  ✂ extracting model files from bundle...")
            _extract_bundle(data_path, js_path)
            # Clean up the raw bundle + js now that we have the files.
            for tmp in (data_path, js_path):
                if tmp.exists():
                    tmp.unlink()
            print("  ✓ bundle cleaned up")
        except Exception as e:
            print(f"  ✗ failed: {e}")
            print("    Download manually and place these files in:")
            print(f"      {STT_DIR}")
            for f in STT_FILES:
                mark = "✓" if (STT_DIR / f).is_file() else "✗"
                print(f"      {mark} {f}")

    print("\nDone. Start the service with:  docker compose up --build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
