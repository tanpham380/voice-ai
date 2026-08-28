"""Voice service — FastAPI app.

Endpoints
  GET  /health                 liveness + engine status
  POST /v1/voice/chat          multipart file (+voice, session_id) ->
                               {transcript, answer, audio (base64 WAV), session_id}
  POST /v1/chat                {message, session_id?, stream?} -> AI text (SSE)
  POST /v1/chat/stream-audio   {message, session_id?, stream?} -> streaming WAV (LLM->TTS)
  GET  /                       static assistant-style voice UI

The voice flow is:  mic → STT (nghi-stt-v3) → AI (consultant) → TTS (VieNeu).
"""
from __future__ import annotations

import base64
import io
import logging
import time
import wave

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import ai, config, stt, tts
from .pipeline import _split_sentences
from .tts import _tts_kwargs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("voice.main")

app = FastAPI(title="Voice Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# The static UI lives next to the project root (voice-service/static/index.html).
STATIC_DIR = config.ROOT / "static"


# ─── helpers ────────────────────────────────────────────────────────────────
def _load_audio_16k_mono(data: bytes) -> np.ndarray:
    """Decode uploaded audio bytes to a 16 kHz mono float32 array (for STT)."""
    buf = io.BytesIO(data)
    try:
        audio, sr = sf.read(buf, dtype="float32", always_2d=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không đọc được file âm thanh: {e}")
    audio = audio[:, 0]  # mono
    if sr != 16000:
        from scipy.signal import resample_poly

        g = _gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // g, sr // g)
    return audio.astype(np.float32)


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _wav_header() -> bytes:
    """A 48 kHz mono PCM16 WAV header with a huge nframes count.

    The oversized frame count lets the browser start playing immediately while
    the PCM body is still streaming in (the real length is unknown up front).
    """
    h = io.BytesIO()
    with wave.open(h, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(tts.SAMPLE_RATE)
        w.setnframes(1_000_000_000)
    return h.getvalue()


def _stream_tts_pcm(message: str, session_id: str | None, voice: str | None):
    """Yield raw PCM16 @48 kHz chunks: LLM streams tokens -> sentence TTS.

    This is the low-latency core shared by both the text and voice paths. The
    first audio chunk arrives after the first sentence (or ~4 words), so the
    browser can start playing immediately instead of waiting for the whole
    answer. The caller wraps it in either a WAV header (StreamingResponse) or
    a raw PCM stream for Web Audio.
    """
    tts_kwargs = _tts_kwargs({"voice": voice} if voice else {})
    pending = ""
    try:
        for token in ai.ask_stream(message, session_id):
            pending += token
            # Normalize markdown/emoji early so sentence splitting is clean.
            clean = ai.format_for_tts(pending)
            sentences = _split_sentences(clean)
            if len(sentences) > 1:
                for sent in sentences[:-1]:
                    for chunk in tts.synthesize_stream(sent, voice, **tts_kwargs):
                        yield chunk
                pending = sentences[-1]
        if pending.strip():
            for chunk in tts.synthesize_stream(pending, voice, **tts_kwargs):
                yield chunk
    except Exception as e:
        logger.exception("stream-tts failed")


# ─── health ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "ok": True,
        "stt": stt.status(),
        "tts": tts.status(),
        "ai": {"url": config.AI_CHAT_URL, "timeout_seconds": config.AI_TIMEOUT_SECONDS},
    }


# ─── Voice flow (STT → AI → TTS) ────────────────────────────────────────────
@app.post("/v1/voice/chat")
async def voice_chat(
    file: UploadFile = File(...),
    voice: str | None = Form(None),
    session_id: str | None = Form(None),
):
    """Run the full voice pipeline on an uploaded utterance.

    Uses app.pipeline (VAD-gated ASR → streaming LLM → sentence TTS), adopting
    the xiaozhi-esp32-server flow. Returns the same JSON shape as before so the
    existing UI keeps working.
    """
    started = time.time()
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File âm thanh trống.")

    try:
        audio = _load_audio_16k_mono(data)
    except Exception as e:
        logger.exception("decode failed")
        raise HTTPException(status_code=400, detail=f"Không đọc được audio: {e}")

    try:
        from . import pipeline

        result = pipeline.chat_once(audio, session_id=session_id, voice=voice)
    except Exception as e:
        logger.exception("pipeline failed")
        raise HTTPException(status_code=500, detail=f"Pipeline lỗi: {e}")

    transcript = result.get("transcript", "")
    answer = result.get("answer", "")
    wav = result.get("wav", b"")

    if not transcript:
        return {
            "transcript": "",
            "answer": answer,
            "audio": None,
            "session_id": session_id,
            "note": "Không nghe rõ nội dung.",
            "elapsed_ms": int((time.time() - started) * 1000),
        }

    audio_b64 = base64.b64encode(wav).decode("ascii") if wav else None
    return {
        "transcript": transcript,
        "answer": answer,
        "audio": audio_b64,
        "session_id": session_id,
        "intents": result.get("intents", []),
        "aborted": result.get("aborted", False),
        "elapsed_ms": int((time.time() - started) * 1000),
    }


# ─── Text chat (boss test) ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    stream: bool = True
    voice: str | None = None


@app.post("/v1/chat")
async def chat(req: ChatRequest):
    """Text-only AI chat for quick testing (no STT/TTS).

    stream=true -> Server-Sent Events (token-by-token). stream=false -> JSON.
    Uses the configured OpenAI-compatible LLM (e.g. Ornith 1.5 35B).
    """
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message trống")

    if not req.stream:
        try:
            result = ai.ask(req.message, req.session_id)
        except ai.AIError as e:
            raise HTTPException(status_code=502, detail=str(e))
        return {
            "answer": result["answer"],
            "session_id": result.get("session_id"),
            "model": config.OPENAI_MODEL,
        }

    def event_gen():
        try:
            for token in ai.ask_stream(req.message, req.session_id):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except ai.AIError as e:
            yield f"event: error\ndata: {e}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ─── Unified streaming (1 LLM call → text SSE + audio PCM) ─────────────────
@app.post("/v1/chat/stream")
async def chat_stream(req: ChatRequest):
    """Single LLM call that streams BOTH text and audio in one response.

    Returns ``text/event-stream`` with two event types:
      • ``event: text``   data: <token>        (incremental LLM text)
      • ``event: audio``  data: <base64 PCM16>  (sentence TTS chunk @48kHz)
      • ``event: done``   data: <full answer>

    Because the LLM runs ONCE, text and audio are perfectly in sync (no
    duplication) and the first audio arrives as soon as the first sentence is
    synthesized — no double latency from two separate LLM calls.
    """
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message trống")

    voice = req.voice if hasattr(req, "voice") else None
    tts_kwargs = _tts_kwargs({"voice": voice} if voice else {})

    def gen():
        pending = ""
        full = ""
        try:
            for token in ai.ask_stream(req.message, req.session_id):
                full += token
                yield f"event: text\ndata: {token}\n\n"
                pending += token
                sentences = _split_sentences(pending)
                if len(sentences) > 1:
                    for sent in sentences[:-1]:
                        for chunk in tts.synthesize_stream(sent, voice, **tts_kwargs):
                            yield f"event: audio\ndata: {base64.b64encode(chunk).decode()}\n\n"
                    pending = sentences[-1]
            if pending.strip():
                for chunk in tts.synthesize_stream(pending, voice, **tts_kwargs):
                    yield f"event: audio\ndata: {base64.b64encode(chunk).decode()}\n\n"
            yield f"event: done\ndata: {full}\n\n"
        except Exception as e:
            logger.exception("stream failed")
            yield f"event: error\ndata: {e}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ─── Streaming audio (LLM token → sentence TTS → WAV) ──────────────────────
@app.post("/v1/chat/stream-audio")
async def chat_stream_audio(req: ChatRequest):
    """Stream a 48 kHz WAV of the LLM answer, sentence by sentence.

    The LLM streams tokens; as soon as a sentence completes it is synthesized
    and pushed to the client, so the first audio plays within ~1 sentence of
    latency (no need to wait for the whole answer). Mirrors the voice pipeline
    but for the text-input path. Returns audio/wav (streaming).
    """
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message trống")

    voice = req.voice if hasattr(req, "voice") else None

    def gen():
        yield _wav_header()
        yield from _stream_tts_pcm(req.message, req.session_id, voice)

    return StreamingResponse(gen(), media_type="audio/wav")


# ─── Streaming audio as raw PCM (for Web Audio incremental playback) ──────
@app.post("/v1/chat/stream-pcm")
async def chat_stream_pcm(req: ChatRequest):
    """Stream raw 48 kHz PCM16 of the LLM answer (no WAV header).

    Same as /v1/chat/stream-audio but without the WAV wrapper, so the browser
    can feed chunks straight into a Web Audio AudioBufferSourceNode and start
    playing the first sentence immediately (true low-latency streaming).
    """
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message trống")
    voice = req.voice if hasattr(req, "voice") else None

    def gen():
        yield from _stream_tts_pcm(req.message, req.session_id, voice)

    return StreamingResponse(gen(), media_type="audio/pcm")


# ─── STT only (decode uploaded audio → transcript) ────────────────────────
@app.post("/v1/stt")
async def stt_only(file: UploadFile = File(...)):
    """Transcribe an uploaded utterance and return {transcript}."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File âm thanh trống.")
    try:
        audio = _load_audio_16k_mono(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không đọc được audio: {e}")
    try:
        transcript = stt.transcribe(audio, 16000)
    except Exception as e:
        logger.exception("stt failed")
        raise HTTPException(status_code=500, detail=f"STT lỗi: {e}")
    return {"transcript": transcript}


# ─── Voice flow, STREAMING (STT → LLM(stream) → TTS(stream) → SSE) ─────────
@app.post("/v1/voice/chat/stream")
async def voice_chat_stream(
    file: UploadFile = File(...),
    voice: str | None = Form(None),
    session_id: str | None = Form(None),
):
    """Full voice pipeline streamed as SSE (transcript + audio PCM).

    Flow:  mic audio → STT (batch) → LLM (token stream, ONCE) → TTS (sentence
    stream) → audio events. Emits:
      • ``event: transcript`` data: <recognized text>
      • ``event: text``        data: <token>
      • ``event: audio``       data: <base64 PCM16 @48kHz>
      • ``event: done``        data: <full answer>

    Single LLM call → text and audio stay in sync (no duplication), first
    audio plays as soon as the first sentence is synthesized.
    """
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File âm thanh trống.")
    try:
        audio = _load_audio_16k_mono(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không đọc được audio: {e}")

    try:
        transcript = stt.transcribe(audio, 16000)
    except Exception as e:
        logger.exception("stt failed")
        raise HTTPException(status_code=500, detail=f"STT lỗi: {e}")

    if not transcript:
        return StreamingResponse(
            iter([f"event: transcript\ndata: \n\n".encode()]),
            media_type="text/event-stream",
        )

    tts_kwargs = _tts_kwargs({"voice": voice} if voice else {})

    def gen():
        yield f"event: transcript\ndata: {transcript}\n\n"
        pending = ""
        full = ""
        try:
            for token in ai.ask_stream(transcript, session_id):
                full += token
                yield f"event: text\ndata: {token}\n\n"
                pending += token
                sentences = _split_sentences(pending)
                if len(sentences) > 1:
                    for sent in sentences[:-1]:
                        for chunk in tts.synthesize_stream(sent, voice, **tts_kwargs):
                            yield f"event: audio\ndata: {base64.b64encode(chunk).decode()}\n\n"
                    pending = sentences[-1]
            if pending.strip():
                for chunk in tts.synthesize_stream(pending, voice, **tts_kwargs):
                    yield f"event: audio\ndata: {base64.b64encode(chunk).decode()}\n\n"
            yield f"event: done\ndata: {full}\n\n"
        except Exception as e:
            logger.exception("voice stream failed")
            yield f"event: error\ndata: {e}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ─── static UI ──────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def index():
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    return JSONResponse(
        status_code=404,
        content={"detail": "Chưa có static/index.html — hãy tạo file UI."},
    )


# ─── startup warmup ─────────────────────────────────────────────────────────
def _warmup_engines():
    """Pre-load STT + TTS models in a background thread so the first user
    utterance doesn't pay the cold-start model-load cost (~1-2s for VieNeu
    ONNX, ~0.5s for sherpa-onnx). Without this the TTS model loads *after*
    the LLM finishes, which is the visible delay sếp noticed."""
    import threading

    def _run():
        try:
            stt.get_recognizer()
            logger.info("Warmup: STT recognizer loaded.")
        except Exception as e:
            logger.warning("Warmup STT skipped: %s", e)
        try:
            tts.get_engine()
            logger.info("Warmup: TTS engine loaded.")
        except Exception as e:
            logger.warning("Warmup TTS skipped: %s", e)

    threading.Thread(target=_run, name="engine-warmup", daemon=True).start()


@app.on_event("startup")
def _startup():
    logger.info("Voice service starting on %s:%s", config.HOST, config.PORT)
    logger.info("STT model dir: %s", config.STT_MODEL_DIR_ABS)
    logger.info("AI endpoint:   %s", config.AI_CHAT_URL)
    _warmup_engines()
