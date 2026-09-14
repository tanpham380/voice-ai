"""Voice pipeline — VAD → ASR → LLM(stream) → TTS(sentence) + xiaozhi extras.

Adopts xiaozhi-esp32-server's flow and adds three production patterns from it:

  1. VAD-gated pipeline (Silero)  — only run ASR when voice is detected.
  2. Streaming LLM + sentence TTS — first audio plays after the first sentence.
  3. Barge-in / interrupt         — if the user cuts in while the bot is
     speaking, abort the current TTS/LLM stream (xiaozhi: abortHandle.py +
     client_is_speaking).
  4. AudioRateController          — pace the outgoing audio to real time so the
     player never under/over-runs (xiaozhi: core/handle/sendAudioHandle.py).
  5. Intent / function-call routing — the LLM may emit a lightweight intent tag
     (e.g. ``[intent:abort]`` / ``[intent:escalate]``) that the pipeline acts on
     without speaking it aloud (xiaozhi: core/providers/intent/function_call).

Transport (HTTP / WebSocket / SIP later) stays separate from this processing
core, so the same feed/run loop can drive a future SIP call.
"""
from __future__ import annotations

import io
import logging
import queue
import re
import threading
import time
import wave

import numpy as np

from . import ai, config, stt, tts
from .tts import _tts_kwargs

logger = logging.getLogger("voice.pipeline")

# 16 kHz, 16-bit, 512-sample frames (32 ms) — matches Silero VAD.
_FRAME = 512
_VAD_PAD = config.VAD_PAD_FRAMES
_VAD_SILENCE = config.VAD_SILENCE_FRAMES

# Pipeline states (mirrors xiaozhi's connection state machine).
STATE_IDLE = "idle"
STATE_LISTENING = "listening"
STATE_THINKING = "thinking"
STATE_SPEAKING = "speaking"

# Intent tag emitted by the LLM, e.g. "Xin chào [intent:escalate]".
_INTENT_RE = re.compile(r"\[intent:([a-z_]+)\]", re.IGNORECASE)

# Word-level TTS chunking: break on spaces once this many words have
# accumulated, so the first audio plays after a short phrase instead of a
# whole sentence. Sentence punctuation still forces an immediate break.
# Tunable: lower = earlier first audio but more robotic / more calls;
# higher = smoother prosody but higher latency.
# 4 words is tuned for low voice latency: the bot starts speaking sooner
# (smaller TTFA) while still keeping reasonable prosody. Learned from
# xiaozhi-esp32-server, which splits by sentence punctuation (。！？) rather
# than per-word to keep prosody continuous.
_MIN_TTS_WORDS = 4

# Punctuation that forces an immediate chunk break (sentence-level).
_SENT_PUNCT = set(".!?;:\n")


def _split_sentences(text: str) -> list[str]:
    """Split text into TTS chunks at word boundaries for incremental synthesis.

    Unlike sentence splitting (which waits for ``.!?``), this breaks on spaces
    once a minimum number of words has accumulated, so TTS starts early.
    Sentence punctuation still forces an immediate break. A minimum buffer is
    kept so each chunk has enough context for natural prosody — pure
    single-word chunks sound robotic and cost one extra inference call each.
    """
    buf = ""
    out: list[str] = []
    for ch in text:
        buf += ch
        if ch in _SENT_PUNCT:
            s = buf.strip()
            if s:
                out.append(s)
            buf = ""
        elif ch.isspace():
            stripped = buf.strip()
            if stripped and len(stripped.split()) >= _MIN_TTS_WORDS:
                out.append(stripped)
                buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out


def _strip_intents(text: str) -> tuple[str, list[str]]:
    """Remove ``[intent:xxx]`` tags from spoken text; return (clean, intents)."""
    intents = [m.group(1).lower() for m in _INTENT_RE.finditer(text)]
    clean = _INTENT_RE.sub("", text).strip()
    return clean, intents


class AudioRateController:
    """Pace outgoing PCM audio to real time (xiaozhi: AudioRateController).

    Instead of dumping every synthesized chunk at once (which can overflow the
    player buffer or sound rushed), we sleep proportionally to the chunk
    duration so audio leaves the pipeline at ~1x playback speed. A small
    look-ahead slack keeps it smooth without adding perceptible latency.
    """

    def __init__(self, sample_rate: int = 48000, slack: float = 0.02) -> None:
        self._sr = sample_rate
        self._slack = slack  # seconds of headroom before blocking
        self._lock = threading.Lock()
        self._sent_samples = 0
        self._start = time.monotonic()

    def reset(self) -> None:
        with self._lock:
            self._sent_samples = 0
            self._start = time.monotonic()

    def pace(self, pcm16: bytes) -> None:
        """Block just long enough that ``pcm16`` is released at real time."""
        n = len(pcm16) // 2  # int16 samples
        with self._lock:
            self._sent_samples += n
            elapsed = time.monotonic() - self._start
            due = self._sent_samples / self._sr
        wait = due - elapsed - self._slack
        if wait > 0:
            time.sleep(wait)


class VoicePipeline:
    def __init__(self, session_id: str | None = None, voice: str | None = None) -> None:
        self.session_id = session_id
        self.voice = voice or config.TTS_DEFAULT_VOICE
        from .vad import get_vad

        self._vad = get_vad()
        self._buf: list[np.ndarray] = []  # voiced frames (with pad)
        self._silence = 0
        self._voiced = False
        self._lock = threading.Lock()
        self._result: queue.Queue = queue.Queue(maxsize=1)
        self._abort = threading.Event()  # set when user barges in
        self._state = STATE_IDLE
        self._rate = AudioRateController(tts.SAMPLE_RATE)
        self._speaking = False

    # ── state helpers ──
    def _set_state(self, s: str) -> None:
        self._state = s

    def state(self) -> str:
        return self._state

    # ── ingest ──
    def feed(self, frame16k_mono_f32: np.ndarray) -> None:
        """Push one 512-sample frame.

        While the bot is SPEAKING, any detected voice triggers barge-in: we
        abort the current reply and switch back to LISTENING.
        """
        with self._lock:
            is_voice = self._vad.is_voice(frame16k_mono_f32)

            # Barge-in: user cuts in while we are speaking.
            if is_voice and self._state == STATE_SPEAKING:
                logger.info("barge-in: user interrupted, aborting current reply")
                self._abort.set()
                self._speaking = False
                self._set_state(STATE_LISTENING)
                # Reset buffers to capture the new utterance from scratch.
                self._buf = []
                self._silence = 0
                self._voiced = True
                self._buf.append(frame16k_mono_f32)
                return

            if is_voice:
                if not self._voiced:
                    self._buf = self._buf[-_VAD_PAD:]  # keep a little pre-roll
                self._buf.append(frame16k_mono_f32)
                self._voiced = True
                self._silence = 0
                if self._state in (STATE_IDLE, STATE_LISTENING):
                    self._set_state(STATE_LISTENING)
            else:
                if self._voiced:
                    self._buf.append(frame16k_mono_f32)
                    self._silence += 1
                    if self._silence >= _VAD_SILENCE:
                        self._flush()
                else:
                    self._buf = self._buf[-_VAD_PAD:]

    def _flush(self) -> None:
        if not self._buf:
            self._voiced = False
            self._silence = 0
            return
        pcm = (np.concatenate(self._buf) * 32767).clip(-32768, 32768).astype(np.int16)
        self._buf = []
        self._voiced = False
        self._silence = 0
        self._vad.reset()
        self._set_state(STATE_THINKING)
        # Run ASR + LLM in a worker so feed() stays non-blocking.
        threading.Thread(target=self._process, args=(pcm,), daemon=True).start()

    def _process(self, pcm16: np.ndarray) -> None:
        """VAD-gated ASR, then STREAM the LLM and TTS sentence-by-sentence.

        Honors barge-in (self._abort) and routes LLM intents.
        """
        try:
            samples = pcm16.astype(np.float32) / 32768.0
            transcript = stt.transcribe(samples, 16000)
            if not transcript:
                self._result.put({"transcript": "", "answer": "", "wav": b"", "intents": []})
                self._set_state(STATE_IDLE)
                return

            self._abort.clear()
            self._rate.reset()
            answer_parts: list[str] = []
            pending = ""
            wav_chunks: list[bytes] = []
            intents_seen: list[str] = []

            logger.info("process: entering LLM stream")
            for token in ai.ask_stream(transcript, self.session_id):
                if self._abort.is_set():
                    logger.info("barge-in abort during LLM stream")
                    break
                answer_parts.append(token)
                pending += token
                clean = ai.format_for_tts(pending)
                sentences = _split_sentences(clean)
                if len(sentences) > 1:
                    for sent in sentences[:-1]:
                        clean, intents = _strip_intents(sent)
                        intents_seen.extend(intents)
                        if clean:
                            for chunk in tts.synthesize_stream(clean, self.voice, **_tts_kwargs()):
                                if self._abort.is_set():
                                    break
                                wav_chunks.append(chunk)
                                self._rate.pace(chunk)
                    pending = sentences[-1]

            # Flush the final remainder (also strip intents from it).
            if pending.strip() and not self._abort.is_set():
                clean, intents = _strip_intents(pending)
                intents_seen.extend(intents)
                if clean:
                    for chunk in tts.synthesize_stream(clean, self.voice, **_tts_kwargs()):
                        if self._abort.is_set():
                            break
                        wav_chunks.append(chunk)
                        self._rate.pace(chunk)

            answer = "".join(answer_parts).strip()
            # If an escalate intent was seen, append a hand-off note (not spoken
            # as part of the streamed answer, but returned to the caller).
            if "escalate" in intents_seen:
                answer += "\n[escalate: chuyển giao tư vấn viên]"
            if "mute" in intents_seen:
                logger.info("intent=mute: suppressing spoken output")
                wav_chunks = []

            wav = b"".join(wav_chunks)
            self._result.put(
                {
                    "transcript": transcript,
                    "answer": answer,
                    "wav": wav,
                    "intents": intents_seen,
                    "aborted": self._abort.is_set(),
                }
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("pipeline error")
            self._result.put(
                {"transcript": "", "answer": f"[lỗi] {e}", "wav": b"", "intents": []}
            )
        finally:
            self._set_state(STATE_IDLE)

    # ── synchronous one-shot (HTTP /v1/voice/chat) ──
    def run_once(self, timeout: float | None = None) -> dict:
        """Block until a voiced segment is recognized and answered."""
        return self._result.get(timeout=timeout)

    # ── streaming WAV (low latency, rate-controlled) ──
    def stream_wav(self, transcript: str):
        """Yield a 48 kHz WAV of the LLM answer, sentence by sentence.

        Uses tts.synthesize_stream + AudioRateController so the first audio
        chunk arrives quickly and the rest plays at real time.
        """
        clean, intents = _strip_intents(transcript)
        if "mute" in intents:
            return
        sentences = _split_sentences(clean)
        if not sentences:
            return
        self._rate.reset()
        chunks: list[bytes] = []
        for sent in sentences:
            for chunk in tts.synthesize_stream(sent, self.voice, **_tts_kwargs()):
                if self._abort.is_set():
                    return
                chunks.append(chunk)
                self._rate.pace(chunk)
        if not chunks:
            return
        pcm = b"".join(chunks)
        yield _wrap_wav(pcm)


def _wrap_wav(pcm16: bytes, sample_rate: int = 48000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16)
    return buf.getvalue()


def _run_pipeline_once(audio_16k_mono_f32: np.ndarray,
                       session_id: str | None = None,
                       voice: str | None = None) -> dict:
    """Run the full voice pipeline on a complete utterance buffer.

    VAD-free path: the caller already sent the whole recording, so we do not
    need to gate ASR on voice activity. We transcribe directly, stream the LLM
    once, and synthesize sentence-by-sentence. Returns {transcript, answer,
    wav, intents}.

    Unlike the VAD-gated ``VoicePipeline.run_once`` (which blocks on a worker
    thread with an artificial timeout), this runs synchronously and waits for
    the LLM to finish, so it never times out mid-stream.
    """
    voice = voice or config.TTS_DEFAULT_VOICE
    try:
        samples = np.asarray(audio_16k_mono_f32, dtype=np.float32).reshape(-1)
        transcript = stt.transcribe(samples, 16000)
        if not transcript:
            return {"transcript": "", "answer": "", "wav": b"", "intents": []}

        answer_parts: list[str] = []
        pending = ""
        wav_chunks: list[bytes] = []
        intents_seen: list[str] = []

        for token in ai.ask_stream(transcript, session_id):
            answer_parts.append(token)
            pending += token
            clean = ai.format_for_tts(pending)
            sentences = _split_sentences(clean)
            if len(sentences) > 1:
                for sent in sentences[:-1]:
                    clean_s, intents = _strip_intents(sent)
                    intents_seen.extend(intents)
                    if clean_s:
                        for chunk in tts.synthesize_stream(clean_s, voice, **_tts_kwargs()):
                            wav_chunks.append(chunk)
                pending = sentences[-1]

        # Flush the final remainder.
        if pending.strip():
            clean, intents = _strip_intents(pending)
            intents_seen.extend(intents)
            if clean:
                for chunk in tts.synthesize_stream(clean, voice, **_tts_kwargs()):
                    wav_chunks.append(chunk)

        answer = "".join(answer_parts).strip()
        wav = b"".join(wav_chunks)
        return {
            "transcript": transcript,
            "answer": answer,
            "wav": wav,
            "intents": intents_seen,
        }
    except Exception as e:  # noqa: BLE001
        logger.exception("pipeline error")
        return {"transcript": "", "answer": f"[lỗi] {e}", "wav": b"", "intents": []}


def chat_once(audio_16k_mono_f32: np.ndarray, session_id: str | None = None,
              voice: str | None = None) -> dict:
    """Run the full voice pipeline on a complete utterance buffer.

    Used by the HTTP endpoint when the browser sends a whole recording.
    """
    return _run_pipeline_once(audio_16k_mono_f32, session_id=session_id, voice=voice)
