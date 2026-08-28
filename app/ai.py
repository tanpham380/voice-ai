"""AI module — OpenAI-compatible LLM client (local-first, streaming).

Adopts the provider pattern from xiaozhi-esp32-server (core/providers/llm/openai):
  - one reusable OpenAI client
  - streaming responses (yield tokens) for low-latency TTS
  - per-session dialogue memory (core/utils/dialogue.py idea)
  - system prompt supports the {{current_time}} placeholder

The voice flow is:  mic → STT (transcript) → AI (this module) → TTS (speech).
"""
from __future__ import annotations

import logging
from datetime import datetime

from openai import OpenAI

from . import config

logger = logging.getLogger("voice.ai")


class AIError(RuntimeError):
    """Raised when the LLM endpoint is unreachable or returns no answer."""


_client = OpenAI(
    api_key=config.OPENAI_API_KEY or "EMPTY",
    base_url=config.OPENAI_BASE_URL or None,
)


# ─── per-session conversation memory ────────────────────────────────────────
# Mirrors xiaozhi's Dialogue: keep recent user/assistant turns per session_id.
_SESSIONS: dict[str, list[dict]] = {}
_MAX_TURNS = 10


def _history_for(session_id: str | None) -> list[dict]:
    if not session_id:
        return []
    return _SESSIONS.setdefault(session_id, [])


def _trim(history: list[dict]) -> None:
    if len(history) > _MAX_TURNS * 2:
        del history[: len(history) - _MAX_TURNS * 2]


def _system_prompt() -> str:
    return config.OPENAI_SYSTEM_PROMPT.replace(
        "{{current_time}}", datetime.now().strftime("%H:%M")
    )


def _messages(session_id: str | None, user_text: str) -> list[dict]:
    return [
        {"role": "system", "content": _system_prompt()},
        *_history_for(session_id),
        {"role": "user", "content": user_text},
    ]


import re as _re

# Characters / patterns that confuse a Vietnamese TTS engine. We strip them
# before synthesis so the spoken output is clean and natural.
_TTS_STRIP_RE = _re.compile(
    r"(\*\*\*|\*\*|\*|__|_|`{1,3}|#{1,6}|>|\[|\]|\||=+|-{3,}|•|·|→|⇒|✓|✔|✗|✘|⚠|🔊|🔉|🔇|"
    r"🎙|🎤|💡|⭐|🌟|💬|📌|📎|🔗|👉|👈|👆|👇|🔔|❓|❗|⚡|🚀|💯|🙏|😊|😉|😍|🥰|😎|🤔|"
    r"©|®|™|§|¶|†|‡|°|′|″|‰|×|÷|±|≈|≠|≤|≥|∞|∑|∏|√|∫|∂|∆|Ω|α|β|γ|δ|μ|π|σ|φ|ω)"
)

# Collapse any run of whitespace (incl. newlines) into a single space.
_WS_RE = _re.compile(r"\s+")

# Map common unicode punctuation to ASCII so the TTS doesn't misread them.
_PUNCT_MAP = {
    "…": "...", "–": "-", "—": "-", "‘": "'", "’": "'",
    "“": '"', "”": '"', "„": '"', "•": "", "·": "", "→": " đến ",
    "⇒": " đến ", "×": " nhân ", "÷": " chia ", "±": " cộng trừ ",
}


def format_for_tts(text: str) -> str:
    """Normalize an LLM answer into clean speech text for the TTS engine.

    - Removes markdown / emoji / bullet / arrow glyphs (TTS would read them
      aloud or mangle pronunciation).
    - Replaces unicode math/arrow symbols with Vietnamese words where useful.
    - Collapses whitespace and fixes spacing around Vietnamese punctuation
      (no space before , . ; : ? ! and exactly one after).
    - Strips leading/trailing whitespace.

    The result is what actually gets synthesized, while the raw LLM text can
    still be shown in the UI panel.
    """
    if not text:
        return ""
    s = text
    # Unicode punctuation / symbol substitution.
    for k, v in _PUNCT_MAP.items():
        s = s.replace(k, v)
    # Strip markdown / emoji / decorative glyphs.
    s = _TTS_STRIP_RE.sub("", s)
    # Remove standalone URLs (read as gibberish otherwise).
    s = _re.sub(r"https?://\S+", "", s)
    s = _re.sub(r"www\.\S+", "", s)
    # Collapse whitespace.
    s = _WS_RE.sub(" ", s).strip()
    # Fix spacing around Vietnamese sentence punctuation: no space before,
    # one space after. Handles both ASCII and common full-width forms.
    s = _re.sub(r"\s*([,.;:!?])\s*", r"\1 ", s)
    s = _re.sub(r"\s+", " ", s).strip()
    # Remove accidental double punctuation left by stripping.
    s = _re.sub(r"([,.;:!?])\1+", r"\1", s)
    return s


def _extra_params() -> dict:
    """Standard OpenAI sampling params (top_p, presence_penalty, stop).

    `repetition_penalty` is a vLLM/Omnith extension, NOT part of the OpenAI
    API spec, so the openai SDK rejects it as a kwarg. We route it through
    `extra_body` (passed verbatim to the server) instead.
    """
    params: dict = {}
    if config.OPENAI_TOP_P:
        params["top_p"] = config.OPENAI_TOP_P
    if config.OPENAI_PRESENCE_PENALTY:
        params["presence_penalty"] = config.OPENAI_PRESENCE_PENALTY
    if config.OPENAI_STOP:
        params["stop"] = config.OPENAI_STOP
    return params


def _extra_body() -> dict:
    """Non-standard params forwarded verbatim to the LLM server (vLLM/Omnith)."""
    body: dict = {}
    if config.OPENAI_REPETITION_PENALTY:
        body["repetition_penalty"] = config.OPENAI_REPETITION_PENALTY
    # enable_thinking=False makes Ornith skip the <think>…</think> block so the
    # first answer token arrives immediately (low latency for voice/TTS).
    if not config.OPENAI_ENABLE_THINKING:
        body["enable_thinking"] = False
    return body


def ask(text: str, session_id: str | None = None) -> dict:
    """Non-streaming helper (used by the HTTP /v1/voice/chat endpoint).

    Returns {answer, session_id, raw}. Appends the turn to session memory.
    """
    if not text or not text.strip():
        raise ValueError("text is empty")

    try:
        resp = _client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=_messages(session_id, text),
            temperature=config.OPENAI_TEMPERATURE,
            max_tokens=config.OPENAI_MAX_TOKENS,
            timeout=config.AI_TIMEOUT_SECONDS,
            extra_body=_extra_body(),
            **_extra_params(),
        )
        answer = (resp.choices[0].message.content or "").strip()
    except Exception as e:  # openai raises a hierarchy of errors; catch all for demo
        logger.error("LLM request failed: %s", e)
        raise AIError(f"Không gọi được LLM: {e}") from e

    if not answer:
        raise AIError("LLM không trả về nội dung.")

    history = _history_for(session_id)
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": answer})
    _trim(history)

    return {"answer": answer, "session_id": session_id, "raw": {"model": config.OPENAI_MODEL}}


def ask_stream(text: str, session_id: str | None = None):
    """Streaming generator: yields answer tokens as they arrive.

    Mirrors xiaozhi's LLMProvider.response() (stream=True). The caller is
    responsible for splitting tokens into sentences and feeding TTS. Session
    memory is updated when the stream is exhausted.
    """
    if not text or not text.strip():
        raise ValueError("text is empty")

    collected: list[str] = []
    try:
        stream = _client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=_messages(session_id, text),
            temperature=config.OPENAI_TEMPERATURE,
            max_tokens=config.OPENAI_MAX_TOKENS,
            stream=True,
            timeout=config.AI_TIMEOUT_SECONDS,
            extra_body=_extra_body(),
            **_extra_params(),
        )
        for chunk in stream:
            delta = chunk.choices[0].delta if getattr(chunk, "choices", None) else None
            token = getattr(delta, "content", "") if delta else ""
            if token:
                collected.append(token)
                yield token
    except Exception as e:
        logger.error("LLM stream failed: %s", e)
        raise AIError(f"Không gọi được LLM: {e}") from e

    answer = "".join(collected).strip()
    if answer:
        history = _history_for(session_id)
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": answer})
        _trim(history)
