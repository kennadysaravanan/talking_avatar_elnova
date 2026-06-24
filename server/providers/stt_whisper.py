"""
STT provider. Primary = OpenAI Whisper API (cheap, no GPU contention — our 5 GPUs are busy
with the avatar). Local openai-whisper is a documented fallback (would need a 6th GPU or CPU).

Mic/voice is secondary in this prototype (text is primary). Interruption is text-triggered.
"""

from __future__ import annotations

import io
import os

import numpy as np
import soundfile as sf
from openai import AsyncOpenAI


class OpenAIWhisperSTT:
    def __init__(self, model: str = None):
        self._client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model or os.environ.get("OPENAI_STT_MODEL", "whisper-1")

    async def transcribe(self, pcm16: np.ndarray, sample_rate: int) -> str:
        # Wrap PCM as an in-memory wav for the API.
        buf = io.BytesIO()
        sf.write(buf, pcm16, sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)
        buf.name = "audio.wav"
        resp = await self._client.audio.transcriptions.create(model=self.model, file=buf)
        return resp.text.strip()
