"""
OpenAI streaming TTS provider.

OpenAI TTS emits 24kHz PCM; the engine/audio-encoder wants 16kHz. We request raw PCM and
resample 24k->16k before yielding int16 chunks. Cancellable for interruption.
"""

from __future__ import annotations

import os
from typing import AsyncIterator

import numpy as np
from openai import AsyncOpenAI

try:
    import librosa  # high-quality resample
    _HAVE_LIBROSA = True
except Exception:  # pragma: no cover
    _HAVE_LIBROSA = False

OPENAI_TTS_RATE = 24000
TARGET_RATE = 16000


def _resample_24k_to_16k(pcm_f32: np.ndarray) -> np.ndarray:
    if _HAVE_LIBROSA:
        return librosa.resample(pcm_f32, orig_sr=OPENAI_TTS_RATE, target_sr=TARGET_RATE)
    # linear-interp fallback (no librosa)
    n_out = int(round(len(pcm_f32) * TARGET_RATE / OPENAI_TTS_RATE))
    x_old = np.linspace(0, 1, num=len(pcm_f32), endpoint=False)
    x_new = np.linspace(0, 1, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, pcm_f32).astype(np.float32)


class OpenAITTS:
    sample_rate = TARGET_RATE

    def __init__(self, model: str = None, voice: str = None):
        self._client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model or os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
        self.voice = voice or os.environ.get("OPENAI_TTS_VOICE", "alloy")

    async def stream_pcm(self, text: str) -> AsyncIterator[np.ndarray]:
        # response_format="pcm" => raw 24kHz s16le mono.
        async with self._client.audio.speech.with_streaming_response.create(
            model=self.model, voice=self.voice, input=text, response_format="pcm",
        ) as resp:
            tail = b""
            async for raw in resp.iter_bytes():
                buf = tail + raw
                # keep byte alignment to int16
                n = (len(buf) // 2) * 2
                tail = buf[n:]
                if n == 0:
                    continue
                pcm16_24k = np.frombuffer(buf[:n], dtype="<i2")
                f32 = pcm16_24k.astype(np.float32) / 32768.0
                f16k = _resample_24k_to_16k(f32)
                out = np.clip(f16k * 32768.0, -32768, 32767).astype(np.int16)
                if out.size:
                    yield out
