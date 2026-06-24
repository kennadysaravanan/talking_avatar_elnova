"""
Provider interfaces. GPT/OpenAI is wired now, but everything the turn manager touches goes
through these ABCs so the LLM/TTS/STT can be swapped later without changing orchestration.
"""

from __future__ import annotations

import abc
from typing import AsyncIterator

import numpy as np


class LLMProvider(abc.ABC):
    @abc.abstractmethod
    def stream_reply(self, user_text: str, history: list[dict]) -> AsyncIterator[str]:
        """Yield response text incrementally (token/word chunks). Must be cancellable: when the
        consuming task is cancelled, stop the upstream stream promptly."""
        raise NotImplementedError


class TTSProvider(abc.ABC):
    # Engine consumes 16kHz mono PCM. Implementations resample to this.
    sample_rate: int = 16000

    @abc.abstractmethod
    def stream_pcm(self, text: str) -> AsyncIterator[np.ndarray]:
        """Yield int16 mono PCM chunks at self.sample_rate for `text`. Cancellable."""
        raise NotImplementedError


class STTProvider(abc.ABC):
    @abc.abstractmethod
    async def transcribe(self, pcm16: np.ndarray, sample_rate: int) -> str:
        """Transcribe a complete utterance (int16 mono PCM) to text."""
        raise NotImplementedError
