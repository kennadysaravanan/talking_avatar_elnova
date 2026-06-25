"""
turn_manager.py  —  conversation turns + interruption (TEXT MODE).

Each user input starts a TURN: LLM stream -> sentence chunking -> TTS stream -> PCM pushed into
the AudioQueue (worker rank 4). A turn runs inside an asyncio Task guarded by a cancel token.

INTERRUPTION (constraint: text-triggered; VAD is future work):
  When a NEW user message arrives while a turn is still speaking, we:
    1) cancel the in-flight turn task   -> closes LLM + TTS streams (their finally blocks abort)
    2) send FLUSH to the worker         -> drops queued-but-unspoken audio
    3) start the new turn
  NOTE (constraint #5): interruption is NOT instant. Audio already sent to rank 4 and in the
  dist ring still renders for a block or two -> expect a ~0.5-1s tail. We do NOT reset the KV
  cache (that would corrupt it); the model naturally returns toward idle as audio drains.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

import numpy as np

from .ipc import OrchestratorIPC
from .providers.base import LLMProvider, TTSProvider

logger = logging.getLogger("turn_manager")

INTERRUPT_MODE = "text"  # <- flips to "vad" when voice-activity interruption is added later

# Start TTS as soon as a sentence/clause boundary is seen, to cut first-frame latency.
_SENT_BOUNDARY = re.compile(r"[.!?。！？]\s|[.!?。！？]$|[,;，；]\s")


class TurnManager:
    def __init__(self, ipc: OrchestratorIPC, llm: LLMProvider, tts: TTSProvider, voice_sink=None):
        self._ipc = ipc
        self._llm = llm
        self._tts = tts
        # voice_sink(pcm16) publishes the SAME TTS audio to LiveKit so the user hears it
        # (the worker only uses it for lip-sync). Set to LiveKitPublisher.play_voice.
        self._voice_sink = voice_sink
        self._task: Optional[asyncio.Task] = None
        self._history: list[dict] = []
        self._lock = asyncio.Lock()

    @property
    def speaking(self) -> bool:
        return self._task is not None and not self._task.done()

    async def on_user_text(self, text: str) -> None:
        """Entry point for a new user message. Interrupts any in-flight turn first."""
        async with self._lock:
            if self.speaking:
                logger.info("interrupt: new text arrived mid-speech -> cancelling turn (mode=%s)",
                            INTERRUPT_MODE)
                await self._cancel_current()
            self._task = asyncio.create_task(self._run_turn(text), name="turn")

    async def _cancel_current(self) -> None:
        assert self._task is not None
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001 — don't let a dying turn break the new one
            logger.exception("error while cancelling turn")
        # Drop in-flight unspoken audio. Tail still renders (~0.5-1s) by design.
        await self._ipc.flush()

    async def _run_turn(self, user_text: str) -> None:
        full_reply = ""
        sentence = ""
        started_talking = False
        try:
            async for delta in self._llm.stream_reply(user_text, self._history):
                full_reply += delta
                sentence += delta
                if _SENT_BOUNDARY.search(sentence):
                    if not started_talking:
                        await self._ipc.send_control("TALK_START")
                        started_talking = True
                    await self._speak(sentence)
                    sentence = ""
            # flush trailing partial sentence
            if sentence.strip():
                if not started_talking:
                    await self._ipc.send_control("TALK_START")
                    started_talking = True
                await self._speak(sentence)
        except asyncio.CancelledError:
            logger.info("turn cancelled (interruption)")
            raise
        finally:
            # Record history only if we produced something; tell worker we expect idle next.
            if full_reply.strip():
                self._history.append({"role": "user", "content": user_text})
                self._history.append({"role": "assistant", "content": full_reply})
                # keep history bounded
                self._history[:] = self._history[-20:]
            if started_talking:
                await self._ipc.send_control("IDLE")

    async def _speak(self, text: str) -> None:
        """Stream TTS for one sentence: feed the worker (lip-sync) AND publish to LiveKit (user hears)."""
        async for pcm16 in self._tts.stream_pcm(text):
            pcm16 = np.ascontiguousarray(pcm16, dtype=np.int16)
            await self._ipc.send_pcm16(pcm16)                 # worker: drives mouth motion
            if self._voice_sink is not None:
                await self._voice_sink(pcm16)                 # LiveKit: user hears the voice
