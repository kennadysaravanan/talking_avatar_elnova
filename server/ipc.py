"""
server/ipc.py  —  Orchestrator side of the ZeroMQ bridge to the avatar worker (rank 4).

Mirror of avatar/ipc_worker.py. The ORCHESTRATOR BINDS both sockets (stable endpoints); the
worker connects. Async (zmq.asyncio) so it composes with FastAPI/asyncio.

Channels:
  AUDIO : orchestrator PUSH (bind)  -> worker PULL    (TTS PCM + control)
  FRAME : worker PUSH               -> orchestrator PULL (bind)   (decoded RGB frames)

Wire format MUST match avatar/ipc_worker.py:
  AUDIO: [b"AUDIO", int16 LE mono 16k bytes] | [b"FLUSH"] | [b"TALK_START"] | [b"IDLE"]
  FRAME: [b"FRAME", json header, uint8 RGB bytes]
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import zmq
import zmq.asyncio

AUDIO_ENDPOINT = os.environ.get("AVATAR_AUDIO_ENDPOINT", "tcp://127.0.0.1:5557")
FRAME_ENDPOINT = os.environ.get("AVATAR_FRAME_ENDPOINT", "tcp://127.0.0.1:5558")
SAMPLE_RATE = 16000


@dataclass
class Frame:
    rgb: np.ndarray   # uint8 [H,W,3]
    seq: int
    kind: str         # "idle" | "talk"
    ts: float


class OrchestratorIPC:
    def __init__(self, audio_endpoint: str = AUDIO_ENDPOINT, frame_endpoint: str = FRAME_ENDPOINT):
        self._ctx = zmq.asyncio.Context.instance()

        self._audio = self._ctx.socket(zmq.PUSH)
        self._audio.setsockopt(zmq.SNDHWM, 1000)
        self._audio.bind(audio_endpoint)

        self._frame = self._ctx.socket(zmq.PULL)
        self._frame.setsockopt(zmq.RCVHWM, 200)
        self._frame.bind(frame_endpoint)

    # ---- audio / control out (orchestrator -> worker) ----
    async def send_pcm16(self, pcm16: np.ndarray) -> None:
        """pcm16: int16 mono 16kHz numpy array."""
        assert pcm16.dtype == np.int16
        await self._audio.send_multipart([b"AUDIO", pcm16.tobytes()])

    async def send_control(self, kind: str) -> None:
        """kind in {'FLUSH','TALK_START','IDLE'}."""
        await self._audio.send_multipart([kind.encode()])

    async def flush(self) -> None:
        await self.send_control("FLUSH")

    async def send_ref(self, path: str) -> None:
        """Tell the worker to hot-swap its reference image to `path` (a same-pod filesystem path)."""
        await self._audio.send_multipart([b"REF", path.encode()])

    # ---- frames in (worker -> orchestrator) ----
    async def recv_frame(self) -> Optional[Frame]:
        parts = await self._frame.recv_multipart()
        if parts[0] != b"FRAME" or len(parts) < 3:
            return None
        hdr = json.loads(parts[1])
        rgb = np.frombuffer(parts[2], dtype=np.uint8).reshape(hdr["h"], hdr["w"], 3)
        return Frame(rgb=rgb, seq=hdr["seq"], kind=hdr["kind"], ts=hdr["ts"])

    def close(self) -> None:
        self._audio.close(0)
        self._frame.close(0)
