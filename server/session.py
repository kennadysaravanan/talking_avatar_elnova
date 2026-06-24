"""
session.py  —  per-session state.

A session is created on photo upload. It owns: the reference image path, the LiveKit room +
publisher, the IPC bridge to the worker, and the turn manager.

WARM-UP FLOW (brief 2A):
  upload photo -> create session -> tell worker to warm up with this ref image -> worker
  generates the idle loop -> first idle frame published -> client clears the loading overlay.

IMPORTANT SINGLE-SESSION NOTE (constraint #6):
  One 8xH100 pod = ONE concurrent conversation (5 GPUs per session). This SessionManager
  enforces a single active session by design and rejects a second concurrent one. Production
  scoping must account for this cost-per-user reality.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .ipc import OrchestratorIPC
from .providers.llm_openai import OpenAILLM
from .providers.tts_openai import OpenAITTS
from .transport_livekit import LiveKitPublisher
from .turn_manager import TurnManager

logger = logging.getLogger("session")

# size "720*400" -> width 720, height 400 (W*H per the engine's --size convention).
DEFAULT_W, DEFAULT_H = (int(x) for x in os.environ.get("AVATAR_SIZE", "720*400").split("*"))


@dataclass
class Session:
    session_id: str
    ref_image_path: str
    ipc: OrchestratorIPC
    publisher: LiveKitPublisher
    turns: TurnManager
    room_name: str
    state: str = "warming"  # warming -> idle -> talking (driven by turn manager + frames)

    async def warm_up(self) -> bool:
        """Start the video track and wait for the first idle frame from the worker.

        The resident worker is already running and producing idle frames for whatever ref image
        it was launched with. For a per-session ref image, the worker must be (re)started with
        REF_IMAGE pointing at this upload (see README — the prototype warms the worker at launch;
        hot ref-image swap without restart is future work). Here we wait for first frame.
        """
        await self.publisher.start(DEFAULT_W, DEFAULT_H)
        ok = await self.publisher.wait_first_frame(timeout=180.0)
        self.state = "idle" if ok else "error"
        return ok


class SessionManager:
    def __init__(self) -> None:
        self._session: Optional[Session] = None
        self._lock = asyncio.Lock()

    @property
    def current(self) -> Optional[Session]:
        return self._session

    async def create(self, ref_image_path: str) -> Session:
        async with self._lock:
            if self._session is not None:
                # constraint #6: single concurrent session per pod.
                raise RuntimeError("A session is already active (one conversation per pod).")
            sid = uuid.uuid4().hex[:12]
            room = f"avatar-{sid}"
            ipc = OrchestratorIPC()
            publisher = LiveKitPublisher(ipc, room_name=room)
            turns = TurnManager(ipc, OpenAILLM(), OpenAITTS())
            self._session = Session(
                session_id=sid, ref_image_path=ref_image_path, ipc=ipc,
                publisher=publisher, turns=turns, room_name=room,
            )
            logger.info("created session %s (room %s)", sid, room)
            return self._session

    async def destroy(self) -> None:
        async with self._lock:
            if self._session is None:
                return
            await self._session.publisher.close()
            self._session.ipc.close()
            logger.info("destroyed session %s", self._session.session_id)
            self._session = None
