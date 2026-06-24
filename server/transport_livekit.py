"""
transport_livekit.py  —  publish generated frames as ONE continuous LiveKit video track.

We use LiveKit Cloud (config via env). A single LocalVideoTrack carries the whole session:
idle and talking are the same track (no reconnect/swap), so transitions are seamless.

Pipeline: worker frames (via OrchestratorIPC) -> jitter ring buffer -> 25fps pacer ->
rtc.VideoFrame (RGBA) -> track source.capture_frame().

Frame cadence: the engine targets sample_fps=25. The pacer publishes at a fixed 25fps; the
ring buffer (a few frames) absorbs idle<->talk crossfade hiccups and network jitter. If the
ring underflows we re-publish the last frame (hold) rather than stalling the track.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import deque
from typing import Optional

import numpy as np
from livekit import api, rtc

from .blend import crossfade
from .ipc import Frame, OrchestratorIPC

logger = logging.getLogger("transport")

PUBLISH_FPS = int(os.environ.get("AVATAR_PUBLISH_FPS", "25"))
JITTER_FRAMES = int(os.environ.get("AVATAR_JITTER_FRAMES", "6"))  # small cushion


def _rgb_to_rgba(rgb: np.ndarray) -> bytes:
    h, w, _ = rgb.shape
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = 255
    return rgba.tobytes()


class LiveKitPublisher:
    """One per session. Connects to a LiveKit room and pumps frames from IPC to a video track."""

    def __init__(self, ipc: OrchestratorIPC, room_name: str, identity: str = "avatar"):
        self._ipc = ipc
        self._room_name = room_name
        self._identity = identity
        self._room: Optional[rtc.Room] = None
        self._source: Optional[rtc.VideoSource] = None
        self._ring: deque[Frame] = deque(maxlen=max(JITTER_FRAMES * 4, 32))
        self._last_rgb: Optional[np.ndarray] = None
        self._prev_kind = "idle"
        self._tasks: list[asyncio.Task] = []
        self._first_frame_evt = asyncio.Event()
        self._stop = asyncio.Event()
        self._width = 0
        self._height = 0

    @staticmethod
    def _token(room: str, identity: str) -> str:
        return (
            api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
            .with_identity(identity)
            .with_name(identity)
            .with_grants(api.VideoGrants(room_join=True, room=room, can_publish=True,
                                         can_subscribe=False))
            .to_jwt()
        )

    async def start(self, width: int, height: int) -> None:
        self._width, self._height = width, height
        self._room = rtc.Room()
        await self._room.connect(os.environ["LIVEKIT_URL"], self._token(self._room_name, self._identity))
        self._source = rtc.VideoSource(width, height)
        track = rtc.LocalVideoTrack.create_video_track("avatar", self._source)
        await self._room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
        )
        logger.info("published avatar track to room=%s (%dx%d)", self._room_name, width, height)
        self._tasks.append(asyncio.create_task(self._ingest_loop(), name="lk-ingest"))
        self._tasks.append(asyncio.create_task(self._publish_loop(), name="lk-publish"))

    async def wait_first_frame(self, timeout: float = 120.0) -> bool:
        try:
            await asyncio.wait_for(self._first_frame_evt.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def _ingest_loop(self) -> None:
        """Pull frames from the worker into the ring. Apply idle<->talk crossfade at boundaries."""
        while not self._stop.is_set():
            frame = await self._ipc.recv_frame()
            if frame is None:
                continue
            # Crossfade at a kind change (idle<->talk) to hide the seam.
            if frame.kind != self._prev_kind and self._last_rgb is not None:
                for blended in crossfade(self._last_rgb, [frame.rgb]):
                    self._ring.append(Frame(rgb=blended, seq=frame.seq, kind=frame.kind, ts=frame.ts))
            self._prev_kind = frame.kind
            self._ring.append(frame)

    async def _publish_loop(self) -> None:
        """Fixed 25fps pacer. Hold last frame on underflow rather than stalling."""
        period = 1.0 / PUBLISH_FPS
        # prime the jitter buffer
        while len(self._ring) < JITTER_FRAMES and not self._stop.is_set():
            await asyncio.sleep(period)
        loop = asyncio.get_running_loop()
        next_t = loop.time()
        while not self._stop.is_set():
            if self._ring:
                self._last_rgb = self._ring.popleft().rgb
            if self._last_rgb is not None:
                vf = rtc.VideoFrame(
                    width=self._width, height=self._height,
                    type=rtc.VideoBufferType.RGBA, data=_rgb_to_rgba(self._last_rgb),
                )
                self._source.capture_frame(vf)
                if not self._first_frame_evt.is_set():
                    self._first_frame_evt.set()
            next_t += period
            await asyncio.sleep(max(0.0, next_t - loop.time()))

    async def close(self) -> None:
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        if self._room is not None:
            await self._room.disconnect()
