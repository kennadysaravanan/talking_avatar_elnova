"""
ipc_worker.py  —  ZeroMQ transport used by the avatar worker (rank 4 ONLY).

WHY THIS EXISTS
---------------
The avatar worker runs as 5 separate OS processes under `torchrun --nproc_per_node=5`.
The FastAPI orchestrator is yet another process. asyncio queues cannot cross process
boundaries, so we bridge the orchestrator <-> worker with ZeroMQ over loopback.

Only worker **rank 4** (the VAE rank) ever touches these queues, because rank 4 is the
only rank that (a) invokes get_audio_callback() and (b) yields decoded frames. See
avatar_worker.py and the plan doc for the verified GPU topology.

TWO CHANNELS (orchestrator binds, worker connects):
  * AUDIO  : orchestrator PUSH  ->  worker PULL   (TTS PCM in, plus control messages)
  * FRAME  : worker PUSH        ->  orchestrator PULL   (decoded RGB frames out)

WIRE FORMAT (must match server/ipc.py exactly):
  AUDIO multipart:
      [b"AUDIO", <int16 LE mono 16kHz PCM bytes>]
      [b"FLUSH"]            # drop all queued-but-unspoken audio (interruption)
      [b"TALK_START"]       # marker: a new spoken turn begins (optional, for metrics)
      [b"IDLE"]             # marker: orchestrator believes we are idle now
  FRAME multipart:
      [b"FRAME", <json header bytes>, <uint8 RGB bytes, H*W*3>]
      header = {"h": int, "w": int, "seq": int, "kind": "idle"|"talk", "ts": float}

CRITICAL CONSTRAINT (load-bearing, see plan constraint #2):
  The audio side of this module is consumed by get_audio_callback(), which runs INSIDE
  the distributed dist.send/recv ring. If rank 4 ever blocks waiting for audio, the ring
  stalls and the entire 5-GPU pipeline DEADLOCKS. Therefore `drain_audio()` is strictly
  non-blocking (zmq.NOBLOCK). It NEVER waits.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import zmq

SAMPLE_RATE = 16000  # engine / audio-encoder input rate (Hz). TTS is resampled to this.

# Endpoints. Orchestrator binds; worker connects. Override via env on the pod if needed.
AUDIO_ENDPOINT = os.environ.get("AVATAR_AUDIO_ENDPOINT", "tcp://127.0.0.1:5557")
FRAME_ENDPOINT = os.environ.get("AVATAR_FRAME_ENDPOINT", "tcp://127.0.0.1:5558")


class WorkerIPC:
    """ZeroMQ client living on rank 4. Background thread drains the audio socket into an
    in-memory PCM ring so the (hot, non-blocking) audio callback never touches the socket
    directly."""

    def __init__(
        self,
        audio_endpoint: str = AUDIO_ENDPOINT,
        frame_endpoint: str = FRAME_ENDPOINT,
    ) -> None:
        self._ctx = zmq.Context.instance()

        # AUDIO: PULL, connect to orchestrator's PUSH.
        self._audio_sock = self._ctx.socket(zmq.PULL)
        self._audio_sock.setsockopt(zmq.RCVHWM, 1000)
        self._audio_sock.connect(audio_endpoint)

        # FRAME: PUSH, connect to orchestrator's PULL.
        self._frame_sock = self._ctx.socket(zmq.PUSH)
        self._frame_sock.setsockopt(zmq.SNDHWM, 200)   # bounded; drop rather than balloon latency
        self._frame_sock.setsockopt(zmq.SNDTIMEO, 0)   # non-blocking send (drop on backpressure)
        self._frame_sock.connect(frame_endpoint)

        # Decoded PCM waiting to be consumed by the audio callback (float32 mono 16kHz).
        self._pcm = np.zeros(0, dtype=np.float32)
        self._pcm_lock = threading.Lock()

        self._talking = False          # set True between TALK_START and queue-drain
        self._pending_ref = None       # new reference-image path (hot-swap), set by orchestrator
        self._frame_seq = 0
        self._stop = threading.Event()

        self._reader = threading.Thread(target=self._reader_loop, name="ipc-audio-reader", daemon=True)
        self._reader.start()

    # ----------------------------------------------------------------- audio in
    def _reader_loop(self) -> None:
        """Background thread: continuously pull from the audio socket and append PCM to the
        ring. Control messages mutate state. This thread MAY block on recv() — it is NOT in
        the dist ring — which is exactly why the callback path is decoupled from it."""
        poller = zmq.Poller()
        poller.register(self._audio_sock, zmq.POLLIN)
        while not self._stop.is_set():
            socks = dict(poller.poll(timeout=100))  # ms
            if self._audio_sock not in socks:
                continue
            try:
                parts = self._audio_sock.recv_multipart()
            except zmq.ZMQError:
                continue
            mtype = parts[0]
            if mtype == b"AUDIO" and len(parts) >= 2:
                pcm16 = np.frombuffer(parts[1], dtype="<i2")
                pcmf = pcm16.astype(np.float32) / 32768.0
                with self._pcm_lock:
                    self._pcm = np.concatenate([self._pcm, pcmf])
                self._talking = True
            elif mtype == b"FLUSH":
                # Interruption: discard everything not yet rendered.
                with self._pcm_lock:
                    self._pcm = np.zeros(0, dtype=np.float32)
                self._talking = False
            elif mtype == b"TALK_START":
                self._talking = True
            elif mtype == b"IDLE":
                self._talking = False
            elif mtype == b"REF" and len(parts) >= 2:
                # New reference image (same-pod filesystem path) -> hot-swap the avatar identity.
                with self._pcm_lock:
                    self._pending_ref = parts[1].decode()

    def drain_audio(self, n_samples: int) -> Optional[np.ndarray]:
        """Pop exactly `n_samples` of float32 PCM for one block, NON-BLOCKING.

        Returns None when there is not a full block of real audio available — the caller
        (get_audio_callback) MUST then substitute ambient/idle audio. This function never
        waits: returning None on underflow is the whole point (see constraint #2).
        """
        with self._pcm_lock:
            if self._pcm.shape[0] >= n_samples:
                out = self._pcm[:n_samples].copy()
                self._pcm = self._pcm[n_samples:]
                return out
            # Underflow. If we have a partial tail and the talker just stopped, we let it
            # drain on the next call; for now signal "no full block" -> ambient.
            return None

    def has_audio(self) -> bool:
        with self._pcm_lock:
            return self._pcm.shape[0] > 0

    @property
    def talking(self) -> bool:
        return self._talking

    def get_pending_ref(self) -> Optional[str]:
        """Return-and-clear a pending hot-swap reference-image path (None if none). Used by the
        VAE rank's hotswap check."""
        with self._pcm_lock:
            p = self._pending_ref
            self._pending_ref = None
            return p

    # ---------------------------------------------------------------- frames out
    def push_frame(self, rgb: np.ndarray, kind: str) -> None:
        """Send one uint8 RGB frame [H,W,3] to the orchestrator. Non-blocking: if the
        orchestrator is briefly behind, the frame is dropped (better a dropped frame than a
        stalled GPU ring)."""
        assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
        h, w, _ = rgb.shape
        header = json.dumps(
            {"h": int(h), "w": int(w), "seq": self._frame_seq, "kind": kind, "ts": time.time()}
        ).encode()
        self._frame_seq += 1
        try:
            self._frame_sock.send_multipart([b"FRAME", header, np.ascontiguousarray(rgb).tobytes()],
                                            flags=zmq.NOBLOCK)
        except zmq.Again:
            pass  # orchestrator behind; drop this frame

    def close(self) -> None:
        self._stop.set()
        try:
            self._reader.join(timeout=1.0)
        except RuntimeError:
            pass
        self._audio_sock.close(0)
        self._frame_sock.close(0)
