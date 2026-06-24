"""
idle.py  —  Idle-audio generation and idle-frame ring buffer.

The avatar must keep moving subtly even when no one is speaking. The engine is audio-driven,
so "idle" means: feed the audio callback a benign audio block that produces small, natural
motion (breathing/blinks) WITHOUT making the avatar appear to mumble words.

IDLE AUDIO STRATEGY (plan constraint #3 — load-bearing):
  The model is speech-driven. Feeding it broadband noise risks the avatar interpreting the
  noise as phonemes and "mumbling". So we DEFAULT TO TRUE SILENCE (zeros) and treat noise as
  an opt-in fallback to be A/B tested on the pod:

      IDLE_AUDIO_MODE=silence   -> zeros            (DEFAULT, try first)
      IDLE_AUDIO_MODE=noise     -> ~-45 dBFS gaussian noise (fallback if silence looks frozen)

  Switch via the env var; do not hard-code a preference. Document the winner in README after
  the on-pod test.

IDLE FRAME RING:
  We also keep a short ring buffer of recently-decoded idle frames. Its primary purpose is a
  jitter cushion / fallback if the generator momentarily stalls (e.g. during the talk->idle
  handoff). The *live* idle motion still comes from the running generator; the ring is a
  safety net, not the primary source.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Optional

import numpy as np

IDLE_AUDIO_MODE = os.environ.get("IDLE_AUDIO_MODE", "silence")  # "silence" | "noise"
# -45 dBFS amplitude in linear terms (10 ** (-45/20)). Only used in noise mode.
_NOISE_AMP = 10.0 ** (-45.0 / 20.0)


def ambient_block(n_samples: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Return one block of ambient (idle) audio, float32 mono 16kHz, length n_samples.

    Silence by default; faint noise only if IDLE_AUDIO_MODE=noise. See constraint #3.
    """
    if IDLE_AUDIO_MODE == "noise":
        rng = rng or np.random.default_rng(0)
        return (rng.standard_normal(n_samples).astype(np.float32) * _NOISE_AMP)
    # silence (default)
    return np.zeros(n_samples, dtype=np.float32)


class IdleFrameRing:
    """Fixed-size ring of recent idle RGB frames (uint8 [H,W,3]) used as a jitter/fallback
    cushion. Capacity defaults to ~4s at 25fps."""

    def __init__(self, capacity: int = 100) -> None:
        self._buf: deque[np.ndarray] = deque(maxlen=capacity)
        self._read = 0

    def add(self, frame: np.ndarray) -> None:
        self._buf.append(frame)

    def __len__(self) -> int:
        return len(self._buf)

    def is_ready(self, min_frames: int = 12) -> bool:
        return len(self._buf) >= min_frames

    def next_loop_frame(self) -> Optional[np.ndarray]:
        """Return the next frame for a seamless loop (ping-pong avoided for simplicity;
        plain wrap is acceptable because idle motion is small). Returns None if empty."""
        if not self._buf:
            return None
        frame = self._buf[self._read % len(self._buf)]
        self._read += 1
        return frame
