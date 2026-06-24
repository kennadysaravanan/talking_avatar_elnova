"""
server/blend.py  —  idle <-> talk crossfade helpers (orchestrator side).

Duplicate of avatar/blend.py kept here so the `server` package is self-contained and does NOT
import the avatar engine package. Pure numpy, no engine deps.

DEFAULT: 8-frame linear crossfade (~0.32s at 25fps). Operates on uint8 RGB frames [H,W,3].
"""

from __future__ import annotations

from typing import List

import numpy as np

DEFAULT_CROSSFADE_FRAMES = 8


def alpha_blend(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear blend: (1-t)*a + t*b, returned as uint8. t in [0,1]."""
    t = float(np.clip(t, 0.0, 1.0))
    out = a.astype(np.float32) * (1.0 - t) + b.astype(np.float32) * t
    return np.clip(out, 0, 255).astype(np.uint8)


def crossfade(from_frame: np.ndarray, to_frames: List[np.ndarray],
              n: int = DEFAULT_CROSSFADE_FRAMES) -> List[np.ndarray]:
    """Crossfade from a single `from_frame` into the first `n` of `to_frames`.

    Returns n blended frames; caller continues with to_frames[n:]. If to_frames is shorter
    than n, blends as many as available.
    """
    n = min(n, len(to_frames))
    out: List[np.ndarray] = []
    for i in range(n):
        t = (i + 1) / (n + 1)
        out.append(alpha_blend(from_frame, to_frames[i], t))
    return out
