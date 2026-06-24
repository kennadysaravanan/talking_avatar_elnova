"""
blend.py  —  idle <-> talk crossfade helpers.

At the idle->talk and talk->idle boundaries the engine's `motion_frames` conditioning already
gives pose continuity, so seams are usually small. A short linear alpha crossfade hides any
residual jump and keeps the perceived cadence smooth.

DEFAULT: 8-frame linear crossfade (≈0.32s at 25fps). Tune on the pod; noted in README.

These helpers operate on uint8 RGB frames [H,W,3]. They are pure / CPU-cheap and run in the
orchestrator's frame path (or the worker, if you prefer to blend before transport).
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
    """Produce a crossfade from a single `from_frame` into the first `n` of `to_frames`.

    Returns a list of n blended frames; the caller then continues with to_frames[n:].
    If to_frames is shorter than n, blends as many as available.
    """
    n = min(n, len(to_frames))
    out: List[np.ndarray] = []
    for i in range(n):
        t = (i + 1) / (n + 1)
        out.append(alpha_blend(from_frame, to_frames[i], t))
    return out
