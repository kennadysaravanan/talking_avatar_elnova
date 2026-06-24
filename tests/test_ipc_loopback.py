"""
ZMQ loopback test — validates the orchestrator<->worker wire protocol WITHOUT any GPU/model.

Exercises:
  * audio PCM roundtrip (orchestrator PUSH -> worker PULL ring -> non-blocking drain)
  * FLUSH control drops queued audio (interruption path)
  * frame roundtrip (worker PUSH -> orchestrator PULL, shape/kind preserved)
  * non-blocking drain returns None on underflow (constraint #2 contract)

Run:  python -m pytest tests/test_ipc_loopback.py   (or: python tests/test_ipc_loopback.py)
"""
import asyncio
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "avatar"))

# Use unique ports so a stray run doesn't clash.
os.environ.setdefault("AVATAR_AUDIO_ENDPOINT", "tcp://127.0.0.1:5657")
os.environ.setdefault("AVATAR_FRAME_ENDPOINT", "tcp://127.0.0.1:5658")

from server.ipc import OrchestratorIPC          # noqa: E402
from ipc_worker import WorkerIPC                 # noqa: E402


async def _main() -> int:
    orch = OrchestratorIPC()
    worker = WorkerIPC()
    await asyncio.sleep(0.3)  # let PUSH/PULL connect

    failures = []

    # --- audio roundtrip ---
    pcm = (np.sin(np.arange(8000) * 0.1) * 10000).astype(np.int16)
    await orch.send_pcm16(pcm)
    await asyncio.sleep(0.3)
    got = worker.drain_audio(8000)
    if got is None or got.shape[0] != 8000:
        failures.append(f"audio drain failed: {None if got is None else got.shape}")
    else:
        # values should match (int16 -> float32/32768 roundtrip)
        if not np.allclose(got, pcm.astype(np.float32) / 32768.0, atol=1e-4):
            failures.append("audio values mismatch after roundtrip")

    # --- underflow returns None (non-blocking contract) ---
    if worker.drain_audio(8000) is not None:
        failures.append("drain_audio should return None on underflow")

    # --- FLUSH drops queued audio ---
    await orch.send_pcm16(pcm)
    await asyncio.sleep(0.2)
    await orch.flush()
    await asyncio.sleep(0.2)
    if worker.has_audio():
        failures.append("FLUSH did not drop queued audio")

    # --- frame roundtrip ---
    rgb = (np.random.rand(400, 720, 3) * 255).astype(np.uint8)
    worker.push_frame(rgb, kind="talk")
    frame = await asyncio.wait_for(orch.recv_frame(), timeout=2.0)
    if frame is None or frame.rgb.shape != (400, 720, 3) or frame.kind != "talk":
        failures.append(f"frame roundtrip failed: {None if frame is None else (frame.rgb.shape, frame.kind)}")
    elif not np.array_equal(frame.rgb, rgb):
        failures.append("frame pixels mismatch after roundtrip")

    worker.close()
    orch.close()

    if failures:
        print("FAIL:")
        for f in failures:
            print("  -", f)
        return 1
    print("PASS: ipc loopback (audio roundtrip, underflow=None, FLUSH, frame roundtrip)")
    return 0


def test_ipc_loopback():
    assert asyncio.run(_main()) == 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
