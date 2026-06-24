# Acceptance tests (run ON THE POD)

All numeric acceptance tests require the 5×H100 pod, the downloaded weights, and the running
worker + orchestrator. Run them in order — **test 0 is a hard gate**.

## 0. Real-time gate (DO THIS FIRST — cheap kill switch)
```
bash scripts/bench_block.sh        # BENCH_BLOCKS=50 by default
```
- Measures per-block compute (4-step denoise + VAE decode), skipping compile-warmup blocks.
- Budget = **480 ms/block** (12 frames @ 25 fps).
- **If p50 > 480 ms → STOP.** Real-time is infeasible on this hardware/config; do not proceed.
- Records `bench_block.metrics`.

## 1. Warm-up time
- Start worker (`scripts/run_worker.sh`) with `REF_IMAGE=<photo>`; start orchestrator.
- `POST /session` with the photo; measure wall time until `wait_first_frame` returns (first idle
  frame published). Target: idle streaming within a few seconds of warm model (first-ever run
  includes one-time compile — report separately).

## 2. End-to-end talk latency
- With a live session, send a text message over the WS and timestamp it.
- Measure to the first `kind="talk"` frame arriving at the orchestrator (`Frame.ts`).
- **Expect a known warm-up on the first block of each response** (constraint #4: live audio only
  flows at autoregressive round r≥2). Report median over several prompts.

## 3. Sustained talking FPS
- During a long reply, count `kind="talk"` frames/sec arriving at the orchestrator over 10 s.
- Target ≈ 25 (sample_fps). Record actual; correlate with the bench p50.

## 4. Interruption (TEXT MODE)
- Send prompt A; while talk frames are flowing, send prompt B.
- Assert: old answer stops and B starts. **Assert stop within ~1 s, NOT instantly**
  (constraint #5: in-flight audio in the dist ring renders a ~0.5–1 s tail). Verify no crash and
  that frames keep flowing (KV cache stays healthy). Confirm clean return to idle after B.

## 5. Long session (stability)
- 10+ minutes alternating idle/talk. Watch `nvidia-smi` memory and FPS.
- Assert: no OOM, no crash, no progressive FPS drift. (KV cache is resident by design; if memory
  climbs, see environment.md OOM notes.)

## Metrics logging
The worker logs `[VAE] decoding ... Xs` per block and `[BENCH]` summaries. The orchestrator logs
turn/interrupt events and frame kinds. Capture stdout to a file per test and tabulate into the
README results table.

## Off-pod sanity (no GPU) — already runnable
```
python tests/test_ipc_loopback.py          # wire protocol (audio/flush/frames)
python -m py_compile server/*.py server/providers/*.py avatar/avatar_worker.py avatar/*.py
```
