# Real-Time 2D Talking Avatar Server (LiveAvatar engine)

A real-time, interactive 2D talking-avatar server. Rendering engine = Alibaba **LiveAvatar**
(Wan2.2-S2V-14B + Live-Avatar LoRA), vendored and pinned in `avatar/liveavatar/`. The live
interaction layer (STT, LLM, TTS, idle loop, streaming transport, interruption) is ours.

Deployment target: **RunPod 8×H100 80GB SXM**, using **5 GPUs** (`CUDA_VISIBLE_DEVICES=0,1,2,3,4`).

> **One pod = ONE concurrent conversation.** A session uses 5 GPUs (4 DiT + 1 VAE); only 3 are
> left over — not enough for a second session. This is single-session by design; factor the
> cost-per-user into any production scoping. (Constraint #6.)

---

## How it works (verified against the real engine source)

The upstream repo ships an **offline** entry script. The real-time streaming seam exists in the
code but is wired to nothing. We verified two seams in
`avatar/liveavatar/models/wan/causal_s2v_pipeline_tpp_blockwise.py` and built the missing driver:

1. **Audio IN** — `get_audio_callback()` is *called* (line 486) but never *defined* upstream. We
   define it (`avatar/avatar_worker.py`) to pop TTS PCM from a queue; on empty it returns ambient
   audio. Live audio activates at autoregressive round `r≥2` (rounds 0–1 are template warm-up).
2. **Frames OUT** — `generate()` is a **generator** that `yield`s decoded RGB per block on the VAE
   rank (line 1012). We iterate it and push frames to the orchestrator. No VAE patching needed.

**GPU topology:** 5 ranks → ranks 0–3 DiT, **rank 4 VAE**. Rank 4 is the *only* rank that touches
audio-in and frames-out, so only it connects to our IPC.

```
Browser (React) ──WS text/state──▶ Orchestrator (FastAPI) ──ZMQ──▶ Worker (torchrun nproc=5)
       ▲                              STT/LLM/TTS, turns,            rank4: get_audio_callback
       └──── WebRTC video ────────────┘ interruption                       generate()→frames
                      LiveKit Cloud (single continuous track per session)
```

## Repo layout
- `avatar/` — vendored engine (`liveavatar/`, Apache-2.0, see `avatar/LICENSE`+`NOTICE`) + our
  `avatar_worker.py` (driver), `ipc_worker.py`, `idle.py`, `blend.py`.
- `server/` — FastAPI orchestrator: `main.py`, `session.py`, `turn_manager.py`,
  `transport_livekit.py`, `ipc.py`, `providers/` (GPT/TTS/Whisper behind ABCs).
- `web/` — React client (upload → loading → LiveKit player → text box → interrupt).
- `scripts/` — `download_models.sh`, `run_worker.sh`, `bench_block.sh`, `acceptance/`.
- `Dockerfile`, `environment.md`, `requirements.txt`, `requirements-lock.txt`.

---

## RunPod run steps

1. **Launch** an 8×H100 80GB SXM pod (PyTorch 2.8 / CUDA 12.8 image) with a **persistent volume
   mounted at `ckpt/`**.
2. **Build env:** `docker build -t avatar .` (or `pip install -r requirements.txt` after the
   torch/FA3 steps in `environment.md`). After a clean install: `pip freeze > requirements-lock.txt`.
3. **Download weights once** (to the persistent volume): `bash scripts/download_models.sh`.
4. **GATE — prove real-time FIRST:** `bash scripts/bench_block.sh`.
   If `p50 > 480 ms/block`, **stop** — real-time is infeasible; nothing else matters.
5. **Start the worker:** `REF_IMAGE=/path/photo.jpg bash scripts/run_worker.sh`
   (first run compiles — minutes; subsequent runs fast).
6. **Set env & start orchestrator:**
   ```
   export OPENAI_API_KEY=...  LIVEKIT_URL=...  LIVEKIT_API_KEY=...  LIVEKIT_API_SECRET=...
   uvicorn server.main:app --host 0.0.0.0 --port 8080
   ```
7. **Serve the client:** `cd web && npm install && VITE_API_BASE=http://<pod>:8080 npm run dev`.

### Configuration
All env vars (endpoints, idle-audio mode, LiveKit, OpenAI) are documented in
[environment.md](environment.md). Notable: `IDLE_AUDIO_MODE=silence` (default) vs `noise`.

---

## Design assumptions (defaults; tune on the pod)
- **Idle audio = true silence** (zeros) by default — the model is speech-driven and broadband
  noise can cause idle "mumbling". `IDLE_AUDIO_MODE=noise` is an A/B fallback (constraint #3).
- **Crossfade = 8 frames** linear alpha at idle↔talk boundaries (`avatar/blend.py`).
- **Idle frame ring = ~4 s** jitter/fallback cushion; live idle motion comes from the generator.
- **IPC = ZeroMQ over loopback**; only worker rank 4 connects.
- **TTS 24 kHz → 16 kHz** resample before queueing.
- **Interruption is text-triggered** (`INTERRUPT_MODE="text"`); VAD is future work.

## Known behaviors that are NOT bugs
- **First-block warm-up per response** — live audio only flows at `r≥2`; expect a short startup
  delay on the first block of each answer (constraint #4).
- **Interruption tail ~0.5–1 s** — we never reset the KV cache (that corrupts it); in-flight audio
  in the dist ring finishes rendering before the avatar returns toward idle (constraint #5).
- **First inference is slow** — `ENABLE_COMPILE=true` compiles on the first run.

## Future work (explicitly out of scope here)
- Hot per-session ref-image swap without restarting the worker (prototype warms at launch).
- VAD-based (voice) interruption.
- Multi-session scaling (needs more pods; see single-session note).

---

## Testing
- **Off-pod (no GPU), runnable now:** `python tests/test_ipc_loopback.py` (wire protocol);
  `python -m py_compile server/*.py server/providers/*.py avatar/avatar_worker.py avatar/*.py`.
- **On-pod acceptance suite + the real-time gate:** see
  [scripts/acceptance/README.md](scripts/acceptance/README.md).

### Acceptance results (fill in after pod run)
| Test | Target | Actual |
|---|---|---|
| 0. Real-time gate (p50/block) | < 480 ms | _TBD_ |
| 1. Warm-up to first idle frame | few s (warm) | _TBD_ |
| 2. Text → first talk frame | report median | _TBD_ |
| 3. Sustained talk FPS | ≈ 25 | _TBD_ |
| 4. Interruption stop | within ~1 s, no crash | _TBD_ |
| 5. 10-min session | no OOM/drift/crash | _TBD_ |

## Attribution / license
`avatar/liveavatar/` is vendored from https://github.com/Alibaba-Quark/LiveAvatar (Apache-2.0).
See `avatar/LICENSE` and `avatar/NOTICE`. Our code is in `server/`, `web/`, `scripts/`, and the
non-vendored files under `avatar/`.
