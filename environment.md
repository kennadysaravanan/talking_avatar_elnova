# Environment — pins & rationale (reproducibility = money on a rented pod)

Target: **RunPod 8×H100 80GB SXM**, 5 GPUs used (`CUDA_VISIBLE_DEVICES=0,1,2,3,4`).
Pod driver: CUDA 12.8. Python 3.10.

## Install order (also encoded in the Dockerfile)
1. System: `python3.10`, `git`, `ffmpeg`, build tools.
2. **PyTorch (cu128 wheels — work on the 12.8 driver):**
   ```
   pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
   pip install torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```
3. **FlashAttention 3 (Hopper / H100):**
   ```
   pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch280 \
               --extra-index-url https://download.pytorch.org/whl/cu128
   ```
   Non-Hopper fallback (documented, NOT our target): `pip install flash-attn==2.8.3 --no-build-isolation`.
4. `pip install -r requirements.txt`
5. **After a clean install on the pod:** `pip freeze > requirements-lock.txt` and commit it.

## Hard pins (DO NOT violate)
| Package | Pin | Why |
|---|---|---|
| transformers | `>=4.49.0,<=4.51.3` | **>4.51.3 breaks S2V / the audio encoder.** Hard cap. |
| numpy | `>=1.23.5,<2` | **numpy 2.x breaks the stack.** Must stay 1.x. |
| peft | `==0.17.1` | Exact, per upstream. LoRA load path depends on it. |
| diffusers | `>=0.31.0` | Scheduler APIs used by the pipeline. |
| tokenizers | `>=0.20.3` | transformers compatibility. |
| accelerate | `>=1.1.1` | upstream baseline. |
| torch | `==2.8.0` (cu128) | FA3 wheel set is built against torch 2.8.0/cu128. |

## Known Issues / Fixes (rental-server debugging)
- **numpy 2.x pulled in transitively** → `pip install "numpy<2"` and re-check. Many engine ops assume numpy 1.x ABI.
- **transformers >4.51.3** → S2V / audio-encoder breakage. Reinstall `transformers<=4.51.3`.
- **FA3 wheel "not found"** → confirm torch is EXACTLY 2.8.0 + cu128. The windreamer index only has cu128_torch280 wheels; any other torch will miss.
- **OOM on long sessions** → lower `--size`, ensure `--fp8` is on, watch KV-cache growth (the worker keeps it resident across turns by design — see constraint #5). Consider `--offload_kv_cache`.
- **First inference very slow** → `ENABLE_COMPILE=true` compiles on first run (expected, minutes). Subsequent runs are fast. The bench harness skips the first few blocks for this reason.
- **Pipeline appears to hang at a block boundary** → almost certainly the audio callback blocked. The callback MUST be non-blocking (constraint #2). It returns ambient audio on underflow; it must never wait on the queue.

## Verified engine facts (from reading the real source)
- Streaming seam (`get_audio_callback`, generator `yield`) lives ONLY in
  `causal_s2v_pipeline_tpp_blockwise.py`; the upstream entry script uses the offline `_tpp` one.
- 5-rank topology: ranks 0–3 DiT, rank 4 VAE. Rank 4 is the only rank touching audio-in/frames-out.
- `num_frames_per_block=3` latent ×4 = 12 pixel frames/block; `sample_fps=25` → **480 ms/block budget**.

## Runtime env vars
| Var | Default | Meaning |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4` | 5 GPUs for the worker |
| `ENABLE_COMPILE` | `true` | compile on first run |
| `IDLE_AUDIO_MODE` | `silence` | `silence` (default, constraint #3) or `noise` |
| `AVATAR_AUDIO_ENDPOINT` | `tcp://127.0.0.1:5557` | orchestrator→worker PCM |
| `AVATAR_FRAME_ENDPOINT` | `tcp://127.0.0.1:5558` | worker→orchestrator frames |
| `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | — | LiveKit Cloud |
| `OPENAI_API_KEY` | — | GPT + TTS + Whisper |
