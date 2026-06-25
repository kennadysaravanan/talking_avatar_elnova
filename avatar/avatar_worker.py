"""
avatar_worker.py  —  THE missing live driver for LiveAvatar (Wan2.2-S2V-14B + Live-Avatar LoRA).

WHAT THIS IS
------------
The upstream repo ships an OFFLINE entry script (minimal_inference/s2v_streaming_interact.py)
that imports the NON-streaming `causal_s2v_pipeline_tpp` pipeline, reads a file audio, and
saves an mp4. The streaming seam — `get_audio_callback()` plus a generator that yields decoded
frames per block — exists ONLY in `causal_s2v_pipeline_tpp_blockwise` and is wired to NO entry
script. This file is that missing entry script: a resident, forever-running driver that

  * implements get_audio_callback()  (audio IN, from the orchestrator via ZeroMQ)   [seam #1]
  * iterates generate() and pushes decoded RGB frames OUT (to the orchestrator)      [seam #2]
  * keeps the model resident across idle/talk turns without resetting the KV cache.

VERIFIED GPU TOPOLOGY (5 ranks, --num_gpus_dit 4 --enable_vae_parallel):
  ranks 0..3 = DiT, rank 4 = VAE. `_initialize_comm_group` (pipeline lines 684-690) wires a
  ring via dist.send/recv. RANK 4 IS THE ONLY RANK THAT:
     - calls get_audio_callback()  (pipeline line 486, gated on `not in_dit_device and r>=2`)
     - yields decoded frames        (pipeline line 1012)
  => only rank 4 attaches to our queues. All other ranks just spin the generator.

LOAD-BEARING CONSTRAINTS (see plan doc):
  #1 Real-time: each block must compute in <480ms (12 frames @25fps). Prove with bench_block.py.
  #2 get_audio_callback() MUST be non-blocking — it runs inside the dist ring. Blocking = deadlock.
  #3 Idle audio defaults to SILENCE (zeros), not noise (avoid idle "mumbling").
  #4 Live audio only flows at r>=2 (rounds 0-1 use template) -> known per-response warm-up.
  #5 Interruption has a ~0.5-1s tail (in-flight audio still renders); we do NOT reset KV cache.

Run via scripts/run_worker.sh (torchrun --nproc_per_node=5).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time

# Make `import liveavatar...` resolve regardless of CWD (engine subtree sits next to this file).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# Heavy deps (torch / engine) are imported lazily inside main() so this module can be
# syntax-checked and partially imported off-GPU. See `_lazy_imports()`.

logger = logging.getLogger("avatar_worker")

SAMPLE_RATE = 16000  # engine audio rate

# Neutral default scene prompt. The model still expects a text prompt; for a talking-head we
# keep it generic. Overridable with --prompt per deployment.
DEFAULT_PROMPT = (
    "A person looking directly at the camera in a well-lit room, calm and natural expression, "
    "subtle head and facial motion, photorealistic."
)


def _lazy_imports():
    import torch
    import torch.distributed as dist
    from liveavatar.models.wan.wan_2_2.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
    from liveavatar.models.wan.causal_s2v_pipeline_tpp_blockwise import WanS2V  # STREAMING pipeline
    from liveavatar.utils.args_config import parse_args_for_training_config
    return torch, dist, MAX_AREA_CONFIGS, WAN_CONFIGS, WanS2V, parse_args_for_training_config


# --------------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser("LiveAvatar resident worker")
    # Mirror the args run_worker.sh passes (subset of the upstream entry script).
    p.add_argument("--task", default="s2v-14B")
    p.add_argument("--size", default="720*400")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--training_config", required=True)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--image", default=None,
                   help="Reference photo. If omitted, --image_env / IMAGE env / session warmup must set it.")
    p.add_argument("--infer_frames", type=int, default=48)
    p.add_argument("--sample_steps", type=int, default=4)
    p.add_argument("--sample_guide_scale", type=float, default=0.0)
    p.add_argument("--sample_solver", default="euler")
    p.add_argument("--sample_shift", type=float, default=None)
    p.add_argument("--base_seed", type=int, default=420)
    p.add_argument("--num_gpus_dit", type=int, default=4)
    p.add_argument("--enable_vae_parallel", action="store_true")
    p.add_argument("--enable_online_decode", action="store_true")
    p.add_argument("--offload_kv_cache", action="store_true")
    p.add_argument("--convert_model_dtype", action="store_true")
    p.add_argument("--start_from_ref", action="store_true")
    p.add_argument("--load_lora", action="store_true")
    p.add_argument("--lora_path_dmd", default=None)
    p.add_argument("--fp8", action="store_true")
    # Bench mode: run N blocks, measure per-block compute, print p50/p95, exit. (constraint #1)
    p.add_argument("--bench_blocks", type=int, default=0,
                   help="If >0, run as a benchmark for this many blocks and exit (no IPC).")
    return p.parse_args()


# ----------------------------------------------------------------- helpers
def make_silent_wav(seconds: float = 6.0) -> str:
    """Write a short silent 16kHz mono wav for the warm-up audio template (rounds r<2 read a
    file via encode_audio). Returns the path. ~6s comfortably covers 2*infer_frames."""
    import soundfile as sf
    n = int(seconds * SAMPLE_RATE)
    path = os.path.join(tempfile.gettempdir(), "liveavatar_warmup_silence.wav")
    sf.write(path, np.zeros(n, dtype=np.float32), SAMPLE_RATE, subtype="PCM_16")
    return path


def block_audio_samples(num_frames_per_block: int, sample_fps: int) -> int:
    """Samples of 16kHz audio the callback must return for one block.
    block pixel frames = num_frames_per_block*4; duration = frames/fps; samples = duration*16000."""
    pixel_frames = num_frames_per_block * 4
    return int(round(pixel_frames / float(sample_fps) * SAMPLE_RATE))


def frames_from_yield(img, _logged={"done": False}):
    """Convert a generator-yielded frame tensor to a list of uint8 RGB [H,W,3] frames.

    The yielded tensor is float in [-1,1] (model space; save_video uses value_range=(-1,1)).
    Exact dim order is confirmed-on-first-run: we LOG the raw shape once and reshape
    defensively to [f,3,H,W]. If the first-pod-run log shows a different layout, adjust here.
    """
    import torch
    x = img.detach().to(torch.float32).cpu()
    if not _logged["done"]:
        logger.info(f"[frames_from_yield] first yielded tensor shape={tuple(x.shape)} "
                    f"min={float(x.min()):.3f} max={float(x.max()):.3f}")
        _logged["done"] = True

    # Squeeze leading singleton batch dims until <=4 dims remain.
    while x.dim() > 4 and x.shape[0] == 1:
        x = x[0]
    # Expected [3, f, H, W] or [f, 3, H, W] or [1,3,f,H,W]->[3,f,H,W].
    if x.dim() == 5:           # [1,3,f,H,W]
        x = x[0]
    if x.dim() == 4:
        if x.shape[0] == 3:    # [3,f,H,W] -> [f,H,W,3]
            x = x.permute(1, 2, 3, 0)
        elif x.shape[1] == 3:  # [f,3,H,W] -> [f,H,W,3]
            x = x.permute(0, 2, 3, 1)
        else:                  # ambiguous; assume [f,H,W,3] already
            pass
    elif x.dim() == 3:         # single frame [3,H,W] or [H,W,3]
        if x.shape[0] == 3:
            x = x.permute(1, 2, 0)
        x = x.unsqueeze(0)

    arr = ((x.clamp(-1, 1) + 1.0) * 0.5 * 255.0).round().to(torch.uint8).numpy()
    return [np.ascontiguousarray(arr[i]) for i in range(arr.shape[0])]


# ----------------------------------------------------------------- build model
def build_model(args, training_settings):
    torch, dist, MAX_AREA_CONFIGS, WAN_CONFIGS, WanS2V, _ = _lazy_imports()

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="[%(asctime)s][rank{}] %(levelname)s: %(message)s".format(rank),
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    assert world_size >= 5 and args.num_gpus_dit == 4 and args.enable_vae_parallel, \
        "Live worker requires 5 ranks: --num_gpus_dit 4 --enable_vae_parallel."

    cfg = WAN_CONFIGS[args.task]
    # Construction is IDENTICAL to the upstream entry script (signatures verified identical
    # between _tpp and _tpp_blockwise); only the imported class differs.
    wan = WanS2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        sp_size=1,
        t5_cpu=False,
        convert_model_dtype=args.convert_model_dtype,
        single_gpu=False,
        offload_kv_cache=args.offload_kv_cache,
    )

    if args.load_lora and args.lora_path_dmd is not None:
        wan.noise_model = wan.add_lora_to_model(
            wan.noise_model,
            lora_rank=training_settings["lora_rank"],
            lora_alpha=training_settings["lora_alpha"],
            lora_target_modules=training_settings["lora_target_modules"],
            init_lora_weights=training_settings["init_lora_weights"],
            pretrained_lora_path=args.lora_path_dmd,
            load_lora_weight_only=False,
        )

    if args.fp8 and hasattr(torch, "_scaled_mm"):
        from liveavatar.utils.fp8_linear import replace_linear_with_scaled_fp8
        replace_linear_with_scaled_fp8(
            wan.noise_model,
            ignore_keys=["text_embedding", "time_embedding", "time_projection",
                         "head.head", "casual_audio_encoder.encoder.final_linear"],
        )

    return wan, cfg, rank, MAX_AREA_CONFIGS


# ----------------------------------------------------------------- run loop
def run(args, training_settings):
    torch, dist, _, _, _, _ = _lazy_imports()
    wan, cfg, rank, MAX_AREA_CONFIGS = build_model(args, training_settings)

    in_dit_device = rank < args.num_gpus_dit
    is_vae_rank = (rank == args.num_gpus_dit)  # rank 4

    n_block_samples = block_audio_samples(wan.num_frames_per_block, cfg.sample_fps)
    logger.info(f"per-block audio samples = {n_block_samples} "
                f"(num_frames_per_block={wan.num_frames_per_block}, fps={cfg.sample_fps})")

    # ----- rank 4 only: IPC + idle + the audio callback (seams #1 and #2) -----
    ipc = None
    idle_ring = None
    if is_vae_rank and args.bench_blocks == 0:
        from ipc_worker import WorkerIPC
        from idle import IdleFrameRing, ambient_block
        ipc = WorkerIPC()
        idle_ring = IdleFrameRing(capacity=100)
        _rng = np.random.default_rng(0)

        def get_audio_callback():
            # CONSTRAINT #2: strictly non-blocking. Pop a full block if available, else ambient.
            chunk = ipc.drain_audio(n_block_samples)
            if chunk is None:
                # idle / underflow -> silence (or faint noise per IDLE_AUDIO_MODE). constraint #3
                chunk = ambient_block(n_block_samples, _rng)
            return chunk.astype(np.float32)

        # ATTACH THE HOOK. This is the line the engine has been calling but never defining.
        wan.get_audio_callback = get_audio_callback

    # ----- bench mode (constraint #1): measure per-block compute, no IPC -----
    if args.bench_blocks > 0 and is_vae_rank:
        # In bench we still need *some* audio source on rank 4; feed silence.
        from idle import ambient_block
        wan.get_audio_callback = lambda: ambient_block(n_block_samples)

    warmup_audio = make_silent_wav(6.0)

    # NOTE: the avatar identity is the reference image passed here (REF_IMAGE at launch). A single
    # resident generate() runs forever to keep KV-cache + motion continuity. Per-upload hot-swap
    # was attempted via a per-round dist.broadcast but that DEADLOCKS the TPP pipeline (ranks are
    # intentionally at different rounds), so it is NOT used — changing identity = relaunch the
    # worker with a new REF_IMAGE. (A non-collective restart protocol is future work.)
    gen = wan.generate(
        input_prompt=args.prompt,
        ref_image_path=args.image,
        audio_path=warmup_audio,           # template for rounds r<2 only; r>=2 uses the callback
        enable_tts=False, num_repeat=1, pose_video=None,
        generate_size=args.size, max_area=MAX_AREA_CONFIGS[args.size],
        infer_frames=args.infer_frames, shift=args.sample_shift,
        sample_solver=args.sample_solver, sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale, seed=args.base_seed,
        offload_model=False, init_first_frame=args.start_from_ref, drop_motion_noisy=False,
        num_gpus_dit=args.num_gpus_dit, enable_vae_parallel=args.enable_vae_parallel,
        input_video_for_sam2=None, enable_online_decode=args.enable_online_decode,
    )

    # Iterate forever. DiT ranks yield None; rank 4 yields decoded frame tensors.
    block_times = []
    last = time.time()
    n_yields = 0
    for item in gen:
        now = time.time(); dt = now - last; last = now

        if is_vae_rank and item is not None:
            frames = frames_from_yield(item)
            kind = "talk" if (ipc and ipc.talking) else "idle"
            for f in frames:
                if idle_ring is not None and kind == "idle":
                    idle_ring.add(f)
                if ipc is not None:
                    ipc.push_frame(f, kind)

        if args.bench_blocks > 0:
            n_yields += 1
            if n_yields > 3:
                block_times.append(dt)
            if len(block_times) >= args.bench_blocks:
                _report_bench(block_times, n_block_samples, wan.num_frames_per_block, cfg.sample_fps)
                break

    if ipc is not None:
        ipc.close()
    if dist.is_initialized():
        dist.destroy_process_group()


def _report_bench(times, n_block_samples, nfpb, fps):
    import statistics
    pixel_frames = nfpb * 4
    budget_ms = pixel_frames / float(fps) * 1000.0
    p50 = statistics.median(times) * 1000.0
    p95 = sorted(times)[int(len(times) * 0.95) - 1] * 1000.0 if len(times) >= 20 else max(times) * 1000.0
    verdict = "PASS (real-time feasible)" if p50 < budget_ms else "FAIL (cannot keep up live)"
    logger.warning("=" * 64)
    logger.warning(f"[BENCH] blocks={len(times)} pixel_frames/block={pixel_frames} "
                   f"budget={budget_ms:.1f}ms/block")
    logger.warning(f"[BENCH] per-block compute p50={p50:.1f}ms  p95={p95:.1f}ms")
    logger.warning(f"[BENCH] VERDICT: {verdict}")
    logger.warning("=" * 64)
    # Also drop a metrics file for the README acceptance table.
    try:
        with open(os.path.join(os.getcwd(), "bench_block.metrics"), "w") as fh:
            fh.write(f"blocks={len(times)}\nbudget_ms={budget_ms:.1f}\np50_ms={p50:.1f}\n"
                     f"p95_ms={p95:.1f}\nverdict={verdict}\n")
    except OSError:
        pass


def main():
    args = parse_args()
    _, _, _, _, _, parse_args_for_training_config = _lazy_imports()
    training_settings = parse_args_for_training_config(args.training_config)
    run(args, training_settings)


if __name__ == "__main__":
    main()
