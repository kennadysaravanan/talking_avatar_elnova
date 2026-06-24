#!/usr/bin/env bash
# CONSTRAINT #1 — PROVE REAL-TIME BEFORE BUILDING/RUNNING THE FULL STACK.
#
# 12 pixel frames/block @ 25fps = 480ms of video per block. The 4-step denoise + VAE decode
# for one block MUST finish in < 480ms or live streaming is impossible. The 45 FPS figure was
# H800 + compile + fp8; H100 should match but is NOT guaranteed.
#
# This runs the SAME model-load path as the real worker (avatar_worker.py --bench_blocks N),
# measures per-block compute time (skipping the first few compile-warmup blocks), and prints
# p50/p95 plus a PASS/FAIL verdict against the 480ms budget. If p50 > 480ms: STOP — the
# real-time premise has failed and there is no point building the rest.
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
export ENABLE_COMPILE="${ENABLE_COMPILE:-true}"
N="${BENCH_BLOCKS:-50}"
REF_IMAGE="${REF_IMAGE:-avatar/liveavatar_examples/ref.jpg}"

torchrun --nproc_per_node=5 --master_port="${MASTER_PORT:-29103}" \
  avatar/avatar_worker.py \
  --task s2v-14B --size "720*400" \
  --ckpt_dir "${CKPT_DIR:-ckpt/Wan2.2-S2V-14B/}" \
  --training_config "${TRAINING_CONFIG:-avatar/liveavatar/configs/s2v_causal_sft.yaml}" \
  --image "${REF_IMAGE}" \
  --infer_frames 48 --sample_steps 4 --sample_guide_scale 0 --sample_solver euler \
  --num_gpus_dit 4 --enable_vae_parallel --enable_online_decode \
  --load_lora --lora_path_dmd "${LORA_DIR:-ckpt/LiveAvatar}" --convert_model_dtype --fp8 \
  --bench_blocks "${N}"

echo "---- bench result ----"
cat bench_block.metrics 2>/dev/null || echo "no metrics file produced"
