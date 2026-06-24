#!/usr/bin/env bash
# Launch the resident avatar worker on 5 GPUs (4 DiT + 1 VAE).
# Mirrors upstream infinite_inference_multi_gpu.sh but points at OUR streaming driver
# (avatar/avatar_worker.py, which imports the _tpp_blockwise pipeline) instead of the
# offline upstream entry script.
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS=OFF
export ENABLE_COMPILE="${ENABLE_COMPILE:-true}"   # first run compiles (slow); subsequent runs fast
export IDLE_AUDIO_MODE="${IDLE_AUDIO_MODE:-silence}"  # constraint #3: silence-first

REF_IMAGE="${REF_IMAGE:-avatar/liveavatar_examples/ref.jpg}"

torchrun --nproc_per_node=5 --master_port="${MASTER_PORT:-29102}" \
  avatar/avatar_worker.py \
  --task s2v-14B \
  --size "720*400" \
  --ckpt_dir "${CKPT_DIR:-ckpt/Wan2.2-S2V-14B/}" \
  --training_config "${TRAINING_CONFIG:-avatar/liveavatar/configs/s2v_causal_sft.yaml}" \
  --image "${REF_IMAGE}" \
  --infer_frames 48 \
  --sample_steps 4 \
  --sample_guide_scale 0 \
  --sample_solver euler \
  --base_seed 420 \
  --num_gpus_dit 4 \
  --enable_vae_parallel \
  --enable_online_decode \
  --load_lora \
  --lora_path_dmd "${LORA_DIR:-ckpt/LiveAvatar/liveavatar.safetensors}" \
  --convert_model_dtype \
  --fp8
