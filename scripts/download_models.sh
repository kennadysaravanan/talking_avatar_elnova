#!/usr/bin/env bash
# Download base model + LoRA onto the PERSISTENT RunPod volume (mount ckpt/ on the volume so
# this ~60GB+ download happens once, not on every pod start).
set -euo pipefail
cd "$(dirname "$0")/.."

# Uncomment if HF is slow/blocked from the pod region:
# export HF_ENDPOINT=https://hf-mirror.com

mkdir -p ckpt
huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir ./ckpt/Wan2.2-S2V-14B
huggingface-cli download Quark-Vision/Live-Avatar --local-dir ./ckpt/LiveAvatar

echo "Done. ckpt/Wan2.2-S2V-14B and ckpt/LiveAvatar ready."
