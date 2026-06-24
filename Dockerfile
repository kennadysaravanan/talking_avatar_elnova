# Real-Time 2D Talking Avatar — pod image.
# Target: RunPod 8xH100 80GB SXM (we use 5 GPUs). Driver is CUDA 12.8; we install torch built
# for cu128. See environment.md for the rationale behind every pin + Known Issues / Fixes.
#
# NOTE: model weights are NOT baked in. Mount ckpt/ on a persistent RunPod volume and run
# scripts/download_models.sh once. (~60GB+; re-downloading per pod start wastes money.)

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- system deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip git wget curl ffmpeg \
        build-essential ninja-build \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- PyTorch (cu128 wheels work on the pod's 12.8 driver) ---
RUN pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 \
 && pip install torchaudio --index-url https://download.pytorch.org/whl/cu128

# --- FlashAttention 3 (Hopper / H100 path). Fallback documented in environment.md. ---
RUN pip install flash_attn_3 \
        --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch280 \
        --extra-index-url https://download.pytorch.org/whl/cu128

# --- application + engine deps ---
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# --- project source (engine subtree is vendored under avatar/liveavatar) ---
COPY . /app

# Worker (torchrun) and orchestrator (uvicorn) are started by scripts, not baked as CMD,
# because they are separate processes. See README "RunPod run steps".
#   GPU worker:    bash scripts/run_worker.sh
#   orchestrator:  uvicorn server.main:app --host 0.0.0.0 --port 8080
EXPOSE 8080
CMD ["bash", "-lc", "echo 'See README: run scripts/run_worker.sh and uvicorn server.main:app'"]
