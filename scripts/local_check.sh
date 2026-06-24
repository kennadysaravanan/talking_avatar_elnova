#!/usr/bin/env bash
# Off-GPU local sanity checks (no model, no CUDA). Safe to run on any dev machine.
#   1) byte-compile all OUR Python (skips the vendored engine, which needs CUDA deps)
#   2) run the ZeroMQ orchestrator<->worker wire-protocol loopback test in a throwaway venv
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"

echo "== 1/2  byte-compile our modules =="
$PY -m py_compile \
  avatar/avatar_worker.py avatar/ipc_worker.py avatar/idle.py avatar/blend.py \
  server/main.py server/ipc.py server/session.py server/turn_manager.py server/transport_livekit.py \
  server/providers/base.py server/providers/llm_openai.py server/providers/tts_openai.py \
  server/providers/stt_whisper.py tests/test_ipc_loopback.py
echo "   OK: all modules compile"

echo "== 2/2  ZMQ loopback test (audio / flush / frames) =="
VENV=".venv-localcheck"
if [ ! -d "$VENV" ]; then
  $PY -m venv "$VENV"
  # shellcheck disable=SC1091
  . "$VENV/bin/activate"
  pip install -q --upgrade pip >/dev/null
  pip install -q pyzmq numpy >/dev/null
else
  # shellcheck disable=SC1091
  . "$VENV/bin/activate"
fi
python tests/test_ipc_loopback.py
deactivate 2>/dev/null || true

echo
echo "All local (off-GPU) checks passed. GPU acceptance tests run on the pod:"
echo "  bash scripts/bench_block.sh   # real-time gate FIRST (<480 ms/block)"
echo "  see scripts/acceptance/README.md"
