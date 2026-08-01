#!/usr/bin/env bash
# Native Metal-accelerated llama-server for Bonsai 27B.
#
# Docker on macOS CANNOT access Metal (Hypervisor.framework exposes no virtual
# GPU to Linux guests), so inference runs here on the host and containers talk
# to it over HTTP. See docker/compose.yaml for the container side.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$ROOT/models/Ternary-Bonsai-27B-gguf"

# Prefer the source build (Metal 4 tensor API enabled) over the prebuilt.
# NOTE: Homebrew's llama.cpp is NOT usable here — upstream cannot read the
# fork's ternary Q2_0 block layout ("tensor 'output_norm.weight' has offset ...").
if [[ -x "$ROOT/src/llama.cpp-prism/build/bin/llama-server" ]]; then
  SERVER="$ROOT/src/llama.cpp-prism/build/bin/llama-server"
else
  SERVER="$ROOT/bin/llama-prism-b9570-0ad1dab/llama-server"
fi

# 0.0.0.0 is REQUIRED: binding 127.0.0.1 makes the server unreachable from the
# Docker VM. Everything is therefore also reachable from your LAN, so the API
# key below is not optional hygiene.
HOST="${BONSAI_HOST:-0.0.0.0}"
PORT="${BONSAI_PORT:-8080}"
CTX="${BONSAI_CTX:-32768}"   # model trains to 262144; that KV cache will not fit in 24 GB

[[ -f "$ROOT/.env" ]] && set -a && . "$ROOT/.env" && set +a
: "${BONSAI_API_KEY:?BONSAI_API_KEY not set — create .env with: BONSAI_API_KEY=bonsai-\$(openssl rand -hex 20)}"

echo "server : $SERVER"
echo "model  : $MODEL_DIR/Ternary-Bonsai-27B-Q2_0.gguf"
echo "listen : http://$HOST:$PORT  (ctx=$CTX, all layers on Metal)"

# Performance flags, all measured (see README "Tuning"):
#   --parallel 1  : n_parallel defaulted to 4, so Open WebUI's background title/tag
#                   requests decoded CONCURRENTLY with the real chat and split
#                   memory bandwidth 4 ways. Single-user box: give one request
#                   everything. This was the dominant cost (4.8 t/s observed).
#   -fa on        : explicit flash attention (default 'auto'); cheaper attention
#                   at long context and a prerequisite for KV quantization.
#   -ctk/-ctv q8_0: halves KV cache memory, negligible quality loss. Only safe
#                   WITH flash attention — without it llama.cpp dequantizes the
#                   cache every attention step and ends up slower.
exec "$SERVER" \
  -m "$MODEL_DIR/Ternary-Bonsai-27B-Q2_0.gguf" \
  --mmproj "$MODEL_DIR/Ternary-Bonsai-27B-mmproj-Q8_0.gguf" \
  -ngl 99 \
  -c "$CTX" \
  --parallel "${BONSAI_PARALLEL:-1}" \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  --host "$HOST" --port "$PORT" \
  --api-key "$BONSAI_API_KEY" \
  --jinja \
  --temp 0.5 --top-p 0.85 --top-k 20 --min-p 0 \
  "$@"
