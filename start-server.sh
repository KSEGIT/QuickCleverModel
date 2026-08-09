#!/usr/bin/env bash
# Native Metal-accelerated llama-server for Bonsai 27B, in ROUTER mode.
#
# Docker on macOS CANNOT access Metal (Hypervisor.framework exposes no virtual
# GPU to Linux guests), so inference runs here on the host and containers talk
# to it over HTTP. See docker/compose.yaml for the container side.
#
# Router mode: launching WITHOUT -m makes this process a router that spawns one
# child llama-server per model in models.ini (server.cpp:94). Both quantizations
# then appear in /v1/models and in the Open WebUI dropdown, switchable without a
# restart.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Prefer the source build (Metal 4 tensor API enabled) over the prebuilt.
# NOTE: Homebrew's llama.cpp is NOT usable here — upstream cannot read the
# fork's ternary Q2_0 block layout ("tensor 'output_norm.weight' has offset ...").
if [[ -x "$ROOT/src/llama.cpp-prism/build/bin/llama-server" ]]; then
  SERVER="$ROOT/src/llama.cpp-prism/build/bin/llama-server"
elif [[ -x "$ROOT/bin/llama-prism-b9570-0ad1dab/llama-server" ]]; then
  SERVER="$ROOT/bin/llama-prism-b9570-0ad1dab/llama-server"
else
  echo "no llama-server binary found. Build the fork:" >&2
  echo "  git clone https://github.com/PrismML-Eng/llama.cpp src/llama.cpp-prism" >&2
  echo "  cmake -S src/llama.cpp-prism -B src/llama.cpp-prism/build -DCMAKE_BUILD_TYPE=Release" >&2
  echo "  cmake --build src/llama.cpp-prism/build --target llama-server -j" >&2
  exit 1
fi

# .env must be sourced BEFORE the defaults below — it is the one source of
# truth, so BONSAI_CTX and friends in .env have to win over the defaults.
[[ -f "$ROOT/.env" ]] && set -a && . "$ROOT/.env" && set +a

# 0.0.0.0 is REQUIRED: binding 127.0.0.1 makes the server unreachable from the
# Docker VM. Everything is therefore also reachable from your LAN, so the API
# key below is not optional hygiene. Children bind 127.0.0.1 and are stripped of
# the key by the router, so it is enforced exactly once, here.
HOST="${BONSAI_HOST:-0.0.0.0}"
PORT="${BONSAI_PORT:-8080}"
CTX="${BONSAI_CTX:-32768}"   # model trains to 262144; that KV cache will not fit in 24 GB

# Keep ONE model resident and swap on selection. Switching in the UI unloads
# the current child and loads the other from disk, measured at +4-5s because the
# mmap'ed weights stay warm in the page cache.
#
# BONSAI_MODELS_MAX=2 keeps both hot and makes switching instant. That was
# tested and WORKS -- both children loaded, served six alternating requests and
# logged no allocation failures, even with the machine already 12 GB into swap.
# One-at-a-time is the default only because it leaves more of the 17.8 GB Metal
# working set (llama.log: "MTL0 : Apple M5 (18186 MiB free)") for everything
# else on the box, not because two do not fit.
MODELS_MAX="${BONSAI_MODELS_MAX:-1}"

: "${BONSAI_API_KEY:?BONSAI_API_KEY not set — create .env with: BONSAI_API_KEY=bonsai-\$(openssl rand -hex 20)}"

# Render the preset. Paths must be absolute: the children are spawned by the
# router and inheriting CWD is fragile.
PRESET="$ROOT/run/models.ini"
mkdir -p "$ROOT/run"
sed -e "s|@ROOT@|$ROOT|g" \
    -e "s|@CTX@|$CTX|g" \
    -e "s|@NKVO@|${BONSAI_NO_KV_OFFLOAD:-false}|g" \
    -e "s|@PARALLEL@|${BONSAI_PARALLEL:-1}|g" \
    -e "s|@SPEC@|${BONSAI_SPEC:-ngram-simple}|g" \
    -e "s|@CACHE_REUSE@|${BONSAI_CACHE_REUSE:-256}|g" \
    -e "s|@CACHE_RAM@|${BONSAI_CACHE_RAM:-2048}|g" \
    "$ROOT/models.ini.in" > "$PRESET"

echo "server : $SERVER (router)"
echo "preset : $PRESET"
echo "listen : http://$HOST:$PORT  (ctx=$CTX, models-max=$MODELS_MAX)"

# No -m: that is what selects router mode. No --mmproj either — it would be
# discarded here and must live in the preset sections.
exec "$SERVER" \
  --models-preset "$PRESET" \
  --models-max "$MODELS_MAX" \
  --host "$HOST" --port "$PORT" \
  --api-key "$BONSAI_API_KEY" \
  "$@"
