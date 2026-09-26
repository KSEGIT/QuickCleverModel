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

# Use the pinned Prism build. Historical prebuilts cannot read current PQ2_0.
if [[ -x "$ROOT/src/llama.cpp-prism/build/bin/llama-server" ]]; then
  SERVER="$ROOT/src/llama.cpp-prism/build/bin/llama-server"
else
  echo "no llama-server binary found. Build the fork:" >&2
  echo "  git clone https://github.com/PrismML-Eng/llama.cpp src/llama.cpp-prism" >&2
  echo "  git -C src/llama.cpp-prism checkout 922be44aa6ac81b46f092716351cddff1c1733a7" >&2
  echo "  cmake -S src/llama.cpp-prism -B src/llama.cpp-prism/build -DCMAKE_BUILD_TYPE=Release" >&2
  echo "  cmake --build src/llama.cpp-prism/build --target llama-server -j" >&2
  exit 1
fi

if [[ -f "$ROOT/runtime-versions.env" ]]; then
  EXPECTED_SHA="$(awk -F= '$1 == "PRISM_SHA" { print $2 }' "$ROOT/runtime-versions.env")"
  SERVER_VERSION="$("$SERVER" --version 2>&1)"
  [[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ && "$SERVER_VERSION" == *"${EXPECTED_SHA:0:7}"* ]] || {
    echo "llama-server does not match the pinned Prism revision $EXPECTED_SHA; rebuild before starting" >&2
    exit 1
  }
  printf 'Prism: %s\n' "$EXPECTED_SHA"
fi

if [[ -f "$ROOT/models/Ternary-Bonsai-27B-gguf/Ternary-Bonsai-27B-Q2_0.gguf" \
      && ! -f "$ROOT/models/Ternary-Bonsai-27B-gguf/Ternary-Bonsai-27B-PQ2_0.gguf" ]]; then
  echo "Legacy ternary file found. Run ./fetch-models.sh bonsai for official PQ2_0; keep Q2_0 for rollback." >&2
  exit 1
fi

# .env must be sourced BEFORE the defaults below — it is the one source of
# truth, so BONSAI_CTX and friends in .env have to win over the defaults.
[[ -f "$ROOT/.env" ]] && set -a && . "$ROOT/.env" && set +a

validate_setting() {
  local name="$1" pattern="$2" value
  value="${!name-}"
  [[ -z "$value" || "$value" =~ $pattern ]] || {
    echo "invalid $name: $value" >&2
    exit 2
  }
}
for setting in BONSAI_CTX BONSAI_CTX_TEXT BONSAI_PARALLEL BONSAI_UBATCH \
    QCM_AGENT_CTX QCM_QWEN9_CTX QCM_QWEN4_CTX QCM_GRANITE_CTX QCM_AGENT_UBATCH; do
  validate_setting "$setting" '^[1-9][0-9]*$'
done
for setting in BONSAI_CTK BONSAI_CTV QCM_QWEN_CTK QCM_QWEN_CTV QCM_GRANITE_CTK QCM_GRANITE_CTV; do
  validate_setting "$setting" '^(f32|f16|bf16|q8_0|q4_0|q4_1|iq4_nl|q5_0|q5_1)$'
done
for setting in BONSAI_FA QCM_QWEN_FA QCM_GRANITE_FA QCM_QWEN_REASONING; do
  validate_setting "$setting" '^(on|off|auto)$'
done
for setting in BONSAI_NGL QCM_QWEN9_NGL QCM_QWEN4_NGL QCM_GRANITE_NGL \
    BONSAI_REASONING_BUDGET QCM_QWEN_REASONING_BUDGET; do
  validate_setting "$setting" '^(-1|[0-9]+)$'
done
for setting in BONSAI_CACHE_RAM BONSAI_CACHE_REUSE QCM_AGENT_CACHE_RAM; do
  validate_setting "$setting" '^[0-9]+$'
done

# 0.0.0.0 is REQUIRED: binding 127.0.0.1 makes the server unreachable from the
# Docker VM. Everything is therefore also reachable from your LAN, so the API
# key below is not optional hygiene. Children bind 127.0.0.1 and are stripped of
# the key by the router, so it is enforced exactly once, here.
HOST="${BONSAI_HOST:-0.0.0.0}"
PORT="${BONSAI_PORT:-8080}"
CTX="${BONSAI_CTX:-8192}"   # conservative shared floor; raise only after measuring
# The text-only preset can usually afford more context than the vision ones:
# no mmproj, and it is the preset coding agents use, where long conversations
# are the norm. Defaults to CTX, so this changes nothing unless you set it.
CTX_TEXT="${BONSAI_CTX_TEXT:-$CTX}"

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
ROOT_SED="$(printf '%s' "$ROOT" | sed 's/[\\&|]/\\&/g')"
sed -e "s|@ROOT@|$ROOT_SED|g" \
    -e "s|@NGL@|${BONSAI_NGL:-99}|g" \
    -e "s|@FA@|${BONSAI_FA:-on}|g" \
    -e "s|@QWEN9_CTX@|${QCM_QWEN9_CTX:-${QCM_AGENT_CTX:-8192}}|g" \
    -e "s|@QWEN4_CTX@|${QCM_QWEN4_CTX:-${QCM_AGENT_CTX:-8192}}|g" \
    -e "s|@GRANITE_CTX@|${QCM_GRANITE_CTX:-${QCM_AGENT_CTX:-8192}}|g" \
    -e "s|@QWEN9_NGL@|${QCM_QWEN9_NGL:-99}|g" \
    -e "s|@QWEN4_NGL@|${QCM_QWEN4_NGL:-99}|g" \
    -e "s|@GRANITE_NGL@|${QCM_GRANITE_NGL:-99}|g" \
    -e "s|@QWEN_FA@|${QCM_QWEN_FA:-auto}|g" \
    -e "s|@GRANITE_FA@|${QCM_GRANITE_FA:-auto}|g" \
    -e "s|@QWEN_CTK@|${QCM_QWEN_CTK:-f16}|g" \
    -e "s|@QWEN_CTV@|${QCM_QWEN_CTV:-f16}|g" \
    -e "s|@GRANITE_CTK@|${QCM_GRANITE_CTK:-f16}|g" \
    -e "s|@GRANITE_CTV@|${QCM_GRANITE_CTV:-f16}|g" \
    -e "s|@AGENT_UBATCH@|${QCM_AGENT_UBATCH:-256}|g" \
    -e "s|@AGENT_CACHE_RAM@|${QCM_AGENT_CACHE_RAM:-1024}|g" \
    -e "s|@QWEN_REASONING@|${QCM_QWEN_REASONING:-off}|g" \
    -e "s|@QWEN_REASONING_BUDGET@|${QCM_QWEN_REASONING_BUDGET:--1}|g" \
    -e "s|@CTX@|$CTX|g" \
    -e "s|@CTX_TEXT@|$CTX_TEXT|g" \
    -e "s|@NKVO@|${BONSAI_NO_KV_OFFLOAD:-false}|g" \
    -e "s|@PARALLEL@|${BONSAI_PARALLEL:-1}|g" \
    -e "s|@SPEC@|${BONSAI_SPEC:-ngram-simple}|g" \
    -e "s|@CACHE_REUSE@|${BONSAI_CACHE_REUSE:-256}|g" \
    -e "s|@CACHE_RAM@|${BONSAI_CACHE_RAM:-2048}|g" \
    -e "s|@CTK@|${BONSAI_CTK:-q4_0}|g" \
    -e "s|@CTV@|${BONSAI_CTV:-q4_0}|g" \
    -e "s|@UBATCH@|${BONSAI_UBATCH:-512}|g" \
    -e "s|@REASONING_BUDGET@|${BONSAI_REASONING_BUDGET:-4096}|g" \
    -e "s|@NGRAM_MOD_N_MATCH@|${BONSAI_NGRAM_MOD_N_MATCH:-24}|g" \
    -e "s|@NGRAM_MOD_N_MIN@|${BONSAI_NGRAM_MOD_N_MIN:-48}|g" \
    -e "s|@NGRAM_MOD_N_MAX@|${BONSAI_NGRAM_MOD_N_MAX:-64}|g" \
    "$ROOT/models.ini.in" > "$PRESET"

# The three new downloads are optional on existing installations. Exclude
# missing agents so discovery and update.sh do not promise absent weights.
qwen9=false; qwen4=false; granite=false
[[ -f "$ROOT/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf" ]] && qwen9=true
[[ -f "$ROOT/models/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf" ]] && qwen4=true
[[ -f "$ROOT/models/granite-4.1-8b-GGUF/granite-4.1-8b-Q4_K_M.gguf" ]] && granite=true
awk -v qwen9="$qwen9" -v qwen4="$qwen4" -v granite="$granite" '
  /^\[/ { show = 1 }
  /^\[qwen3\.5-9b-q4_k_m\]/ { show = qwen9 == "true" }
  /^\[qwen3\.5-4b-q4_k_m\]/ { show = qwen4 == "true" }
  /^\[granite-4\.1-8b-q4_k_m\]/ { show = granite == "true" }
  show { print }
' "$PRESET" > "$PRESET.available"
mv "$PRESET.available" "$PRESET"

echo "server : $SERVER (router)"
echo "preset : $PRESET"
echo "listen : http://$HOST:$PORT  (ctx=$CTX, text-ctx=$CTX_TEXT, models-max=$MODELS_MAX)"

# No -m: that is what selects router mode. No --mmproj either — it would be
# discarded here and must live in the preset sections.
exec "$SERVER" \
  --models-preset "$PRESET" \
  --models-max "$MODELS_MAX" \
  --host "$HOST" --port "$PORT" \
  --api-key "$BONSAI_API_KEY" \
  "$@"
