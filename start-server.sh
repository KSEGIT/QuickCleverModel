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

# `bash -x` and SHELLOPTS=xtrace print every expansion, so a key would land in
# the trace. Every block that handles a key value turns tracing off first
# (xtrace_off) and puts it back after (xtrace_restore).
xtrace_was_on=0
xtrace_off() {
  case $- in *x*) xtrace_was_on=1 ;; esac
  set +x
}
xtrace_restore() {
  if [[ "$xtrace_was_on" == 1 ]]; then set -x; fi
  return 0
}

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
# Tracing stays off from here until the key is out of the environment: .env
# holds BONSAI_API_KEY and may hold other secrets, and xtrace would print them.
xtrace_off
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
    QCM_AGENT_CTX QCM_QWEN9_CTX QCM_QWEN4_CTX QCM_GRANITE_CTX \
    QCM_QWEN36_CTX QCM_GEMMA4_CTX QCM_AGENT_UBATCH QCM_QWEN36_FIT_TARGET; do
  validate_setting "$setting" '^[1-9][0-9]*$'
done
for setting in BONSAI_CTK BONSAI_CTV QCM_QWEN_CTK QCM_QWEN_CTV QCM_GRANITE_CTK QCM_GRANITE_CTV \
    QCM_QWEN36_CTK QCM_QWEN36_CTV QCM_GEMMA4_CTK QCM_GEMMA4_CTV; do
  validate_setting "$setting" '^(f32|f16|bf16|q8_0|q4_0|q4_1|iq4_nl|q5_0|q5_1)$'
done
for setting in BONSAI_FA QCM_QWEN_FA QCM_GRANITE_FA QCM_QWEN_REASONING \
    QCM_QWEN36_FA QCM_GEMMA4_FA QCM_QWEN36_REASONING; do
  validate_setting "$setting" '^(on|off|auto)$'
done
for setting in BONSAI_NGL QCM_QWEN9_NGL QCM_QWEN4_NGL QCM_GRANITE_NGL \
    QCM_QWEN36_NGL QCM_GEMMA4_NGL BONSAI_REASONING_BUDGET \
    QCM_QWEN_REASONING_BUDGET QCM_QWEN36_REASONING_BUDGET; do
  validate_setting "$setting" '^(-1|[0-9]+)$'
done
case "${QCM_QWEN36_QUANT:-Q4_K_XL}" in
  Q4_K_XL) QWEN36_FILE=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf ;;
  Q4_K_M) QWEN36_FILE=Qwen3.6-35B-A3B-UD-Q4_K_M.gguf ;;
  IQ4_XS) QWEN36_FILE=Qwen3.6-35B-A3B-UD-IQ4_XS.gguf ;;
  *) echo "invalid QCM_QWEN36_QUANT: $QCM_QWEN36_QUANT" >&2; exit 2 ;;
esac
case "${QCM_GEMMA4_QUANT:-QAT_Q4_0}" in
  QAT_Q4_0) GEMMA4_REL=gemma-4-E4B-it-qat-q4_0-gguf/gemma-4-E4B_q4_0-it.gguf ;;
  Q5_K_M) GEMMA4_REL=gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q5_K_M.gguf ;;
  *) echo "invalid QCM_GEMMA4_QUANT: $QCM_GEMMA4_QUANT" >&2; exit 2 ;;
esac
for setting in BONSAI_CACHE_RAM BONSAI_CACHE_REUSE QCM_AGENT_CACHE_RAM QCM_AGENT_CACHE_REUSE; do
  validate_setting "$setting" '^[0-9]+$'
done

# 0.0.0.0 is REQUIRED: binding 127.0.0.1 makes the server unreachable from the
# Docker VM. Everything is therefore also reachable from your LAN, so the API
# key below is not optional hygiene. Children bind 127.0.0.1; the key file step
# at the end says how they treat the key.
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
# The keys reach llama-server through --api-key-file (below), which reads one
# key per line and skips lines that start with '#'. BONSAI_API_KEY keeps the
# --api-key meaning: a comma separates several keys, and each one goes on its
# own line. Reject what the file format would change. A '#' key is the
# dangerous one: the server skips it, and with no other key it runs with no
# auth. --api-key also read double quotes as CSV quoting, which the file does
# not, so no key may contain one. Never print a key.
key_error() { echo "invalid BONSAI_API_KEY: $1" >&2; exit 2; }
case "$BONSAI_API_KEY" in
  *$'\n'*|*$'\r'*) key_error "it must not contain a line break" ;;
  *'"'*) key_error "a key must not contain a double quote (\")" ;;
  ,*|*,|*,,*) key_error "a comma-separated key is empty" ;;
esac
IFS=',' read -r -a api_keys <<< "$BONSAI_API_KEY"
for api_key in "${api_keys[@]}"; do
  [[ "$api_key" != '#'* ]] || key_error "a key must not start with '#'"
done
unset api_key

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
    -e "s|@QWEN36_CTX@|${QCM_QWEN36_CTX:-${QCM_AGENT_CTX:-8192}}|g" \
    -e "s|@GEMMA4_CTX@|${QCM_GEMMA4_CTX:-${QCM_AGENT_CTX:-8192}}|g" \
    -e "s|@QWEN9_NGL@|${QCM_QWEN9_NGL:-99}|g" \
    -e "s|@QWEN4_NGL@|${QCM_QWEN4_NGL:-99}|g" \
    -e "s|@GRANITE_NGL@|${QCM_GRANITE_NGL:-99}|g" \
    -e "s|@QWEN36_NGL@|${QCM_QWEN36_NGL:-auto}|g" \
    -e "s|@GEMMA4_NGL@|${QCM_GEMMA4_NGL:-99}|g" \
    -e "s|@QWEN36_FIT_TARGET@|${QCM_QWEN36_FIT_TARGET:-1536}|g" \
    -e "s|@QWEN36_MODEL@|$ROOT_SED/models/Qwen3.6-35B-A3B-GGUF/$QWEN36_FILE|g" \
    -e "s|@GEMMA4_MODEL@|$ROOT_SED/models/$GEMMA4_REL|g" \
    -e "s|@QWEN_FA@|${QCM_QWEN_FA:-auto}|g" \
    -e "s|@GRANITE_FA@|${QCM_GRANITE_FA:-auto}|g" \
    -e "s|@QWEN36_FA@|${QCM_QWEN36_FA:-auto}|g" \
    -e "s|@GEMMA4_FA@|${QCM_GEMMA4_FA:-auto}|g" \
    -e "s|@QWEN_CTK@|${QCM_QWEN_CTK:-f16}|g" \
    -e "s|@QWEN_CTV@|${QCM_QWEN_CTV:-f16}|g" \
    -e "s|@GRANITE_CTK@|${QCM_GRANITE_CTK:-f16}|g" \
    -e "s|@GRANITE_CTV@|${QCM_GRANITE_CTV:-f16}|g" \
    -e "s|@QWEN36_CTK@|${QCM_QWEN36_CTK:-f16}|g" \
    -e "s|@QWEN36_CTV@|${QCM_QWEN36_CTV:-f16}|g" \
    -e "s|@GEMMA4_CTK@|${QCM_GEMMA4_CTK:-f16}|g" \
    -e "s|@GEMMA4_CTV@|${QCM_GEMMA4_CTV:-f16}|g" \
    -e "s|@AGENT_UBATCH@|${QCM_AGENT_UBATCH:-256}|g" \
    -e "s|@AGENT_CACHE_RAM@|${QCM_AGENT_CACHE_RAM:-1024}|g" \
    -e "s|@AGENT_CACHE_REUSE@|${QCM_AGENT_CACHE_REUSE:-256}|g" \
    -e "s|@QWEN_REASONING@|${QCM_QWEN_REASONING:-off}|g" \
    -e "s|@QWEN_REASONING_BUDGET@|${QCM_QWEN_REASONING_BUDGET:--1}|g" \
    -e "s|@QWEN36_REASONING@|${QCM_QWEN36_REASONING:-off}|g" \
    -e "s|@QWEN36_REASONING_BUDGET@|${QCM_QWEN36_REASONING_BUDGET:--1}|g" \
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
qwen9=false; qwen4=false; granite=false; qwen36=false; gemma4=false
[[ -f "$ROOT/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf" ]] && qwen9=true
[[ -f "$ROOT/models/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf" ]] && qwen4=true
[[ -f "$ROOT/models/granite-4.1-8b-GGUF/granite-4.1-8b-Q4_K_M.gguf" ]] && granite=true
[[ -f "$ROOT/models/Qwen3.6-35B-A3B-GGUF/$QWEN36_FILE" ]] && qwen36=true
[[ -f "$ROOT/models/$GEMMA4_REL" ]] && gemma4=true
if [[ "$qwen36" == true ]]; then
  ram_kib=""
  if [[ -r /proc/meminfo ]]; then
    ram_kib="$(awk '$1 == "MemTotal:" { print $2 }' /proc/meminfo)"
    available_kib="$(awk '$1 == "MemAvailable:" { print $2 }' /proc/meminfo)"
    if [[ -n "$available_kib" && "$available_kib" -lt 29360128 ]]; then
      echo "WARNING: Qwen3.6 hybrid inference has under 28 GiB available RAM; expect load failure or swapping." >&2
    fi
  elif command -v sysctl >/dev/null; then
    ram_bytes="$(sysctl -n hw.memsize 2>/dev/null || true)"
    if [[ "$ram_bytes" =~ ^[0-9]+$ ]]; then ram_kib=$((ram_bytes / 1024)); fi
  fi
  if [[ "$ram_kib" =~ ^[0-9]+$ && "$ram_kib" -lt 33554432 ]]; then
    echo "WARNING: Qwen3.6 hybrid inference has under 32 GiB physical RAM; 64 GiB is preferred." >&2
  fi
fi
# QCM_MODELS: the only presets the router may serve, as a comma list of
# section names. Empty serves every preset whose weights exist. Every other
# section is dropped from the rendered file (awk below), so the router cannot
# list or load it and a request for it cannot evict the main model. A listed
# name that is not in the preset, or whose model file is missing, stops the
# start. Spaces around a name are fine; an empty item, or a space or line
# break inside a name, is an error, not something to guess at. The preset
# lookup accepts CRLF lines and `model=/path` without spaces, as llama-server
# does.
allowed=""
allowed_models=()
models_shown="all presets with weights"
trim() { local s="${1#"${1%%[![:space:]]*}"}"; printf '%s' "${s%"${s##*[![:space:]]}"}"; }
models_error() { echo "invalid QCM_MODELS: $1" >&2; exit 2; }
if [[ -n "${QCM_MODELS:-}" ]]; then
  case "$QCM_MODELS" in
    *$'\n'*|*$'\r'*) models_error "it must not contain a line break" ;;
  esac
  IFS=',' read -r -a requested <<< "$QCM_MODELS,"
  for name in ${requested[@]+"${requested[@]}"}; do
    name="$(trim "$name")"
    [[ -n "$name" ]] || models_error "a comma-separated item is empty"
    [[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
      models_error "entry '$name' is not a preset ID; separate IDs with commas"
    # Prints nothing if there is no [name] section. Else "found", then one
    # line per model line in every [name] section (a name may appear twice;
    # the section filter below keeps all of them, so all must be checked).
    # Text after the closing ] (a comment) is ignored.
    info="$(awk -v want="$name" '
      { sub(/\r$/, "") }
      /^\[/ { inside = substr($0, 2, index($0, "]") - 2) == want
              if (inside) found = 1
              next }
      inside && /^[ \t]*model[ \t]*=/ {
        sub(/^[ \t]*model[ \t]*=[ \t]*/, ""); sub(/[ \t]+$/, ""); models[++n] = ($0 == "" ? "<empty>" : $0) }
      END { if (found) { print "found"; for (i = 1; i <= n; i++) print models[i] } }
    ' "$PRESET")"
    [[ -n "$info" ]] || models_error "$name is not in the preset"
    # An empty `model =` value is printed as <empty>, because command
    # substitution would drop a trailing empty line.
    if [[ "$info" == found ]]; then model_lines=""; else model_lines="${info#found?}"; fi
    [[ -n "$model_lines" ]] ||
      models_error "$name names weights that are missing: no model line"
    while IFS= read -r model; do
      [[ "$model" != "<empty>" && -f "$model" ]] ||
        models_error "$name names weights that are missing: ${model/<empty>/no model line}"
    done <<< "$model_lines"
    allowed_models+=("$name")
  done
  [[ ${#allowed_models[@]} -gt 0 ]] || models_error "it lists no model"
  allowed=",$(IFS=','; printf '%s' "${allowed_models[*]}"),"
  models_shown="$(IFS=','; printf '%s' "${allowed_models[*]}")"
fi

awk -v qwen9="$qwen9" -v qwen4="$qwen4" -v granite="$granite" \
    -v qwen36="$qwen36" -v gemma4="$gemma4" -v allowed="$allowed" '
  /^\[/ { show = 1 }
  /^\[qwen3\.5-9b-q4_k_m\]/ { show = qwen9 == "true" }
  /^\[qwen3\.5-4b-q4_k_m\]/ { show = qwen4 == "true" }
  /^\[granite-4\.1-8b-q4_k_m\]/ { show = granite == "true" }
  /^\[qwen3\.6-35b-a3b\]/ { show = qwen36 == "true" }
  /^\[gemma4-e4b\]/ { show = gemma4 == "true" }
  /^\[/ && allowed != "" {
    name = substr($0, 2, index($0, "]") - 2)
    if (name != "*" && index(allowed, "," name ",") == 0) show = 0
  }
  /^ngl = auto$/ { next }
  show { print }
' "$PRESET" > "$PRESET.available"
mv "$PRESET.available" "$PRESET"

if [[ -n "$allowed" ]]; then
  # The router also lists and loads models from the llama.cpp download cache
  # (load_from_cache). LLAMA_CACHE is the first place Prism 922be44 looks
  # (common/hf-cache.cpp), so point it at an empty directory of our own,
  # emptied on every start. It overrides any LLAMA_CACHE from .env.
  export LLAMA_CACHE="$ROOT/run/empty-model-cache"
  rm -rf -- "$LLAMA_CACHE"
  mkdir -p "$LLAMA_CACHE"

  grep -Eq '^load-on-startup[[:space:]]*=[[:space:]]*true' "$PRESET" ||
    echo "WARNING: no model in QCM_MODELS has load-on-startup = true; nothing loads until the first request." >&2
  # DEFAULT_MODELS is Open WebUI's pre-selected model list. Warn only: it may
  # name a model on another connection.
  IFS=',' read -r -a default_models <<< "${DEFAULT_MODELS:-}"
  for name in ${default_models[@]+"${default_models[@]}"}; do
    name="$(trim "$name")"
    [[ -z "$name" || "$allowed" == *",$name,"* ]] ||
      echo "WARNING: DEFAULT_MODELS names $name, which QCM_MODELS does not serve here." >&2
  done
fi

# The keys go to the server in a file, never on its command line: argv is
# readable by every local user (ps, /proc/<pid>/cmdline), in a container too.
# printf is a bash builtin, so no process ever carries a key in argv. mktemp
# makes a new 0600 file; it is made 0400 and then renamed into place in one
# step, so a reader never sees a half-written file. The key leaves the
# environment before exec, so the server and its children do not inherit it.
#
# Router mode, Prism 922be44: the router strips --api-key from the children it
# spawns, but not --api-key-file (upstream fixed this later, 982a3329af).
# The children (127.0.0.1 only) therefore get the file path, not the key, and
# check the key too, reading the file when each one starts. Proxied requests
# keep their Authorization header, so clients see no change.
#
# One file per server: run/api-key.<pid>. $$ stays the server's PID after
# exec, so a second run of this script writes its own file and never changes
# the keys a running server's children read. Files of servers that are gone
# are removed. Only "No such process" counts as gone: a PID owned by another
# user answers "Operation not permitted" and its file stays.
#
# run/ MUST NOT be shared between containers, or between a container and the
# host. "Is that PID alive?" is asked in this script's own PID namespace. In
# another namespace the same number is another process (or none), so a live
# server's key file would look gone and be removed, and its children could no
# longer read the key when they start. One run/ per PID namespace.
#
# Also removed at start, and only if this script owns them (a regular file
# owned by this user; a symlink is never followed or removed):
#  - run/.api-key-tmp.*  left by a start that died between mktemp and mv. Only
#    those older than a minute: a younger one may belong to a start that is
#    running right now.
#  - run/api-key  the single key file of the older launcher.
remove_own_file() {
  if [[ -f "$1" && ! -L "$1" && -O "$1" ]]; then rm -f -- "$1"; fi
  return 0
}
xtrace_off
for old_key_file in "$ROOT"/run/api-key.*; do
  old_pid="${old_key_file##*/api-key.}"
  [[ "$old_pid" =~ ^[0-9]+$ && "$old_pid" != "$$" ]] || continue
  if [[ "$(LC_ALL=C kill -0 "$old_pid" 2>&1)" == *"No such process"* ]]; then
    remove_own_file "$old_key_file"
  fi
done
unset old_key_file old_pid
remove_own_file "$ROOT/run/api-key"
find "$ROOT/run" -maxdepth 1 -type f -name '.api-key-tmp.*' -user "$(id -u)" \
     -mmin +1 -exec rm -f -- {} + ||
  echo "WARNING: could not remove old temporary key files in $ROOT/run" >&2
KEY_FILE="$ROOT/run/api-key.$$"
key_tmp="$(umask 077 && mktemp "$ROOT/run/.api-key-tmp.XXXXXX")" || {
  echo "cannot create a temporary key file in $ROOT/run" >&2
  exit 1
}
if ! printf '%s\n' "${api_keys[@]}" > "$key_tmp" \
   || ! chmod 400 "$key_tmp" || ! mv -f -- "$key_tmp" "$KEY_FILE"; then
  rm -f -- "$key_tmp"
  echo "cannot write the key file $KEY_FILE" >&2
  exit 1
fi
unset BONSAI_API_KEY BONSAI_SERVER_KEY api_keys
xtrace_restore

echo "server : $SERVER (router)"
echo "preset : $PRESET"
echo "models : $models_shown"
echo "listen : http://$HOST:$PORT  (ctx=$CTX, text-ctx=$CTX_TEXT, models-max=$MODELS_MAX)"

# No -m: that is what selects router mode. No --mmproj either — it would be
# discarded here and must live in the preset sections.
exec "$SERVER" \
  --models-preset "$PRESET" \
  --models-max "$MODELS_MAX" \
  --host "$HOST" --port "$PORT" \
  --api-key-file "$KEY_FILE" \
  "$@"
