#!/usr/bin/env bash
# Control the benchmark llama-server on the RTX worker over SSH.
#
#   WORKER_SSH=user@host tests/bench/worker.sh up      # stop live, start test server
#   WORKER_SSH=user@host tests/bench/worker.sh down    # stop test server, restore live
#   WORKER_SSH=user@host tests/bench/worker.sh vram    # used GPU memory, MiB
#   WORKER_SSH=user@host tests/bench/worker.sh health  # test server /health JSON
#   WORKER_SSH=user@host tests/bench/worker.sh ctx     # router mode: the context
#                                                      # size MODEL_ALIAS really got
#
# Two ways to start the test server (MODE):
#   router (default)  the image's own entrypoint (start-server.sh) in router
#                     mode, so every models.ini preset runs with its production
#                     settings. Needs MODELS_DIR (mounted at /app/models),
#                     MODEL_ALIAS (the preset name, warmed up after start) and
#                     BENCH_API_KEY (the test server's bearer key). CTX is
#                     optional: empty keeps the preset's own default; a value
#                     sets every preset ctx variable (QCM_AGENT_CTX,
#                     BONSAI_CTX, BONSAI_CTX_TEXT).
#   direct            one llama-server with --parallel, for the concurrency
#                     suite (presets fix parallel = 1). Needs MODELS_DIR,
#                     TEMPLATE_FILE, MODEL_FILE, MODEL_ALIAS; CTX defaults to
#                     32768, PARALLEL to 1.
#
# The GPU has room for one model, so `up` stops the live container first and
# `down` must always run afterwards: it brings the live container back and
# fails (exit 1) if its health check does not return 200 in time.
#
# The test server listens on the worker's loopback only. Reach it from the
# caller with:  ssh -f -N -L 18080:127.0.0.1:$TEST_PORT "$WORKER_SSH"
#
# Every docker/curl/nvidia-smi command runs ON THE WORKER, so paths such as
# MODELS_DIR, TEMPLATE_FILE and LIVE_ENV_FILE are worker paths.
#
# Secrets: the live server wants a bearer key. It is read from the
# BONSAI_API_KEY= line of LIVE_ENV_FILE on the worker and passed to curl on
# stdin (-H @-), so it never crosses the SSH link, never appears in argv on
# either side, and is never printed.
#
# BENCH_API_KEY (router mode) is a throwaway key made per run. It goes to
# curl on stdin too; `docker run` needs it as -e, so it does appear in that
# one remote argv. It guards a server that listens on the worker's loopback
# only, and it is never printed.
set -euo pipefail

LIVE_CONTAINER="${LIVE_CONTAINER:-bonsai-llama-1}"
LIVE_HEALTH_URL="${LIVE_HEALTH_URL:-http://127.0.0.1:8080/health}"
LIVE_ENV_FILE="${LIVE_ENV_FILE:-}"
TEST_CONTAINER="${TEST_CONTAINER:-qcm-bench}"
TEST_IMAGE="${TEST_IMAGE:-qcm-rtx-validation:922be44}"
TEST_PORT="${TEST_PORT:-18080}"
MODE="${MODE:-router}"
# Empty CTX means: router mode keeps the preset default; direct mode, 32768.
CTX="${CTX:-}"
PARALLEL="${PARALLEL:-1}"
# Tunables, mainly so tests do not wait for real timeouts.
UP_TIMEOUT="${UP_TIMEOUT:-180}"
# Loading a model from disk on the first request can take minutes.
WARM_TIMEOUT="${WARM_TIMEOUT:-600}"
DOWN_TIMEOUT="${DOWN_TIMEOUT:-120}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"
START_ATTEMPTS="${START_ATTEMPTS:-5}"
RETRY_SLEEP="${RETRY_SLEEP:-3}"

die() { echo "worker.sh: $*" >&2; exit 1; }

usage() {
  echo "usage: WORKER_SSH=user@host $0 up|down|vram|health|ctx" >&2
  exit 2
}

need() {
  local name
  for name in "$@"; do
    [[ -n "${!name:-}" ]] || die "$name is not set"
  done
}

is_int() { [[ "$1" =~ ^[0-9]+$ ]]; }

# Run one shell command string on the worker. BatchMode: never prompt for a
# password. StrictHostKeyChecking=yes: the host key must already be in
# known_hosts; an unknown or changed key is a hard failure.
remote() {
  ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15 \
    "$WORKER_SSH" "$1"
}

q() { printf '%q' "$1"; }

# Poll CHECK (a function name) until it succeeds or TIMEOUT seconds pass.
wait_for() {
  local timeout="$1" check="$2" deadline=$((SECONDS + $1))
  while (( SECONDS < deadline )); do
    if "$check"; then return 0; fi
    sleep "$POLL_INTERVAL"
  done
  "$check" || { echo "worker.sh: $check not ready after ${timeout}s" >&2; return 1; }
}

test_ready() {
  local out
  # Exit 3 from the remote side means the container is gone: with --rm a
  # crashed llama-server removes itself, so waiting longer cannot help.
  out="$(remote "docker inspect -f '{{.State.Running}}' $(q "$TEST_CONTAINER") >/dev/null 2>&1 || exit 3; curl -fsS --max-time 5 http://127.0.0.1:$(q "$TEST_PORT")/health" 2>/dev/null)" && return 0
  local rc=$?
  if (( rc == 3 )); then die "test container $TEST_CONTAINER exited during start-up"; fi
  [[ -n "$out" ]] && echo "worker.sh: waiting for test server: $out" >&2
  return 1
}

live_ready() {
  local code
  code="$(remote "$(live_health_script)" 2>/dev/null)" || true
  [[ "$code" == "200" ]]
}

# Worker-side script that prints the live health HTTP status code. The key
# is read and used on the worker only.
live_health_script() {
  local url envfile
  url="$(q "$LIVE_HEALTH_URL")"
  envfile="$(q "$LIVE_ENV_FILE")"
  cat <<EOF
key=''
f=$envfile
if [ -n "\$f" ] && [ -r "\$f" ]; then key=\$(sed -n 's/^BONSAI_API_KEY=//p' "\$f" | tail -n 1 | tr -d "\\"'"); fi
if [ -n "\$key" ]; then printf 'Authorization: Bearer %s\\n' "\$key" | curl -s -o /dev/null -w '%{http_code}' --max-time 5 -H @- $url; else curl -s -o /dev/null -w '%{http_code}' --max-time 5 $url; fi
EOF
}

# Print "Authorization: Bearer <key>" for curl -H @- on the worker. Through
# a pipe into ssh, so the key is never in an argv.
bench_auth() { printf 'Authorization: Bearer %s\n' "$BENCH_API_KEY"; }

# Router mode: ask MODEL_ALIAS for one token, so the router loads it now.
# An unknown preset fails here (the router answers 4xx), not mid-suite.
warm_model() {
  local body
  body="$(printf '{"model":"%s","messages":[{"role":"user","content":"ok"}],"max_tokens":1}' "$MODEL_ALIAS")"
  bench_auth | remote "curl -sf -o /dev/null -m $(q "$WARM_TIMEOUT") -H @- \
-H 'Content-Type: application/json' -d $(q "$body") \
http://127.0.0.1:$(q "$TEST_PORT")/v1/chat/completions"
}

# Validate everything before the first SSH call: a bad setting must not
# stop the live container.
check_up_settings() {
  is_int "$TEST_PORT" || die "TEST_PORT must be an integer, got '$TEST_PORT'"
  case "$MODE" in
    router)
      need MODELS_DIR MODEL_ALIAS BENCH_API_KEY
      [[ "$MODEL_ALIAS" =~ ^[A-Za-z0-9._-]+$ ]] || die "MODEL_ALIAS must be a preset name, got '$MODEL_ALIAS'"
      [[ -z "$CTX" ]] || is_int "$CTX" || die "CTX must be an integer, got '$CTX'"
      ;;
    direct)
      need MODELS_DIR TEMPLATE_FILE MODEL_FILE MODEL_ALIAS
      CTX="${CTX:-32768}"
      is_int "$CTX" || die "CTX must be an integer, got '$CTX'"
      is_int "$PARALLEL" || die "PARALLEL must be an integer, got '$PARALLEL'"
      ;;
    *) die "MODE must be router or direct, got '$MODE'" ;;
  esac
}

# The image's normal entrypoint: start-server.sh in router mode over the
# image's models.ini. HOST/PORT come from BONSAI_HOST/BONSAI_PORT.
router_run_cmd() {
  local ctx_env=""
  if [[ -n "$CTX" ]]; then
    ctx_env="-e QCM_AGENT_CTX=$(q "$CTX") -e BONSAI_CTX=$(q "$CTX") -e BONSAI_CTX_TEXT=$(q "$CTX") "
  fi
  printf '%s' "docker run -d --rm --gpus all --name $(q "$TEST_CONTAINER") \
-p 127.0.0.1:$(q "$TEST_PORT"):8080 \
-v $(q "$MODELS_DIR"):/app/models:ro \
-e BONSAI_API_KEY=$(q "$BENCH_API_KEY") -e BONSAI_HOST=0.0.0.0 -e BONSAI_PORT=8080 \
-e BONSAI_MODELS_MAX=1 ${ctx_env}$(q "$TEST_IMAGE") >/dev/null"
}

# One llama-server with its own slot count: the recorded server command.
direct_run_cmd() {
  printf '%s' "docker run -d --rm --gpus all --name $(q "$TEST_CONTAINER") \
-p 127.0.0.1:$(q "$TEST_PORT"):8080 \
-v $(q "$MODELS_DIR"):/models:ro \
-v $(q "$TEMPLATE_FILE"):/template.jinja:ro \
--entrypoint /app/src/llama.cpp-prism/build/bin/llama-server \
$(q "$TEST_IMAGE") \
--host 0.0.0.0 --port 8080 \
--model $(q "$MODEL_FILE") --alias $(q "$MODEL_ALIAS") \
--chat-template-file /template.jinja \
--ctx-size $(q "$CTX") --n-gpu-layers 99 --flash-attn auto \
--cache-type-k f16 --cache-type-v f16 --reasoning off \
--parallel $(q "$PARALLEL") >/dev/null"
}

cmd_up() {
  check_up_settings

  echo "worker.sh: stopping live container $LIVE_CONTAINER" >&2
  remote "docker stop $(q "$LIVE_CONTAINER") >/dev/null 2>&1 || true"

  # A second `up` (phase 2, other slot count) replaces the running test
  # container instead of failing on the name clash.
  local run_cmd
  if [[ "$MODE" == router ]]; then
    echo "worker.sh: starting $TEST_CONTAINER (router, ctx ${CTX:-preset default}, $MODEL_ALIAS)" >&2
    run_cmd="$(router_run_cmd)"
  else
    echo "worker.sh: starting $TEST_CONTAINER (ctx $CTX, parallel $PARALLEL, $MODEL_ALIAS)" >&2
    run_cmd="$(direct_run_cmd)"
  fi
  remote "docker rm -f $(q "$TEST_CONTAINER") >/dev/null 2>&1 || true; $run_cmd"

  wait_for "$UP_TIMEOUT" test_ready || die "test server did not become healthy"
  if [[ "$MODE" == router ]]; then
    echo "worker.sh: loading $MODEL_ALIAS" >&2
    warm_model || die "model $MODEL_ALIAS did not load (missing from the image's models.ini?)"
  fi
  echo "worker.sh: test server healthy on worker 127.0.0.1:$TEST_PORT" >&2
}

# Router mode: the context size the router gave MODEL_ALIAS, from the
# child's /props (default_generation_settings.n_ctx). Parsed here, on the
# caller, so the worker needs nothing beyond curl.
cmd_ctx() {
  need MODEL_ALIAS BENCH_API_KEY
  is_int "$TEST_PORT" || die "TEST_PORT must be an integer, got '$TEST_PORT'"
  [[ "$MODEL_ALIAS" =~ ^[A-Za-z0-9._-]+$ ]] || die "MODEL_ALIAS must be a preset name, got '$MODEL_ALIAS'"
  local props n_ctx
  props="$(bench_auth | remote "curl -sf --max-time 10 -H @- \
'http://127.0.0.1:$TEST_PORT/props?model=$MODEL_ALIAS'")" || die "could not read /props for $MODEL_ALIAS"
  n_ctx="$(printf '%s' "$props" | python3 -c 'import json, sys; print(json.load(sys.stdin)["default_generation_settings"]["n_ctx"])' 2>/dev/null)" \
    || die "no default_generation_settings.n_ctx in /props for $MODEL_ALIAS"
  if ! is_int "$n_ctx" || (( n_ctx == 0 )); then
    die "unexpected n_ctx for $MODEL_ALIAS: $n_ctx"
  fi
  echo "$n_ctx"
}

# The path that must not give up: production depends on it. Each attempt
# stops the test container (best effort: it may already be gone, or SSH may
# drop) and starts the live one. `docker start` on a running container is a
# no-op, so a retry after a lost reply is safe.
cmd_down() {
  local attempt started=0
  is_int "$START_ATTEMPTS" && (( START_ATTEMPTS > 0 )) || START_ATTEMPTS=5
  for (( attempt = 1; attempt <= START_ATTEMPTS; attempt++ )); do
    echo "worker.sh: stopping $TEST_CONTAINER, starting $LIVE_CONTAINER (attempt $attempt/$START_ATTEMPTS)" >&2
    remote "docker stop $(q "$TEST_CONTAINER") >/dev/null 2>&1 || true" \
      || echo "worker.sh: could not reach worker to stop $TEST_CONTAINER; continuing" >&2
    if remote "docker start $(q "$LIVE_CONTAINER") >/dev/null"; then
      started=1
      break
    fi
    if (( attempt < START_ATTEMPTS )); then sleep "$RETRY_SLEEP"; fi
  done
  (( started )) || die "could not start $LIVE_CONTAINER after $START_ATTEMPTS attempts"
  wait_for "$DOWN_TIMEOUT" live_ready || die "live container $LIVE_CONTAINER is not healthy (expected HTTP 200)"
  echo "worker.sh: live container healthy" >&2
}

cmd_vram() {
  local mib
  mib="$(remote "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1")"
  mib="${mib//[[:space:]]/}"
  is_int "$mib" || die "unexpected nvidia-smi output: $mib"
  echo "$mib"
}

cmd_health() {
  remote "curl -sS --max-time 5 http://127.0.0.1:$(q "$TEST_PORT")/health"
  echo
}

main() {
  [[ $# -eq 1 ]] || usage
  need WORKER_SSH
  case "$1" in
    up) cmd_up ;;
    down) cmd_down ;;
    vram) cmd_vram ;;
    health) cmd_health ;;
    ctx) cmd_ctx ;;
    *) usage ;;
  esac
}

main "$@"
