#!/usr/bin/env bash
# Control the benchmark llama-server on the RTX worker over SSH.
#
#   WORKER_SSH=user@host tests/bench/worker.sh up      # stop live, start test server
#   WORKER_SSH=user@host tests/bench/worker.sh down    # stop test server, restore live
#   WORKER_SSH=user@host tests/bench/worker.sh vram    # used GPU memory, MiB
#   WORKER_SSH=user@host tests/bench/worker.sh health  # test server /health JSON
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
set -euo pipefail

LIVE_CONTAINER="${LIVE_CONTAINER:-bonsai-llama-1}"
LIVE_HEALTH_URL="${LIVE_HEALTH_URL:-http://127.0.0.1:8080/health}"
LIVE_ENV_FILE="${LIVE_ENV_FILE:-}"
TEST_CONTAINER="${TEST_CONTAINER:-qcm-bench}"
TEST_IMAGE="${TEST_IMAGE:-qcm-rtx-validation:922be44}"
TEST_PORT="${TEST_PORT:-18080}"
CTX="${CTX:-32768}"
PARALLEL="${PARALLEL:-1}"
# Tunables, mainly so tests do not wait for real timeouts.
UP_TIMEOUT="${UP_TIMEOUT:-180}"
DOWN_TIMEOUT="${DOWN_TIMEOUT:-120}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"
START_ATTEMPTS="${START_ATTEMPTS:-5}"
RETRY_SLEEP="${RETRY_SLEEP:-3}"

die() { echo "worker.sh: $*" >&2; exit 1; }

usage() {
  echo "usage: WORKER_SSH=user@host $0 up|down|vram|health" >&2
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

cmd_up() {
  need MODELS_DIR TEMPLATE_FILE MODEL_FILE MODEL_ALIAS
  is_int "$CTX" || die "CTX must be an integer, got '$CTX'"
  is_int "$PARALLEL" || die "PARALLEL must be an integer, got '$PARALLEL'"
  is_int "$TEST_PORT" || die "TEST_PORT must be an integer, got '$TEST_PORT'"

  echo "worker.sh: stopping live container $LIVE_CONTAINER" >&2
  remote "docker stop $(q "$LIVE_CONTAINER") >/dev/null 2>&1 || true"

  # A second `up` (phase 2, other slot count) replaces the running test
  # container instead of failing on the name clash.
  echo "worker.sh: starting $TEST_CONTAINER (ctx $CTX, parallel $PARALLEL, $MODEL_ALIAS)" >&2
  remote "docker rm -f $(q "$TEST_CONTAINER") >/dev/null 2>&1 || true; \
docker run -d --rm --gpus all --name $(q "$TEST_CONTAINER") \
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

  wait_for "$UP_TIMEOUT" test_ready || die "test server did not become healthy"
  echo "worker.sh: test server healthy on worker 127.0.0.1:$TEST_PORT" >&2
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
    *) usage ;;
  esac
}

main "$@"
