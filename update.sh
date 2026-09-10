#!/usr/bin/env bash
# Update the Linux/Docker Bonsai stack, prove it still serves, roll back if not.
#
#   ./update.sh              update, smoke-test every model, roll back on failure
#   ./update.sh --check      report drift only; changes nothing; exit 1 if behind
#   ./update.sh --rollback   return to the last recorded good state
#
# Linux/Docker only. macOS runs everything natively — see stack.sh.
#
# Why this smoke-tests instead of trusting a health check: on 2026-09-10 the
# box reported "Up 29 hours (healthy)" and /health returned {"status":"ok"}
# while every single model failed to load. The repo had been moved with
# `mv QuickCleverModel/ ../firesandbrain1/` under a running stack, Docker
# recreated the missing bind-mount source as an empty directory, and
# /app/models held no weights. Container liveness says nothing about whether
# the thing can answer. Only a real generation does.
set -uo pipefail

# BONSAI_UPDATE_ROOT is set by the re-exec below, where this file has been
# copied to a temp dir and dirname no longer points at the repo.
ROOT="${BONSAI_UPDATE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
COMPOSE_FILE="$ROOT/docker/compose.linux.yaml"
ENV_FILE="$ROOT/.env"
STATE="$ROOT/run/update-state"
LLAMA_IMAGE="bonsai-llama:cuda"
ROLLBACK_IMAGE="bonsai-llama:rollback"
WEBUI_IMAGE="ghcr.io/open-webui/open-webui:main"
LLAMA_CONTAINER="bonsai-llama-1"

log()  { printf '%s  %s\n' "$(date -u +%H:%M:%S)" "$*"; }
warn() { printf '%s  WARN %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die()  { printf '%s  FAIL %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; exit 1; }

# --- pure helpers, unit-tested in tests/test_update.py -----------------------

models_present() {
  # True when at least one weight file is visible under $1. Guards the exact
  # 2026-09-10 failure: a bind-mount source that exists but is empty.
  local dir="${1:-}"
  [[ -n "$dir" && -d "$dir" ]] || return 1
  # -print -quit stops at the first hit; the tree is ~11 GB and the only
  # question is whether it holds any weights at all.
  [[ -n "$(find "$dir" -name '*.gguf' -type f -print -quit 2>/dev/null)" ]]
}

parse_model_ids() {
  # /v1/models on stdin -> one alias per line. The smoke test loads all of
  # them, because the router loads each alias as its own child process and
  # one working model proves nothing about the others.
  python3 -c 'import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(1)
for m in d.get("data", []):
    if m.get("id"):
        print(m["id"])'
}

smoke_ok() {
  # A /v1/chat/completions reply on stdin. Exit 0 only for a real generation.
  python3 -c 'import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(1)          # curl writes nothing when the connection fails
if not isinstance(d, dict) or "error" in d:
    sys.exit(1)
choices = d.get("choices") or []
if not choices:
    sys.exit(1)
msg = choices[0].get("message") or {}
# Bonsai is a thinking model: a short reply can hit max_tokens while still
# inside its reasoning block, arriving as content="" with the text in
# reasoning_content. That is a healthy generation, not a failure — scoring it
# as one would roll back a stack that works.
text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
sys.exit(0 if text.strip() else 1)'
}

# Sourced by the test suite to exercise the helpers above without running an
# update. Executing the script normally leaves BONSAI_UPDATE_LIB unset, the
# && short-circuits, and the `return` is never reached.
[[ -n "${BONSAI_UPDATE_LIB:-}" ]] && return 0

# --- run from a copy, never from the file we are about to rewrite -----------
#
# `git pull --ff-only` and `git reset --hard` both rewrite update.sh whenever
# an update touches this script. Bash reads a script lazily, by byte offset,
# so a file that changes underneath a running shell makes it resume at a stale
# offset and execute whatever bytes now sit there. That is worst during a
# rollback: the one code path whose whole job is to bring a broken stack back.
#
# Copying to a temp file and exec-ing that leaves bash reading a file nothing
# will touch. The copy deletes itself on exit.
if [[ -z "${BONSAI_UPDATE_REEXEC:-}" ]]; then
  copy="$(mktemp "${TMPDIR:-/tmp}/bonsai-update.XXXXXX")" || die "mktemp failed"
  cat "${BASH_SOURCE[0]}" > "$copy" || die "could not copy myself to $copy"
  chmod +x "$copy"
  export BONSAI_UPDATE_REEXEC=1
  export BONSAI_UPDATE_ROOT="$ROOT"
  exec "$copy" "$@"
fi

# Captured now: inside a function BASH_SOURCE[0] is the function's source, so
# reading it at trap time would delete the wrong path.
SELF="${BASH_SOURCE[0]}"
trap 'rm -f -- "$SELF"' EXIT

# --- everything below needs a real box ---------------------------------------

# .env carries BONSAI_API_KEY and the bind addresses. Exported because compose
# reads them for interpolation too.
set -a
if [[ -f "$ENV_FILE" ]]; then
  # .env is per-machine and never in the repo, so there is nothing to follow.
  # shellcheck source=/dev/null
  . "$ENV_FILE"
fi
set +a
API="http://${LLAMA_BIND:-127.0.0.1}:8080"

compose() { docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"; }

running_working_dir() {
  docker inspect "$LLAMA_CONTAINER" \
    --format '{{index .Config.Labels "com.docker.compose.project.working_dir"}}' \
    2>/dev/null
}

preflight() {
  command -v docker >/dev/null 2>&1 \
    || die "docker not found. This is the Linux/Docker path; macOS uses ./stack.sh"
  [[ -f "$COMPOSE_FILE" ]] || die "no compose file at $COMPOSE_FILE"
  [[ -f "$ENV_FILE" ]]     || die "no .env at $ENV_FILE"
  [[ -n "${BONSAI_API_KEY:-}" ]] || die "BONSAI_API_KEY is not set in $ENV_FILE"

  models_present "$ROOT/models" || die \
"no *.gguf under $ROOT/models

Refusing to restart into an empty bind mount. That is the 2026-09-10 failure
mode: the container comes up healthy and every model load fails. Either run
./fetch-models.sh, or check the repo was not moved while the stack was up."

  # A running stack pinned to a different directory means compose would mount
  # that old path, not this one. Restarting would recreate the outage.
  local live; live="$(running_working_dir)"
  if [[ -n "$live" && "$live" != "$ROOT/docker" ]]; then
    die \
"the running stack was started from a different checkout

  running:  $live
  this one: $ROOT/docker

Bring it down from here first:
  docker compose --env-file .env -f docker/compose.linux.yaml up -d --force-recreate"
  fi
}

wait_for_health() {
  # Condition polling, not a fixed sleep: a cold 27B load off a spinning disk
  # is minutes, a warm restart is seconds.
  local deadline=$((SECONDS + 180))
  while (( SECONDS < deadline )); do
    if curl -fsS -m 5 "$API/health" 2>/dev/null | grep -q '"ok"'; then
      return 0
    fi
    sleep 3
  done
  return 1
}

smoke() {
  # Load every alias and require a real generation from each. Slow by design:
  # a cold 27B load is the thing being tested.
  local ids id body failed=0
  ids="$(curl -fsS -m 15 -H "Authorization: Bearer $BONSAI_API_KEY" \
         "$API/v1/models" 2>/dev/null | parse_model_ids)"
  [[ -n "$ids" ]] || { warn "/v1/models returned nothing"; return 1; }

  while read -r id; do
    [[ -n "$id" ]] || continue
    log "  smoke: $id"
    body="$(curl -fsS -m 900 \
      -H "Authorization: Bearer $BONSAI_API_KEY" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"$id\",\"messages\":[{\"role\":\"user\",\"content\":\"say OK\"}],\"max_tokens\":8}" \
      "$API/v1/chat/completions" 2>/dev/null)"
    if printf '%s' "$body" | smoke_ok; then
      log "  smoke: $id OK"
    else
      warn "smoke: $id FAILED — ${body:0:200}"
      failed=1
    fi
  done <<< "$ids"
  return "$failed"
}

record_rollback_point() {
  mkdir -p "$(dirname "$STATE")"
  local sha webui
  sha="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null)"
  webui="$(docker image inspect "$WEBUI_IMAGE" --format '{{.Id}}' 2>/dev/null)"
  # Retag rather than copy: the old layers stay on disk under a second name,
  # so rollback is a tag move and needs no network.
  docker image inspect "$LLAMA_IMAGE" >/dev/null 2>&1 \
    && docker tag "$LLAMA_IMAGE" "$ROLLBACK_IMAGE"
  printf 'sha=%s\nwebui=%s\n' "$sha" "$webui" > "$STATE"
  log "rollback point: ${sha:0:12} / ${webui:0:19}"
}

do_rollback() {
  [[ -f "$STATE" ]] || die "no rollback point recorded at $STATE"
  local sha webui
  sha="$(sed -n 's/^sha=//p' "$STATE")"
  webui="$(sed -n 's/^webui=//p' "$STATE")"

  log "rolling back to ${sha:0:12}"
  [[ -n "$sha" ]] && git -C "$ROOT" reset --hard "$sha" >/dev/null 2>&1
  docker image inspect "$ROLLBACK_IMAGE" >/dev/null 2>&1 \
    && docker tag "$ROLLBACK_IMAGE" "$LLAMA_IMAGE"
  [[ -n "$webui" ]] && docker image inspect "$webui" >/dev/null 2>&1 \
    && docker tag "$webui" "$WEBUI_IMAGE"
  compose up -d --force-recreate || warn "compose failed during rollback"

  if wait_for_health && smoke; then
    log "rollback verified — stack is serving again"
    return 0
  fi
  warn "ROLLBACK DID NOT RECOVER THE STACK — needs a human"
  return 1
}

cmd_check() {
  preflight
  local behind=0 rc=0
  if git -C "$ROOT" fetch --quiet 2>/dev/null; then
    behind="$(git -C "$ROOT" rev-list --count 'HEAD..@{u}' 2>/dev/null || echo 0)"
  else
    warn "could not fetch; commit drift unknown"
  fi
  if (( behind > 0 )); then
    log "behind origin by $behind commit(s)"
    rc=1
  else
    log "code up to date"
  fi

  local local_id remote_digest
  local_id="$(docker image inspect "$WEBUI_IMAGE" --format '{{.Id}}' 2>/dev/null)"
  remote_digest="$(docker manifest inspect "$WEBUI_IMAGE" 2>/dev/null | head -c 40)"
  if [[ -z "$local_id" ]]; then
    log "open-webui image not present locally"
    rc=1
  elif [[ -z "$remote_digest" ]]; then
    warn "could not reach the registry; image drift unknown"
  fi
  return "$rc"
}

cmd_update() {
  preflight
  record_rollback_point

  log "pulling code"
  if ! git -C "$ROOT" pull --ff-only; then
    die "git pull --ff-only failed — the checkout has diverged, fix it by hand"
  fi

  log "building llama image"
  compose build llama || { warn "build failed"; do_rollback; exit 1; }

  log "pulling open-webui"
  compose pull open-webui || warn "pull failed; keeping the current image"

  log "restarting stack"
  compose up -d || { warn "compose up failed"; do_rollback; exit 1; }

  log "waiting for health"
  wait_for_health || { warn "never became healthy"; do_rollback; exit 1; }

  log "smoke testing every model"
  if ! smoke; then
    warn "smoke test failed after update — rolling back"
    do_rollback
    exit 1
  fi

  log "update complete and verified"
}

case "${1:-}" in
  "")         cmd_update;;
  --check)    cmd_check;;
  --rollback) preflight; do_rollback;;
  -h|--help)  sed -n '2,10p' "$0" | sed 's/^# \?//';;
  *)          sed -n '2,10p' "$0" | sed 's/^# \?//'; exit 2;;
esac
