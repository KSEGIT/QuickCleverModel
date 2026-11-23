#!/usr/bin/env bash
# Chat UI safety check for the Docker paths (docker/compose.linux.yaml and
# docker/compose.yaml). update.sh runs it after every start or recreate of the
# chat UI; run it by hand after you start the stack with WEBUI_AUTH=true.
#
#   ./webui-guard.sh [--wait SECONDS]
#
# --wait: with login on, wait up to SECONDS for the chat UI to answer before
# checking it (default 0: check once). A UI that is still starting is not a
# UI that is safe.
#
# Exit codes:
#   0  nothing wrong found (or nothing to check)
#   1  the default no-login admin account works: the UI container was STOPPED
#   2  .env holds a WEBUI_MODEL that is not a plain model id
#   3  the default admin works and the UI container could NOT be stopped:
#      it is STILL RUNNING and must be stopped by hand
#
# Why: Open WebUI with WEBUI_AUTH=false creates one admin account,
# admin@localhost with password "admin" (backend/open_webui/routers/auths.py,
# signin). Turn login on for that same data volume and the login page accepts
# those public credentials, so anyone who reaches the UI is admin. A fresh
# open-webui-data volume is the fix; this check catches the case where that
# step was missed.
#
# Settings, all optional: QCM_ENV_FILE (default ./.env), QCM_WEBUI_URL
# (default http://$WEBUI_BIND:9090), QCM_WEBUI_CONTAINER (default
# bonsai-webui, the container_name in both compose files).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${QCM_ENV_FILE:-$ROOT/.env}"

log()  { printf 'webui-guard: %s\n' "$*"; }
warn() { printf 'webui-guard: WARN %s\n' "$*" >&2; }

# shellcheck source=lib/webui.sh
. "$ROOT/lib/webui.sh" || { warn "cannot read $ROOT/lib/webui.sh"; exit 2; }

WAIT=0
case "${1:-}" in
  --wait)
    WAIT="${2:-0}"
    [[ "$WAIT" =~ ^[0-9]{1,6}$ ]] || { warn "--wait needs a number of seconds"; exit 2; }
    WAIT=$((10#$WAIT));;   # base 10: "08" is not an octal error
  "") ;;
  *) warn "usage: $0 [--wait SECONDS]"; exit 2;;
esac

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  . "$ENV_FILE"
  set +a
fi

if ! webui_model_ok "${WEBUI_MODEL:-}"; then
  warn "WEBUI_MODEL='${WEBUI_MODEL}' in $ENV_FILE is not a plain model id
(allowed: letters, digits and . _ : / -). Fix it before you start the stack."
  exit 2
fi

URL="${QCM_WEBUI_URL:-http://${WEBUI_BIND:-127.0.0.1}:9090}"
CONTAINER="${QCM_WEBUI_CONTAINER:-bonsai-webui}"

# Wait only for a UI that .env says starts with login; otherwise look once.
webui_auth_on || WAIT=0
deadline=$(( SECONDS + WAIT ))
until curl -fsS -m 5 -o /dev/null "$URL/health" 2>/dev/null; do
  if (( SECONDS >= deadline )); then
    log "the chat UI at $URL does not answer; nothing to check"
    exit 0
  fi
  sleep 3
done

# Decide by what the RUNNING UI does, not by .env: compose lets a shell
# variable override .env, so the UI can run with login on while .env says
# off. /api/config reports it without a login (features.auth). Only when
# that cannot be read does .env decide. A no-login UI is never probed: it
# answers any sign-in with its own admin, by design.
running_auth="$(curl -s -m 10 "$URL/api/config" 2>/dev/null | python3 -c '
import json, sys
try:
    v = json.load(sys.stdin).get("features", {}).get("auth")
except Exception:
    v = None
print({True: "true", False: "false"}.get(v, ""))' 2>/dev/null)"
case "$running_auth" in
  true) ;;
  false)
    log "the running chat UI has no login; no login to check"
    exit 0;;
  *)
    if ! webui_auth_on; then
      log "cannot read the running UI's login mode, and WEBUI_AUTH is not true; nothing to check"
      exit 0
    fi;;
esac

code="$(curl -s -m 15 -o /dev/null -w '%{http_code}' \
  -H 'Content-Type: application/json' \
  -d '{"email":"admin@localhost","password":"admin"}' \
  "$URL/api/v1/auths/signin" 2>/dev/null)"

if [[ "$code" != "200" ]]; then
  log "the default no-login admin does not work here (HTTP ${code:-none}); OK"
  exit 0
fi

warn "!!! admin@localhost / admin CAN SIGN IN to the chat UI at $URL !!!
This data volume was made with WEBUI_AUTH=false. With login on, those public
credentials give anyone who reaches the UI full admin rights.
Fix: back up the open-webui-data volume, then start again with a FRESH
volume and WEBUI_AUTH=true, and sign up the real admin first. Or set
WEBUI_AUTH=false in .env to go back to the no-login, loopback-only setup."
if docker stop "$CONTAINER" >/dev/null 2>&1; then
  warn "stopped the $CONTAINER container"
  exit 1
fi
warn "!!! COULD NOT STOP $CONTAINER — the chat UI is STILL RUNNING and STILL
accepts admin@localhost / admin. Stop it by hand NOW: docker stop $CONTAINER"
exit 3
