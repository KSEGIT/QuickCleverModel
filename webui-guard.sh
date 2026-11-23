#!/usr/bin/env bash
# Chat UI safety check for the Docker paths (docker/compose.linux.yaml and
# docker/compose.yaml). update.sh runs it after every update; run it by hand
# after you start the stack with WEBUI_AUTH=true.
#
#   ./webui-guard.sh
#
# Exit codes:
#   0  nothing wrong found (or nothing to check)
#   1  the default no-login admin account works: the UI container was stopped
#   2  .env holds a WEBUI_MODEL that is not a plain model id
#   3  the default admin works, but the UI container could NOT be stopped
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

# Model ids go into OPENAI_API_CONFIGS as JSON. A quote, comma or bracket
# would break that JSON or add entries, so allow only plain id characters.
# update.sh uses the same pattern; tests/test_webui_guard.py keeps them equal.
MODEL_ID_RE='^[A-Za-z0-9._:/-]+$'

webui_model_ok() {
  [[ -z "${1:-}" || "$1" =~ $MODEL_ID_RE ]]
}

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

auth="$(printf '%s' "${WEBUI_AUTH:-false}" | tr '[:upper:]' '[:lower:]')"
if [[ "$auth" != "true" ]]; then
  log "WEBUI_AUTH is not true; no login to check"
  exit 0
fi

URL="${QCM_WEBUI_URL:-http://${WEBUI_BIND:-127.0.0.1}:9090}"
CONTAINER="${QCM_WEBUI_CONTAINER:-bonsai-webui}"

if ! curl -fsS -m 5 -o /dev/null "$URL/health" 2>/dev/null; then
  log "the chat UI at $URL does not answer; nothing to check"
  exit 0
fi

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
Stopping the $CONTAINER container now.
Fix: back up the open-webui-data volume, then start again with a FRESH
volume and WEBUI_AUTH=true, and sign up the real admin first. Or set
WEBUI_AUTH=false in .env to go back to the no-login, loopback-only setup."
if docker stop "$CONTAINER" >/dev/null 2>&1; then
  warn "stopped $CONTAINER"
  exit 1
fi
warn "COULD NOT STOP $CONTAINER — stop it by hand now: docker stop $CONTAINER"
exit 3
