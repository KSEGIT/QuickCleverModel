#!/usr/bin/env bash
# Open WebUI, running natively on the host.
#
# Originally this was a container (see docker/compose.yaml), but Docker Desktop's
# internal registry proxy (http.docker.internal:3128) wedged and every image pull
# hangs — including docker.io. Running natively avoids the pull entirely and does
# not disturb the containers already running on this machine.
#
# Inference still comes from start-server.sh on :8080 (native, Metal).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
[[ -f "$ROOT/.env" ]] && set -a && . "$ROOT/.env" && set +a
: "${BONSAI_API_KEY:?BONSAI_API_KEY not set — see .env}"

# Talking to llama-server on the host directly, so loopback (not host.docker.internal).
export OPENAI_API_BASE_URL="http://127.0.0.1:${BONSAI_PORT:-8080}/v1"
export OPENAI_API_KEY="$BONSAI_API_KEY"
export ENABLE_OLLAMA_API=False
# Only valid while the DB has zero users. Open WebUI refuses to start with
# WEBUI_AUTH=False if any user exists ("You can't turn off authentication because
# there are existing users"). If you ever create one, either flip this to True or
# delete .webui-data/ to get back to a fresh install.
export WEBUI_AUTH=False

# Open WebUI fires background generations (chat title, tags, follow-ups,
# autocomplete) as SEPARATE model requests that decode concurrently with your
# actual reply. On a single-GPU box they steal memory bandwidth from the response
# you are waiting for — this is why the UI showed 4.8 t/s while the same model
# benchmarked at ~12 t/s. Off by default here; re-enable individually if wanted.
export ENABLE_TITLE_GENERATION=False
export ENABLE_TAGS_GENERATION=False
export ENABLE_FOLLOW_UP_GENERATION=False
export ENABLE_AUTOCOMPLETE_GENERATION=False
export ENABLE_RETRIEVAL_QUERY_GENERATION=False
export ENABLE_SEARCH_QUERY_GENERATION=False
export DATA_DIR="$ROOT/.webui-data"
export PORT="${WEBUI_PORT:-9090}"
export HOST="127.0.0.1"            # UI stays loopback-only; only llama-server needs 0.0.0.0

mkdir -p "$DATA_DIR"
echo "webui  : http://127.0.0.1:$PORT"
echo "backend: $OPENAI_API_BASE_URL"

exec "$ROOT/.venv/bin/open-webui" serve --port "$PORT" --host "$HOST"
