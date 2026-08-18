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

# .env is the ONLY source of truth for connections. Open WebUI otherwise
# persists them in its own DB (openai.api_base_urls / openai.api_keys in
# DEFAULT_CONFIG) and the DB wins over the environment on every later start,
# so an edited .env silently has no effect. That is not hypothetical: it is
# how a stale key survived in .webui-data and made every request fail with
# "Invalid API Key" while .env held the correct one. With this False, the
# environment is authoritative on every start.
#
# Trade-off: connection/model settings changed in the admin UI no longer
# survive a restart. Change them in .env instead.
export ENABLE_PERSISTENT_CONFIG=False

# Talking to llama-server on the host directly, so loopback (not host.docker.internal).
LOCAL_BASE_URL="http://127.0.0.1:${BONSAI_PORT:-8080}/v1"

# Optional second provider: Hetzner's hosted open-weight models, added to the
# same dropdown as the local Bonsai models. Only wired up when a key is
# present, so the default stays entirely local. Open WebUI reads the plural
# forms as semicolon-separated lists (config.py: OPENAI_API_BASE_URLS /
# OPENAI_API_KEYS), and they take precedence over the singular ones.
#
# NOTE this sends prompts off this machine to Hetzner. Everything else in this
# stack is local. Hetzner state they store usage metadata only, not request or
# response content — but that is their infrastructure and their claim.
if [[ -n "${HETZNER_API_KEY:-}" ]]; then
  HETZNER_BASE="${HETZNER_API_BASE:-https://inference.hetzner.com/api/v1}"
  export OPENAI_API_BASE_URLS="$LOCAL_BASE_URL;$HETZNER_BASE"
  export OPENAI_API_KEYS="$BONSAI_API_KEY;$HETZNER_API_KEY"
  export OPENAI_API_BASE_URL="$LOCAL_BASE_URL"
  export OPENAI_API_KEY="$BONSAI_API_KEY"
else
  export OPENAI_API_BASE_URL="$LOCAL_BASE_URL"
  export OPENAI_API_KEY="$BONSAI_API_KEY"
fi
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

# Sourced rather than executed (tests/test_webui_providers.py) — stop here so
# the environment above can be inspected without launching a server.
[[ "${BASH_SOURCE[0]}" == "${0}" ]] || return 0

mkdir -p "$DATA_DIR"
echo "webui  : http://127.0.0.1:$PORT"
if [[ -n "${OPENAI_API_BASE_URLS:-}" ]]; then
  echo "backend: ${OPENAI_API_BASE_URLS//;/ + }"
else
  echo "backend: $OPENAI_API_BASE_URL"
fi

exec "$ROOT/.venv/bin/open-webui" serve --port "$PORT" --host "$HOST"
