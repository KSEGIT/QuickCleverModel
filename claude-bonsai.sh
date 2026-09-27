#!/usr/bin/env bash
# Launch Claude Code against this server instead of a hosted Anthropic model.
#
#   ./claude-bonsai.sh                 interactive
#   ./claude-bonsai.sh -p 'your task'  one-shot
#
# Every argument is passed through to `claude`.
#
# No proxy is involved. llama-server answers /v1/messages -- the Anthropic
# Messages API -- alongside the OpenAI routes, so Claude Code talks to it
# directly. See docs/claude-code.md.
#
# This is a launcher rather than ~/.claude/settings.json on purpose: settings
# repoint EVERY Claude Code session at this box, including the ones you want on
# a hosted model. Here the environment lives and dies with the process.
set -euo pipefail

ENV_FILE="${BONSAI_ENV_FILE:-$(cd "$(dirname "$0")" && pwd)/.env}"

err() { echo "err  $*" >&2; exit 1; }

command -v claude >/dev/null || err "claude is not on PATH — https://claude.com/claude-code"
[[ -f "$ENV_FILE" ]] || err "no $ENV_FILE — copy .env.example and set BONSAI_API_KEY"

# set -a exports everything the file assigns, which is what we want here.
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

# Same connect/bind split as setup-opencode.sh: BONSAI_HOST is the address the
# SERVER binds, never a connect target, and 0.0.0.0 is a wildcard nobody dials.
if [[ -n "${BONSAI_SERVER_URL:-}" ]]; then
  # Claude Code appends /v1 itself; a base ending in /v1 yields /v1/v1/messages.
  BASE_URL="${BONSAI_SERVER_URL%/v1}"
else
  CONNECT_HOST="${BONSAI_HOST:-127.0.0.1}"
  [[ "$CONNECT_HOST" == "0.0.0.0" ]] && CONNECT_HOST=127.0.0.1
  BASE_URL="http://${CONNECT_HOST}:${BONSAI_PORT:-8080}"
fi

API_KEY="${BONSAI_SERVER_KEY:-${BONSAI_API_KEY:-}}"
[[ -n "$API_KEY" ]] || err "no BONSAI_SERVER_KEY or BONSAI_API_KEY in $ENV_FILE"

# Keep the existing default until the target-GPU agent matrix is measured.
MODEL="${BONSAI_CLAUDE_MODEL:-bonsai-27b-ternary-text}"

# Match the selected preset, including the separate agent context controls.
case "$MODEL" in
  qwen3.5-9b-q4_k_m) CONTEXT="${QCM_QWEN9_CTX:-${QCM_AGENT_CTX:-8192}}" ;;
  qwen3.5-4b-q4_k_m) CONTEXT="${QCM_QWEN4_CTX:-${QCM_AGENT_CTX:-8192}}" ;;
  granite-4.1-8b-q4_k_m) CONTEXT="${QCM_GRANITE_CTX:-${QCM_AGENT_CTX:-8192}}" ;;
  qwen3.6-35b-a3b) CONTEXT="${QCM_QWEN36_CTX:-${QCM_AGENT_CTX:-8192}}" ;;
  gemma4-e4b) CONTEXT="${QCM_GEMMA4_CTX:-${QCM_AGENT_CTX:-8192}}" ;;
  bonsai-27b-ternary-text) CONTEXT="${BONSAI_CTX_TEXT:-${BONSAI_CTX:-8192}}" ;;
  *) CONTEXT="${BONSAI_CTX:-8192}" ;;
esac

export ANTHROPIC_BASE_URL="$BASE_URL"
export ANTHROPIC_AUTH_TOKEN="$API_KEY"
export ANTHROPIC_MODEL="$MODEL"
# Background jobs (titles, summaries) run on a Haiku-class model. Left unset,
# Claude Code asks this server for a model it has never heard of.
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL"
export ANTHROPIC_SMALL_FAST_MODEL="$MODEL"
export CLAUDE_CODE_MAX_CONTEXT_TOKENS="$CONTEXT"

echo "==>  $MODEL at $BASE_URL (context $CONTEXT)" >&2
exec claude "$@"
