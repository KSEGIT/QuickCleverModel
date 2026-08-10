#!/usr/bin/env bash
# setup-opencode.sh — make OpenCode talk to the Bonsai API.
#
# Does two things and nothing else:
#   1. checks opencode is installed; offers to install it if not
#      (mac: Homebrew, Linux: install script, Windows/git-bash: npm)
#   2. writes the provider config (~/.config/opencode/opencode.json)
#
# One source of truth: the repo's .env. It needs:
#   BONSAI_HOST=...      host/IP of the llama-server, e.g. 100.109.50.118
#                        (use 127.0.0.1 when the server runs on this machine)
#   BONSAI_API_KEY=...   the API key (must match the server you point at)
# Missing values are resolved at run time and saved back into .env:
# the key is fetched over SSH when possible, otherwise you are prompted.
#
# MCP servers (e.g. Playwright) are a separate concern — add them to the
# config yourself, see https://opencode.ai/docs/mcp-servers/
set -euo pipefail

ENV_FILE="$(cd "$(dirname "$0")" && pwd)/.env"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/opencode"
CONFIG="$CONFIG_DIR/opencode.json"

info() { echo "==>  $*"; }
ok()   { echo " ok  $*"; }
err()  { echo "err  $*" >&2; exit 1; }

if [[ ! -f "$ENV_FILE" ]]; then
  info "creating $ENV_FILE"
  touch "$ENV_FILE"; chmod 600 "$ENV_FILE"
fi
set -a; . "$ENV_FILE"; set +a

if [[ -z "${BONSAI_HOST:-}" ]]; then
  printf 'BONSAI_HOST not in .env — host/IP of the llama-server (e.g. 100.109.50.118, or 127.0.0.1 for this machine): '
  read -r BONSAI_HOST
  [[ -n "$BONSAI_HOST" ]] || err "no host given"
  printf 'BONSAI_HOST=%s\n' "$BONSAI_HOST" >> "$ENV_FILE"
  ok "BONSAI_HOST saved to .env"
fi

if [[ -z "${BONSAI_API_KEY:-}" ]]; then
  info "no BONSAI_API_KEY in .env — trying SSH to $BONSAI_HOST"
  BONSAI_API_KEY=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$BONSAI_HOST" \
    'grep ^BONSAI_API_KEY= ~/QuickCleverModel/.env 2>/dev/null' 2>/dev/null \
    | head -1 | cut -d= -f2- || true)
  if [[ -n "$BONSAI_API_KEY" ]]; then
    ok "key fetched over SSH"
  else
    printf 'SSH fetch failed — paste BONSAI_API_KEY (from .env on the server): '
    read -r BONSAI_API_KEY
    [[ -n "$BONSAI_API_KEY" ]] || err "no key given"
  fi
  printf 'BONSAI_API_KEY=%s\n' "$BONSAI_API_KEY" >> "$ENV_FILE"
  ok "BONSAI_API_KEY saved to .env"
fi

if ! command -v opencode >/dev/null; then
  info "opencode is not installed"
  case "$(uname -s)" in
    Darwin) install_cmd="brew install opencode" ;;
    Linux)  install_cmd="curl -fsSL https://opencode.ai/install | bash" ;;
    MINGW*|MSYS*|CYGWIN*) install_cmd="npm install -g opencode-ai" ;;
    *) err "unsupported OS — install opencode manually: https://opencode.ai/docs/" ;;
  esac
  printf 'Install it with:  %s   [y/N] ' "$install_cmd"
  read -r answer
  [[ "$answer" =~ ^[Yy]$ ]] || err "aborted — install opencode and re-run"
  eval "$install_cmd"
  command -v opencode >/dev/null || err "install finished but opencode is not on PATH — open a new shell and re-run"
fi
ok "opencode installed: $(opencode --version 2>/dev/null || echo present)"

info "writing $CONFIG (host $BONSAI_HOST)"
# limit.context tells opencode when to compact. It must match the server's
# BONSAI_CTX (in .env on the machine running llama-server), else opencode
# compacts too early or the server rejects long requests.
CTX="${BONSAI_CTX:-8192}"
mkdir -p "$CONFIG_DIR"
cat > "$CONFIG" <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "bonsai": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Bonsai local",
      "options": {
        "baseURL": "http://$BONSAI_HOST:8080/v1",
        "apiKey": "$BONSAI_API_KEY"
      },
      "models": {
        "bonsai-27b-1bit":    { "name": "bonsai-27b-1bit",
                                "limit": { "context": $CTX, "output": 4096 } },
        "bonsai-27b-ternary": { "name": "bonsai-27b-ternary",
                                "limit": { "context": $CTX, "output": 4096 } },
        "bonsai-27b-ternary-text": { "name": "bonsai-27b-ternary-text",
                                "limit": { "context": $CTX, "output": 4096 } }
      }
    }
  },
  "model": "bonsai/bonsai-27b-ternary-text"
}
JSON
chmod 600 "$CONFIG"
ok "config written (mode 600 — it holds the API key)"

echo
echo "==> done. Run:  opencode"
