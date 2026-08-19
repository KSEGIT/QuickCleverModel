#!/usr/bin/env bash
# setup-opencode.sh — make OpenCode talk to the Bonsai API.
#
# Does two things and nothing else:
#   1. checks opencode is installed; offers to install it if not
#      (mac: Homebrew, Linux: install script, Windows/git-bash: npm)
#   2. writes the provider config (~/.config/opencode/opencode.json)
#
# One source of truth: the repo's .env. It reads:
#   BONSAI_SERVER_URL=...  where to CONNECT, e.g. http://127.0.0.1:8080/v1
#                          (defaults to this machine; set it to another box's
#                          address when the server runs elsewhere)
#   BONSAI_SERVER_KEY=...  that server's API key; defaults to BONSAI_API_KEY,
#                          which is correct only when the server is local
#
# BONSAI_HOST is deliberately NOT used as a connect address: start-server.sh
# uses it as the server's BIND address, and a machine cannot bind an address
# it does not own. Connect and bind are different facts, so different names.
# Missing values are resolved at run time and saved back into .env:
# the key is fetched over SSH when possible, otherwise you are prompted.
#
# This is a ONE-SHOT render, not a live link: the config is a copy of what
# .env said at the moment you ran it. Change BONSAI_API_KEY, BONSAI_HOST or
# BONSAI_PORT afterwards and opencode keeps the old values until you re-run.
# That drift is silent and presents as auth failures, so:
#
#   ./setup-opencode.sh --check     compare the config against .env, change
#                                   nothing, exit 1 if they disagree
#
# Re-running without --check re-renders and fixes any drift.
#
# MCP servers (e.g. Playwright) are a separate concern — add them to the
# config yourself, see https://opencode.ai/docs/mcp-servers/
set -euo pipefail

MODE="write"
case "${1:-}" in
  --check) MODE="check" ;;
  "")      ;;
  *)       echo "unknown argument: $1 (only --check is supported)" >&2; exit 2 ;;
esac

# Overridable so the test suite can point at a fixture .env instead of the
# real one; defaults to the repo's .env, which is the normal case.
ENV_FILE="${BONSAI_ENV_FILE:-$(cd "$(dirname "$0")" && pwd)/.env}"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/opencode"
CONFIG="$CONFIG_DIR/opencode.json"

info() { echo "==>  $*"; }
ok()   { echo " ok  $*"; }
err()  { echo "err  $*" >&2; exit 1; }

if [[ ! -f "$ENV_FILE" ]]; then
  [[ "$MODE" == check ]] && err "no $ENV_FILE to check against"
  info "creating $ENV_FILE"
  touch "$ENV_FILE"; chmod 600 "$ENV_FILE"
fi
set -a; . "$ENV_FILE"; set +a

# Only ask where the server is when nothing already says so. BONSAI_SERVER_URL
# answers it outright; otherwise fall back to BONSAI_HOST, and failing that
# assume this machine. --check never prompts.
# Only ask when there is someone to ask: a non-interactive run (CI, the test
# suite, a script) takes the localhost default silently rather than dying on
# EOF from `read` under `set -e`.
if [[ -z "${BONSAI_SERVER_URL:-}" && -z "${BONSAI_HOST:-}" && "$MODE" != check && -t 0 ]]; then
  printf 'Where is the llama-server? [http://127.0.0.1:%s/v1] ' "${BONSAI_PORT:-8080}"
  read -r reply || reply=""
  if [[ -n "$reply" ]]; then
    # Accept either a bare host/IP or a full URL.
    [[ "$reply" == http*://* ]] || reply="http://$reply:${BONSAI_PORT:-8080}/v1"
    BONSAI_SERVER_URL="$reply"
    printf 'BONSAI_SERVER_URL=%s\n' "$BONSAI_SERVER_URL" >> "$ENV_FILE"
    ok "BONSAI_SERVER_URL saved to .env"
  fi
fi

if [[ -z "${BONSAI_SERVER_KEY:-}${BONSAI_API_KEY:-}" ]]; then
  [[ "$MODE" == check ]] && err "no BONSAI_SERVER_KEY or BONSAI_API_KEY in $ENV_FILE"
  info "no key in .env — trying SSH to ${BONSAI_HOST:-the server}"
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

if [[ "$MODE" != check ]] && ! command -v opencode >/dev/null; then
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
[[ "$MODE" == check ]] || ok "opencode installed: $(opencode --version 2>/dev/null || echo present)"

# limit.context tells opencode when to compact. It must match the server's
# BONSAI_CTX (in .env on the machine running llama-server), else opencode
# compacts too early or the server rejects long requests.
#
# The default mirrors the server launcher for THIS platform (start-server.sh
# uses 32768 on macOS; docker/compose.linux.yaml uses 8192). When the server
# runs on a different machine, that guess can be wrong in either direction —
# set BONSAI_CTX explicitly to whatever that machine serves.
if [[ -n "${BONSAI_CTX:-}" ]]; then
  CTX="$BONSAI_CTX"
  CTX_EXPLICIT=1
else
  case "$(uname -s)" in
    Darwin) CTX=32768 ;;
    *)      CTX=8192 ;;
  esac
  CTX_EXPLICIT=0
fi

# Where the CLIENT connects. Deliberately NOT BONSAI_HOST: start-server.sh
# uses that as the server's BIND address (`--host "$HOST"`), so a tailnet IP
# there makes this machine try to bind an address it does not own and
# llama-server fails to start. Connect-address and bind-address are different
# facts; they get different variables.
#
# Falls back to BONSAI_HOST for configs written before this split existed.
if [[ -n "${BONSAI_SERVER_URL:-}" ]]; then
  BASE_URL="$BONSAI_SERVER_URL"
else
  # Default is this machine. BONSAI_HOST is only consulted for configs written
  # before the split; 0.0.0.0 is a bind wildcard, never a connect target.
  CONNECT_HOST="${BONSAI_HOST:-127.0.0.1}"
  [[ "$CONNECT_HOST" == "0.0.0.0" ]] && CONNECT_HOST=127.0.0.1
  BASE_URL="http://${CONNECT_HOST}:${BONSAI_PORT:-8080}/v1"
fi

# The key belongs to whichever server BASE_URL names. BONSAI_API_KEY is this
# machine's server key — right when the target is local, wrong (401) when it
# is another box with its own key.
API_KEY="${BONSAI_SERVER_KEY:-$BONSAI_API_KEY}"

if [[ "$MODE" == check ]]; then
  [[ -f "$CONFIG" ]] || err "no config at $CONFIG — run ./setup-opencode.sh to create it"
  BASE_URL="$BASE_URL" API_KEY="$API_KEY" CTX="$CTX" CONFIG="$CONFIG" \
  CTX_EXPLICIT="$CTX_EXPLICIT" \
  python3 - <<'PYCHECK'
import json, os, sys

cfg_path = os.environ["CONFIG"]
try:
    with open(cfg_path) as fh:
        cfg = json.load(fh)
except (OSError, ValueError) as exc:
    print(f"err  {cfg_path} is unreadable or not valid JSON: {exc}", file=sys.stderr)
    sys.exit(1)

opts = cfg.get("provider", {}).get("bonsai", {}).get("options", {})
models = cfg.get("provider", {}).get("bonsai", {}).get("models", {})
want_ctx = int(os.environ["CTX"])

drift = []
if opts.get("baseURL") != os.environ["BASE_URL"]:
    drift.append(f"baseURL: config has {opts.get('baseURL')!r}, .env implies {os.environ['BASE_URL']!r}")
if opts.get("apiKey") != os.environ["API_KEY"]:
    # Never print either key; a prefix is enough to tell them apart.
    have = (opts.get("apiKey") or "")[:14]
    want = os.environ["API_KEY"][:14]
    drift.append(f"apiKey: config has {have}…, .env has {want}…")
# Only a real mismatch when .env states the context. Otherwise `want_ctx` is
# this platform's default, which says nothing about a remote server — report
# it as a note so it is visible without failing the check.
notes = []
for name, m in sorted(models.items()):
    got = m.get("limit", {}).get("context")
    if got == want_ctx:
        continue
    msg = f"{name}: context {got}, .env implies {want_ctx}"
    (drift if os.environ["CTX_EXPLICIT"] == "1" else notes).append(msg)

if drift:
    print("err  opencode config has drifted from .env:", file=sys.stderr)
    for d in drift:
        print(f"       - {d}", file=sys.stderr)
    print("     fix with: ./setup-opencode.sh", file=sys.stderr)
    sys.exit(1)
for n in notes:
    print(f"note  {n} (BONSAI_CTX unset — set it to the server's value to pin this)")
print(f" ok  opencode config matches .env ({os.environ['BASE_URL']})")
PYCHECK
  exit $?
fi

info "writing $CONFIG ($BASE_URL)"
mkdir -p "$CONFIG_DIR"
cat > "$CONFIG" <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "bonsai": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Bonsai local",
      "options": {
        "baseURL": "$BASE_URL",
        "apiKey": "$API_KEY"
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
