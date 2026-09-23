#!/usr/bin/env bash
# setup-opencode.sh — make OpenCode talk to the Bonsai API.
#
# Does two things and nothing else:
#   1. checks opencode is installed; offers to install it if not
#      (all supported platforms: pinned npm release)
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
#
# Your `mcp` block survives re-runs. It did not always: this script rendered the
# whole document from its own template, so every run deleted the servers this
# header tells you to add, and --check then called them drift. Keys outside the
# `provider`/`model` pair it owns are now carried across (PRESERVED_KEYS).
#
# Worth knowing if you came here from Codex: opencode is the way to drive MCP
# servers against this stack. Codex sends each MCP server as one tool of
# `type: "namespace"` with the real tools nested inside, which llama.cpp does
# not implement, so the model cannot call them — measured, same tool flat gets
# called and namespaced does not. opencode sends them flat
# (playwright_browser_navigate and friends) and they work.
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

# The model asked for /props below. Any id the router knows will do; this one
# is the smallest, so if it is not already resident the load is the cheapest.
DEFAULT_MODEL="bonsai-27b-1bit"

# Appending to a file that does not end in a newline silently glues the new
# assignment onto the previous line, corrupting both.
append_env() {
  [[ -s "$ENV_FILE" && -n "$(tail -c1 "$ENV_FILE")" ]] && printf '\n' >> "$ENV_FILE"
  printf '%s\n' "$1" >> "$ENV_FILE"
}

# A bare host becomes a full URL; a URL is validated rather than trusted. An
# unusable value here is persisted to .env, so re-running never repairs it.
normalise_server_url() {
  local v="$1"
  if [[ "$v" != *://* ]]; then
    # Bracket a bare IPv6 literal, else host:port parsing is ambiguous.
    [[ "$v" == *:*:* && "$v" != \[*\] ]] && v="[$v]"
    # Only append default port if the input has no port already.
    # Detect port: bracketed IPv6 with :port after ], or plain host:port (single colon).
    if [[ "$v" == \]*:* || ( "$v" == *:* && "$v" != *:*:* && "$v" != \[*\] ) ]]; then
      # Port already present
      v="http://$v/v1"
    else
      # No port, add default
      v="http://$v:${BONSAI_PORT:-8080}/v1"
    fi
  fi
  local scheme="${v%%://*}"
  scheme="$(printf '%s' "$scheme" | tr '[:upper:]' '[:lower:]')"
  [[ "$scheme" == http || "$scheme" == https ]] \
    || err "server URL must be http:// or https://, got: $1"
  # Normalise the scheme's case so the rendered config and --check agree
  # regardless of how it was typed.
  v="$scheme://${v#*://}"
  # Strip trailing slashes before checking/appending /v1 to avoid /v1/v1 or /v1/
  v="${v%/}"
  # llama-server serves the OpenAI-compatible API under /v1; without it every
  # completions request 404s with nothing to explain why.
  [[ "$v" == */v1 ]] || v="$v/v1"
  printf '%s' "$v"
}

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
  # Persist even when the default is accepted: otherwise the prompt re-fires on
  # every run, including the re-runs this script tells you to do to fix drift.
  BONSAI_SERVER_URL="$(normalise_server_url "${reply:-127.0.0.1}")"
  append_env "BONSAI_SERVER_URL=$BONSAI_SERVER_URL"
  ok "BONSAI_SERVER_URL=$BONSAI_SERVER_URL saved to .env"
fi

# Validate whatever .env supplied, too — it may have been hand-edited.
if [[ -n "${BONSAI_SERVER_URL:-}" ]]; then
  BONSAI_SERVER_URL="$(normalise_server_url "$BONSAI_SERVER_URL")"
fi

if [[ -z "${BONSAI_SERVER_KEY:-}${BONSAI_API_KEY:-}" ]]; then
  [[ "$MODE" == check ]] && err "no BONSAI_SERVER_KEY or BONSAI_API_KEY in $ENV_FILE"
  # Only SSH when there is a host to SSH to. The prompt that used to guarantee
  # BONSAI_HOST was set is gone, and an unguarded reference dies under `set -u`.
  SSH_TARGET=""
  if [[ -n "${BONSAI_HOST:-}" && "${BONSAI_HOST}" != "0.0.0.0" ]]; then
    SSH_TARGET="$BONSAI_HOST"
  elif [[ -n "${BONSAI_SERVER_URL:-}" ]]; then
    # host[:port] out of scheme://host[:port]/path, minus any IPv6 brackets.
    SSH_TARGET="${BONSAI_SERVER_URL#*://}"; SSH_TARGET="${SSH_TARGET%%/*}"
    # Strip port: for [host]:port, remove ]:port; for plain host:port (single colon), remove :port
    if [[ "$SSH_TARGET" == \]*:* ]]; then
      # Bracketed IPv6 with port: [::1]:8080 -> [::1] -> ::1
      SSH_TARGET="${SSH_TARGET%]:*}"
      SSH_TARGET="${SSH_TARGET#[}"
    elif [[ "$SSH_TARGET" == *:* && "$SSH_TARGET" != *:*:* ]]; then
      # Plain host:port (single colon): host:8080 -> host
      SSH_TARGET="${SSH_TARGET%:*}"
    else
      # No port or bracketed IPv6 without port: [::1] -> ::1
      SSH_TARGET="${SSH_TARGET#[}"; SSH_TARGET="${SSH_TARGET%]}"
    fi
  fi

  FETCHED=""
  if [[ -n "$SSH_TARGET" && "$SSH_TARGET" != "127.0.0.1" && "$SSH_TARGET" != "localhost" ]]; then
    info "no key in .env — trying SSH to $SSH_TARGET"
    FETCHED=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$SSH_TARGET" \
      'grep ^BONSAI_API_KEY= ~/QuickCleverModel/.env 2>/dev/null' 2>/dev/null \
      | head -1 | cut -d= -f2- || true)
    [[ -n "$FETCHED" ]] && ok "key fetched over SSH from $SSH_TARGET"
  fi

  if [[ -z "$FETCHED" ]]; then
    # Fail with a clear message rather than blocking forever on a prompt
    # nobody can answer (CI, a cron, the test suite).
    [[ -t 0 ]] || err "no API key in $ENV_FILE and cannot prompt (non-interactive) — set BONSAI_SERVER_KEY or BONSAI_API_KEY"
    printf 'Paste the API key for %s: ' "${BONSAI_SERVER_URL:-this machine}"
    read -r FETCHED || FETCHED=""
    [[ -n "$FETCHED" ]] || err "no key given"
  fi

  # A key fetched from another box is THAT server's key. Writing it to
  # BONSAI_API_KEY would hand it to start-server.sh, which would then serve
  # this machine with a foreign key and 401 every local client.
  if [[ -n "$SSH_TARGET" && "$SSH_TARGET" != "127.0.0.1" && "$SSH_TARGET" != "localhost" ]]; then
    append_env "BONSAI_SERVER_KEY=$FETCHED"
    BONSAI_SERVER_KEY="$FETCHED"
    ok "BONSAI_SERVER_KEY saved to .env (key for $SSH_TARGET, not this machine)"
  else
    append_env "BONSAI_API_KEY=$FETCHED"
    BONSAI_API_KEY="$FETCHED"
    ok "BONSAI_API_KEY saved to .env"
  fi
fi

if [[ "$MODE" != check ]] && ! command -v opencode >/dev/null; then
  info "opencode is not installed"
  case "$(uname -s)" in
    Darwin|Linux|MINGW*|MSYS*|CYGWIN*) install_cmd="npm install -g opencode-ai@1.18.31" ;;
    *) err "unsupported OS — install opencode manually: https://opencode.ai/docs/" ;;
  esac
  command -v npm >/dev/null || err "npm is required to install the pinned OpenCode release"
  printf 'Install it with:  %s   [y/N] ' "$install_cmd"
  read -r answer
  [[ "$answer" =~ ^[Yy]$ ]] || err "aborted — install opencode and re-run"
  eval "$install_cmd"
  command -v opencode >/dev/null || err "install finished but opencode is not on PATH — open a new shell and re-run"
fi
[[ "$MODE" == check ]] || ok "opencode installed: $(opencode --version 2>/dev/null || echo present)"
# Where the CLIENT connects. Deliberately NOT BONSAI_HOST: start-server.sh uses
# that as the server's BIND address (`--host "$HOST"`), and a machine cannot
# bind an address it does not own. Connect and bind are different facts.
#
# BONSAI_HOST is still accepted as a fallback for configs written before the
# split — except 0.0.0.0, a bind wildcard that is never a connect target.
if [[ -n "${BONSAI_SERVER_URL:-}" ]]; then
  BASE_URL="$BONSAI_SERVER_URL"
else
  CONNECT_HOST="${BONSAI_HOST:-127.0.0.1}"
  [[ "$CONNECT_HOST" == "0.0.0.0" ]] && CONNECT_HOST=127.0.0.1
  BASE_URL="http://${CONNECT_HOST}:${BONSAI_PORT:-8080}/v1"
fi

# The key belongs to whichever server BASE_URL names. BONSAI_API_KEY is this
# machine's server key — right when the target is local, a guaranteed 401 when
# it is another box with its own key.
API_KEY="${BONSAI_SERVER_KEY:-${BONSAI_API_KEY:-}}"
[[ -n "$API_KEY" ]] || err "no BONSAI_SERVER_KEY or BONSAI_API_KEY in $ENV_FILE"

# limit.context tells opencode when to compact, and must match what the server
# actually serves: too high and long requests are rejected mid-session, too low
# and it compacts away context you paid for.
#
# Ask the server rather than guess. Guessing from this machine's `uname` is
# wrong whenever the server is elsewhere — a Mac pointed at the 8 GB Linux box
# would advertise 32768 against a server serving 8192. /props?model=<id>
# reports the resident child's real n_ctx.
if [[ -n "${BONSAI_CTX:-}" ]]; then
  CTX="$BONSAI_CTX"
  CTX_SOURCE="BONSAI_CTX in .env"
else
  CTX="$(curl -s -m 5 -H "Authorization: Bearer $API_KEY" \
           "${BASE_URL%/v1}/props?model=$DEFAULT_MODEL" 2>/dev/null \
         | python3 -c 'import sys,json
try:
    n = json.load(sys.stdin).get("default_generation_settings", {}).get("n_ctx")
    print(n if isinstance(n, int) and n > 0 else "")
except Exception:
    print("")' 2>/dev/null || true)"
  if [[ -n "$CTX" ]]; then
    CTX_SOURCE="the server ($BASE_URL)"
  else
    # Unreachable at setup time. 8192 is the conservative floor across both
    # shipped configurations (docker/compose.linux.yaml) — advertising less
    # than the server allows only costs early compaction, while advertising
    # more gets requests rejected.
    CTX=8192
    CTX_SOURCE="fallback (server unreachable; set BONSAI_CTX to pin it)"
  fi
fi

# The text-only preset can carry a bigger context than the vision ones (see
# models.ini.in), so one number for all three would either reject long requests
# against the text model or compact it away early. Ask about that model
# specifically; fall back to the shared CTX when it cannot be reached.
TEXT_MODEL="bonsai-27b-ternary-text"
if [[ -n "${BONSAI_CTX_TEXT:-}" ]]; then
  CTX_TEXT="$BONSAI_CTX_TEXT"
  CTX_TEXT_SOURCE="BONSAI_CTX_TEXT in .env"
else
  CTX_TEXT="$(curl -s -m 5 -H "Authorization: Bearer $API_KEY" \
           "${BASE_URL%/v1}/props?model=$TEXT_MODEL" 2>/dev/null \
         | python3 -c 'import sys,json
try:
    n = json.load(sys.stdin).get("default_generation_settings", {}).get("n_ctx")
    print(n if isinstance(n, int) and n > 0 else "")
except Exception:
    print("")' 2>/dev/null || true)"
  if [[ -n "$CTX_TEXT" ]]; then
    CTX_TEXT_SOURCE="the server ($BASE_URL)"
  else
    CTX_TEXT="$CTX"
    CTX_TEXT_SOURCE="same as the shared context"
  fi
fi

# Render and compare in Python: a shell heredoc cannot escape these values, so
# a context of "32k" or a key containing a quote silently produced invalid JSON
# while the script reported success. json.dump cannot.
#
# The key goes over stdin, not argv or the environment: `ps eww` exposes both.
PYHELPER="$(mktemp)"
trap 'rm -f "$PYHELPER"' EXIT
cat > "$PYHELPER" <<'PYEOF'
import hashlib, json, os, sys

mode, cfg_path, base_url, ctx_raw, ctx_source, ctx_text_raw = sys.argv[1:7]
api_key = sys.stdin.read()

try:
    ctx = int(ctx_raw)
    if ctx <= 0:
        raise ValueError
except ValueError:
    sys.exit(f"err  context must be a positive integer, got {ctx_raw!r} (from {ctx_source})")

try:
    ctx_text = int(ctx_text_raw)
    if ctx_text <= 0:
        raise ValueError
except ValueError:
    sys.exit(f"err  text context must be a positive integer, got {ctx_text_raw!r}")

MODELS = ("bonsai-27b-1bit", "bonsai-27b-ternary", "bonsai-27b-ternary-text")
TEXT_MODEL = "bonsai-27b-ternary-text"
AGENT_CONTEXT_KEYS = {
    "qwen3.5-9b-q4_k_m": "QCM_QWEN9_CTX",
    "granite-4.1-8b-q4_k_m": "QCM_GRANITE_CTX",
    "qwen3.5-4b-q4_k_m": "QCM_QWEN4_CTX",
    "qwen3.6-35b-a3b": "QCM_QWEN36_CTX",
    "gemma4-e4b": "QCM_GEMMA4_CTX",
}
contexts = {m: ctx_text if m == TEXT_MODEL else ctx for m in MODELS}
for model, key in AGENT_CONTEXT_KEYS.items():
    value = os.environ.get(key) or os.environ.get("QCM_AGENT_CTX") or "8192"
    try:
        contexts[model] = int(value)
        if contexts[model] <= 0:
            raise ValueError
    except ValueError:
        sys.exit(f"err  {key} must be a positive integer, got {value!r}")
MODELS += tuple(AGENT_CONTEXT_KEYS)
# Top-level keys this script does not own. opencode keeps MCP servers here,
# and rewriting the file from `expected` alone silently deleted them.
PRESERVED_KEYS = ("mcp", "agent", "instructions", "permission", "keybinds",
                  "formatter", "lsp", "theme", "share", "autoupdate", "plugin")
expected = {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
        "bonsai": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Bonsai local",
            "options": {"baseURL": base_url, "apiKey": api_key},
            # The text preset gets its own context: it is the agent/coding
            # preset and models.ini.in lets it run larger than the vision ones.
            "models": {m: {"name": m,
                           "limit": {"context": contexts[m],
                                     "output": 4096}}
                       for m in MODELS},
        }
    },
    "model": "bonsai/bonsai-27b-ternary-text",
}

def digest(value):
    """Compare secrets without printing them. A 14-char prefix of a live key
    leaks 7 hex characters of it into scrollback and CI logs."""
    return hashlib.sha256((value or "").encode()).hexdigest()[:8]

if mode == "check":
    try:
        with open(cfg_path) as fh:
            actual = json.load(fh)
        if not isinstance(actual, dict):
            raise ValueError("config must be a JSON object, not " + type(actual).__name__)
    except (OSError, ValueError) as exc:
        sys.exit(f"err  {cfg_path} is unreadable or not valid JSON: {exc}")

    # Compare the whole rendered document, not a hand-picked subset: a config
    # with the right url and key but no models, or a stale model list, is still
    # a config opencode cannot use.
    ctx_explicit = ctx_source.startswith("BONSAI_CTX")

    def strip_ctx(doc):
        clone = json.loads(json.dumps(doc))
        for m in clone.get("provider", {}).get("bonsai", {}).get("models", {}).values():
            m.get("limit", {}).pop("context", None)
        # Keys this script does not own are not drift. Without this, adding the
        # `mcp` block the header tells you to add reported the config as broken.
        for key in PRESERVED_KEYS:
            clone.pop(key, None)
        return clone

    drift, notes = [], []
    a_opts = actual.get("provider", {}).get("bonsai", {}).get("options", {})
    if a_opts.get("baseURL") != base_url:
        drift.append(f"baseURL: config has {a_opts.get('baseURL')!r}, .env implies {base_url!r}")
    if digest(a_opts.get("apiKey")) != digest(api_key):
        drift.append(f"apiKey: config sha256:{digest(a_opts.get('apiKey'))}, .env sha256:{digest(api_key)}")

    a_models = actual.get("provider", {}).get("bonsai", {}).get("models", {})
    if set(a_models) != set(MODELS):
        drift.append(f"models: config has {sorted(a_models)}, expected {sorted(MODELS)}")
    for name in sorted(set(a_models) & set(MODELS)):
        got = a_models[name].get("limit", {}).get("context")
        want = contexts[name]
        if got != want:
            if name in AGENT_CONTEXT_KEYS:
                explicit = bool(os.environ.get(AGENT_CONTEXT_KEYS[name]) or os.environ.get("QCM_AGENT_CTX"))
                source = AGENT_CONTEXT_KEYS[name] + "/QCM_AGENT_CTX"
            else:
                explicit = ctx_explicit or (name == TEXT_MODEL and bool(os.environ.get("BONSAI_CTX_TEXT")))
                source = ctx_source
            msg = f"{name}: context {got}, {source} says {want}"
            (drift if explicit else notes).append(msg)

    if strip_ctx(actual) != strip_ctx(expected):
        found_specific = False
        for field in ("npm", "name"):
            a = actual.get("provider", {}).get("bonsai", {}).get(field)
            e = expected["provider"]["bonsai"][field]
            if a != e:
                drift.append(f"provider.{field}: config has {a!r}, expected {e!r}")
                found_specific = True
        if actual.get("model") != expected["model"]:
            drift.append(f"default model: config has {actual.get('model')!r}, expected {expected['model']!r}")
            found_specific = True
        if not found_specific:
            # Structural difference not covered by specific checks above
            drift.append("config structure differs from expected (possibly $schema, limit.output, or provider keys)")

    if drift:
        print("err  opencode config has drifted from .env:", file=sys.stderr)
        for d in drift:
            print(f"       - {d}", file=sys.stderr)
        print("     fix with: ./setup-opencode.sh", file=sys.stderr)
        sys.exit(1)
    for n in notes:
        print(f"note  {n}")
    print(f" ok  opencode config matches .env ({base_url})")
    sys.exit(0)

# Write mode. umask before creating: chmod after the fact leaves a window in
# which a file containing the API key is world-readable.
os.makedirs(os.path.dirname(cfg_path), exist_ok=True)

# Carry over the keys this script does not own. Writing `expected` alone wiped
# out any `mcp` block -- the very thing the header tells you to add by hand --
# on every run, with no warning.
document = dict(expected)
try:
    with open(cfg_path) as fh:
        previous = json.load(fh)
    if isinstance(previous, dict):
        kept = [k for k in PRESERVED_KEYS if k in previous]
        for k in kept:
            document[k] = previous[k]
        if kept:
            print(f"note  kept your {', '.join(kept)} from the existing config")
except (OSError, ValueError):
    pass  # no readable config yet: nothing to preserve

old = os.umask(0o077)
try:
    with open(cfg_path, "w") as fh:
        json.dump(document, fh, indent=2)
        fh.write("\n")
finally:
    os.umask(old)
os.chmod(cfg_path, 0o600)
print(f" ok  config written to {cfg_path} (mode 600 — it holds the API key)")
PYEOF

if [[ "$MODE" == check ]]; then
  [[ -f "$CONFIG" ]] || err "no config at $CONFIG — run ./setup-opencode.sh to create it"
else
  info "writing $CONFIG ($BASE_URL, context $CTX from $CTX_SOURCE)"
fi

printf '%s' "$API_KEY" | python3 "$PYHELPER" "$MODE" "$CONFIG" "$BASE_URL" "$CTX" "$CTX_SOURCE" "$CTX_TEXT"
RC=$?
[[ "$MODE" == check ]] && exit $RC

echo
echo "==> done. Run:  opencode"
