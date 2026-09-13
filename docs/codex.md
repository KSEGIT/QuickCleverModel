# Codex CLI against Bonsai

Use your own server as the model behind [Codex CLI](https://developers.openai.com/codex/cli).
No proxy in the middle: Codex talks to llama-server directly.

Written against codex-cli 0.154. Codex moves fast — if a key here is rejected,
check `codex --help` and the current docs before assuming this page is right.

## Before you start

Three things must be true.

1. The stack is up. `docker compose --env-file .env -f docker/compose.linux.yaml ps`
   shows the `llama` service running.
2. You can reach it. The API binds a Tailscale address, not loopback, so
   Tailscale must be up on both machines. `curl localhost:8080` from the server
   returns nothing — that is expected, see
   [architecture.md](architecture.md#ports--security).
3. You have `BONSAI_API_KEY` from the server's `.env`.

Check all three at once:

```bash
# Missing key: 401
curl -sS -o /dev/null -w '%{http_code}\n' \
  -H 'Content-Type: application/json' \
  -d '{"model":"bonsai-27b-ternary-text","input":"hi","max_output_tokens":1}' \
  http://100.x.y.z:8080/v1/responses

# Invalid key: 401
curl -sS -o /dev/null -w '%{http_code}\n' \
  -H 'Authorization: Bearer not-the-bonsai-key' \
  -H 'Content-Type: application/json' \
  -d '{"model":"bonsai-27b-ternary-text","input":"hi","max_output_tokens":1}' \
  http://100.x.y.z:8080/v1/responses

# Valid key: 200
curl -sS -o /dev/null -w '%{http_code}\n' \
  -H "Authorization: Bearer $BONSAI_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"bonsai-27b-ternary-text","input":"hi","max_output_tokens":1}' \
  http://100.x.y.z:8080/v1/responses
```

The three calls should return `401`, `401`, and `200`, respectively. `200` and
you are ready.

**Do not test with `/v1/models`.** That route answers without a key, so it
returns `200` even when your key is wrong, and it will tell you the setup works
when it does not. Use `/v1/responses` to check the key for this Codex setup.

## Configure it

Two files. The provider goes in `~/.codex/config.toml`:

```toml
[model_providers.bonsai]
name = "Bonsai"
base_url = "http://100.x.y.z:8080/v1"
env_key = "BONSAI_API_KEY"
wire_api = "responses"
request_max_retries = 2
stream_max_retries = 3
# A slow local model streams for a long time. The default gives up too early.
stream_idle_timeout_ms = 600000
```

The profile goes in its own file, `~/.codex/bonsai.config.toml`:

```toml
model = "bonsai-27b-ternary-text"
model_provider = "bonsai"
model_context_window = 65536
```

Two traps here:

- **`wire_api` must be `responses`.** Codex removed `wire_api = "chat"`; it is
  now a hard error that tells you to use `responses`. This works only because
  llama-server serves `/v1/responses` as well as `/v1/chat/completions`.
- **The profile does not go in `config.toml`.** A `[profiles.bonsai]` table
  there is refused, with an error telling you to move it to
  `~/.codex/bonsai.config.toml`. Put it in its own file from the start.

## Run it

Codex reads the key from the environment. On the client, keep only this key in
a client-side environment file, then export it:

```bash
BONSAI_ENV_FILE="$HOME/.config/bonsai.env" # contains only BONSAI_API_KEY=...
set -a && . "$BONSAI_ENV_FILE" && set +a
codex -p bonsai                    # interactive
codex exec -p bonsai 'your prompt' # one-shot
```

Without the key you get `Missing environment variable: BONSAI_API_KEY`. To
avoid exporting it every session, put `export BONSAI_API_KEY=...` in your
`~/.zshrc`, or wrap it:

```bash
BONSAI_REPO=/path/to/QuickCleverModel
BONSAI_ENV_FILE="$HOME/.config/bonsai.env"
bonsai() { (cd "$BONSAI_REPO" && set -a && . "$BONSAI_ENV_FILE" && set +a && codex -p bonsai "$@"); }
```

## Which model

`bonsai-27b-ternary-text` is the default in the profile above, and it is the
right one for agent work. It is the same weights as `bonsai-27b-ternary` with
no `mmproj`, so it cannot see images — but loading vision disables the prompt
cache for **every** turn, and an agent re-sends a long conversation on each
step. Text-only keeps multi-turn caching, which matters far more here than
image input. See the comment on that section in `models.ini.in`.

Switch per run with `-m`:

| model | when |
|---|---|
| `bonsai-27b-ternary-text` | agent and coding work — the default |
| `bonsai-27b-ternary` | you need to send an image |
| `bonsai-27b-1bit` | smaller and faster, lower quality |

Only one model is resident at a time (`--models-max 1`). Switching unloads the
other, which takes seconds and can briefly return
`model name=... failed to load` while it swaps. Retry.

## What to expect

Around 15 tokens per second on an RTX 3070 Ti. A short turn with one tool call
takes 15 to 40 seconds. That is usable for small, well-scoped tasks and painful
for a long refactor. Numbers: [tuning.md](tuning.md).

Tool calling works — the model returns proper `tool_calls` with valid JSON
arguments, and Codex completes the full round trip.

Codex prints `Model metadata for bonsai-27b-ternary-text not found` on every
run. It is harmless: Codex keeps a catalogue of its own models and yours is not
in it, so it falls back to defaults. `model_context_window` in the profile is
what supplies the number that matters.

## The chat template

The template baked into the Bonsai GGUFs only renders the first system message
and raises `System message must be at the beginning.` for any after it. Codex
sends two — one from the Responses API `instructions` field, one from a
`developer` item it includes in every request — so **every** Codex call
returned `400 Unable to generate parser for this template`.

`bonsai-chat-template.jinja` in the repo root fixes this, and each model
section in `models.ini.in` points at it with `chat-template-file`. If you see
that 400 again, the server is running without it: check that
`chat-template-file` reached `run/models.ini`, and rebuild so the template is
in the image.

Other clients were never affected. OpenCode and plain `curl` send one system
message, so they never hit the raise.

## Other tools

Anything that speaks the OpenAI API works the same way — point it at
`http://100.x.y.z:8080/v1` with your key. `setup-opencode.sh` does this for
OpenCode, including a drift check against `.env`.

Claude Code speaks the Anthropic Messages API rather than the OpenAI one, and
needs no proxy: this server answers `/v1/messages` as well. See
[claude-code.md](claude-code.md).

## MCP does not work here

Codex sends each MCP server as a single tool of `type: "namespace"`, with the
real tools nested inside it. llama.cpp does not implement that type, so the
model never sees them. Measured against this server with one identical tool
sent both ways:

| tool shape | model calls it |
|---|---|
| namespace-wrapped, as Codex sends MCP | no |
| the same tool, flat | yes |

So Playwright, context7 and the rest are unreachable from Codex here, however
you prompt. The model is not at fault and a bigger model would not help.

Use Claude Code ([claude-code.md](claude-code.md)) or OpenCode for MCP work:
both send tool definitions flat, and both drive Playwright against this server.
Codex remains fine for everything that does not need MCP.
