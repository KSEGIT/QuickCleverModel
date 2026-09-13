# Claude Code against Bonsai

Use your own server as the model behind Claude Code. **No proxy.** llama-server
answers `/v1/messages`, the Anthropic Messages API, alongside the OpenAI routes.

Written against Claude Code 2.1. Unlike Codex, MCP servers work here, so this is
the client to use when you want Playwright.

## Run it

`claude-bonsai.sh` in the repo root sets the environment and launches Claude Code:

```bash
./claude-bonsai.sh                 # interactive
./claude-bonsai.sh -p 'your task'  # one-shot
```

Every argument is passed straight through. It reads `.env` for the key and
`BONSAI_SERVER_URL`, so it follows the same settings as everything else here.

## Why a launcher and not settings.json

Putting `ANTHROPIC_BASE_URL` in `~/.claude/settings.json` repoints **every**
Claude Code session at this server, including the ones you want running on a
hosted model. The launcher scopes it to one session, and removing it needs no
edit. Set it globally only if this server is the only model you use.

## What it sets, and why

| variable | why |
|---|---|
| `ANTHROPIC_BASE_URL` | the server. No `/v1` suffix — Claude Code appends it |
| `ANTHROPIC_AUTH_TOKEN` | your `BONSAI_API_KEY` |
| `ANTHROPIC_MODEL` | `bonsai-27b-ternary-text` — the text preset, see below |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` | Claude Code runs small background jobs on a Haiku-class model. Unset, it asks this server for a model it does not have |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | Claude Code does not know this model, and assumes 200k. The text preset serves 131072, so without this it compacts too late and the server truncates |

## Which model

`bonsai-27b-ternary-text` — the text-only preset. It carries no `mmproj`, and
loading vision disables the prompt cache for **every** turn. An agent re-sends a
long conversation on each step, so that cache matters far more here than image
input. It is also the only preset with 128k of context (`BONSAI_CTX_TEXT`).

## MCP, including Playwright

Claude Code sends tool definitions flat, which is what llama.cpp understands, so
MCP servers work. Add one for a single session:

```bash
cat > mcp.json <<'JSON'
{ "mcpServers": { "playwright": { "command": "npx", "args": ["-y", "@playwright/mcp@latest"] } } }
JSON

./claude-bonsai.sh --mcp-config mcp.json -p 'Open https://example.com and tell me the page title.'
```

Measured: the model calls `mcp__playwright__browser_navigate`, then
`browser_snapshot`, and reports the title. Two turns, about two minutes.

This is the difference from Codex, which wraps each MCP server in one tool of
`type: "namespace"` that llama.cpp cannot read — see [codex.md](codex.md).

## What to expect

Slower than Codex, for one specific reason: Claude Code's system prompt and
tools are large. Measured here, a trivial one-shot sent **34,888** input tokens
against Codex's ~11,000. Prefill runs at roughly 500 tok/s, so that is about a
minute before the first token, then generation at ~15 tok/s.

`--bare` skips hooks, plugins and `CLAUDE.md` files. The same prompt then cost
5,802 tokens instead of 34,888 — worth it for quick one-shots, though you lose
the setup you configured.

Long conversations are what the 128k context is for, but each turn re-sends the
lot. Keep sessions tight.

## Noise you can ignore

- **`claude.ai connectors are disabled ...`** — expected. An auth token is set,
  so your claude.ai login is not used.
- **`total_cost_usd`** in `--output-format json` — meaningless. Claude Code
  prices the tokens as if they went to Anthropic. This server costs electricity.
