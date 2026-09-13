# Claude Code against Bonsai

Use your own server as the model behind Claude Code. **No proxy.** llama-server
answers `/v1/messages`, the Anthropic Messages API, alongside the OpenAI routes.

Written against Claude Code 2.1. Unlike Codex, MCP servers work here, so this is
the client to use when you want Playwright.

## Quick start

Everything you need, in order.

```bash
# 1. The server must be up. On the Linux box:
docker compose --env-file .env -f docker/compose.linux.yaml up -d

# 2. Check the key works. 200 means ready; 401 means the key is wrong.
#    Do NOT test with /v1/models -- that route answers without a key.
curl -sS -o /dev/null -w '%{http_code}\n' \
  -H "x-api-key: $BONSAI_API_KEY" -H 'anthropic-version: 2023-06-01' \
  -H 'Content-Type: application/json' \
  -d '{"model":"bonsai-27b-ternary-text","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}' \
  http://100.x.y.z:8080/v1/messages

# 3. Run it. Arguments pass straight through to claude.
./claude-bonsai.sh                 # interactive
./claude-bonsai.sh -p 'your task'  # one-shot

# 4. Browser automation, or any other MCP server.
echo '{"mcpServers":{"playwright":{"command":"npx","args":["-y","@playwright/mcp@latest"]}}}' > mcp.json
./claude-bonsai.sh --mcp-config mcp.json -p 'Open https://example.com, give me the title.'

# 5. Agents. Subagents run on this server too -- nothing extra to set.
./claude-bonsai.sh -p 'Use the Task tool to check X, then summarise.'
```

The launcher reads `.env` for the key and `BONSAI_SERVER_URL`, so it follows the
same settings as everything else here.

If you want this without the script, it is only environment — see
[what it sets](#what-it-sets-and-why) and export those yourself.

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

## Agents

Subagents work with no extra setup. The launcher points every model name Claude
Code can resolve at this server — `ANTHROPIC_MODEL`, the `sonnet`/`opus`/`haiku`
/`fable` aliases behind `/model`, the background classifier, and
`CLAUDE_CODE_SUBAGENT_MODEL`. Anything left unmapped would be a request leaving
the machine, which then 401s, because the token is a Bonsai key. It also sets
`CLAUDE_CODE_NO_MODEL_FALLBACK=1`, so a failure shows you this server's error
instead of silently retrying somewhere else.

Measured: a one-shot that spawns a general-purpose subagent to run a command and
report back completes in two turns, with `bonsai-27b-ternary-text` as the only
model used.

To run agents on a different preset:

```bash
BONSAI_CLAUDE_AGENT_MODEL=bonsai-27b-ternary ./claude-bonsai.sh
```

Think before you do. The server keeps **one** model resident (`--models-max 1`),
so every hand-off between the main model and the agent unloads one set of
weights and loads the other. On a 27B model that is seconds to a minute per
switch, each way. Same preset for both is almost always faster, and the vision
preset also disables the prompt cache for every turn.

Agents are also where the speed below hurts most: each one carries its own
prompt, and they run one after another on a single GPU, not in parallel.

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
