#!/usr/bin/env bash
# Playwright MCP server — gives Bonsai browser tools inside Open WebUI.
#
# Open WebUI's MCP client (utils/mcp/client.py) speaks STREAMABLE HTTP only —
# it uses streamablehttp_client, not stdio and not SSE. So this must run as an
# HTTP server, not as a stdio subprocess.
#
# Endpoint: http://localhost:8931/mcp
#
# IMPORTANT: use the hostname "localhost", not "127.0.0.1". Playwright MCP host-
# checks incoming requests against the bound host and returns 403 for a
# mismatch — 127.0.0.1 is rejected even though it resolves to the same machine.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
# .env must be sourced BEFORE the defaults below — it is the one source of
# truth, so PW_MCP_* in .env wins over them. Every sibling launcher does this
# (start-server.sh, start-webui.sh, bench.sh); without it the PW_MCP_* entries
# in .env.example are dead config that silently does nothing.
[[ -f "$ROOT/.env" ]] && set -a && . "$ROOT/.env" && set +a

PORT="${PW_MCP_PORT:-8931}"
BROWSER="${PW_MCP_BROWSER:-chrome}"
OUTPUT_DIR="${PW_MCP_OUTPUT_DIR:-$ROOT/.playwright-mcp}"

# --isolated keeps the profile in memory so runs don't accumulate cookies or
# touch your real Chrome profile. Drop it to reuse a logged-in profile.
# Headed by default: for a "guided session" you want to watch what it does.
# --snapshot-mode none is the difference between usable and unusable here.
# The default ("full") attaches the page's entire accessibility tree to EVERY
# tool response. On a real commercial site that is enormous — allegro.pl
# produced 56,928 characters (~15,800 tokens) in one response. At ~50 t/s
# prefill on a 27B that is ~7 minutes of prompt processing before a single
# token is generated, and it compounds because every later step resends the
# whole conversation. Measured with this flag: 279 chars, a 99.5% reduction.
# The model can still call browser_snapshot explicitly when it needs the tree,
# and browser_find gives targeted results.
#
# --output-dir is where the tools that write files (screenshots, saved
# sessions, console/network dumps a tool chooses to persist) put them. It
# replaces the old --output-mode flag, which upstream removed — passing it to
# @playwright/mcp 0.0.79 aborts the server with "unknown option
# '--output-mode'". Note this is only a LOCATION, not a routing switch: 0.0.79
# has no size threshold that diverts a tool response to disk, so response text
# still reaches the model inline. The context saving here comes entirely from
# --snapshot-mode above, which is untouched.
#
# --output-max-size caps the output directory's total bytes, evicting
# oldest-first. Upstream's implementation opens with `if (!maxSize) return`, so
# leaving it unset means no eviction at all and screenshots/snapshots/traces
# accumulate until the disk fills. Default to a generous 512 MiB — far above
# anything a normal session produces, so in practice it only ever trims
# long-forgotten artifacts. Set PW_MCP_OUTPUT_MAX_SIZE=0 to opt out entirely
# and keep everything forever.
#
# Note this DELETES files once the cap is exceeded, oldest first. Point
# PW_MCP_OUTPUT_DIR somewhere else if you need artifacts kept permanently.
OUTPUT_MAX_SIZE="${PW_MCP_OUTPUT_MAX_SIZE:-536870912}"
MAX_SIZE_ARGS=(--output-max-size "$OUTPUT_MAX_SIZE")
[[ "$OUTPUT_MAX_SIZE" == "0" ]] && MAX_SIZE_ARGS=()

exec npx -y @playwright/mcp@latest \
  --port "$PORT" \
  --host 127.0.0.1 \
  --browser "$BROWSER" \
  --isolated \
  --snapshot-mode "${PW_MCP_SNAPSHOT:-none}" \
  --output-dir "$OUTPUT_DIR" \
  "${MAX_SIZE_ARGS[@]}" \
  --image-responses omit \
  "$@"
