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

PORT="${PW_MCP_PORT:-8931}"
BROWSER="${PW_MCP_BROWSER:-chrome}"

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
# --output-mode file sends snapshot/console/network dumps to disk instead of
# into the model's context, for the same reason.
exec npx -y @playwright/mcp@latest \
  --port "$PORT" \
  --host 127.0.0.1 \
  --browser "$BROWSER" \
  --isolated \
  --snapshot-mode "${PW_MCP_SNAPSHOT:-none}" \
  --output-mode file \
  --image-responses omit \
  "$@"
