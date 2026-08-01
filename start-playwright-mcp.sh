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
exec npx -y @playwright/mcp@latest \
  --port "$PORT" \
  --host 127.0.0.1 \
  --browser "$BROWSER" \
  --isolated \
  "$@"
