#!/usr/bin/env bash
# Configure this client only; never install Codex or change the server.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/codex-client/setup.py" "$@"
