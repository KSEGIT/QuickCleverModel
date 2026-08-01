#!/usr/bin/env bash
# One-command control for the whole local Bonsai stack.
#
#   ./stack.sh up | down | restart | status | logs [svc] | open
#
# Three processes, all native on the host — nothing runs in Docker, because
# Docker on macOS cannot reach Metal (see README).
#
#   llama       :8080  Bonsai 27B on Metal            start-server.sh
#   playwright  :8931  browser tools over MCP         start-playwright-mcp.sh
#   webui       :9090  Open WebUI chat interface      start-webui.sh
#
# Start order matters: llama first (webui probes it for the model list),
# playwright before webui so the tool server is live when the UI connects.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
RUN="$ROOT/run"
LOGS="$RUN/logs"
mkdir -p "$LOGS"

SERVICES=(llama playwright webui)

port_of() { case "$1" in llama) echo 8080;; webui) echo 9090;; playwright) echo 8931;; esac; }
script_of() {
  case "$1" in
    llama)      echo "$ROOT/start-server.sh";;
    webui)      echo "$ROOT/start-webui.sh";;
    playwright) echo "$ROOT/start-playwright-mcp.sh";;
  esac
}
# Readiness is "is the port accepting connections", not "did the process start" —
# llama-server maps 6.7 GB of weights and Open WebUI runs DB migrations, so both
# are up for a while before they can actually serve.
listening() { lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null | head -1; }

start_one() {
  local svc="$1" port pid
  port="$(port_of "$svc")"
  pid="$(listening "$port")"
  if [[ -n "$pid" ]]; then
    printf "  %-11s already running (pid %s)\n" "$svc" "$pid"
    return 0
  fi
  printf "  %-11s starting" "$svc"
  # nohup + disown so services outlive this shell and any parent that spawned it
  nohup "$(script_of "$svc")" > "$LOGS/$svc.log" 2>&1 < /dev/null &
  disown
  for _ in $(seq 1 120); do
    pid="$(listening "$port")"
    [[ -n "$pid" ]] && { printf "\r  %-11s ready on :%s (pid %s)\n" "$svc" "$port" "$pid"; return 0; }
    # bail out early if the process died rather than waiting the full 4 minutes
    if grep -qaiE "Traceback|error loading model|failed to load|Address already in use" "$LOGS/$svc.log" 2>/dev/null; then
      printf "\r  %-11s FAILED — see %s\n" "$svc" "$LOGS/$svc.log"; return 1
    fi
    printf "."; sleep 2
  done
  printf "\r  %-11s TIMEOUT — see %s\n" "$svc" "$LOGS/$svc.log"; return 1
}

stop_one() {
  local svc="$1" port pid
  port="$(port_of "$svc")"
  pid="$(listening "$port")"
  if [[ -z "$pid" ]]; then printf "  %-11s not running\n" "$svc"; return 0; fi
  kill "$pid" 2>/dev/null
  for _ in $(seq 1 15); do
    [[ -z "$(listening "$port")" ]] && { printf "  %-11s stopped\n" "$svc"; return 0; }
    sleep 1
  done
  kill -9 "$pid" 2>/dev/null
  printf "  %-11s force-killed\n" "$svc"
}

cmd_up() {
  [[ -f "$ROOT/.env" ]] || { echo "missing .env — create it with:"; echo "  printf 'BONSAI_API_KEY=bonsai-%s\\n' \"\$(openssl rand -hex 20)\" > .env && chmod 600 .env"; exit 1; }
  echo "starting stack..."
  for s in "${SERVICES[@]}"; do start_one "$s" || exit 1; done
  echo
  cmd_status
  echo
  echo "  chat UI -> http://127.0.0.1:9090"
}

cmd_down() { echo "stopping stack..."; for i in 2 1 0; do stop_one "${SERVICES[$i]}"; done; }

cmd_status() {
  printf "  %-11s %-6s %-9s %s\n" SERVICE PORT STATE PID
  for s in "${SERVICES[@]}"; do
    local port pid; port="$(port_of "$s")"; pid="$(listening "$port")"
    printf "  %-11s %-6s %-9s %s\n" "$s" "$port" "$([[ -n $pid ]] && echo up || echo down)" "${pid:--}"
  done
}

cmd_logs() {
  local svc="${1:-}"
  if [[ -z "$svc" ]]; then tail -n 40 -f "$LOGS"/*.log; else tail -n 60 -f "$LOGS/$svc.log"; fi
}

case "${1:-}" in
  up)      cmd_up;;
  down)    cmd_down;;
  restart) cmd_down; echo; cmd_up;;
  status)  cmd_status;;
  logs)    cmd_logs "${2:-}";;
  open)    open http://127.0.0.1:9090;;
  *) sed -n '2,12p' "$0" | sed 's/^# \?//'; exit 1;;
esac
