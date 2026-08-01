#!/usr/bin/env bash
# One-command control for the whole local Bonsai stack.
#
#   ./stack.sh up | down | restart | status | logs [svc] | open | reap
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
port_pid() { lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null | head -1; }

# A bare port check is not enough: anything else bound to 8080 (a stray
# `python -m http.server`, another project) would be reported as our service and
# `up` would skip starting it. Match the process command line too.
sig_of() {
  case "$1" in
    llama)      echo "llama-server";;
    webui)      echo "open-webui";;
    playwright) echo "playwright-mcp";;
  esac
}
listening() {
  local svc="$1" pid
  pid="$(port_pid "$(port_of "$svc")")"
  [[ -z "$pid" ]] && return 1
  if ps -p "$pid" -o command= 2>/dev/null | grep -q "$(sig_of "$svc")"; then
    echo "$pid"; return 0
  fi
  return 1   # port taken by something that is not this service
}

# llama-server binds the port BEFORE it finishes loading the model — measured at
# 0.6s to bind vs 1.8s to serve on a warm cache, and much wider on a cold start
# or with a draft model. Treating "port open" as ready makes the first requests
# fail, so probe the health endpoint for the HTTP services.
ready() {
  local svc="$1"
  listening "$svc" >/dev/null || return 1
  case "$svc" in
    llama|webui)
      [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 \
            "http://127.0.0.1:$(port_of "$svc")/health" 2>/dev/null)" == "200" ]] ;;
    *) return 0 ;;   # playwright has no health route; a bound port is enough
  esac
}
# Distinguish "free" from "occupied by a foreign process" so `up` can say so.
foreign_on_port() {
  local svc="$1" pid
  pid="$(port_pid "$(port_of "$svc")")"
  [[ -n "$pid" ]] && ! listening "$svc" >/dev/null && echo "$pid"
}

start_one() {
  local svc="$1" port pid intruder
  port="$(port_of "$svc")"
  pid="$(listening "$svc")"
  if [[ -n "$pid" ]]; then
    printf "  %-11s already running (pid %s)\n" "$svc" "$pid"
    return 0
  fi
  intruder="$(foreign_on_port "$svc")"
  if [[ -n "$intruder" ]]; then
    printf "  %-11s PORT %s TAKEN by pid %s (%s) — not ours, refusing to start\n" \
      "$svc" "$port" "$intruder" "$(ps -p "$intruder" -o comm= 2>/dev/null)"
    return 1
  fi
  printf "  %-11s starting" "$svc"
  # nohup + disown so services outlive this shell and any parent that spawned it
  nohup "$(script_of "$svc")" > "$LOGS/$svc.log" 2>&1 < /dev/null &
  disown
  for _ in $(seq 1 120); do
    if ready "$svc"; then
      pid="$(listening "$svc")"
      printf "\r  %-11s ready on :%s (pid %s)\n" "$svc" "$port" "$pid"; return 0
    fi
    # bail out early if the process died rather than waiting the full 4 minutes
    if grep -qaiE "Traceback|error loading model|failed to load|Address already in use" "$LOGS/$svc.log" 2>/dev/null; then
      printf "\r  %-11s FAILED — see %s\n" "$svc" "$LOGS/$svc.log"; return 1
    fi
    printf "."; sleep 2
  done
  printf "\r  %-11s TIMEOUT — see %s\n" "$svc" "$LOGS/$svc.log"; return 1
}

stop_one() {
  local svc="$1" port pid ppid
  port="$(port_of "$svc")"
  pid="$(listening "$svc")"
  if [[ -z "$pid" ]]; then printf "  %-11s not running\n" "$svc"; return 0; fi

  # playwright runs as `npx` -> `npm exec` -> `node playwright-mcp`. Only the node
  # child holds the port, so killing just that orphans the npm wrapper — they
  # accumulate one per up/down cycle. Take the parent too when it is the wrapper.
  ppid="$(ps -p "$pid" -o ppid= 2>/dev/null | tr -d ' ')"
  if [[ -n "$ppid" && "$ppid" != 1 ]] && ps -p "$ppid" -o command= 2>/dev/null | grep -q "$(sig_of "$svc")"; then
    kill "$ppid" 2>/dev/null
  fi
  kill "$pid" 2>/dev/null

  for _ in $(seq 1 15); do
    [[ -z "$(listening "$svc")" ]] && { printf "  %-11s stopped\n" "$svc"; return 0; }
    sleep 1
  done
  kill -9 "$pid" 2>/dev/null; [[ -n "$ppid" && "$ppid" != 1 ]] && kill -9 "$ppid" 2>/dev/null
  printf "  %-11s force-killed\n" "$svc"
}

# Reap anything left from earlier crashes or interrupted runs. Matches on the
# service signature, so it will not touch unrelated node/python processes.
#
# CAUTION: it matches by name, not by port — every llama-server on the machine
# dies, including one you started by hand on another port for benchmarking.
# Use `down` for the normal case; `reap` only to clean up after a crash.
cmd_reap() {
  echo "reaping stale processes..."
  for s in "${SERVICES[@]}"; do
    local sig n; sig="$(sig_of "$s")"
    n="$(pgrep -f "$sig" 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "$n" -gt 0 ]]; then
      pkill -f "$sig" 2>/dev/null; sleep 1; pkill -9 -f "$sig" 2>/dev/null
      printf "  %-11s killed %s process(es)\n" "$s" "$n"
    else
      printf "  %-11s nothing stale\n" "$s"
    fi
  done
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
    local port pid; port="$(port_of "$s")"; pid="$(listening "$s")"
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
  reap)    cmd_reap;;
  logs)    cmd_logs "${2:-}";;
  open)    open http://127.0.0.1:9090;;
  *) sed -n '2,12p' "$0" | sed 's/^# \?//'; exit 1;;
esac
