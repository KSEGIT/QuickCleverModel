#!/usr/bin/env bash
# Repeatable throughput benchmark against the running llama-server.
#
# `make bench` fires ONE request, which is far too noisy to compare flag
# changes: back-to-back runs on this box vary by more than most optimisations
# are worth. This runs N iterations and reports the median, so an A/B is
# actually decidable.
#
#   ./bench.sh                          # 5 reps, default model
#   ./bench.sh -n 9 -m bonsai-27b-1bit  # 9 reps, the 1-bit model
#   ./bench.sh -w repetitive            # workload that reuses input tokens
#
# Prints median generation tok/s, prompt tok/s, and the speculative-decoding
# draft acceptance rate — the last one matters because a low acceptance rate
# means the drafter is burning bandwidth for nothing.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
[[ -f "$ROOT/.env" ]] && set -a && . "$ROOT/.env" && set +a
: "${BONSAI_API_KEY:?BONSAI_API_KEY not set}"

REPS=5
MODEL="bonsai-27b-ternary"
WORKLOAD="generic"
MAXTOK=200
PORT="${BONSAI_PORT:-8080}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) REPS="$2"; shift 2;;
    -m) MODEL="$2"; shift 2;;
    -w) WORKLOAD="$2"; shift 2;;
    -t) MAXTOK="$2"; shift 2;;
    *) echo "usage: $0 [-n reps] [-m model] [-w generic|repetitive] [-t max_tokens]" >&2; exit 1;;
  esac
done

# Two workloads, because n-gram speculation behaves completely differently on
# them: "generic" invents new tokens, "repetitive" echoes input tokens back,
# which is the case n-gram drafting is designed to win.
case "$WORKLOAD" in
  generic)
    PROMPT="Write a short paragraph explaining why the sky appears blue."
    ;;
  repetitive)
    PROMPT="Repeat the following list back to me exactly, unchanged: alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango"
    ;;
  *) echo "unknown workload: $WORKLOAD" >&2; exit 1;;
esac

# --reasoning off is not settable per-request, so allow enough tokens that a
# thinking block does not consume the entire budget and leave nothing measured.
printf "model=%s workload=%s reps=%s max_tokens=%s\n" "$MODEL" "$WORKLOAD" "$REPS" "$MAXTOK"

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

for i in $(seq 1 "$REPS"); do
  curl -s --max-time 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" -H "Authorization: Bearer $BONSAI_API_KEY" \
    -d "$(printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%s,"cache_prompt":false}' \
          "$MODEL" "$PROMPT" "$MAXTOK")" \
  | python3 -c "
import json,sys
d=json.load(sys.stdin)
if 'error' in d:
    print('ERR', d['error'].get('message','')[:120], file=sys.stderr); raise SystemExit(1)
t=d.get('timings',{})
print('%.3f %.3f %d %d' % (t.get('predicted_per_second',0), t.get('prompt_per_second',0),
                           t.get('draft_n',0), t.get('draft_n_accepted',0)))
" >> "$TMP"
  printf "  rep %s/%s: %s\n" "$i" "$REPS" "$(tail -1 "$TMP")"
done

python3 - "$TMP" <<'PY'
import sys, statistics
rows = [l.split() for l in open(sys.argv[1]) if l.strip()]
gen  = sorted(float(r[0]) for r in rows)
pp   = sorted(float(r[1]) for r in rows)
dn   = sum(int(r[2]) for r in rows)
da   = sum(int(r[3]) for r in rows)
print()
print("  generation  median %.2f tok/s   min %.2f   max %.2f   spread %.1f%%"
      % (statistics.median(gen), gen[0], gen[-1],
         (gen[-1]-gen[0])/statistics.median(gen)*100 if gen else 0))
print("  prompt      median %.2f tok/s" % statistics.median(pp))
if dn:
    print("  draft       %d/%d accepted (%.1f%%)" % (da, dn, da/dn*100))
else:
    print("  draft       none (speculation off or unused)")
PY
