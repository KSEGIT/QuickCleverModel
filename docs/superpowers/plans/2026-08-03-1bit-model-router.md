# 1-bit Bonsai 27B + Router Model Selection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve both Bonsai 27B quantizations (ternary `Q2_0` and 1-bit `Q1_0`) from one `llama-server` router on `:8080`, selectable from the Open WebUI dropdown without a restart.

**Architecture:** `llama-server` launched with no `-m` enters router mode and spawns one child server per model defined in an INI preset. Section names become model IDs. The router owns `--host/--port/--api-key`; children bind `127.0.0.1` unauthenticated.

**Tech Stack:** bash, GNU make, Python 3 stdlib (no pytest — `cache-viz.py` is deliberately dependency-free), `hf` CLI (Homebrew), `llama-server` from `src/llama.cpp-prism` @ `4dd1656`.

**Spec:** `docs/superpowers/specs/2026-08-03-1bit-model-router-design.md`

## Global Constraints

- **Binary:** always prefer `$ROOT/src/llama.cpp-prism/build/bin/llama-server`, falling back to `$ROOT/bin/llama-prism-b9570-0ad1dab/llama-server`. Never Homebrew's — it cannot load ternary `Q2_0`.
- **Model IDs are exactly** `bonsai-27b-ternary` and `bonsai-27b-1bit`. These strings appear in `models.ini.in` section headers, `Makefile` bench default, and README. They must match verbatim everywhere.
- **`--mmproj` must never be passed on the router command line** — `unset_reserved_args(base_preset, true)` (`server-models.cpp:210`) discards it. Per-section only.
- **Absolute paths only** inside the generated preset.
- **`cache-viz.py` stays stdlib-only.** Its docstring promises "Stdlib only; no install step." Tests use `unittest`, not pytest.
- **Never commit weights.** `models/`, `run/`, `src/`, `bin/` are already gitignored.
- **`.env` holds `BONSAI_API_KEY`** and is required by `start-server.sh`; keep that check.
- Existing tunables keep their names and defaults: `BONSAI_HOST` (`0.0.0.0`), `BONSAI_PORT` (`8080`), `BONSAI_CTX` (`32768`), `BONSAI_PARALLEL` (`1`), `BONSAI_SPEC` (`ngram-simple`), `BONSAI_CACHE_REUSE` (`256`). New: `BONSAI_MODELS_MAX` (default `1`).

---

### Task 1: Fetch the 1-bit weights

**Files:**
- Create: `fetch-models.sh`
- Modify: `Makefile` (add `models` target to `.PHONY` and a rule)

**Interfaces:**
- Consumes: nothing.
- Produces: `models/Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf` (3,803,452,480 bytes) and `models/Bonsai-27B-gguf/Bonsai-27B-mmproj-Q8_0.gguf` (629,246,880 bytes). Task 2's preset hardcodes these paths.

- [ ] **Step 1: Write `fetch-models.sh`**

```bash
#!/usr/bin/env bash
# Download the GGUF weights this stack needs from Hugging Face.
#
# Not wired into `make up` — 4.4 GB should be an explicit decision, not a
# side effect of starting the stack.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

command -v hf >/dev/null || {
  echo "hf CLI not found. Install it with: brew install huggingface-cli" >&2
  exit 1
}

# repo <TAB> file <TAB> expected bytes
FILES=$(cat <<'EOF'
prism-ml/Bonsai-27B-gguf	Bonsai-27B-Q1_0.gguf	3803452480
prism-ml/Bonsai-27B-gguf	Bonsai-27B-mmproj-Q8_0.gguf	629246880
EOF
)

# `stat -f%z` is BSD/macOS; this stack is macOS-only (Metal).
size_of() { stat -f%z "$1" 2>/dev/null || echo 0; }

while IFS=$'\t' read -r repo file want; do
  [[ -z "$repo" ]] && continue
  dest_dir="$ROOT/models/${repo#*/}"
  dest="$dest_dir/$file"
  have="$(size_of "$dest")"
  if [[ "$have" == "$want" ]]; then
    printf "  %-32s present (%s bytes)\n" "$file" "$have"
    continue
  fi
  [[ "$have" != 0 ]] && printf "  %-32s size %s != expected %s, re-downloading\n" "$file" "$have" "$want"
  printf "  %-32s downloading...\n" "$file"
  mkdir -p "$dest_dir"
  hf download "$repo" "$file" --local-dir "$dest_dir"
  got="$(size_of "$dest")"
  [[ "$got" == "$want" ]] || { echo "ERROR: $file is $got bytes, expected $want" >&2; exit 1; }
  printf "  %-32s ok (%s bytes)\n" "$file" "$got"
done <<< "$FILES"

echo "weights ready in $ROOT/models"
```

- [ ] **Step 2: Make it executable and add the make target**

```bash
chmod +x fetch-models.sh
```

In `Makefile`, add `models` to the `.PHONY` line so it reads:

```make
.PHONY: help up down restart status logs open bench reap cache-viz models
```

and add this rule immediately after the `help` rule:

```make
models:         ## Download the GGUF weights (4.4 GB, explicit — not part of `make up`)
	@./fetch-models.sh
```

- [ ] **Step 3: Run it**

Run: `make models`
Expected: both files report `ok` with the exact byte counts above. Takes a few minutes on first run.

- [ ] **Step 4: Verify idempotence**

Run: `make models`
Expected: both files report `present`, no download, returns in under a second.

- [ ] **Step 5: Verify on-disk state**

Run: `ls -l models/Bonsai-27B-gguf/`
Expected: `Bonsai-27B-Q1_0.gguf` at 3803452480 and `Bonsai-27B-mmproj-Q8_0.gguf` at 629246880.

- [ ] **Step 6: Commit**

```bash
git add fetch-models.sh Makefile
git commit -m "Add fetch-models.sh: pull 1-bit Bonsai 27B weights from HF"
```

---

### Task 2: Router mode — preset template and `start-server.sh`

**Files:**
- Create: `models.ini.in`
- Modify: `start-server.sh` (full rewrite of the exec section)
- Generated at runtime: `run/models.ini` (already gitignored via `run/`)

**Interfaces:**
- Consumes: the two weight paths from Task 1, plus the existing `models/Ternary-Bonsai-27B-gguf/` files.
- Produces: a router on `$BONSAI_PORT` whose `GET /v1/models` lists exactly `bonsai-27b-ternary` and `bonsai-27b-1bit`. Task 3 (`make bench`) and Task 4 (`cache-viz.py`) depend on those IDs and on the router's log format.

- [ ] **Step 1: Write `models.ini.in`**

`@ROOT@` is replaced with the repo's absolute path at launch.

```ini
; Model presets for llama-server router mode. Rendered to run/models.ini by
; start-server.sh, which substitutes @ROOT@ for the repo path.
;
; Section names ARE the model IDs: server-models.cpp:154 force-sets
; LLAMA_ARG_ALIAS to the section name, so these strings are what appears in
; /v1/models and in the Open WebUI dropdown.
;
; mmproj MUST be declared per-section. On the router's own command line it is
; silently discarded by unset_reserved_args(base_preset, true).
version = 1

; Shared by both children. Every value here is measured — see README "Tuning".
[*]
ngl = 99
c = @CTX@
parallel = @PARALLEL@
fa = on
ctk = q8_0
ctv = q8_0
spec-type = @SPEC@
cache-reuse = @CACHE_REUSE@
metrics = true
jinja = true
temp = 0.5
top-p = 0.85
top-k = 20
min-p = 0

; 6.7 GB, 94.6% of FP16 per the model card. The default.
[bonsai-27b-ternary]
model = @ROOT@/models/Ternary-Bonsai-27B-gguf/Ternary-Bonsai-27B-Q2_0.gguf
mmproj = @ROOT@/models/Ternary-Bonsai-27B-gguf/Ternary-Bonsai-27B-mmproj-Q8_0.gguf
load-on-startup = true

; 3.8 GB, 89.5% of FP16 per the model card. Loads on first request.
[bonsai-27b-1bit]
model = @ROOT@/models/Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf
mmproj = @ROOT@/models/Bonsai-27B-gguf/Bonsai-27B-mmproj-Q8_0.gguf
load-on-startup = false
```

- [ ] **Step 2: Rewrite `start-server.sh`**

Replace the whole file with this. The comment block explaining the measured flags moves into `models.ini.in`'s `[*]` section context, so keep the tuning rationale here only where it concerns the router itself.

```bash
#!/usr/bin/env bash
# Native Metal-accelerated llama-server for Bonsai 27B, in ROUTER mode.
#
# Docker on macOS CANNOT access Metal (Hypervisor.framework exposes no virtual
# GPU to Linux guests), so inference runs here on the host and containers talk
# to it over HTTP. See docker/compose.yaml for the container side.
#
# Router mode: launching WITHOUT -m makes this process a router that spawns one
# child llama-server per model in models.ini (server.cpp:94). Both quantizations
# then appear in /v1/models and in the Open WebUI dropdown, switchable without a
# restart.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Prefer the source build (Metal 4 tensor API enabled) over the prebuilt.
# NOTE: Homebrew's llama.cpp is NOT usable here — upstream cannot read the
# fork's ternary Q2_0 block layout ("tensor 'output_norm.weight' has offset ...").
if [[ -x "$ROOT/src/llama.cpp-prism/build/bin/llama-server" ]]; then
  SERVER="$ROOT/src/llama.cpp-prism/build/bin/llama-server"
else
  SERVER="$ROOT/bin/llama-prism-b9570-0ad1dab/llama-server"
fi

# 0.0.0.0 is REQUIRED: binding 127.0.0.1 makes the server unreachable from the
# Docker VM. Everything is therefore also reachable from your LAN, so the API
# key below is not optional hygiene. Children bind 127.0.0.1 and are stripped of
# the key by the router, so it is enforced exactly once, here.
HOST="${BONSAI_HOST:-0.0.0.0}"
PORT="${BONSAI_PORT:-8080}"
CTX="${BONSAI_CTX:-32768}"   # model trains to 262144; that KV cache will not fit in 24 GB

# Metal's working set on this box is 17.8 GB, not 24 (llama.log: "MTL0 : Apple
# M5 (18186 MiB free)"). Two 27B models resident is 10.5 GB of weights plus up
# to two ~873 MiB mmproj allocations, leaving too little for two KV caches at
# c=32768. So: keep ONE model loaded and swap on selection. Raise to 2 to keep
# both hot if you have lowered BONSAI_CTX.
MODELS_MAX="${BONSAI_MODELS_MAX:-1}"

[[ -f "$ROOT/.env" ]] && set -a && . "$ROOT/.env" && set +a
: "${BONSAI_API_KEY:?BONSAI_API_KEY not set — create .env with: BONSAI_API_KEY=bonsai-\$(openssl rand -hex 20)}"

# Render the preset. Paths must be absolute: the children are spawned by the
# router and inheriting CWD is fragile.
PRESET="$ROOT/run/models.ini"
mkdir -p "$ROOT/run"
sed -e "s|@ROOT@|$ROOT|g" \
    -e "s|@CTX@|$CTX|g" \
    -e "s|@PARALLEL@|${BONSAI_PARALLEL:-1}|g" \
    -e "s|@SPEC@|${BONSAI_SPEC:-ngram-simple}|g" \
    -e "s|@CACHE_REUSE@|${BONSAI_CACHE_REUSE:-256}|g" \
    "$ROOT/models.ini.in" > "$PRESET"

echo "server : $SERVER (router)"
echo "preset : $PRESET"
echo "listen : http://$HOST:$PORT  (ctx=$CTX, models-max=$MODELS_MAX)"

# No -m: that is what selects router mode. No --mmproj either — it would be
# discarded here and must live in the preset sections.
exec "$SERVER" \
  --models-preset "$PRESET" \
  --models-max "$MODELS_MAX" \
  --host "$HOST" --port "$PORT" \
  --api-key "$BONSAI_API_KEY" \
  "$@"
```

- [ ] **Step 3: Restart the stack**

Run: `make down && make up`
Expected: all three services reach `up`. The `llama` line should appear faster than before — the router answers `/health` as soon as it binds, without waiting for the startup child.

- [ ] **Step 4: Verify the preset rendered with absolute paths**

Run: `grep -E '^(model|mmproj) =' run/models.ini`
Expected: four lines, every path absolute and starting `/Users/`, no remaining `@ROOT@`.

- [ ] **Step 5: Verify both models are listed**

```bash
set -a; . ./.env; set +a
curl -s -H "Authorization: Bearer $BONSAI_API_KEY" http://127.0.0.1:8080/v1/models \
  | python3 -c "import json,sys; [print(m['id'], m.get('status',{}).get('value')) for m in json.load(sys.stdin)['data']]"
```

Expected: exactly two lines — `bonsai-27b-ternary loaded` and `bonsai-27b-1bit unloaded`.

- [ ] **Step 6: Verify auth is still enforced at the router**

```bash
set -a; . ./.env; set +a
echo "no-key  $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/v1/models)"
echo "bad-key $(curl -s -o /dev/null -w '%{http_code}' -H 'Authorization: Bearer nope' http://127.0.0.1:8080/v1/models)"
echo "good    $(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $BONSAI_API_KEY" http://127.0.0.1:8080/v1/models)"
```

Expected: `no-key 401`, `bad-key 401`, `good 200`.

- [ ] **Step 7: Verify a completion from each model**

```bash
set -a; . ./.env; set +a
for m in bonsai-27b-ternary bonsai-27b-1bit; do
  echo "== $m"
  curl -s --max-time 600 http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" -H "Authorization: Bearer $BONSAI_API_KEY" \
    -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in five words.\"}],\"max_tokens\":300}" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message'].get('content') or d['choices'][0]['message'].get('reasoning_content'))"
done
```

Expected: both return text. The 1-bit request takes noticeably longer the first time — that is the on-demand load.

- [ ] **Step 8: Verify the 1-bit model is on Metal, not CPU**

Run: `grep -aiE "Metal|offloaded|CPU buffer" run/logs/llama.log | tail -20`
Expected: layer offload to Metal for the 1-bit child too. If it shows all layers on CPU, `Q1_0` is not being GPU-accelerated and Task 5's measurements will show it — stop and report rather than proceeding.

- [ ] **Step 9: Capture a real child log line — needed by Task 4**

Run: `grep -aE "spawning server instance|task [0-9]+" run/logs/llama.log | head -20`
Expected: a `spawning server instance with name=... on port ...` line, and slot/task lines carrying the router's `[port]` prefix. **Copy two or three of these verbatim into a scratch note — Task 4 uses them as test fixtures instead of a guessed format.**

- [ ] **Step 10: Commit**

```bash
git add models.ini.in start-server.sh
git commit -m "start-server.sh: run llama-server in router mode with both 27B quants

Section names in models.ini become the model IDs in Open WebUI's dropdown.
mmproj is declared per-section because the router discards it from its own
command line (unset_reserved_args(base_preset, true))."
```

---

### Task 3: `make bench` against a chosen model

**Files:**
- Modify: `Makefile` (the `bench` rule)

**Interfaces:**
- Consumes: model IDs from Task 2.
- Produces: `make bench [MODEL=<id>]`. Task 5 uses it to fill in the README comparison table.

In router mode the `"model"` field is **required** on POST — the current payload omits it and the router answers 400 `model name is missing from the request`.

- [ ] **Step 1: Confirm the current bench actually breaks**

Run: `make bench`
Expected: failure — a 400 from the router, surfacing as a `KeyError: 'timings'` from the parsing python. This confirms the fix is needed rather than assumed.

- [ ] **Step 2: Replace the `bench` rule**

```make
bench:          ## Measure throughput — make bench MODEL=bonsai-27b-1bit
	@set -a; . ./.env; set +a; \
	m="$(or $(MODEL),bonsai-27b-ternary)"; \
	echo "  model: $$m"; \
	curl -s --max-time 600 http://127.0.0.1:8080/v1/chat/completions \
	  -H "Content-Type: application/json" -H "Authorization: Bearer $$BONSAI_API_KEY" \
	  -d "{\"model\":\"$$m\",\"messages\":[{\"role\":\"user\",\"content\":\"Count to twenty.\"}],\"max_tokens\":150}" \
	| python3 -c "import json,sys;t=json.load(sys.stdin)['timings'];print('  %.1f tok/s generation, %.1f t/s prompt'%(t['predicted_per_second'],t['prompt_per_second']))"
```

- [ ] **Step 3: Verify the default**

Run: `make bench`
Expected: prints `model: bonsai-27b-ternary` then a tok/s line.

- [ ] **Step 4: Verify the override**

Run: `make bench MODEL=bonsai-27b-1bit`
Expected: prints `model: bonsai-27b-1bit` then a tok/s line. Record both numbers — Task 5 needs them.

- [ ] **Step 5: Commit**

```bash
git add Makefile
git commit -m "make bench: send the model field, add MODEL= override

Router mode requires \"model\" on POST; the old payload 400s."
```

---

### Task 4: `cache-viz.py` — stop conflating the two children

**Files:**
- Create: `tests/test_cache_viz.py`
- Modify: `cache-viz.py` (parser section, lines ~28-66, plus `snapshot()` and the table markup/JS)

**Interfaces:**
- Consumes: the real log lines captured in Task 2 Step 9.
- Produces: nothing downstream depends on this.

Each child numbers its tasks from 0 independently, and `_pending` keys on task id alone — so task 0 from the ternary child and task 0 from the 1-bit child merge into one record. The router prefixes forwarded child output with the child's port (`server-models.cpp:816`), which is the discriminator.

**Before writing the test, replace the fixture strings below with the real lines captured in Task 2 Step 9.** The shapes here are derived from the source but the exact prefix position depends on the router's log formatting — use observed reality, not this approximation.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cache_viz.py`:

```python
"""Tests for cache-viz.py's log parser.

stdlib unittest, not pytest — cache-viz.py promises "Stdlib only; no install
step" and its tests should not break that.

Run: python3 -m unittest discover -s tests -v
"""
import importlib.util
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_cache_viz():
    """cache-viz.py has a dash in its name, so it is not directly importable."""
    spec = importlib.util.spec_from_file_location(
        "cache_viz", os.path.join(ROOT, "cache-viz.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ParserTest(unittest.TestCase):
    def setUp(self):
        self.cv = load_cache_viz()
        self.cv._requests.clear()
        self.cv._pending.clear()
        self.cv._ports.clear()

    def feed(self, *lines):
        for line in lines:
            self.cv._parse_line(line)

    def test_learns_port_to_model_mapping(self):
        self.feed("0.00.1 I srv spawning server instance with name=bonsai-27b-1bit on port 41001")
        self.assertEqual(self.cv._ports.get("41001"), "bonsai-27b-1bit")

    def test_two_children_reusing_task_0_do_not_merge(self):
        self.feed(
            "0.00.1 I srv spawning server instance with name=bonsai-27b-ternary on port 41000",
            "0.00.2 I srv spawning server instance with name=bonsai-27b-1bit on port 41001",
            # ternary child, task 0: 100 fresh prompt tokens
            "0.01.0 I [41000] slot update_slots: id 0 | task 0 | prompt eval time = 500.00 ms / 100 tokens",
            "0.01.1 I [41000] slot release: id 0 | task 0 | eval time = 1000.00 ms / 10 tokens ( 100.00 ms per token, 10.00 tokens per second)",
            # 1-bit child, also task 0: 7 fresh prompt tokens
            "0.02.0 I [41001] slot update_slots: id 0 | task 0 | prompt eval time = 50.00 ms / 7 tokens",
            "0.02.1 I [41001] slot release: id 0 | task 0 | eval time = 200.00 ms / 20 tokens ( 10.00 ms per token, 20.00 tokens per second)",
        )
        self.assertEqual(len(self.cv._requests), 2)
        by_model = {r["model"]: r for r in self.cv._requests}
        self.assertEqual(by_model["bonsai-27b-ternary"]["new"], 100)
        self.assertEqual(by_model["bonsai-27b-1bit"]["new"], 7)
        self.assertEqual(by_model["bonsai-27b-1bit"]["gen_tok_s"], 20.0)

    def test_reused_tokens_attributed_to_the_right_child(self):
        self.feed(
            "0.00.1 I srv spawning server instance with name=bonsai-27b-ternary on port 41000",
            "0.00.2 I srv spawning server instance with name=bonsai-27b-1bit on port 41001",
            "0.01.0 W [41000] slot update_slots: id 0 | task 0 | restored context checkpoint (pos_min = 484, pos_max = 484, n_tokens = 485, n_past = 485, size = 149.626 MiB)",
            "0.01.1 I [41000] slot update_slots: id 0 | task 0 | prompt eval time = 10.00 ms / 3 tokens",
            "0.01.2 I [41000] slot release: id 0 | task 0 | eval time = 100.00 ms / 5 tokens ( 20.00 ms per token, 5.00 tokens per second)",
            "0.02.0 I [41001] slot update_slots: id 0 | task 0 | prompt eval time = 90.00 ms / 40 tokens",
            "0.02.1 I [41001] slot release: id 0 | task 0 | eval time = 100.00 ms / 5 tokens ( 20.00 ms per token, 5.00 tokens per second)",
        )
        by_model = {r["model"]: r for r in self.cv._requests}
        self.assertEqual(by_model["bonsai-27b-ternary"]["reused"], 485)
        self.assertEqual(by_model["bonsai-27b-1bit"]["reused"], 0)

    def test_unprefixed_lines_still_parse(self):
        """Single-model logs and pre-router history must keep working."""
        self.feed(
            "0.01.0 I slot update_slots: id 0 | task 7 | prompt eval time = 500.00 ms / 100 tokens",
            "0.01.1 I slot release: id 0 | task 7 | eval time = 1000.00 ms / 10 tokens ( 100.00 ms per token, 12.90 tokens per second)",
        )
        self.assertEqual(len(self.cv._requests), 1)
        self.assertEqual(self.cv._requests[0]["new"], 100)
        self.assertEqual(self.cv._requests[0]["gen_tok_s"], 12.9)

    def test_unknown_port_falls_back_to_the_port_number(self):
        """A child whose spawn line scrolled out of the log is still distinct."""
        self.feed(
            "0.01.0 I [41007] slot update_slots: id 0 | task 0 | prompt eval time = 500.00 ms / 100 tokens",
            "0.01.1 I [41007] slot release: id 0 | task 0 | eval time = 100.00 ms / 5 tokens ( 20.00 ms per token, 5.00 tokens per second)",
        )
        self.assertEqual(self.cv._requests[0]["model"], ":41007")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m unittest discover -s tests -v`
Expected: FAIL — `AttributeError: module 'cache_viz' has no attribute '_ports'`.

- [ ] **Step 3: Implement the parser change**

In `cache-viz.py`, replace the regex block and `_parse_line` (currently lines 30-66) with:

```python
RE_TASK = re.compile(r"task\s+(-?\d+)")
RE_RESTORED = re.compile(r"restored context checkpoint .*?n_tokens = (\d+)")
RE_PROMPT = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
RE_EVAL = re.compile(r"\beval time =\s*([\d.]+) ms /\s*(\d+) tokens .*?([\d.]+) tokens per second")
RE_FULL = re.compile(r"forcing full prompt re-processing")
# Router mode: the router forwards each child's output prefixed with that
# child's port (server-models.cpp:816 -> LOG("[%5d] %s", port, buffer)), and
# announces the mapping once when it spawns the child. %5d right-aligns, so a
# 4-digit port arrives as "[ 8081]".
RE_CHILD = re.compile(r"\[\s*(\d{4,5})\]")
RE_SPAWN = re.compile(r"spawning server instance with name=(\S+) on port (\d+)")

_lock = threading.Lock()
_requests = []          # completed request records, oldest first
_pending = {}           # (port, task id) -> partial record
_ports = {}             # child port -> model name


def _parse_line(line: str) -> None:
    s = RE_SPAWN.search(line)
    if s:
        _ports[s.group(2)] = s.group(1)
        return
    m = RE_TASK.search(line)
    if not m:
        return
    # Each child numbers its tasks from 0 independently, so the task id alone is
    # NOT unique once the router has two children logging into the same file.
    c = RE_CHILD.search(line)
    port = c.group(1) if c else None
    task = m.group(1)
    key = (port, task)
    rec = _pending.setdefault(key, {"task": int(task), "port": port,
                                    "reused": 0, "new": 0,
                                    "prefill_ms": 0.0, "gen_tok_s": 0.0,
                                    "gen_tokens": 0, "full_reprocess": False})
    if RE_FULL.search(line):
        rec["full_reprocess"] = True
    r = RE_RESTORED.search(line)
    if r:
        rec["reused"] = int(r.group(1))
    p = RE_PROMPT.search(line)
    if p:
        rec["prefill_ms"] = float(p.group(1))
        rec["new"] = int(p.group(2))
    e = RE_EVAL.search(line)
    if e and "prompt eval" not in line:
        rec["gen_tokens"] = int(e.group(2))
        rec["gen_tok_s"] = float(e.group(3))
        # generation timing is the last line of a request -> it is complete
        with _lock:
            rec["model"] = _model_of(rec["port"])
            _requests.append(_pending.pop(key))
            del _requests[:-200]        # keep the last 200


def _model_of(port):
    """Label a record by model. Falls back to the bare port when the spawn line
    is not in the log (rotated away, or attached mid-run), so two unknown
    children are still told apart. Single-model runs have no port at all."""
    if port is None:
        return "single"
    return _ports.get(port, ":" + port)
```

- [ ] **Step 4: Clear `_ports` on log truncation**

In `tail_log`, the truncation branch currently clears `_requests` and `_pending`. Add `_ports` so a restarted server's new child ports are not shadowed by stale ones. The branch becomes:

```python
            if size < pos:              # truncated by a restart
                pos = 0
                with _lock:
                    _requests.clear()
                _pending.clear()
                _ports.clear()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m unittest discover -s tests -v`
Expected: 5 tests, all PASS.

- [ ] **Step 6: Surface the model in the dashboard**

In `PAGE`, change the table header (line ~220) from:

```html
    <table id="tbl"><thead><tr><th>Req</th><th>Reused</th><th>Fresh</th><th>Prefill ms</th><th>Gen t/s</th></tr></thead>
```

to:

```html
    <table id="tbl"><thead><tr><th>Req</th><th>Model</th><th>Reused</th><th>Fresh</th><th>Prefill ms</th><th>Gen t/s</th></tr></thead>
```

and the row template (line ~299) from:

```javascript
    `<tr><td>${i+1}</td><td>${r.reused}</td><td>${r.new}</td><td>${r.prefill_ms.toFixed(0)}</td><td>${r.gen_tok_s.toFixed(1)}</td></tr>`).join("");
```

to:

```javascript
    `<tr><td>${i+1}</td><td>${r.model||"—"}</td><td>${r.reused}</td><td>${r.new}</td><td>${r.prefill_ms.toFixed(0)}</td><td>${r.gen_tok_s.toFixed(1)}</td></tr>`).join("");
```

Also update the subtitle (line ~191) so it no longer implies a single model:

```html
<div class="sub">Prompt tokens reused from the KV cache vs processed from scratch · llama-server router :8080 · refreshes every 2s</div>
```

- [ ] **Step 7: Verify against the live stack**

```bash
make cache-viz
set -a; . ./.env; set +a
for m in bonsai-27b-ternary bonsai-27b-1bit; do
  curl -s --max-time 600 http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" -H "Authorization: Bearer $BONSAI_API_KEY" \
    -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"Count to ten.\"}],\"max_tokens\":200}" > /dev/null
done
curl -s http://127.0.0.1:8090/api/data | python3 -m json.tool | head -40
```

Expected: separate records, each carrying the correct `model`, with no two requests sharing a `(port, task)`. Open http://127.0.0.1:8090 and confirm the Model column is populated.

- [ ] **Step 8: Commit**

```bash
git add tests/test_cache_viz.py cache-viz.py
git commit -m "cache-viz: key requests by (port, task), label rows by model

Router children each number tasks from 0, so keying on task id alone merged
two models' requests into one record. The router prefixes forwarded child
output with the child port (server-models.cpp:816) — use it."
```

---

### Task 5: Measure, then document what was measured

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the tok/s numbers from Task 3, the model IDs from Task 2.
- Produces: nothing.

Nothing in this task may be written from expectation. Every number in the README comes from a command run in this task.

- [ ] **Step 1: Measure generation speed for both models**

```bash
make bench MODEL=bonsai-27b-ternary
make bench MODEL=bonsai-27b-1bit
```

Record both. Note that with `--models-max 1` the second invocation includes a model swap; re-run each twice and take the second reading so the swap is not counted.

- [ ] **Step 2: Test whether both models fit simultaneously**

```bash
BONSAI_MODELS_MAX=2 ./stack.sh restart
set -a; . ./.env; set +a
for m in bonsai-27b-ternary bonsai-27b-1bit; do
  curl -s --max-time 600 http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" -H "Authorization: Bearer $BONSAI_API_KEY" \
    -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi.\"}],\"max_tokens\":100}" \
    -o /dev/null -w "$m HTTP %{http_code}\n"
done
curl -s -H "Authorization: Bearer $BONSAI_API_KEY" http://127.0.0.1:8080/v1/models \
  | python3 -c "import json,sys; [print(m['id'], m.get('status',{}).get('value')) for m in json.load(sys.stdin)['data']]"
grep -aiE "ggml_metal|failed to allocate|out of memory|unable to load" run/logs/llama.log | tail
```

Expected: either both report `loaded` and both return 200 — in which case `2` is viable and the README says so — or one fails to allocate, in which case `1` stays the documented default. **Record which actually happened.**

- [ ] **Step 3: Verify vision still works on both models**

This is what the per-section `mmproj` finding protects, so it must be checked rather than assumed.

```bash
set -a; . ./.env; set +a
python3 - <<'PY' > /tmp/redcircle.b64
import base64, zlib, struct
def chunk(t, d):
    c = t + d
    return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
W = H = 64
rows = b""
for y in range(H):
    row = b"\x00"
    for x in range(W):
        inside = (x - 32) ** 2 + (y - 32) ** 2 <= 26 ** 2
        row += b"\xd0\x20\x20" if inside else b"\xff\xff\xff"
    rows += row
png = (b"\x89PNG\r\n\x1a\n"
       + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
       + chunk(b"IDAT", zlib.compress(rows))
       + chunk(b"IEND", b""))
print(base64.b64encode(png).decode())
PY
B64=$(cat /tmp/redcircle.b64)
for m in bonsai-27b-ternary bonsai-27b-1bit; do
  echo "== $m"
  curl -s --max-time 600 http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" -H "Authorization: Bearer $BONSAI_API_KEY" \
    -d "{\"model\":\"$m\",\"max_tokens\":300,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What shape and colour is this? Answer in under ten words.\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$B64\"}}]}]}" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('choices',[{}])[0].get('message',{}).get('content') or d)"
done
```

Expected: both describe a red circle. If the 1-bit model errors about missing multimodal support, its `mmproj` line in `models.ini.in` is wrong — fix before continuing.

- [ ] **Step 4: Verify no orphaned children survive shutdown**

Children hold several GB each; leaking them across `make restart` would be a slow-motion memory exhaustion.

```bash
make down
pgrep -fl llama-server || echo "no llama-server processes remain"
```

Expected: `no llama-server processes remain`. If children survive, record it as a known issue in the README and add `pkill -f llama-server` guidance via the existing `make reap` — do not silently ignore it.

- [ ] **Step 5: Bring the stack back up**

Run: `make up && make status`
Expected: all three `up`.

- [ ] **Step 6: Update the README**

Make these edits, filling every number from Steps 1-4:

1. In the architecture diagram near the top, show the router with two children rather than a single `llama-server`.
2. Under **Usage**, add `make models` to the command table: *"download the GGUF weights (4.4 GB) — run once before the first `make up`"*.
3. Add a **Model selection** section after **Usage**:
   - the two model IDs and how to pick them in the Open WebUI dropdown;
   - the measured size/quality/speed table (ternary 6.7 GB / 94.6% / _X_ tok/s vs 1-bit 3.8 GB / 89.5% / _Y_ tok/s);
   - `BONSAI_MODELS_MAX`, its default, and the Step 2 finding about whether `2` fits, quoting the measured Metal working set of 17.8 GB;
   - that `models.ini.in` is the template and `run/models.ini` the generated file, and that adding a third model is a new section;
   - that `--mmproj` must stay per-section.
4. Update `make bench` in the command table to mention `MODEL=`.
5. In **Layout**, add `models.ini.in`, `fetch-models.sh`, `tests/`, and `models/Bonsai-27B-gguf/`.
6. In **Verified**, add rows for: both models listed, 1-bit completion on Metal, vision on both, and the `--models-max 2` result.
7. Note that router mode is marked experimental in the fork (`server.cpp:340`).

- [ ] **Step 7: Verify the README's claims against reality one last time**

Run: `make status && python3 -m unittest discover -s tests -v`
Expected: three services `up`, 5 tests PASS. Re-read the README diff and confirm no number in it was written from expectation rather than from a command in this task.

- [ ] **Step 8: Commit**

```bash
git add README.md
git commit -m "README: document router model selection and the 1-bit 27B

Records measured tok/s for both quantizations, the 17.8 GB Metal working set
that drives the models-max default, and the verified vision path on both."
```

---

## Self-Review

**Spec coverage** — every section maps to a task:

| Spec section | Task |
|---|---|
| `models.ini.in` → `run/models.ini` | 2 |
| `start-server.sh` rewrite, no fallback branch | 2 |
| `fetch-models.sh` + `make models` | 1 |
| `cache-viz.py` `(port, task)` fix | 4 |
| `Makefile` bench `model` field + `MODEL=` | 3 |
| `stack.sh` unchanged | verified in 2 (Step 3) and 5 (Step 4) |
| Memory / `--models-max` decision | 5 (Step 2) |
| Verification items 1-9 | 2 (Steps 4-8), 3 (Step 4), 4 (Step 7), 5 (Steps 1-4) |
| Risk: orphaned children | 5 (Step 4) |
| Risk: router experimental | 5 (Step 6.7) |

**Ordering note:** Task 4 deliberately follows Task 2 so the parser is written against a captured real log rather than a guessed prefix format. Task 2 Step 9 exists solely to hand that fixture forward.

**Naming consistency:** `bonsai-27b-ternary` / `bonsai-27b-1bit` appear identically in `models.ini.in` section headers (Task 2), the bench default and override (Task 3), test fixtures (Task 4), and README (Task 5). `_ports`, `_pending`, `_model_of` are consistent between the test in Task 4 Step 1 and the implementation in Step 3. `BONSAI_MODELS_MAX` is consistent between Task 2 Step 2 and Task 5 Step 2.

**Known approximation, deliberately flagged rather than hidden:** the `RE_CHILD` pattern and the test fixtures in Task 4 encode a log format derived from reading `server-models.cpp:816`, not from observed output. Task 4's preamble instructs the implementer to replace them with the lines captured in Task 2 Step 9. This is the one place the plan cannot be authoritative in advance.
