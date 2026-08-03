# Add 1-bit Bonsai 27B with in-UI model selection

Date: 2026-08-03

## Goal

Run both Bonsai 27B quantizations from one server and pick between them in the
Open WebUI dropdown, without a restart:

| model | file | size | quality (vendor) |
|---|---|---|---|
| ternary `Q2_0` (today) | `Ternary-Bonsai-27B-Q2_0.gguf` | 6.7 GB | 94.6% of FP16 |
| 1-bit `Q1_0` (new) | `Bonsai-27B-Q1_0.gguf` | 3.80 GB | 89.5% of FP16 |

Same architecture, same vision support, same context — so the pair is a
like-for-like quality/speed comparison rather than two unrelated models.

## Why this is possible without new infrastructure

Two facts about the pinned fork (`PrismML-Eng/llama.cpp` @ `4dd1656`), both
verified against the source tree in `src/llama.cpp-prism/`:

1. **`Q1_0` is Metal-accelerated here.** `llama-quantize --help` lists
   `40 or Q1_0 : 1.125 bpw quantization`, and `ggml/src/ggml-metal/ggml-metal.metal`
   contains 49 `q1_0` references. This is a real GPU path, not a CPU fallback.
   (Contrast: `TQ1_0`/`TQ2_0`, the upstream BitNet types, have **zero** Metal
   kernels in this tree — an upstream BitNet GGUF would run CPU-only. Staying in
   the Bonsai family avoids that trap.)
2. **`llama-server` already has a model router.** `--models-dir`,
   `--models-preset`, `--models-max`, `--models-autoload`. It spawns one child
   `llama-server` per model and forwards each request to the right child.

So the work is configuration and wiring, not new code.

## Architecture

```
                    router (this process, loads no model)
  Open WebUI  ──►   :8080  --api-key  --models-preset models.ini
                      │
                      ├─► child :PORT_A   bonsai-27b-ternary   Q2_0  (127.0.0.1)
                      └─► child :PORT_B   bonsai-27b-1bit      Q1_0  (127.0.0.1)
```

Router mode is entered by launching **without** `-m`
(`tools/server/server.cpp:94`: `is_router_server = params.model.path.empty()`).

### Routing and identity

- POST endpoints route on the JSON `"model"` field; GET endpoints on a `?model=`
  query parameter.
- `server_model_meta::update_args` (`server-models.cpp:154`) force-sets
  `LLAMA_ARG_ALIAS` to the preset section name. **The section name is therefore
  the model ID** in `/v1/models` and the label in the Open WebUI dropdown.

### What may live where

`unset_reserved_args` is called two different ways, and the difference dictates
the whole file layout:

| call site | `unset_model_args` | effect |
|---|---|---|
| `server-models.cpp:210` (router's own CLI) | `true` | strips `MODEL`, `MMPROJ`, `ALIAS`, `HF_REPO` |
| `server-models.cpp:156` (per-model preset) | `false` | keeps them; strips only ssl/api-key/`models-*` |

Consequence: **`--mmproj` on the router command line is silently discarded.**
Vision must be declared per-section, or the 27B loses image input. Both models
get their own `mmproj` line.

`--host`, `--port` and `--api-key` stay on the router CLI. Children are stripped
of the API key and bind `127.0.0.1` (`CHILD_ADDR`), so the key is enforced once,
at the router, and the existing `0.0.0.0` exposure story is unchanged.

## Files

### `models.ini.in` (new, committed) → `run/models.ini` (generated, gitignored)

The committed file is a **template**; `start-server.sh` interpolates `@ROOT@`
into it at launch and writes the result to `run/models.ini`, which is what
`--models-preset` receives. This keeps absolute paths out of git while still
giving the children absolute paths.

```ini
version = 1

[*]
# every measured flag from start-server.sh, inherited by both children
ngl = 99
c = 32768
parallel = 1
fa = on
ctk = q8_0
ctv = q8_0
spec-type = ngram-simple
cache-reuse = 256
metrics = true
jinja = true
temp = 0.5
top-p = 0.85
top-k = 20
min-p = 0

[bonsai-27b-ternary]
model = @ROOT@/models/Ternary-Bonsai-27B-gguf/Ternary-Bonsai-27B-Q2_0.gguf
mmproj = @ROOT@/models/Ternary-Bonsai-27B-gguf/Ternary-Bonsai-27B-mmproj-Q8_0.gguf
load-on-startup = true

[bonsai-27b-1bit]
model = @ROOT@/models/Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf
mmproj = @ROOT@/models/Bonsai-27B-gguf/Bonsai-27B-mmproj-Q8_0.gguf
load-on-startup = false
```

Precedence is CLI > section > `[*]`, so `BONSAI_SPEC`/`BONSAI_CTX` overrides
still work by being passed on the router command line.

Paths must be absolute. The fork's docs allow CWD-relative paths, but the child
is spawned by the router and relying on inherited CWD is fragile.

### `start-server.sh` (rewritten)

Drops `-m` and `--mmproj`, renders `run/models.ini` from `models.ini.in`, and
execs the router with `--models-preset run/models.ini`, `--models-max`,
`--host/--port/--api-key`, and the pass-through `"$@"`.

No single-model fallback branch — decided explicitly. If the experimental router
misbehaves, the working single-model launcher is one `git revert` away.

### `fetch-models.sh` + `make models` (new)

Downloads via `hf` (already installed through Homebrew at `/opt/homebrew/bin/hf`):

```
prism-ml/Bonsai-27B-gguf  Bonsai-27B-Q1_0.gguf        3,803,452,480 B
                          Bonsai-27B-mmproj-Q8_0.gguf   629,246,880 B
```

4.43 GB total, against 475 GB free. Idempotent: skips files already present at
the expected size. Not run automatically by `make up` — a 4.4 GB download should
be explicit.

### `cache-viz.py` (bug fix, required)

Child output reaches the router log prefixed with the child's port
(`server-models.cpp:816`: `LOG("[%5d] %s", port, buffer)`).

- The regexes use `re.search`, so they still match through the prefix. No change
  needed there.
- **But `_pending` is keyed by task id alone, and each child numbers its tasks
  from 0 independently.** With both models loaded, task 0 from the ternary child
  and task 0 from the 1-bit child merge into a single record, silently mixing
  two models' cache statistics.

Fix: parse the leading `[port]`, key `_pending` on `(port, task)`, and carry the
port into each record so rows can be labelled by model. The dashboard then shows
the two quantizations side by side, which is the comparison this whole change
exists to enable. Records with no port prefix (single-model runs, older logs)
key on `(None, task)` and keep working.

### `Makefile` — `make bench`

In router mode the `"model"` field is **required** on POST; the current payload
omits it and would 400. Add the field, plus a `MODEL=` variable so either model
can be benchmarked:

```
make bench                      # defaults to bonsai-27b-ternary
make bench MODEL=bonsai-27b-1bit
```

### `stack.sh` — no change

Verified rather than assumed:

- `/health` is registered unconditionally at `server.cpp:179`, outside the
  `if (is_router_server)` block, and is public (no API key). The readiness probe
  works as-is.
- The early-failure grep still fires, because child stdout/stderr is forwarded
  into the router's log.

Note a behaviour change worth documenting: the router answers `/health` as soon
as *it* is up, which may precede the `load-on-startup` child finishing. `make up`
will return sooner, and the first chat request absorbs any remaining load time.

## Memory

Measured, not estimated — from `run/logs/llama.log`:

```
MTL0 : Apple M5 (18186 MiB, 18185 MiB free)
srv load_model: [mtmd] estimated worst-case memory usage of mmproj is 873.10 MiB
```

The Metal working set is **17.8 GB, not the machine's 24 GB**. Both models
resident is 6.7 + 3.8 = 10.5 GB of weights plus up to two mmproj allocations,
leaving roughly 5 GB for two KV caches and compute buffers at `-c 32768`. Too
tight to promise.

**Default `--models-max 1`** (override: `BONSAI_MODELS_MAX`). The router keeps
one model loaded and swaps on selection; switching costs a reload of 3.8–6.7 GB.
Whether `2` actually fits is an open question to be **measured during
implementation**, not guessed — and the README records the measurement either
way.

## Verification

| # | Check | Method |
|---|---|---|
| 1 | Both models listed | `GET /v1/models` returns both section names |
| 2 | Auth unchanged | no-key 401, bad-key 401, good-key 200 against router |
| 3 | Ternary still works | completion via `"model": "bonsai-27b-ternary"` |
| 4 | 1-bit works on Metal | completion via `"model": "bonsai-27b-1bit"`; log shows Metal offload, not CPU fallback |
| 5 | Vision survives both | image prompt to each model — this is what the `mmproj` finding protects |
| 6 | Speed comparison | `make bench` both ways; record tok/s |
| 7 | Memory | observed footprint with `--models-max 2`; decide the documented default |
| 8 | cache-viz correct | both models active, confirm no task-id collision and correct labelling |
| 9 | Stack lifecycle | `make up`/`down`/`status` clean, no orphaned children after `down` |

Item 9 deserves attention: `stop_one` kills the process holding the port (the
router) and its wrapper parent. Whether the router reaps its children on SIGTERM
needs checking — orphaned children holding several GB of weights would be a
nasty leak across `make restart` cycles.

## Risks

- **Router mode is experimental.** `server.cpp:340` prints
  `NOTE: router mode is experimental`. It has test coverage
  (`tools/server/tests/unit/test_router.py`), but is newer than the path in use
  today. Accepted; rollback is `git revert`.
- **Orphaned children** — see verification item 9.
- **Open WebUI model-list caching** — the new model may need a UI refresh or a
  reconnect of the connection to appear.

## Out of scope

- Smaller Bonsai sizes (1.7B/4B/8B). The preset format makes adding one later a
  four-line change.
- Draft-model speculation. Already measured as a 38% regression; unchanged.
- Deleting the unused `Ternary-Bonsai-27B-dspark-Q4_1.gguf` (1.8 GB).
