# Bonsai 27B on macOS — native Metal inference + Docker app layer

## Why this shape

Docker on macOS **cannot** use your GPU. Containers run inside a Linux VM on
Apple's Hypervisor/Virtualization framework, which exposes virtual CPUs and
memory but **no virtual GPU** — there is no Metal device to pass through, and no
`--gpus` equivalent. Docker's own Model Runner sidesteps this by running
inference engines as sandboxed *host* processes rather than in containers.

So: **inference runs natively on the host with Metal; clients talk to it over HTTP.**

The container→host bridge is verified working (see below), but the UI currently
runs natively too, because Docker Desktop's registry proxy wedged mid-setup —
see "Docker registry proxy" below.

```
┌─────────────────────────── macOS host (Apple M5) ───────────────────────────┐
│                                                                             │
│   llama-server ROUTER  ──────────────────────────────  0.0.0.0:8080         │
│   (prism fork, source-built)                       ▲        ▲               │
│     ├─ child: bonsai-27b-ternary  Q2_0  6.7 GB  ─┐ │        │               │
│     └─ child: bonsai-27b-1bit     Q1_0  3.8 GB  ─┤ │        │               │
│          both on Metal / MTLGPUFamilyApple10     │ │        │               │
│          one resident at a time (--models-max 1) ┘ │        │               │
│                                                    │        │               │
│   open-webui (native, :9090) ──────────────────────┘        │ host.docker.internal
│                                                             │               │
│   ┌─────────────── Docker Desktop Linux VM ──────────────────┼────────────┐ │
│   │   any container ─────────────────────────────────────────┘            │ │
│   │   (verified: curl container got a completion at 13.0 tok/s)           │ │
│   └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

The router loads no model itself. It spawns one child `llama-server` per model in
`models.ini` and proxies each request to the right child, so both quantizations
appear in the Open WebUI dropdown and you switch between them without a restart.

## Docker registry proxy (known issue on this machine)

Every `docker pull` hangs — including `hello-world` from Docker Hub — while the
daemon itself stays responsive (`docker ps`, `docker version` return instantly).
Docker Desktop's internal proxy (`http.docker.internal:3128`) stopped forwarding
partway through the Open WebUI pull. The host's own network is fine
(`curl https://ghcr.io/v2/` → 401 in 0.14s, the correct unauthenticated ping).

Restarting Docker Desktop should clear it. `docker/compose.yaml` is kept ready
for that; until then `start-webui.sh` runs the same UI natively.

Note: freshly brew-installed binaries (`skopeo`, `nc`) also failed to reach
ghcr.io by IP while `curl` succeeded to the same address — suggesting a
per-binary network filter on this machine, separate from the Docker issue.

## Runtime choice — why not Homebrew's llama.cpp

`brew install llama.cpp` (b10200) is installed and is fine for **1-bit** Bonsai
weights, but it **cannot load ternary `Q2_0`**:

```
gguf_init_from_reader: tensor 'output_norm.weight' has offset 337715200, expected 357580800
```

The ternary block layout is specific to the PrismML fork. We therefore build the
fork from source — which also compiles the **Metal 4 tensor API** that the
upstream prebuilt binary could not (it shipped with AppleClang 15):

| build | ternary Q2_0 | `has tensor` | prompt proc | generation |
|---|---|---|---|---|
| brew b10200 | ✗ fails to load | ✓ | — | — |
| prism prebuilt b9570 | ✓ | ✗ | 53.98 t/s | 9.71 t/s |
| **prism source build** | **✓** | **✓** | **160.65 t/s** | **11.70 t/s** |

Measured with `llama-bench -p 128 -n 128 -ngl 99 -r 2` on Apple M5 / 24 GB.

## Usage

Fetch the weights once (4.4 GB), then one command starts everything:

```bash
make models    # download GGUF weights — only needed the first time
make up        # or: ./stack.sh up
```

```
  SERVICE     PORT   STATE     PID
  llama       8080   up        43261     Bonsai 27B router on Metal
  playwright  8931   up        43431     browser tools over MCP
  webui       9090   up        43469     chat UI

  chat UI -> http://127.0.0.1:9090
```

| command | does |
|---|---|
| `make models` | download the GGUF weights (4.4 GB). Skips files already present, so re-running is a sub-second no-op |
| `make up` | start all three, waiting until each port actually accepts connections |
| `make down` | stop everything (reverse order) |
| `make restart` | down then up |
| `make status` | what's running, with PIDs |
| `make logs` | tail all logs — `make logs SVC=llama` for one |
| `make open` | open the chat UI |
| `make bench` | measure tok/s — `make bench MODEL=bonsai-27b-1bit` for the other model |

Logs land in `run/logs/<service>.log` (gitignored). `stack.sh` waits on port
readiness rather than process start, because llama-server maps 6.7 GB of weights
and Open WebUI runs DB migrations — both are alive well before they can serve.

Individual services, if you need them separately:

```bash
./start-server.sh          # inference      :8080
./start-playwright-mcp.sh  # browser tools  :8931
./start-webui.sh           # chat UI        :9090
```

Nothing survives a reboot — these are plain user processes, not launchd services.
Run `make up` again.

Once the Docker registry proxy is working again, `cd docker && docker compose up -d`
runs the same UI in a container instead — it points at `host.docker.internal:8080`.

Tunables (env): `BONSAI_CTX` (default 32768), `BONSAI_PORT`, `BONSAI_HOST`,
`BONSAI_MODELS_MAX` (default 1 — see "Model selection"). The model trains to
262144 context; that KV cache will not fit in 24 GB, hence the lower default.
Extra `llama-server` flags pass straight through to the **router**:

```bash
./start-server.sh --reasoning off        # disable thinking mode
./start-server.sh --reasoning-budget 256 # cap thinking tokens
```

## Model selection

Two quantizations of the same 27B are served from one port and chosen in the
Open WebUI dropdown:

| model id | quant | file | vendor quality | vision |
|---|---|---|---|---|
| `bonsai-27b-ternary` | `Q2_0`, 2.125 bpw | 6.7 GB | 94.6% of FP16 | ✓ |
| `bonsai-27b-1bit` | `Q1_0`, 1.125 bpw | 3.8 GB | 89.5% of FP16 | ✓ |

Both run on Metal — `Q1_0` is a first-class type in this fork (`llama-quantize`
lists it, and `ggml-metal.metal` carries its kernels), so the 1-bit model is GPU
accelerated, not a CPU fallback. Note this does **not** hold for `TQ1_0`/`TQ2_0`,
the upstream BitNet types: they have no Metal kernels here, so an upstream BitNet
GGUF would run CPU-only. Staying in the Bonsai family avoids that.

**One model is resident at a time** (`--models-max 1`). Selecting the other in
the dropdown unloads the current child and loads the new one from disk. Measured
cost of a switch, with the machine under other load:

| | wall clock |
|---|---|
| request to the model already loaded | 4–7 s |
| request that switches model | 8–12 s |

So a switch costs roughly **+4–5 s**. The weights are `mmap`ed, so the OS page
cache keeps them warm and a reload is far cheaper than a cold 3.8–6.7 GB read.

`BONSAI_MODELS_MAX=2` keeps both resident and makes switching instant. This was
tested and **works** — both children loaded, served six alternating requests, and
logged no allocation failures, even while the machine was already 12 GB into
swap. It is not the default only because one-at-a-time leaves more headroom for
everything else on the box.

### Adding another model

`models.ini.in` is the committed template; `start-server.sh` interpolates `@ROOT@`
and writes `run/models.ini`, which is what the router reads. A new model is a new
section — the section name becomes the model id in the dropdown, because the
router force-sets `--alias` to it (`server-models.cpp:154`).

Two traps, both of which cost time here:

- **`mmproj` must be declared per-section.** On the router's own command line it
  is silently discarded (`unset_reserved_args(base_preset, true)`,
  `server-models.cpp:210`), and the model loses image input with no error.
- **Do not add the `version = 1` line** that the fork's own README example shows.
  Any key before the first `[section]` header lands in a section named `default`
  (`preset.cpp:254`), which then inherits the `[*]` globals and appears in
  `/v1/models` as a phantom model with no weights. Observed: with that line
  present `/v1/models` returned three entries instead of two.

### API

In router mode the `"model"` field is **required** on POST — without it the
router answers `400 model name is missing from the request`. GET endpoints
(`/props`, `/metrics`) take a `?model=` query parameter instead.

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $BONSAI_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"bonsai-27b-1bit","messages":[{"role":"user","content":"hi"}]}'
```

Router mode is marked experimental upstream (`server.cpp:340` prints
`NOTE: router mode is experimental`).

## Thinking mode

The model reasons by default: short `max_tokens` yields an **empty `content`**
with the text in `message.reasoning_content`. Either allow enough tokens
(~200+), or run with `--reasoning off`.

## Security

`start-server.sh` binds `0.0.0.0` because binding `127.0.0.1` would be
unreachable from the Docker VM — which also makes it reachable from your LAN.
An API key in `.env` (gitignored, mode 600) is therefore required, and enforced.
Re-verified against the router:

```
/v1/chat/completions   no-key 401   bad-key 401   good-key 200
/props                 no-key 401
/models/load           no-key 401      (the load/unload control endpoint)
```

The key is enforced once, at the router. Children bind `127.0.0.1` and are
stripped of it (`unset_reserved_args`), so it is not duplicated per model —
confirmed absent from the unauthenticated `/v1/models` response.

Two endpoints are public by upstream design, not by regression:
`/health` and `/v1/models` both return 200 without a key
(`tools/server/tests/unit/test_security.py` asserts this). On a `0.0.0.0` bind
that means anyone on your LAN can read the model list, **including each child's
full command line and absolute weight paths**. No credentials leak, but if that
bothers you, put the box behind a firewall rule rather than relying on the key.

## Layout

```
start-server.sh                     native Metal inference launcher (router mode)
models.ini.in                       model preset template -> run/models.ini
fetch-models.sh                     weight downloader (make models)
start-webui.sh                      native Open WebUI launcher (:9090)
cache-viz.py                        live prompt-cache dashboard (:8090)
tests/                              stdlib unittest suite for cache-viz.py
docker/compose.yaml                 containerised UI, parked pending Docker fix
.env                                BONSAI_API_KEY (gitignored, 0600)
.venv/                              python 3.11 env for open-webui
.webui-data/                        Open WebUI database + cache
models/Ternary-Bonsai-27B-gguf/     Q2_0 weights (6.7G) + mmproj Q8_0 (600M)
models/Bonsai-27B-gguf/             Q1_0 weights (3.8G) + mmproj Q8_0 (629M)
src/llama.cpp-prism/                fork source build (preferred binary)
bin/llama-prism-b9570-0ad1dab/      prebuilt fallback
```

Tests: `python3 -m unittest discover -s tests -v` (stdlib only, no install step).

## Prompt-cache dashboard

`make cache-viz` serves a live view on :8090 of how many prompt tokens each
request reused from the KV cache versus processed fresh, parsed out of
`run/logs/llama.log`. Under the router it labels each request with the model it
came from, so the two quantizations can be compared side by side.

**Expect a 0% hit rate as currently configured.** Every child logs:

```
srv load_model: cache_reuse is not supported by multimodal, it will be disabled
```

Both models declare an `mmproj`, so `--cache-reuse` is silently a no-op and
nothing is ever reused. This is pre-existing behaviour, not a router regression.
Drop the `mmproj` line from a section to get prompt-cache reuse back for that
model — at the cost of its vision support.

## Browser tools (Playwright MCP)

Bonsai can drive Chrome from the chat UI.

```bash
./start-playwright-mcp.sh    # MCP server on http://localhost:8931/mcp
```

Already registered in Open WebUI as tool server `server:mcp:playwright`. In the
UI, enable it via the **+** / tools control in the message box, then ask it to
browse. A visible Chrome window opens (headed by default — add `--headless` to
`start-playwright-mcp.sh` to hide it).

Two things that cost time and are easy to hit again:

- **Open WebUI's MCP client is streamable-HTTP only** (`streamablehttp_client`
  in `utils/mcp/client.py`) — not stdio, not SSE. Playwright MCP must run with
  `--port`, not as a stdio subprocess.
- **Use `localhost`, not `127.0.0.1`.** Playwright MCP host-checks requests and
  returns **403** for `127.0.0.1` even though it resolves to the same machine.

Tool execution runs in the UI's chat flow. Calling `/api/chat/completions`
directly returns the `tool_calls` for the caller to execute rather than looping
server-side — so verify browser behaviour in the UI, not via curl.

Expect it to handle simple, well-specified tasks and to struggle with
open-ended ones: a 1.58-bit 27B is far weaker at multi-step agentic work than a
frontier model, and at 10–13 tok/s with a thinking block per step each hop is
slow.

## Tuning — measured on this machine

Generation on a 27B is **memory-bandwidth-bound**: every token streams the
6.7 GB working set, so flag-tuning has a hard ceiling. The only way past it is
speculative decoding, which amortises one weight read over several tokens.

`llama-bench`-style comparison, `-c 8192 --parallel 1 -fa on -ctk/-ctv q8_0`,
two workloads — "generic" prose vs output that reuses input tokens:

| `--spec-type` | generic | repetitive | verdict |
|---|---|---|---|
| none (baseline) | 12.9 | 12.9 | |
| **ngram-simple** | 12.7 | **19.6** | **default** — free, 1.5x when it hits |
| ngram-map-k | 11.3 | **26.5** | 2.05x repetitive, −12% generic |
| ngram-mod | 9.9 | 14.0 | worse at both |
| draft-dspark | **7.8** | 7.6 | **−38%, do not use** |
| draft-dspark + ngram | 4.1 | 3.2 | −67% |

Override with `BONSAI_SPEC=ngram-map-k ./start-server.sh` if your workload is
mostly summarising/rewriting/code-editing.

**Draft-model speculation loses badly here.** The repo ships an undocumented
`Ternary-Bonsai-27B-dspark-Q4_1.gguf` (1.95 GB) and the fork has a matching
`--spec-type draft-dspark`. It needs `--spec-draft-n-max 4` to load at all
(the drafter's `block_size` is 4; the default 3 is a hard error). Even then, at
**87% draft acceptance** it still ran 38% slower — evaluating the drafter on
Metal costs more than it saves, the same root cause as
[llama.cpp#23752](https://github.com/ggml-org/llama.cpp/issues/23752). That
issue is written about MTP, but the result generalises to draft models here.
The dspark file is unused; delete it to reclaim 1.8 GB.

## Verified

| Check | Result |
|---|---|
| Ternary Q2_0 loads | prism source build ✓ (brew b10200 ✗) |
| Metal | `MTLGPUFamilyApple10`, `has tensor = true` |
| API key auth | completions/props/models-load: no-key 401 · bad-key 401 · good-key 200 |
| Container → host GPU server | ✓ completion at 13.0 tok/s |
| Vision (mmproj) | ✓ identified a red circle |
| Open WebUI → llama-server | ✓ model listed, completion returned |
| **Router lists exactly both models** | ✓ `bonsai-27b-ternary`, `bonsai-27b-1bit` — no phantom `default` |
| **1-bit Q1_0 runs on Metal** | ✓ child maps `IOAccelerator` + `AGXMetalG17G`, not a CPU fallback |
| **Vision on both models** | ✓ both answered "It is a red circle." — per-section `mmproj` works |
| **Model switching in the UI** | ✓ both appear in the dropdown and select cleanly |
| **`--models-max 1` swap** | ✓ exactly one child resident; switch costs +4–5 s |
| **`--models-max 2` fits** | ✓ both resident, 6 alternating requests, 0 allocation failures |
| **No orphaned children** | ✓ `make down` leaves no `llama-server` process behind |
| **cache-viz under the router** | ✓ 8/8 tests; requests keyed by `(port, task)` and labelled by model |
| Throughput: ternary vs 1-bit | **not yet measured** — see below |

### Not yet measured

The tok/s comparison between the two quantizations is **outstanding**. Every
attempt during this work ran on a machine at load average 46–120 with 11–12 GB
of swap in use (Teams at 181% CPU, Docker Desktop's VM at 113%), which pushed
even the ternary model to 2.8–3.3 tok/s against its documented 12.9 baseline.
Those numbers measure the machine's contention, not the models, so they are
deliberately not recorded here.

To fill this in on a quiet box:

```bash
make bench MODEL=bonsai-27b-ternary
make bench MODEL=bonsai-27b-1bit
```

Run each twice and take the second reading — with `--models-max 1` the first
call after a switch includes the model swap.
