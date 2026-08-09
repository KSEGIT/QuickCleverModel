<p align="center">
  <img src="assets/hero.svg" alt="QuickCleverModel — Bonsai 27B thinking models on Apple Metal and CUDA" width="100%">
</p>

# QuickCleverModel — Bonsai 27B, fast on consumer GPUs

Two run modes: **native Metal on macOS** (with a Docker app layer) and a
**full Docker stack on Linux + NVIDIA**. The models reason (thinking mode) and
hold up surprisingly well for 1-bit/ternary weights.

## Quick setup

**Linux + NVIDIA** (tested on Ubuntu, RTX 3070 Ti 8 GB):

```bash
git clone https://github.com/KSEGIT/QuickCleverModel.git && cd QuickCleverModel
./install.sh
```

That's it — it checks/installs the GPU driver, Docker, and the container
toolkit, downloads the weights (~11 GB), generates an API key, builds the CUDA
image (~10–20 min first time), starts everything, waits until it serves, and
opens the UI. Safe to re-run after any failure — every step skips what's
already done. Details: "Linux + NVIDIA (Docker)" below.

Then:

- UI on `http://127.0.0.1:9090` — pick **`bonsai-27b-1bit`** in the dropdown
  first (comfortable on 8 GB VRAM); the first request takes ~10–20 s while the
  model loads.
- **Browsing from another machine?** Ports are loopback-only by default.
  Either tunnel (`ssh -N -L 9090:127.0.0.1:9090 user@host`) or expose on the
  network: add `WEBUI_BIND=0.0.0.0` to `.env`, set `WEBUI_AUTH: "true"` in
  `docker/compose.linux.yaml` (first signup becomes admin — without it, the
  whole LAN gets an unauthenticated UI), then
  `docker compose --env-file .env -f docker/compose.linux.yaml up -d`.
  Details: "Remote access" below.

**macOS (Apple Silicon):** see "First-time setup (macOS)" — build the fork
with cmake, create the venv and `.env`, then `make models && make up`.

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

## First-time setup (macOS)

A fresh clone is missing four things the repo deliberately does not track —
weights, the fork source, the Python env, and the API key. Prerequisites:
cmake (`brew install cmake`), Python 3.11, `hf` (`brew install huggingface-cli`),
and node/npx (for the Playwright MCP, optional).

```bash
# 1. Build the prism fork's llama-server (Metal is enabled by default on macOS).
#    brew's llama.cpp cannot load the ternary Q2_0 layout — see "Runtime choice".
git clone https://github.com/PrismML-Eng/llama.cpp src/llama.cpp-prism
git -C src/llama.cpp-prism checkout 4dd165625bb6c020285eec8b342af25cf60233dd
cmake -S src/llama.cpp-prism -B src/llama.cpp-prism/build -DCMAKE_BUILD_TYPE=Release
cmake --build src/llama.cpp-prism/build --target llama-server -j

# 2. Python env for Open WebUI
python3.11 -m venv .venv && .venv/bin/pip install open-webui

# 3. API key (enforced by the router — the server binds 0.0.0.0)
printf 'BONSAI_API_KEY=bonsai-%s\n' "$(openssl rand -hex 20)" > .env && chmod 600 .env
```

## Usage

Fetch the weights once (~11 GB, both models), then one command starts everything:

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
| `make models` | download the GGUF weights (~11 GB, both models). Skips files already present, so re-running is a sub-second no-op |
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

Once the Docker registry proxy is working again, the same UI runs in a
container instead — from the repo root, so interpolation finds the key:

```bash
docker compose --env-file .env -f docker/compose.yaml up -d
```

It points at `host.docker.internal:8080`.

Tunables (env): `BONSAI_CTX` (default 32768), `BONSAI_PORT`, `BONSAI_HOST`,
`BONSAI_MODELS_MAX` (default 1 — see "Model selection"). The model trains to
262144 context; that KV cache will not fit in 24 GB, hence the lower default.
Extra `llama-server` flags pass straight through to the **router**:

```bash
./start-server.sh --reasoning off        # disable thinking mode
./start-server.sh --reasoning-budget 256 # cap thinking tokens
```

## Linux + NVIDIA (Docker)

The same stack runs fully containerised on a Linux box with an NVIDIA GPU —
developed against an RTX 3070 Ti (8 GB VRAM, sm_86). `docker/Dockerfile` builds
the prism fork's `llama-server` with CUDA (the fork's Q1_0/Q2_0 kernels work on
the standard MMQ path, sm_86 included), and `docker/compose.linux.yaml` runs it
alongside Open WebUI.

Quick start — one command installs the prerequisites (driver/Docker/toolkit
via `setup-nvidia.sh`), the `hf` CLI, downloads the weights (~11 GB),
generates `.env`, builds and starts the stack, and waits until the UI
answers:

```bash
git clone https://github.com/KSEGIT/QuickCleverModel.git && cd QuickCleverModel
./install.sh             # --check to dry-run; safe to re-run, every step idempotent
```

After a fresh driver install it will tell you to reboot and re-run — it
resumes where it left off. The first build takes 10–20 minutes (CUDA
compile).

<details>
<summary>Manual steps (exactly what install.sh runs, if you prefer to drive it yourself)</summary>

Prerequisites on the host — either run `./setup-nvidia.sh` (auto-detects and
installs driver/Docker/nvidia-container-toolkit, `--check` to dry-run), or set
up manually:

- NVIDIA driver **>= 525** (required by CUDA 12.4)
- Docker (with the Compose plugin) + **nvidia-container-toolkit**, then
  `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`
- the `hf` CLI for the weight download — on Ubuntu >= 23.04 use
  `pipx install huggingface_hub[cli]` (a bare `pip install` hits PEP 668's
  externally-managed error); older releases can use `pip install -U "huggingface_hub[cli]"`

```bash
./fetch-models.sh        # ~11 GB, both models — lands in models/.
                         # Run this BEFORE first compose up: it creates models/
                         # itself; if docker creates the bind-mount source
                         # first, it is root-owned and the download fails.
printf 'BONSAI_API_KEY=bonsai-%s\n' "$(openssl rand -hex 20)" > .env && chmod 600 .env
# --env-file .env is REQUIRED: compose interpolation reads the shell env and
# the project-dir .env (docker/), not the repo-root one — without this flag
# the OPENAI_API_KEY line fails with "required variable is missing a value".
docker compose --env-file .env -f docker/compose.linux.yaml up --build -d
```

</details>

UI on http://127.0.0.1:9090, API on :8080 — same router mode, same model ids,
same `.env` key as the macOS stack.

**8 GB VRAM caveat.** Compose defaults to `BONSAI_CTX=8192` so the KV cache
fits next to the weights. The ternary Q2_0 (7.3 GB with mmproj) is tight at
that ctx; the 1-bit Q1_0 is comfortable and can go much higher:

```bash
BONSAI_CTX=32768 docker compose --env-file .env -f docker/compose.linux.yaml up -d
```

**Both ports are loopback-only by default** (`127.0.0.1:8080` / `:9090`) — the
webui runs with `WEBUI_AUTH=false`, so publishing it would give your whole LAN
an unauthenticated chat UI. The API key on :8080 is enforced either way.

**Remote access** (e.g. you SSH into the box over Tailscale and browse from
another machine): two options.

- Preferred — an SSH tunnel from your local machine, no config change:

  ```bash
  ssh -N -L 9090:127.0.0.1:9090 -L 8080:127.0.0.1:8080 user@<tailscale-ip>
  # then open http://127.0.0.1:9090 locally
  ```

- Or bind to the Tailscale interface: add `WEBUI_BIND=100.x.y.z` (the box's
  Tailscale IP) to `.env`, then `docker compose --env-file .env -f docker/compose.linux.yaml up -d`
  to recreate. This exposes the **unauthenticated** UI to your whole tailnet —
  fine on a personal tailnet, otherwise flip `WEBUI_AUTH` to `true` first.
  `LLAMA_BIND` does the same for :8080 (the API key is enforced there).

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
install.sh                          one-command Linux+NVIDIA installer (setup -> weights -> .env -> compose up)
setup-nvidia.sh                     Ubuntu NVIDIA driver/Docker/toolkit installer (--check to dry-run)
start-server.sh                     native Metal inference launcher (router mode)
models.ini.in                       model preset template -> run/models.ini
fetch-models.sh                     weight downloader (make models)
start-webui.sh                      native Open WebUI launcher (:9090)
cache-viz.py                        live prompt-cache dashboard (:8090)
tests/                              stdlib unittest suite for cache-viz.py
docker/compose.yaml                 containerised UI, parked pending Docker fix
docker/Dockerfile                   CUDA llama-server build (Linux + NVIDIA)
docker/compose.linux.yaml           full Linux/NVIDIA stack (llama + webui)
.env                                BONSAI_API_KEY (gitignored, 0600)
.venv/                              python 3.11 env for open-webui
.webui-data/                        Open WebUI database + cache
models/Ternary-Bonsai-27B-gguf/     Q2_0 weights (6.7G) + mmproj Q8_0 (629M)
models/Bonsai-27B-gguf/             Q1_0 weights (3.8G) + mmproj Q8_0 (629M)
src/llama.cpp-prism/                fork source build (preferred binary, gitignored)
bin/llama-prism-b9570-0ad1dab/      prebuilt fallback (gitignored)
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
whole weight file, so flag-tuning has a hard ceiling. Except for the 1-bit
build, where the Metal kernel does per-weight bit-test ALU work and decode is
**compute-bound** — which is why 1-bit is only +37% over ternary (16.1 vs 10.5
tok/s clean) despite having 47% fewer bytes, and why no flag fixes it.

### KV cache type (measured Aug 2026, llama-bench, stack down, r=3)

| KV type | Q1_0 tg128 | Q2_0 tg128 | verdict |
|---|---|---|---|
| **q4_0** | **16.14** | 10.50 | **default** — +15% on 1-bit |
| q8_0 | 14.06 | 10.25 | previous default |
| f16 | 13.85 | 10.67 | no benefit here |

Upstream advice says quantized KV is unoptimized on Metal
([llama.cpp#23011](https://github.com/ggml-org/llama.cpp/issues/23011)); the
prism fork's own kernels invert that. Measurement beats citation.

### Reality check vs discrete GPUs

An RTX 4060 Laptop runs this same Q1_0 file at ~21.6–22.4 tok/s — but that card
has **256 GB/s** of bandwidth against the base M5's **153 GB/s**. Per GB/s the
M5 is already *more* efficient (35–38% of peak vs 32–33%). Matching a 4060m on
this chip via llama.cpp is not physically on the table; the vendor's own
numbers scale to ~14.6 tok/s for a base M5, which is exactly where we measure.

The one evidenced path past 20 tok/s on this hardware is the **MLX 1-bit
runtime** (`prism-ml/Bonsai-27B-mlx-1bit`): its kernels consume packed weights
without FP16 expansion, and a *base M4 mini* (120 GB/s) measured 21.5 tok/s —
projected ~25–27 here. That is a separate runtime (e.g. oMLX ≥ 0.5.2 serves an
OpenAI-compatible endpoint), not a llama.cpp flag. Unbuilt; see the model card.

### Speculative decoding

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

On the **1-bit** build, speculation shows no benefit (measured Aug 2026):
generic chat never fires a draft, and even a repetitive workload at 49.8%
draft acceptance did not beat spec-off — break-even on this box needs ~57%+.
`ngram-simple` stays the default because it costs nothing when idle, but it is
not a lever on the 1-bit model. Independent corroboration: mlx-dspark measured
1-bit verify as a net 0.71–0.77x *loss* on Apple Silicon.

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
