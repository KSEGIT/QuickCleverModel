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
│   llama-server  ──  Metal / MTLGPUFamilyApple10  ──  0.0.0.0:8080           │
│   (prism fork, source-built)                       ▲        ▲               │
│                                                    │        │               │
│   open-webui (native, :9090) ──────────────────────┘        │ host.docker.internal
│                                                             │               │
│   ┌─────────────── Docker Desktop Linux VM ──────────────────┼────────────┐ │
│   │   any container ─────────────────────────────────────────┘            │ │
│   │   (verified: curl container got a completion at 13.0 tok/s)           │ │
│   └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

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

One command starts everything:

```bash
make up        # or: ./stack.sh up
```

```
  SERVICE     PORT   STATE     PID
  llama       8080   up        43261     Bonsai 27B on Metal
  playwright  8931   up        43431     browser tools over MCP
  webui       9090   up        43469     chat UI

  chat UI -> http://127.0.0.1:9090
```

| command | does |
|---|---|
| `make up` | start all three, waiting until each port actually accepts connections |
| `make down` | stop everything (reverse order) |
| `make restart` | down then up |
| `make status` | what's running, with PIDs |
| `make logs` | tail all logs — `make logs SVC=llama` for one |
| `make open` | open the chat UI |
| `make bench` | measure tok/s against the running server |

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

Tunables (env): `BONSAI_CTX` (default 32768), `BONSAI_PORT`, `BONSAI_HOST`.
The model trains to 262144 context; that KV cache will not fit in 24 GB, hence
the lower default. Extra `llama-server` flags pass straight through:

```bash
./start-server.sh --reasoning off        # disable thinking mode
./start-server.sh --reasoning-budget 256 # cap thinking tokens
```

## Thinking mode

The model reasons by default: short `max_tokens` yields an **empty `content`**
with the text in `message.reasoning_content`. Either allow enough tokens
(~200+), or run with `--reasoning off`.

## Security

`start-server.sh` binds `0.0.0.0` because binding `127.0.0.1` would be
unreachable from the Docker VM — which also makes it reachable from your LAN.
An API key in `.env` (gitignored, mode 600) is therefore required, and enforced:

```
no-key HTTP=401   bad-key HTTP=401   good-key HTTP=200
```

## Layout

```
start-server.sh                     native Metal inference launcher
start-webui.sh                      native Open WebUI launcher (:9090)
docker/compose.yaml                 containerised UI, parked pending Docker fix
.env                                BONSAI_API_KEY (gitignored, 0600)
.venv/                              python 3.11 env for open-webui
.webui-data/                        Open WebUI database + cache
models/Ternary-Bonsai-27B-gguf/     Q2_0 weights (6.7G) + mmproj Q8_0 (600M)
src/llama.cpp-prism/                fork source build (preferred binary)
bin/llama-prism-b9570-0ad1dab/      prebuilt fallback
```

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
| API key auth | no-key 401 · bad-key 401 · good-key 200 |
| Container → host GPU server | ✓ completion at 13.0 tok/s |
| Vision (mmproj) | ✓ identified a red circle |
| Open WebUI → llama-server | ✓ model listed, completion returned |
