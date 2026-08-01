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

```bash
./start-server.sh    # native Metal inference server on 0.0.0.0:8080
./start-webui.sh     # Open WebUI (native) on http://127.0.0.1:9090
```

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

## Verified

| Check | Result |
|---|---|
| Ternary Q2_0 loads | prism source build ✓ (brew b10200 ✗) |
| Metal | `MTLGPUFamilyApple10`, `has tensor = true` |
| API key auth | no-key 401 · bad-key 401 · good-key 200 |
| Container → host GPU server | ✓ completion at 13.0 tok/s |
| Vision (mmproj) | ✓ identified a red circle |
| Open WebUI → llama-server | ✓ model listed, completion returned |
