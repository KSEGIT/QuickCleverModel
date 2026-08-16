# macOS operations

Everything specific to running the stack on Apple Silicon. For why inference
runs natively rather than in Docker, see
[architecture.md](architecture.md#macos-shape--native-metal).

## First-time setup (macOS)

A fresh clone is missing four things the repo deliberately does not track —
weights, the fork source, the Python env, and the API key. Prerequisites:
cmake (`brew install cmake`), Python 3.11, `hf` (`brew install huggingface-cli`),
and node/npx (for the Playwright MCP, optional).

```bash
# 1. Build the prism fork's llama-server (Metal is enabled by default on macOS).
#    brew's llama.cpp cannot load the ternary Q2_0 layout — see docs/decisions.md.
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

## Menu bar control

A small native app shows stack state and offers whole-stack Restart and Stop.

```bash
make menubar            # build it (needs Xcode)
make menubar-install    # build + start at login
make menubar-uninstall  # remove the login item
```

The icon is a leaf: filled when all three services are up, outline when all
are down, a warning triangle when only some are. Status refreshes every 15
seconds.

With the stack down, **Restart** is what starts it — `stack.sh restart` is
`down` then `up`, so there is no separate Start item.

The app is a thin client over `stack.sh`; it reads `./stack.sh status --json`
and shells out for actions. Ports come from `port_of`, so `BONSAI_PORT`,
`PW_MCP_PORT` and `WEBUI_PORT` in `.env` are honored.
Run `make up` again.

Once the Docker registry proxy is working again, the same UI runs in a
container instead — from the repo root, so interpolation finds the key:

```bash
docker compose --env-file .env -f docker/compose.yaml up -d
```

It points at `host.docker.internal:8080`.

Tunables (env): `BONSAI_CTX` (default 32768), `BONSAI_PORT`, `BONSAI_HOST`,
`BONSAI_MODELS_MAX` (default 1 — see
[architecture.md](architecture.md#model-selection)). The model trains to
262144 context; that KV cache will not fit in 24 GB, hence the lower default.
Extra `llama-server` flags pass straight through to the **router**:

```bash
./start-server.sh --reasoning off        # disable thinking mode
./start-server.sh --reasoning-budget 256 # cap thinking tokens
```

## Thinking mode

The model reasons by default: short `max_tokens` yields an **empty `content`**
with the text in `message.reasoning_content`. Either allow enough tokens
(~200+), or run with `--reasoning off`.

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
| Throughput: ternary vs 1-bit | **not yet measured** — see [tuning.md](tuning.md#not-yet-measured) |

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
docs/PRD.md                         product requirements document
docs/architecture.md                system architecture (both platforms, request flow, ports & security)
docs/decisions.md                   design decisions and why
docs/tuning.md                      measured performance tuning
docs/macos.md                       this file — macOS operations
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
