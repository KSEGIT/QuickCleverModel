<p align="center">
  <img src="assets/hero.svg" alt="QuickCleverModel — Bonsai 27B thinking models on Apple Metal and CUDA" width="100%">
</p>

# QuickCleverModel — Bonsai 27B, fast on consumer GPUs

[![tests](https://github.com/KSEGIT/QuickCleverModel/actions/workflows/tests.yml/badge.svg)](https://github.com/KSEGIT/QuickCleverModel/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![release](https://img.shields.io/github/v/release/KSEGIT/QuickCleverModel)](https://github.com/KSEGIT/QuickCleverModel/releases/latest)

## What is this?

This runs a 27B thinking AI model on your own computer. You need a Mac with
Apple Silicon, or a Linux PC with an NVIDIA card (tested on an RTX 3070 Ti
with 8 GB). The model weights are free and open. Everything is private —
nothing leaves your machine. You talk to the model in a web page in your
browser.

## Quick setup

This branch upgrades Prism and adds Qwen3.5 9B/4B and Granite 4.1 8B Q4_K_M.
Read the [runtime migration and version record](docs/runtime-versions.md)
before updating an existing installation. The legacy ternary Q2_0 file must
remain for rollback; `./fetch-models.sh bonsai` downloads its official PQ2_0
replacement separately. Target RTX validation is still pending.

Download new text models with `./fetch-models.sh agents` (about 13.8 GB), then
restart to expose them in model discovery. They use native templates and 8K
starting contexts. `./version.sh` prints pins and installed build information
without network access. See [API and client tests](docs/runtime-testing.md).
Current PASS, FAIL and SKIPPED results are in the
[runtime validation record](docs/runtime-validation.md).

For the browser-agent expansion, download one model at a time:

```bash
./fetch-models.sh --model qwen3.6-35b-a3b           # Q4_K_XL, 22.36 GB
./fetch-models.sh --model gemma4-e4b                 # official QAT Q4, 5.15 GB
./fetch-models.sh --model qwen3.6-35b-a3b --quant IQ4_XS
```

The last command is an optional comparison, not a replacement for the default
file. `--all` still downloads every catalogue artifact, including alternatives
(about 92 GiB of weights); check disk space first. See [browser-agent models](docs/browser-agent-models.md)
and [the opt-in Playwright benchmark](docs/playwright-agent-benchmark.md).
See [validation status](docs/browser-agent-validation.md) before choosing a
production browser-agent model; RTX 3070 Ti comparisons are still pending.
OpenCode's generated model list is static; it may show either new alias before
its selected GGUF exists. The router advertises an alias only after that file
is installed. Keep `QCM_QWEN36_QUANT` or `QCM_GEMMA4_QUANT` aligned with the
downloaded variant.

**Linux + NVIDIA** (tested on Ubuntu, RTX 3070 Ti 8 GB) — two commands:

```bash
git clone https://github.com/KSEGIT/QuickCleverModel.git && cd QuickCleverModel
./install.sh
```

The first run takes 10–20 minutes (it builds the CUDA image) and downloads
about 11 GB of model weights. It is safe to re-run after any failure.
Every setting the stack understands is documented in
[.env.example](.env.example).

Then open `http://127.0.0.1:9090` and pick **`bonsai-27b-1bit`** in the
dropdown. The first answer takes 10–20 seconds while the model loads.

**macOS (Apple Silicon):** the setup is a few steps more — see
[docs/macos.md](docs/macos.md#first-time-setup-macos).

## Remote access

Ports listen on loopback only. Nothing outside the box can reach them until
you choose one of these.

**Option 1 — SSH tunnel (safest, no config change).** Run this on your laptop,
then open `http://127.0.0.1:9090` as usual:

```bash
ssh -N -L 9090:127.0.0.1:9090 -L 8080:127.0.0.1:8080 user@<box-ip>
```

**Option 2 — Tailscale bind (good for daily use).** The box keeps the ports
open on its Tailscale address, so any machine on your tailnet can connect. On
the box, add this to `.env` (use the box's own Tailscale IP, from
`tailscale ip -4`):

```bash
WEBUI_BIND=100.x.y.z
LLAMA_BIND=100.x.y.z
```

Then restart: `docker compose --env-file .env -f docker/compose.linux.yaml up -d`.

Warning: the chat UI has no login by default. Binding it opens the UI to your
whole tailnet. On a shared tailnet, set `WEBUI_AUTH: "true"` in
`docker/compose.linux.yaml` first. The API on :8080 always asks for the key.

Never use `0.0.0.0` unless you mean to share with the whole office LAN.

From another machine, the API is now at `http://100.x.y.z:8080/v1` — point any
OpenAI-compatible tool there (OpenCode, Aider, scripts) with your
`BONSAI_API_KEY`. More detail: [docs/architecture.md](docs/architecture.md#ports--security).
For Codex CLI specifically, see [docs/codex.md](docs/codex.md); for Claude
Code, [docs/claude-code.md](docs/claude-code.md) — it is the one that can also
use MCP servers such as Playwright.

## Using it

Two versions of the same model. Pick one in the dropdown:

| model | size | quality | pick it when |
|---|---|---|---|
| `bonsai-27b-ternary` | 6.7 GB | 94.6% of FP16 | you want the best answers and have the memory |
| `bonsai-27b-1bit` | 3.8 GB | 89.5% of FP16 | you want speed, or you have 8 GB VRAM |

The model thinks before it answers. Short answers can hide inside the
thinking; `--reasoning off` turns thinking off. Details:
[docs/macos.md](docs/macos.md#thinking-mode).

API example (OpenAI-compatible):

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $BONSAI_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"bonsai-27b-1bit","messages":[{"role":"user","content":"hi"}]}'
```

Useful commands:

- macOS: `make up` start, `make down` stop, `make status` check, `make logs`
  watch logs, `make bench` measure speed, `make menubar-install` for a menu
  bar icon with stack status and restart/stop.
- Linux: `docker compose --env-file .env -f docker/compose.linux.yaml up -d`
  to start, same command with `down` to stop.

## Keeping it up to date

Linux only. `update.sh` pulls new code, rebuilds the image, restarts the
stack, and then asks every model to answer. If any model cannot answer, it
puts the old version back and tells you.

```bash
./update.sh            # update now, and undo it if the stack stops working
./update.sh --check    # only report what is behind; changes nothing
./update.sh --rollback # go back to the last version that worked
```

If the new code breaks the stack, `update.sh` remembers that commit and will
not move to it again. It starts working again by itself once a newer commit
lands. Failures that are not the code's fault, like a lost network, do not
count. See [docs/updating.md](docs/updating.md).

It checks the weights are really there before it restarts anything. A stack
can report itself healthy while every model is dead, so a health check is not
enough. Only a real answer proves the stack works.

To update every week on its own:

```bash
sed -e "s|__ROOT__|$PWD|g" -e "s|__USER__|$USER|g" \
    docker/bonsai-update.service | sudo tee /etc/systemd/system/bonsai-update.service
sudo cp docker/bonsai-update.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now bonsai-update.timer
```

Read what it did with `journalctl -u bonsai-update.service -n 50`.

On macOS, update by hand: `git pull` then `make restart`.

## Documentation

- [docs/PRD.md](docs/PRD.md) — what this project is for and what it must do.
- [docs/architecture.md](docs/architecture.md) — how the parts fit together on both platforms.
- [docs/decisions.md](docs/decisions.md) — why we built it this way, with measured evidence.
- [docs/tuning.md](docs/tuning.md) — speed numbers measured on this machine.
- [docs/macos.md](docs/macos.md) — setup and day-to-day use on a Mac.
- [docs/updating.md](docs/updating.md) — how updates and the weekly timer work.
- [docs/codex.md](docs/codex.md) — using Codex CLI with this server as the model.
- [docs/claude-code.md](docs/claude-code.md) — same for Claude Code, which also gets MCP.
- [docs/benchmarks.md](docs/benchmarks.md) — browser-agent benchmark findings, all in one place.
- [docs/benchmark-action.md](docs/benchmark-action.md) — the manual GitHub Actions job that runs the benchmark on the RTX worker.

## License

MIT — see [LICENSE](LICENSE). That covers the scripts, compose files, menu bar
app, tests, and docs in this repository. The inference engine and the model
weights are not distributed here; they are fetched from their own sources at
install time and carry their own terms. See [NOTICE](NOTICE).
