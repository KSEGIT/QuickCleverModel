<p align="center">
  <img src="assets/hero.svg" alt="QuickCleverModel — Bonsai 27B thinking models on Apple Metal and CUDA" width="100%">
</p>

# QuickCleverModel — Bonsai 27B, fast on consumer GPUs

## What is this?

This runs a 27B thinking AI model on your own computer. You need a Mac with
Apple Silicon, or a Linux PC with an NVIDIA card (tested on an RTX 3070 Ti
with 8 GB). The model weights are free and open. Everything is private —
nothing leaves your machine. You talk to the model in a web page in your
browser.

## Quick setup

**Linux + NVIDIA** (tested on Ubuntu, RTX 3070 Ti 8 GB) — two commands:

```bash
git clone https://github.com/KSEGIT/QuickCleverModel.git && cd QuickCleverModel
./install.sh
```

The first run takes 10–20 minutes (it builds the CUDA image) and downloads
about 11 GB of model weights. It is safe to re-run after any failure.

Then open `http://127.0.0.1:9090` and pick **`bonsai-27b-1bit`** in the
dropdown. The first answer takes 10–20 seconds while the model loads.

**macOS (Apple Silicon):** the setup is a few steps more — see
[docs/macos.md](docs/macos.md#first-time-setup-macos).

**Browsing from another machine?** Ports are loopback-only by default. Use an
SSH tunnel (`ssh -N -L 9090:127.0.0.1:9090 user@host`), or set
`WEBUI_BIND=0.0.0.0` in `.env` and `WEBUI_AUTH: "true"` in
`docker/compose.linux.yaml`, then re-run compose up. Details:
[docs/architecture.md](docs/architecture.md#ports--security).

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
  watch logs, `make bench` measure speed.
- Linux: `docker compose --env-file .env -f docker/compose.linux.yaml up -d`
  to start, same command with `down` to stop.

## Documentation

- [docs/PRD.md](docs/PRD.md) — what this project is for and what it must do.
- [docs/architecture.md](docs/architecture.md) — how the parts fit together on both platforms.
- [docs/decisions.md](docs/decisions.md) — why we built it this way, with measured evidence.
- [docs/tuning.md](docs/tuning.md) — speed numbers measured on this machine.
- [docs/macos.md](docs/macos.md) — setup and day-to-day use on a Mac.
