# Design decisions

What we chose, why, and the measured evidence where it exists. The summary
table from the PRD first, then the detail per decision.

| Decision | Rationale |
|---|---|
| Inference native on macOS, not Docker | Docker Desktop's Linux VM exposes no GPU to containers |
| Fork pinned by commit in Dockerfile | Reproducible CUDA builds; upstream can't read Q2_0 anyway |
| Router with one resident model | 24 GB Mac / 8 GB GPU can't hold both + KV cache comfortably; swap is +4–5 s |
| `BONSAI_CTX=8192` on the 8 GB path | Ternary weights + mmproj are 7.3 GB; KV cache must fit alongside |
| API key even on loopback-docker | Router binds 0.0.0.0 for the container bridge; key is the only barrier |
| Loopback-only published ports | Open WebUI runs with `WEBUI_AUTH=false`; LAN exposure must be deliberate |

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

## Open WebUI runs natively on macOS (for now)

The plan was the UI in a container (`docker/compose.yaml`, pointing at
`host.docker.internal:8080`). Docker Desktop's registry proxy wedged mid-setup
and every `docker pull` now hangs, so `start-webui.sh` runs the same UI
natively instead. `docker/compose.yaml` is kept ready for when the proxy is
fixed. Symptoms and diagnosis:
[macos.md](macos.md#docker-registry-proxy-known-issue-on-this-machine).

## One model resident by default (`--models-max 1`)

Selecting the other model in the dropdown unloads the current child and loads
the new one. Measured cost of a switch, with the machine under other load: a
request to the loaded model takes 4–7 s, a request that switches model takes
8–12 s — so a switch costs roughly **+4–5 s**. The weights are `mmap`ed, so the
OS page cache keeps them warm and a reload is far cheaper than a cold
3.8–6.7 GB read.

`BONSAI_MODELS_MAX=2` keeps both resident and makes switching instant. This was
tested and **works** — both children loaded, served six alternating requests,
and logged no allocation failures, even while the machine was already 12 GB
into swap. It is not the default only because one-at-a-time leaves more
headroom for everything else on the box.

## Context size and KV cache defaults

The model trains to 262144 context; that KV cache will not fit in 24 GB, hence
the lower default of `BONSAI_CTX=32768` on macOS. On the 8 GB VRAM Linux path
compose defaults to `BONSAI_CTX=8192`: the ternary weights + mmproj are 7.3 GB,
so the KV cache must fit alongside them.

KV cache type is **q4_0**, not q8_0, measured on this fork's Metal kernels
(`llama-bench`, stack down, r=3): Q1_0 tg128 **16.14** (q4_0) vs 14.06 (q8_0)
vs 13.85 (f16) — +15% on the 1-bit model. Upstream advice says quantized KV is
unoptimized on Metal
([llama.cpp#23011](https://github.com/ggml-org/llama.cpp/issues/23011)); the
prism fork's own kernels invert that. Measurement beats citation.

`cache-ram` is capped below the llama-server default of 8192 MiB. On a 24 GB
box already holding 6.7 GB of weights wired by Metal, letting the prompt cache
grow to 8 GB is what pushes the weights out to swap — measured at 13.5 GB of
page-ins during a single benchmark run. The cap keeps multi-turn reuse (a few
checkpoints of ~150 MiB each) without competing with the weights for RAM.

## Menu bar app is native Swift, not SwiftBar

SwiftBar or xbar would have made this a 30-line bash script that prints menu
lines. Rejected because it puts a third-party app between the user and the
stack: the thing in the menu bar would be SwiftBar, with its formatting
constraints, and `brew install --cask swiftbar` becomes a prerequisite for a
repo that otherwise needs only what macOS ships.

Xcode is already present on this machine, and a single-file SwiftUI
`MenuBarExtra` has no runtime dependencies at all. The cost is a few hundred
lines of Swift instead of ~30 lines of bash, plus a build step — `make
menubar` needs Xcode, while `make up` does not.

Compiled with `-swift-version 5`: Swift 6 strict concurrency rejects the
`DispatchQueue` hand-off between the status poller and the UI, and adopting
full actor isolation is not worth it for one file.
