# Runtime upgrade status

This is an **unreleased, RTX-unvalidated upgrade**. The branch's production
`docker/Dockerfile` pins the maintained Prism revision. Before deploying it,
download the official PQ2_0 ternary artifact; the historical Q2_0 file cannot
load in this revision. Keep the old image and Q2_0 file for rollback.
Run `./version.sh` offline to distinguish configured pins from an installed
binary. It does not claim to identify an already running remote service.

## Verified runtime identity

| Field | Previous release | This branch |
| --- | --- | --- |
| Prism repository | https://github.com/PrismML-Eng/llama.cpp | Same repository; no redirect or replacement |
| Full SHA | `4dd165625bb6c020285eec8b342af25cf60233dd` | `922be44aa6ac81b46f092716351cddff1c1733a7` |
| Commit date | 2026-07-31T00:07:28-07:00 | 2026-09-21T16:31:15Z |
| Development line | Historical checkout | `prism`, developed as `prism-v7` |
| Upstream baseline | Not established | `5ea87ddad22541a37053c7ba92b02ec1923617c6`, 2026-08-25T05:08:06Z |
| CUDA | 12.4.1 | 12.4.1 retained |
| OS | Ubuntu 22.04 | Ubuntu 22.04 retained |

The new pin is from the canonical fork of `ggml-org/llama.cpp`. Its
[README at the exact SHA](https://github.com/PrismML-Eng/llama.cpp/blob/922be44aa6ac81b46f092716351cddff1c1733a7/README.md)
identifies `prism` as the maintained line and warns against `prism-v6`.
The upstream baseline is discoverable in that branch's merge history; it is
not a claim that every upstream change as of September 22 is included.

| Component changed | Previous | New | Reason |
| --- | --- | --- | --- |
| Prism llama.cpp | `4dd165625bb6c020285eec8b342af25cf60233dd` | `922be44aa6ac81b46f092716351cddff1c1733a7` | Qwen3.5, Granite 4.1 and current API/tool parser support. |
| Ternary Bonsai GGUF | Legacy `Q2_0` type ID 42 | Official `PQ2_0` type ID 142 | Current Prism rejects the legacy tensor layout; keep both files for rollback. |
| Docker build | Single stage with source and build tools | Pinned two stage server build | Keep compilers, sources and package caches out of the runtime image. |
| Docker base reference | CUDA 12.4.1 Ubuntu 22.04 tag | Same version with resolved digest | Fix the selected image contents while retaining the supported CUDA and OS line. |

Dependency version changes are listed in [dependency-audit.md](dependency-audit.md).

## Ternary format migration

The existing `Ternary-Bonsai-27B-Q2_0.gguf` stores group-128 tensors under GGML
type ID 42. Current upstream uses that ID for group-64 tensors. Current Prism
deliberately rejects the older layout. This was reproduced against the actual
existing file: loader exit 1, expected offset 357580800, found 337715200.
Both `bonsai-27b-ternary` and `bonsai-27b-ternary-text` use that same file.

The [maintainer explanation](https://github.com/PrismML-Eng/llama.cpp/issues/167)
states that the official `PQ2_0` replacement contains the same packed tensor
bytes with the corrected private type ID 142. The two ternary presets now point
to that separate official file. No existing model has been overwritten or
removed. The old Q2_0 file itself **does not load** with new Prism; retain the
old image to use that exact file.

The replacement pinned in `models.lock.tsv` is:

```
prism-ml/Ternary-Bonsai-27B-gguf
revision: 86e89f34c93201c3dfd5e5880fedb0022fc7e34d
file: Ternary-Bonsai-27B-PQ2_0.gguf
bytes: 7165121600
SHA256: e4781999f1997ef97ce0c58d05750835acc999d18d83ee6489ba7ac7b14cb5f6
```

Download it with `./fetch-models.sh bonsai` before replacing the old image.
Rerun both aliases with the custom Bonsai template on the target GPU before
deploying. Do not silently relabel the legacy tensor type in the loader.

## Container audit

The Dockerfile uses the exact revision's [build instructions](https://github.com/PrismML-Eng/llama.cpp/blob/922be44aa6ac81b46f092716351cddff1c1733a7/docs/build.md)
and CMake definitions. CUDA support uses `GGML_CUDA=ON`, with explicit
`CMAKE_CUDA_ARCHITECTURES=86` for RTX 3070/3070 Ti. `LLAMA_CUDA` is not used.
`GGML_NATIVE=OFF` avoids compiling CPU instructions specific to a build host.
C++17 and the CUDA CMake minimum 3.18 fit Ubuntu 22.04's toolchain.

Prism's release workflow still builds CUDA 12.4 on Ubuntu 22.04, and its
September 18 release publishes CUDA 12.4 binaries. No verified requirement
justifies a CUDA or OS major upgrade. This source evidence does not replace a
CUDA build or a driver/runtime test on the target machine.

Base image indexes were resolved from NVIDIA's Docker registry:

- Development: `nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:da6791294b0b04d7e65d87b7451d6f2390b4d36225ab0701ee7dfec5769829f5`
- Runtime: `nvidia/cuda:12.4.1-runtime-ubuntu22.04@sha256:517da2300c184c9999ec203c2665244bdebd3578d12fcc7065e83667932643d9`

The Dockerfile fetches one exact Git revision, verifies it, and builds only the
server. Runtime contains its executable/shared libraries, certificates,
OpenSSL, OpenMP and curl for diagnostics. Source, compilers, git and package
caches stay outside the runtime stage. Python is not needed by these build
targets. Modern HTTPS uses `LLAMA_OPENSSL=ON`, `libssl-dev` in build and
`libssl3` at runtime; the old curl build integration is deprecated.

Both `LLAMA_BUILD_UI=OFF` and `LLAMA_USE_PREBUILT_UI=OFF` are required. A native
test build with only the first flag downloaded an upstream `latest` UI after
its versioned download failed. The clean Docker build disables both paths;
Open WebUI remains the separate UI. Upstream emits a warning when no embedded
UI assets exist. This is expected for this build, not a suppressed warning.

The x86 CUDA container build completed under emulation on this ARM Mac. The
runtime image is 1,550,294,857 bytes (`docker image inspect`) and its labels
contain the pinned Prism SHA, upstream baseline and CUDA version. Its server
binary cannot start without `libcuda.so.1` on this GPU-less Docker host; the
NVIDIA container runtime must supply the driver on the target. CUDA inference
on the RTX target remains **SKIPPED**: SSH to the configured
remote host was rejected for the available key. The old CUDA image is not
available in this Docker daemon for a size comparison. Base digests and source
are pinned; apt repository contents are not a bit-for-bit package lock.

## Server option audit

Checked against the pinned revision's `common/arg.cpp`, `common/preset.cpp` and
`tools/server/README.md`. All current preset keys remain recognised.

| Area | Decision |
| --- | --- |
| Jinja/templates | Keep `jinja=true`. Custom template only for three Bonsai aliases; native GGUF templates for Qwen/Granite. |
| Offload | Keep layer count configurable per model; 99 requests full layer offload. Fit is not proof that the full model fits an 8 GB GPU. |
| Flash attention | Keep prior Bonsai `on`; new families start at `auto`, separately configurable. No CUDA performance claim. |
| Context | 8192 common floor; separate Qwen9/Qwen4/Granite settings. 12K/16K target-GPU validation remains required. |
| KV cache | Keep previous Bonsai Q4 setting; new agents begin with f16. Expose symmetric q8_0 for comparison. Quantised V needs FA. |
| Batch/ubatch | Retain runtime batch default; new agents use ubatch 256. No blanket increase. |
| Parallel/residency | One slot per new agent; one resident model by default. Actual switch latency remains unmeasured. |
| Prompt cache | Retain supported cache controls, cap new agents' host cache at 1024 MiB. Cache allocation is distinct from GPU KV memory. |
| Reasoning | Qwen starts `reasoning=off`, separate optional budget. Granite uses native `auto` and unrestricted budget rather than Bonsai's 4096. |
| Template arguments | Use supported `reasoning` CLI option; `enable_thinking` CLI kwargs are deprecated. Per-request kwargs remain supported. |
| Speculation | New agents use `none`; retain existing Bonsai ngram configuration. No unmeasured drafter added. |
| Router | Keep `models-preset`, `models-max`, per-section `load-on-startup`. Missing optional agent downloads are excluded from discovery. |
| Multimodal | Keep existing Bonsai projectors; new presets are text-only. Qwen vision is not tested or advertised. |

Do not treat older Metal tuning notes as measurements for the new CUDA build.
CUDA graphs, batching, cache changes and optional drafters have not been tuned.
Reliability tests and target-GPU measurements must precede default changes.

## Dependency changes

The [dependency update record](dependency-audit.md) lists every changed package,
the previous and new versions, sources, and checks. Current Actions major pins
and Jinja2 were already current and remain unchanged. The runtime and Docker
changes are included on this branch; CUDA acceptance remains pending.

## Model sources and reproducibility

`models.lock.tsv` records immutable revisions, exact filenames, byte counts
and SHA256 values. The downloader checks the hash even for existing files,
preserves mismatched originals, and keeps partial downloads for retry.
`./fetch-models.sh agents` downloads the three new text models; no model is
downloaded merely by starting the stack.
OpenCode's static local provider configuration lists the three IDs even when
their files are absent. The router omits absent optional models from discovery;
install the files before selecting those IDs in OpenCode.

Granite is the official IBM `ibm-granite/granite-4.1-8b-GGUF` language model,
not Guardian. Qwen's specified 4B/9B Q4_K_M artifacts are from Unsloth's GGUF
repositories, whose metadata names the official Qwen checkpoints as bases.
All three downloaded files matched the registry's SHA256 values locally.

See [runtime testing](runtime-testing.md) for runnable API/client checks and
the required 8K/12K/16K, FA and KV comparisons. Live results are saved under
`run/` and summarised in `runtime-validation.md` when available.
