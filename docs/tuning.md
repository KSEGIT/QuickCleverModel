# Tuning — measured on this machine

Generation on a 27B is **memory-bandwidth-bound**: every token streams the
whole weight file, so flag-tuning has a hard ceiling. Except for the 1-bit
build, where the Metal kernel does per-weight bit-test ALU work and decode is
**compute-bound** — which is why 1-bit is only +37% over ternary (16.1 vs 10.5
tok/s clean) despite having 47% fewer bytes, and why no flag fixes it.

## KV cache type (measured Aug 2026, llama-bench, stack down, r=3)

| KV type | Q1_0 tg128 | Q2_0 tg128 | verdict |
|---|---|---|---|
| **q4_0** | **16.14** | 10.50 | **default** — +15% on 1-bit |
| q8_0 | 14.06 | 10.25 | previous default |
| f16 | 13.85 | 10.67 | no benefit here |

Upstream advice says quantized KV is unoptimized on Metal
([llama.cpp#23011](https://github.com/ggml-org/llama.cpp/issues/23011)); the
prism fork's own kernels invert that. Measurement beats citation.

## Reality check vs discrete GPUs

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

## Speculative decoding

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

## Not yet measured

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
