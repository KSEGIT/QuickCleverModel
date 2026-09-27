# Browser-agent model expansion

The target is a single-user Playwright/MCP agent on an RTX 3070 Ti with 8 GB
VRAM. Choose models by successful **end-to-end task time**, not by model size or
generation tokens per second alone. The two new models below are candidates,
not measured production defaults.

## Verified model artifacts

The revision, filename, byte count, and SHA-256 values below came from the
Hugging Face model registry on 2026-09-23. Pin downloads to these repository
revisions and verify both size and SHA-256. The listed sizes cover the GGUF
weights only; reserve more disk space for downloads, cache, and alternatives.
The default agent presets should be text/tool-only: **do not attach an mmproj**.

| Candidate | Repository and exact revision | Exact GGUF filename | Bytes (approx. GB) | SHA-256 |
| --- | --- | --- | ---: | --- |
| Qwen3.6 35B A3B, Q4_K_XL (first test) | [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/a483e9e6cbd595906af30beda3187c2663a1118c), `a483e9e6cbd595906af30beda3187c2663a1118c` | `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` | 22,360,456,160 (22.36) | `707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450` |
| Qwen3.6 35B A3B, Q4_K_M (comparison) | Same revision | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | 22,134,528,992 (22.13) | `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61` |
| Qwen3.6 35B A3B, IQ4_XS (comparison) | Same revision | `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` | 17,730,509,792 (17.73) | `649d7508507b84638732c4f52c24c8b15843c6dca2f3ff793ae07c14a67ebbb3` |
| Gemma 4 E4B, official QAT Q4 (first test) | [`google/gemma-4-E4B-it-qat-q4_0-gguf`](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/tree/4b4a2c1d584be7264f87aac328a1bc739ce81b6c), `4b4a2c1d584be7264f87aac328a1bc739ce81b6c` | `gemma-4-E4B_q4_0-it.gguf` | 5,154,941,280 (5.15) | `676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee` |
| Gemma 4 E4B, Q5_K_M (comparison) | [`unsloth/gemma-4-E4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/tree/bfc15c382204943c3a8fff0c750b94ae2364d7a3), `bfc15c382204943c3a8fff0c750b94ae2364d7a3` | `gemma-4-E4B-it-Q5_K_M.gguf` | 5,481,798,784 (5.48) | `a5d6e634db151368d2caa4270fd32ab604d96c2d1291950586989f46d8e17d36` |

The registry reports the Qwen GGUF architecture as `qwen35moe` and Gemma's
as `gemma4`. Both contain an embedded Jinja chat template. The Qwen template
includes Unsloth's developer-role and tool-call fixes; the official Google
Gemma QAT file includes Google's current tool-call template. Use the embedded
template for each model, not the Bonsai override.

The pinned Prism revision
[`922be44aa6ac81b46f092716351cddff1c1733a7`](https://github.com/PrismML-Eng/llama.cpp/tree/922be44aa6ac81b46f092716351cddff1c1733a7)
contains `qwen35moe` and `gemma4` architecture loaders. Its
[`common/chat.cpp`](https://github.com/PrismML-Eng/llama.cpp/blob/922be44aa6ac81b46f092716351cddff1c1733a7/common/chat.cpp)
selects the Qwen XML tool parser from `<tool_call>`, `<function=`, and
`<parameter=` markers, and the Gemma 4 PEG parser from the
`'<|tool_call>call:'` marker. The above embedded templates contain those
markers. This is **static compatibility evidence only**. It does not prove that
either GGUF loads, generates valid calls, or fits the RTX card.

## Intended roles

| Model ID | Role | When to use it |
| --- | --- | --- |
| `qwen3.5-9b-q4_k_m` | Primary fast Playwright agent | Give it an ordinary browser task and let it finish the task. Keep it as the provisional everyday choice until measured results say otherwise. |
| `qwen3.6-35b-a3b` | Complex agent, planner, and recovery candidate | Escalate a difficult page, ambiguous form, unexpected state, or repeated failed browser action at a task boundary. Hybrid CPU/GPU inference and model loading may cost more than its reasoning saves. |
| `gemma4-e4b` | Fast full-GPU challenger | Compare complete browser workflows against Qwen3.5 9B, especially tool calling, DOM reasoning, forms, and extraction. GPU residency is a target to verify, not a known result. |
| `qwen3.5-4b-q4_k_m` | Very fast worker | Give it an entire simple, deterministic workflow, such as known-value form fill or extraction. |
| `granite-4.1-8b-q4_k_m` | Structured tool-call baseline | Keep as a comparison for reliable JSON arguments and multi-turn tools. |

Retain all existing Bonsai presets for comparison and rollback. Do not route
each click or field fill to a different model. Start and finish with one model
when possible; escalate only when task-level evidence calls for it. Measure
switch/load time in the total task time.

## Host and RTX validation gates

Qwen3.6 Q4_K_XL has 22.36 GB of weights. It cannot fit fully in 8 GB VRAM.
Treat **32 GB system RAM as a minimum to assess**, with **64 GB preferred** for
the model, browser, OS, and prompt cache together. These are planning limits,
not measured pass thresholds. Warn on clearly insufficient *available* RAM;
do not reject a 32 GB machine solely because it is below 64 GB. Start at an
8,192-token context and test 12,288 and 16,384 only where the model remains
stable. Test Q4_K_XL against Q4_K_M or IQ4_XS; smaller GGUF size does not prove
lower task time for a MoE model.

On the RTX 3070 Ti, sweep GPU offload for Qwen3.6 instead of setting
`n-gpu-layers=99` by assumption. Record the actual offloaded layers, peak VRAM,
peak system RAM, load time, time to first token, prefill and generation speed,
and complete browser-task time. Leave headroom for KV cache, CUDA buffers,
the browser/display, and long sessions. Start with f16 KV and an 8K context;
compare flash attention and conservative KV quantisation only after correct
tools and stable inference are established. Use model-specific context,
reasoning, sampling, and offload settings, not a shared Bonsai/Qwen/Gemma
profile.

The initial `qwen3.6-35b-a3b` preset deliberately leaves `ngl` unset and uses
Prism `fit=on` with a 1,536 MiB device margin. The pinned Prism fit code can
place excess MoE expert tensors in system RAM while selecting GPU layers.
`QCM_QWEN36_FIT_TARGET` changes the margin; setting `QCM_QWEN36_NGL` chooses
an explicit layer limit and stops automatic layer fitting when that limit would
otherwise need to change. In explicit-NGL mode, `fit-target` does **not**
guarantee that margin; measure peak VRAM yourself. Neither route is a measured
RTX recommendation yet.
The Gemma preset starts with full offload requested, subject to its actual load
and peak-memory tests.

The [upstream Qwen3.6 KV-cache issue](https://github.com/ggml-org/llama.cpp/issues/23589)
reports that some llama.cpp builds reprocess one batch of an otherwise shared
prompt prefix on follow-up turns. It has **not** been measured on this pinned
Prism build. Compare a cold call with repeated calls that share the same large
system prompt, tool schema, and candidate profile. Record cached tokens and
prefill time. A fast decode rate cannot offset repeated multi-thousand-token
prefill on every browser action.

Before promoting either model, run local deterministic Playwright fixtures:
basic and multi-page forms, job-listing extraction into strict JSON,
conditional application, popup recovery, wrong-page recovery, an injected tool
failure, a large page state, and a synthetic job application with a CV-upload
simulation. The candidate profile must be explicit and unsupported experience
must not be invented. Preserve the MCP setting that omits automatic full-page
snapshots and image payloads; request targeted findings or a deliberate
snapshot only when needed.

For each model, test one, five, and 20+ declared tools plus the real Playwright
MCP set. Check `tool_choice=auto` and `required`, streaming, argument JSON,
multi-turn continuation, wrong tool names, hallucinated selectors, and recovery
after a tool error. Compare supported thinking-on and thinking-off/reduced
modes on the *same* tasks. Keep API probes distinct from actual Codex, Claude
Code, OpenCode, and Playwright MCP client runs. Report each as `PASS`, `FAIL`,
or `SKIPPED — reason`.

The decision metric is successful end-to-end task time. Record success count
and rate, median and p95 task time, total inference versus browser time, LLM
turns, input/output tokens, prompt and generation tokens/s, time to first
token, malformed calls, hallucinated actions, recovery success, peak VRAM/RAM,
and model load time. Do not name a winner before those RTX measurements exist.
