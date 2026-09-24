# Browser-agent validation status

The first checks ran on Apple Metal on 2026-09-23. RTX testing began on
2026-09-24 after the worker became accessible. Results below are partial: no
CUDA fit/offload setting or production default for the new models is claimed.
Raw reports are under the ignored `run/` directories.

## RTX worker update, 2026-09-24

SSH access to `firesand-worker` became available. Its RTX 3070 Ti has 8,192 MiB
VRAM, NVIDIA driver 595.84, about 60 GiB system RAM, and the installed app
runs in Docker. The live image reports Prism `4dd165625`; a separate test image
was built from pinned Prism `922be44aa6ac81b46f092716351cddff1c1733a7`
with CUDA 12.4.1 and Ampere (`sm_86`) compilation. The installed Git checkout,
its model files, and WebUI data were not replaced. Each GPU comparison paused
only the live llama container, used a loopback-only test port, and restored the
old container afterward. Its authenticated `/health` endpoint was checked
afterward.

The isolated Linux worktree passed `python3 -m unittest discover -s tests -q`:
391 tests, 10 skipped. The CUDA image's offline `version.sh` reported Prism
`922be44`, CUDA 12.4.1, the exact pinned base digests, and revision `8c29b60`.

The first matched browser comparison used the same Bonsai 1-bit GGUF, mmproj,
template SHA, 65,536-token context, host-side q8_0 KV cache, GPU layer setting,
flash attention and sampling flags. Browser/Playwright MCP ran on the Mac via
an SSH tunnel; model inference and VRAM measurement ran on the RTX worker. This
adds tunnel latency, but it was the same route in both trials.

| Bonsai 27B 1-bit basic form, one trial | Old `4dd165625` | New `922be44` |
| --- | ---: | ---: |
| Task oracle / tool errors | PASS / 0 | PASS / 0 |
| Complete task wall time | 68.47 s | 59.07 s |
| Model inference / browser time | 67.25 / 1.22 s | 57.84 / 1.23 s |
| LLM turns / browser calls | 6 / 5 | 6 / 5 |
| Input / output tokens | 14,435 / 624 | 14,435 / 624 |
| Server-reported prompt / generation speed | 229.6 / 12.63 t/s | 409.3 / 12.58 t/s |
| First token | 11.35 s | 4.50 s |
| Peak RTX VRAM | 4,989 MiB | 4,989 MiB |
| Startup log to model loaded | about 12.24 s | about 4.36 s |

This is a single run, not a success-rate or performance guarantee. The new
run evaluated 2,917 uncached prompt tokens versus 3,707 in the old run; the
reported prompt rates therefore do not isolate a kernel-speed change. Peak
VRAM came from worker `nvidia-smi` telemetry sampled every 500 ms, and load
times came from the corresponding Docker startup logs, not the browser JSON
reports (which leave those fields empty). The raw telemetry and logs are in
the ignored worker `run/rtx/` directory.

The new
image loaded the GGUF and mmproj and served `/v1/chat/completions`,
`/v1/responses`, and `/v1/messages` tool round trips. In a short 128-token API
probe, both builds failed the same plain-text cases because reasoning consumed
the cap. One two-tool Responses case passed on the old build and failed on the
new build; repeat it with a larger token limit before assigning a parser
regression. Both builds logged that `cache-reuse` is disabled when mmproj is
attached. The new-model RTX comparison is pending live tests; the GGUF weights
have now downloaded and passed the catalogue's size and SHA-256 checks.

The sections below describe the earlier Apple Metal checks. Their prior
statement that the RTX host was unavailable applies to **2026-09-23 only**.

| Model | Host and test | Result | Measured detail |
| --- | --- | --- | --- |
| Qwen3.5 4B Q4_K_M | Apple Metal, pinned Prism, one local Playwright basic form | PASS | 12.93 s, 6 LLM turns, 5 tool calls, 0 tool errors; mixed prompt 716 t/s and generation 34.4 t/s. This was one trial, not a reliability rate. |
| Gemma 4 E4B official QAT Q4 | Apple Metal, pinned Prism, GGUF load | PASS | File size and SHA-256 verified. Load took about 2.26 s. No mmproj was attached. |
| Gemma 4 E4B official QAT Q4 | Apple Metal, synthetic API probe | PASS 17 / FAIL 3 | Chat and Responses text/tool round trips passed, including streaming and two distinct tools. Messages text passed, but Messages tool continuation failed in plain, streamed, and two-tool cases. Forcing reasoning off did not fix the tested Messages failures. |
| Gemma 4 E4B official QAT Q4 | Real Playwright MCP tool selection | PASS | With 1, 5, 20 and the full safe inventory advertised, both `auto` and `required` selected the correct navigation tool and reached the fixture. This tests selection, not form completion. |
| Gemma 4 E4B official QAT Q4 | Apple Metal, one local Playwright basic form | FAIL | 120.1 s cap; navigation and targeted `browser_find` worked, but it used label-like targets instead of element refs in fill/type calls, then repeated bad calls. 5 tool errors (3 from MCP). No task completion. |
| Gemma 4 E4B official QAT Q4 | Real Codex CLI | FAIL | Local macOS sandbox blocked app-server initialisation before a model request. This is an environment failure, not evidence of model incompatibility. |
| Gemma 4 E4B official QAT Q4 | Real Claude Code and Claude + Playwright MCP | FAIL | Client's initial prompt exceeded the 8K server context before tool use. Test with larger measured context or a smaller client tool set. |
| Gemma 4 E4B official QAT Q4 | Real OpenCode | FAIL | Read the fixture file, then timed out at 120 s before safe edit. |
| Qwen3.6 35B A3B, all quants | CUDA load, Playwright and real clients | SKIPPED on 2026-09-23 | The RTX worker became available the next day; weight transfer and runtime checks are in progress. Static Prism loader/parser inspection is not a runtime test. |
| Qwen3.5 9B, Granite 4.1 8B, Bonsai 1-bit | New full Playwright comparison | SKIPPED on 2026-09-23 | The Bonsai 1-bit RTX basic-form result appears above; all-model comparison remains pending. |

The Gemma synthetic tool passes do not cancel the browser failure. A tool call
with valid JSON can still name the wrong browser target. Do not promote Gemma
as a fast default from text or API probes alone.

The Qwen3.5 4B prefix-cache probe reused a server that had handled an earlier
task, so its first call was **not a true cold call**. With a roughly 5,402-token
prompt, the two later shared-prefix calls took 0.267 and 0.271 s, while the
`cache_prompt=false` control took 7.98 s. This is useful local evidence that
prompt reuse matters, not an RTX speed estimate.

To complete the decision, run at least three repetitions per model of the
[Playwright suite](playwright-agent-benchmark.md) on the RTX 3070 Ti at 8K.
First compare Qwen3.5 9B, Qwen3.5 4B, Granite 4.1 8B, Gemma 4 E4B and
Qwen3.6 35B A3B Q4_K_XL. Then compare Qwen3.6 Q4_K_M or IQ4_XS and Gemma
Q5_K_M, plus 12K/16K context only when stable. Record load time, GPU offload,
whole-host RAM and peak VRAM, cold/warm prompt processing, full task time and
success. Keep the current Qwen3.5 9B everyday role **provisional** until that
comparison is complete.

Local commands used:

```sh
./fetch-models.sh --model gemma4-e4b
python3 tests/smoke_agent_fixture.py
python3 tests/runtime_probe.py --base-url http://127.0.0.1:18082 --model gemma4-e4b --tools --max-tokens 384 --output run/gemma4-api.json
python3 tests/runtime_clients.py --base-url http://127.0.0.1:18082 --model gemma4-e4b --client all --claude-playwright --output run/gemma4-clients.json
python3 tests/playwright_agent_bench.py --base-url http://127.0.0.1:18082 --model gemma4-e4b --tasks basic_form --repetitions 1 --task-timeout 120 --output run/gemma4-browser-basic.json
```

The exact runtime-test flags in the raw files may include extra optional
limits. Keep those raw reports with any RTX comparison.
