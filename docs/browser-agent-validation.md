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
attached. The downloaded GGUF weights passed the catalogue's size and SHA-256
checks. The new-model RTX tests below are single runs unless a repeat count is
stated.

Gemma 4 E4B official QAT Q4 loaded on the RTX with an 8,192-token context,
f16 K/V cache, automatic flash attention, and 99 requested GPU layers. Its
startup log reached `model loaded` after about 181 seconds while the worker
showed sustained disk I/O pressure; do not treat that as a steady-state load
benchmark. One live basic-form task **failed** in 29.6 seconds. Gemma navigated
and found the Name textbox, then repeatedly passed invalid `browser_type`
targets (`ref=e6` and `Name`) despite Playwright's errors. It made six
malformed browser actions before a 512-token response cap ended the task.
Observed VRAM during the task was 3,135 MiB (a single sample, not a peak).
The RTX API probe passed 17/20 checks: Chat and Responses tool round trips,
streaming, and two distinct tools passed; three Messages tool-continuation
checks failed. This reproduces the earlier Apple Metal pattern and is not a
successful Playwright run.

With the shared prompt's bare-ref syntax explained, Gemma improved from the
first browser failure to five passes in ten single-run cases. Basic form took
14.0 seconds and popup 19.5 seconds. It passed both application decisions and
wrong-state recovery, but failed the three-step form, strict JSON extraction,
injected-error recovery, synthetic job application, and large-page extraction.
The job fixture ended in `Application incomplete` despite the model claiming it
had submitted. Six-case total wall time was 120.7 seconds, with two passes;
sampled peak VRAM was 3,135 MiB and a later warm model load logged 11.4
seconds. These outcomes do not support Gemma as the everyday browser default
on this stack, despite fast token generation.

Qwen3.5 9B Q4_K_M loaded on the RTX at 8K with f16 K/V, automatic flash
attention, reasoning off, and 99 GPU layers. Its startup log reached `model
loaded` after 32.8 seconds under the same disk pressure; sampled peak VRAM was
5,427 MiB. One run of ten browser cases passed seven in 81.5 seconds total;
the median of successful cases was 7.1 seconds. It passed basic form,
three-step form, popup, injected tool failure, synthetic job application,
eligible application decision, and wrong-state recovery. It failed extraction
by repeating `browser_find` until the 30-turn cap, returned a correct no-apply
decision in prose plus a fenced JSON block instead of strict JSON for the
ineligible case, and hit HTTP 400 on the large-page case at 8K context. The
last two failures need separate prompt/context investigation; no success rate
beyond these single runs is claimed. Aggregate server-reported rates over the
ten cases were about 987 prompt and 88 generation tokens/s; the prompt figure
includes cached turns and is not a cold-prefill rate. Real Playwright MCP
selection with 20 tools passed `auto` and `required`.

Qwen3.5 9B's native GGUF template rejects a second system message after the
first. Four Responses API regression cases returned HTTP 500 in the RTX API
probe, including tool continuation and streaming. Chat and Messages tool
round trips passed. This is a Codex compatibility blocker until the template
is corrected and retested; ordinary browser tasks use Chat Completions and
are unaffected by this specific failure.

The first Qwen3.6 Q4_K_XL browser pass (six tasks, original prompt) passed
four and failed two in 260.6 seconds. Its 66.9-second basic form involved 12
tool errors from using `[ref=e6]`-style targets where Playwright expects the
bare `e6` ID. A shared prompt change now states that exact syntax, prohibits
identical error retries, and asks for unfenced JSON. With the revised prompt,
Q4_K_XL completed the basic form in 17.2 seconds with zero tool errors, but
still fenced the extraction JSON and retried an injected failed click before
inspecting, so those two oracles failed. Its RTX API probe passed all 20 cases,
and 20-tool Playwright selection passed `auto` and `required`. Sampled peak
VRAM was 6,241 MiB; the container used about 14.7 GiB RAM during a task.

Qwen3.6 IQ4_XS, with the same 8K hybrid fit settings and revised prompt,
passed basic form (21.8 seconds cold, 15.1 seconds in a warm repeat),
three-step form (24.9 seconds), and synthetic job application (48.6 seconds),
all with zero tool errors. Sampled peak VRAM was 6,281 MiB. Its 5.4K-token
prefix probe took 19.66 seconds with prompt caching disabled, versus 0.54 and
0.53 seconds on two shared-prefix follow-ups. The first probe call already had
1,710 cached tokens, so it is not a pure cold-load number. These are too few
trials to choose between Q4_K_XL and IQ4_XS by quant speed; the browser task
and cache state must match.

Qwen3.5 4B, with the revised prompt, passed five of six selected workflows in
42.9 seconds total (successful-case median 5.85 seconds). Basic form took 4.55
seconds, injected-error recovery 5.85 seconds, and the synthetic application
17.7 seconds. The extraction facts were right, but a Markdown fence failed
strict JSON. In four additional cases, wrong-state recovery passed; eligible
and ineligible application decisions and large-page extraction had correct
facts/actions but failed strict JSON formatting. Its sampled peak VRAM was
3,157 MiB and startup log reached `model loaded` at 6.44 seconds. Selection
with 20 real Playwright tools passed `auto` and `required`.

In a later repeat of the four core workflows (basic form, three-step form,
injected tool failure, synthetic job application), Qwen3.5 4B passed all 12
cases across three repetitions with the patched template. Total wall time was
127.4 seconds; the successful-case median was 8.53 seconds and p95 was 18.43
seconds. Nine browser-action errors occurred, but the agent recovered in every
case. Sampled peak VRAM was 3,157 MiB. Mixed cached-turn server timings were
about 1,372 prompt and 135 generation tokens/s; neither is a cold-prefill
measurement. This core-workflow result does **not** erase its strict-JSON
failures in the broader ten-case set.

The Qwen3.5 GGUF templates for 9B and 4B have identical SHA-256 and the same
late-system guard. A model-specific copy with only that guard changed is now
in the PR, leaving the native Qwen tool and thinking grammar intact. Mounted
into the RTX test server, it passed all 20 API probe cases, including the four
Responses cases that previously returned HTTP 500. The real Codex CLI test
still failed before its first request: the macOS filesystem confinement blocked
in-process app-server initialisation. That is an environment failure, not a
model pass; the actual CLI round trip remains unverified.

A later real Claude Code CLI test passed the isolated file read/edit round trip
with Qwen3.5 9B after the harness bounded output to 2,048 tokens and disabled
compaction for an explicit 8K context. The first run copied a line-number
prefix from the Read tool; a clearer fixture instruction fixed that on repeat.
Real OpenCode read the fixture and wrote a file, but copied its Read-tool line
number despite the instruction, so its exact-output oracle failed. Claude Code
plus the full Playwright MCP inventory did not complete at 8K, 12K, or 16K:
it progressed farther at 16K (navigate, snapshot, tabs, close) but then
reported `Prompt is too long`. This is a real-client limit, not a synthetic
Messages API pass. The harness still keeps Playwright snapshots explicit and
image responses omitted.

Qwen3.5 9B with the revised prompt and patched template passed eight of ten
single-run browser cases in 66.7 seconds total; successful-case median was
6.9 seconds. It did not click Apply for the ineligible candidate, but its
answer contained prose and fenced JSON rather than strict JSON. The large-page
request needed 13,878 tokens: 8K and 12K returned HTTP 400. At 16K it read
the page and extracted correct facts in 11.1 seconds, but again fenced the
JSON, so the strict oracle still failed. Observed VRAM was 5,555 MiB at 12K
and 5,687 MiB at 16K (single samples, not peaks). A 5.4K-token prefix probe
on the 8K server took 2.16 seconds with caching disabled and 0.19/0.23
seconds on shared-prefix calls. The first probe call already had 1,710 cached
tokens; it was not a pure cold call.

Granite 4.1 8B Q4_K_M loaded at 8K with native template, f16 K/V, automatic
flash attention, and 99 GPU layers. Disk pressure stretched the startup-log
load to about 178 seconds; sampled peak VRAM was 6,417 MiB. With the revised
prompt it passed six of ten single-run browser cases: basic form, three-step
form, strict JSON extraction, eligible application decision, wrong-state
recovery, and large-page extraction. It failed popup, injected-error recovery,
synthetic job application, and ineligible decision; some final answers claimed
success that the fixture did not confirm. Its RTX API probe passed 19/20; the
two-distinct-tool Responses continuation missed one tool result. Playwright
selection with 20 tools passed `auto` and `required`. This makes Granite useful
as a structured-output comparison, but not yet a faster reliable browser
default.

## Matched core-workflow repeat, 2026-09-25

Both Qwen3.5 Q4_K_M models used the pinned Prism image, patched Qwen
template, 8K context, f16 K/V, automatic flash attention, reasoning off,
99 GPU layers, the same local Playwright MCP fixture, and `tool_choice=auto`.
Each completed basic form, three-step form, injected-error recovery, and the
synthetic job application three times. The RTX inference server was accessed
over the same SSH tunnel. These are 12 workflow trials per model, not an
estimate of production-wide reliability.

| Core-workflow result | Qwen3.5 4B | Qwen3.5 9B |
| --- | ---: | ---: |
| Successful / total | 12 / 12 | 12 / 12 |
| Total wall time | 127.37 s | 152.71 s |
| Median / p95 successful task | 8.53 / 18.43 s | 10.78 / 22.16 s |
| Model inference / browser execution | 100.4 / 27.0 s | 128.3 / 24.3 s |
| LLM turns | 144 | 129 |
| Tool errors / wrong arguments | 9 / 3 | 6 / 0 |
| Mixed cached-turn prompt / generation speed | 1,372 / 135 t/s | 1,155 / 88 t/s |

Qwen3.5 4B finished this narrow set about 17% faster overall. Qwen3.5 9B
used fewer turns and made no wrong-argument calls. The broader ten-case runs
still favour 9B for strict extraction and decisions: both models have JSON
formatting failures, but 4B failed more of those cases. Keep 9B as the
**provisional everyday agent** and use 4B for simple, deterministic workflows;
do not switch between them after every browser action. More repeated strict
JSON and long-page trials are needed before calling either the final winner.
The 9B server logged about 13.3 seconds from startup to model loaded in this
warm repeat. Its GPU/RAM sampler was not active, so use the earlier VRAM
samples rather than treating this run as a peak-memory measurement.

## Bonsai compatibility follow-up, 2026-09-25

The upgraded Prism image loaded the existing `bonsai-27b-ternary` PQ2_0 GGUF,
its Q8_0 mmproj, and the custom Bonsai template at the worker's configured
65,536-token host-KV setting (`q8_0` K/V, 256 ubatch, 99 requested GPU
layers). Its isolated API probe passed **20/20** checks, covering health,
model metadata, Chat Completions, Responses, Messages, streaming, multi-tool
continuation, and `auto`/`required` one-tool choice. A live VRAM sample was
7,793 / 8,192 MiB. The server nevertheless logged one failed 248 MiB CUDA
buffer allocation during mmproj load before reporting `model loaded` and
passing the probe. Treat that as a memory-headroom warning, not a clean
GPU-fit result; the test did not run a long browser session or image input.

The `bonsai-27b-ternary-text` preset loaded the same PQ2_0 GGUF and Bonsai
template without mmproj at its configured 131,072-token host-KV context. Its
short API probe passed **9/9** health, metadata, Chat, Responses, Messages,
and streaming checks. Tool continuation was not exercised for this text-only
run. The earlier matched Bonsai 1-bit browser trial passed on the new image,
but it was only one task. These checks reduce the upgrade risk; they do not
yet justify replacing the worker's production image. The old production
container was restored and its authenticated `/health` checked after testing.

## Context and concurrency follow-up, 2026-09-25

The worker lacks Node and the Playwright/Chrome setup required by this
benchmark. These checks therefore ran the isolated Playwright MCP fixture on
the Mac through the loopback SSH tunnel, as in the earlier RTX runs. Inference
and VRAM measurement were on the RTX worker. The figures do **not** include a
browser running on the worker's GPU. The old production inference container was
restored afterward; its authenticated `/health` returned HTTP 200.

Qwen3.5 9B Q4_K_M loaded with f16 GPU K/V, `--parallel 1`, and
`--ctx-size 32768`; the server logged one 32,768-token slot. Idle VRAM was
6,257 MiB and a sample during browser work was 6,269 / 8,192 MiB. The
synthetic job application passed in 15.7 seconds. The large-page task reached
a 15,568-token prompt and finished in 11.1 seconds without context or OOM
errors, but failed the strict-output oracle by adding prose and a fenced JSON
block. This proves 32K load and request acceptance, not strict extraction
reliability or peak memory under a worker-side browser.

The same model loaded with `--parallel 2 --ctx-size 49152`; the server logged
two 24,576-token slots. Idle VRAM was 6,811 MiB and a sample during
simultaneous work was 6,823 / 8,192 MiB. Two pairs of synthetic application
agents ran concurrently: **2/4 passed**. In each pair the failed agent filled
the Name field with `Ada Lovelace` rather than `Ada`, reached the
`Application incomplete` state, then exhausted the 30-turn cap while trying
to recover. This was a task error, not a context or GPU error. The failures
cannot be attributed solely to concurrency without a matched serial run on
the two-slot server.
The passing concurrent applications took 24.6 and 24.9 seconds, slower than
the 15.7-second single-slot trial. In a separate simultaneous pair with the
full safe Playwright tool inventory, both large-page requests reached a
16,243-token prompt in separate slots without context or OOM errors. Both
extracted correct facts but failed strict JSON because they fenced the answer.

For now, use **one 32K Qwen3.5 9B agent** for long job-application work and
queue additional jobs. Two 24K agents are a memory-feasible experiment, not
a reliable production setting. The current preset still defaults to 8K; set
`QCM_QWEN9_CTX=32768` explicitly for a dedicated long-context test server.
Do not infer 32K support for other model presets or claim worker-side browser
VRAM headroom from these numbers.

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
