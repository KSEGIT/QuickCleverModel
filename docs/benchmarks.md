# Benchmark findings, all in one place

This page brings together the browser-agent benchmark findings recorded so
far. It is a summary. For full detail, raw report paths, and the caveats
behind each number, read the source pages:

- [docs/browser-agent-validation.md](browser-agent-validation.md) — RTX and
  Metal browser-agent test log, in date order.
- [docs/runtime-validation.md](runtime-validation.md) — native API and model
  checks (Metal host).
- [docs/playwright-agent-benchmark.md](playwright-agent-benchmark.md) — what
  the benchmark tool tests and how to run it.
- [docs/browser-agent-models.md](browser-agent-models.md) — candidate models,
  their intended roles, and the checks a model must pass before promotion.

None of the numbers below are invented. Each one comes from a run recorded in
those pages. Most are single runs or small repeat counts, not a measured
success rate. Treat every result as evidence, not as a final ranking, unless
the text says otherwise.

## Hardware

| Item | Detail |
| --- | --- |
| RTX worker | `firesand-worker` — RTX 3070 Ti, 8,192 MiB VRAM, NVIDIA driver 595.84, about 60 GiB system RAM, runs the stack in Docker |
| Test image | Pinned Prism `922be44` (full revision `922be44aa6ac81b46f092716351cddff1c1733a7`), CUDA 12.4.1, built for Ampere (`sm_86`) |
| Live (production) image | Prism `4dd165625`, as recorded at the time of testing |
| Comparison host | Apple M5, Metal, 24 GB unified memory |

Metal figures are **not** RTX VRAM figures. Apple's unified memory and an
NVIDIA card's VRAM are different things; a number from one host never stands
in for the other.

Every RTX check paused only the live model container for the length of its
own test, used a loopback-only test port, and restored the live container
afterward. Its authenticated `/health` endpoint was checked after each
restore.

## Models tried

| Model | Role | RTX browser result so far |
| --- | --- | --- |
| `qwen3.5-9b-q4_k_m` | Primary fast Playwright agent; provisional everyday choice | Passed 12/12 in the matched core-workflow repeat. Passed 7/10, then 8/10, in the broader ten-case set; the failures were JSON formatting or a context limit, not wrong facts. |
| `qwen3.5-4b-q4_k_m` | Very fast worker, for simple deterministic workflows | Passed 12/12 in the matched core-workflow repeat, about 17% faster overall than 9B on that narrow set. Passed 5/6 selected workflows in an earlier single run; of four more cases, only wrong-state recovery passed, the rest had correct facts but failed strict JSON. |
| `qwen3.6-35b-a3b` (Q4_K_XL) | Complex agent / planner candidate | First pass, original prompt: 4/6. After a prompt fix: basic form passed in 17.2 s with zero tool errors, but extraction JSON was still fenced and an injected failure was retried before inspection, so those two tasks still failed. |
| `qwen3.6-35b-a3b` (IQ4_XS) | Same role, alternate quant | Passed basic form, three-step form, and the synthetic job application, all with zero tool errors. Too few trials yet to choose over Q4_K_XL. |
| `gemma4-e4b` | Fast full-GPU challenger | Failed its first live task. After a prompt fix, passed 5 of 10 single-run cases. Not recommended as a browser default despite fast token generation. |
| `granite-4.1-8b-q4_k_m` | Structured tool-call baseline | Passed 6 of 10 single-run cases. Useful as a structured-output comparison, not yet a faster reliable default. |

## Results

### Fixture suite: core-workflow repeats, 2026-09-25

Both Qwen3.5 models ran the same four core tasks (basic form, three-step
form, injected-error recovery, synthetic job application) three times each,
at 8K context, on the pinned test image. This is 12 trials per model, not a
production-wide reliability estimate.

| Result | Qwen3.5 4B | Qwen3.5 9B |
| --- | ---: | ---: |
| Successful / total | 12 / 12 | 12 / 12 |
| Total wall time | 127.37 s | 152.71 s |
| Median / p95 successful task | 8.53 / 18.43 s | 10.78 / 22.16 s |
| Model inference / browser execution | 100.4 / 27.0 s | 128.3 / 24.3 s |
| LLM turns | 144 | 129 |
| Tool errors / wrong arguments | 9 / 3 | 6 / 0 |
| Mixed cached-turn prompt / generation speed | 1,372 / 135 t/s | 1,155 / 88 t/s |

Qwen3.5 4B finished this set about 17% faster overall. Qwen3.5 9B used fewer
turns and made no wrong-argument calls. The broader ten-case runs still favor
9B for strict extraction and decisions; both models have JSON-formatting
failures, but 4B failed more of those cases.

### 32K context, one agent, 2026-09-25

Qwen3.5 9B loaded with one 32,768-token slot, f16 GPU K/V cache.

| Setting | Value |
| --- | --- |
| Idle VRAM | 6,257 MiB |
| VRAM during browser work | 6,269 / 8,192 MiB |
| Synthetic job application | PASS, 15.7 s |
| Large-page task | Prompt reached 15,568 tokens; finished in 11.1 s with no context or out-of-memory error; failed the strict-output check because the answer had prose plus a fenced JSON block |

This proves the model can load at 32K and accept a large request. It does not
prove strict-extraction reliability or peak memory once a browser also runs
on the worker's GPU.

### Two 24K slots, concurrency, 2026-09-25

The same model loaded with two 24,576-token slots (49,152 total), so two
agents could run at once.

| Setting | Value |
| --- | --- |
| Idle VRAM | 6,811 MiB |
| VRAM during simultaneous work | 6,823 / 8,192 MiB |
| Concurrent runs | Two pairs of synthetic-application agents (4 agents total); 2 / 4 passed |
| Passing run times | 24.6 s and 24.9 s — slower than the 15.7 s single-slot run above |
| Failure cause | Both failures filled the Name field with "Ada Lovelace" instead of "Ada," reached "Application incomplete," then hit the 30-turn cap trying to recover |

### Serial control, same two-slot server, 2026-09-25

A matched control ran the same server, image, and flags, but with one agent
at a time instead of two.

| Setting | Value |
| --- | --- |
| Result | 3 / 4 passed |
| Passing run times | 17.7 s, 14.9 s, 15.0 s (18 turns each, peak prompt about 4,870 tokens) |
| One failure | Same Name-field error as above; hit the 30-turn cap after 25.3 s |

The Name-field error happens with one agent alone, not only with two agents
at once, so it is a task error, not a concurrency error. Four trials each way
are too few to say whether concurrency makes it more common (2/4 versus 1/4
failures). Running two agents at once did add about 9 seconds to each
passing run.

### 64K context, 2026-09-26

Qwen3.5 9B loaded with one 65,536-token slot, f16 GPU K/V cache, all layers
on the GPU.

| Setting | Value |
| --- | --- |
| Idle VRAM | 7,313 / 8,192 MiB |
| Prompt size tested | 59,906 tokens |
| Result | PASS, 26.1 s — the model quoted the requested line correctly |
| Peak VRAM | 7,323 MiB (about 870 MiB free) |

870 MiB free is enough for inference alone. A browser running on the same GPU
at the same time is not measured here.

### Live web: DuckDuckGo image search, 2026-09-26

A one-off harness gave the model real headless Chrome through Playwright MCP
0.0.82: search DuckDuckGo for oranges, open Images, pick a picture, find its
full-size URL, and call a `save_image` tool. That tool only accepts real
JPEG/PNG/GIF/WebP bytes, and navigation stayed on duckduckgo.com.

| Run | Context | Result | Time | Note |
| --- | --- | --- | --- | --- |
| 1 | 32K | FAIL | 55.3 s | Context overflow; DuckDuckGo showed no images |
| 2 | 32K | PASS | 43.7 s | Saved an orange photo (verywellhealth.com) |
| 3 | 64K | FAIL | 133.6 s | Found image URLs but wrote its plan as text instead of calling `save_image` |
| 4 | 64K | PASS | 69.5 s | Saved an orange photo (wallpapers.com) |
| 5 | 64K | PASS | 42.7 s | Saved the same photo as run 2 |

With the context fixes described below, 3 of 4 runs passed. The one failure
was a model error (it did not call the save tool), not a context error.

## Decide suite

A fixed-answer classification benchmark, separate from the browser-agent
suites above. It checks three ways to pick one label from a short list. It
does not drive a browser.

### What it measures

Three runners answer the same items:

| Runner | How it answers |
| --- | --- |
| `qwen-free` | The LLM, free text, told to answer with exactly one label |
| `qwen-schema` | The same LLM, but the request also carries a JSON-schema `response_format` that limits the answer to the label list |
| `gliner` | A GLiNER2.5-Decide model — no LLM call |

Every reply counts as correct, wrong, or invalid (not one of the given
labels — including a reply that names more than one label). Requests use
temperature 0 and turn the LLM's own reasoning off, so repeat runs are
comparable. For each runner and task, the suite reports:

- **accuracy** — correct answers over total. The dashboard shows this as
  the suite's pass rate.
- **invalid answers** — how many replies matched no label. The dashboard
  does not chart this; read it from the run's summary JSON (the
  `bench-summary` workflow artifact, or the published `data/index.json`).
- **time per decision** — median milliseconds for one answer, pooled across
  all three tasks for that runner. The first item is a warm-up and does not
  count.
- **server time** (Qwen runners only) — median of llama-server's own time
  for one answer (`timings.prompt_ms + timings.predicted_ms`). The summary
  JSON calls it `median_server_seconds`; it is null when the server sent no
  timings.

If one request fails (an HTTP error or a time-out), that runner stops. Its
results for the run are then only that error; the other runners still run.

### Comparing times fairly

The three times do not measure the same path:

- Qwen's time per decision is wall time on the GitHub runner. It includes
  the tailnet and the SSH tunnel to the worker, both ways.
- GLiNER runs on the GitHub runner's own CPU. It has no network hop, but
  also no GPU.
- Qwen's server time is the fair, model-only number. Compare GLiNER with
  that, not with Qwen's wall time.

### The dataset

`tests/fixtures/decide/*.jsonl`: 60 synthetic items, 20 per task.

| Task | Labels | Based on |
| --- | --- | --- |
| `route` | `browser`, `code`, `search`, `chat`, `none` | Requests a browser agent could receive |
| `apply` | `apply`, `skip` | The job/candidate facts from the `conditional_application` fixture |
| `page_state` | `form`, `results`, `error`, `done` | Trimmed Playwright page-snapshot text |

Each line is one JSON object: `id`, `task`, `text`, `labels`, `answer`.

### On the dashboard

A run that includes `decide` adds three suite cards — "Decide: Qwen free
text", "Decide: Qwen + schema", and "Decide: GLiNER (CPU)" — plus a
dedicated time-per-decision chart. GLiNER's results carry the host label
"cpu (CI runner)". The GitHub Actions job runs GLiNER on the runner's own
CPU, not the RTX worker's GPU. Accuracy does not depend on the hardware it
runs on. Each decide card shows time per decision in milliseconds, the
Qwen server time, the invalid-answer count, and where the runner ran. The
"Median time over runs" chart leaves the decide runners out; they have
their own chart.

Each decision model loads from a pinned Hugging Face commit, listed in
`tests/fixtures/decide/decision_models.json`. The raw `decide.json` records
it as `decision_model_revision`. To move to a newer model
version, update the commit there.

This page gives no GLiNER numbers. Read them from the dashboard after a
run. That way, this page never goes stale.

### Run it locally

Start a model on a local llama-server (see
[docs/playwright-agent-benchmark.md](playwright-agent-benchmark.md)), then:

```sh
python3 tests/decide_bench.py \
  --base-url http://127.0.0.1:18080 \
  --model qwen3.5-9b-q4_k_m \
  --output run/decide.json
```

This runs all three runners against the 60-item dataset: `gliner` is in
the default `--runners`. Without `gliner2` installed, the `gliner` runner
records an `ImportError` (`ModuleNotFoundError: No module named 'gliner2'`)
as its result, and the two LLM runners still run. For a server-only run,
name the two LLM runners:

```sh
python3 tests/decide_bench.py \
  --base-url http://127.0.0.1:18080 \
  --model qwen3.5-9b-q4_k_m \
  --runners qwen-free qwen-schema \
  --output run/decide.json
```

The `gliner` runner needs its own install, in a virtual environment:

```sh
python3 -m venv .venv-decide
source .venv-decide/bin/activate
pip install "gliner2[local]==2.1.0"
```

Pick the decision model with `--decision-model` (default
`fastino/GLiNER2.5-Decide`). It loads the pinned commit from
`decision_models.json`; `--decision-model-revision` picks another one. The GitHub Actions job installs the same
package, but with CPU-only PyTorch first, to keep a large CUDA wheel off
the runner; a local run can use whatever PyTorch is already on your
machine.

See [docs/benchmark-action.md](benchmark-action.md) for the `decide` and
`decision_model` inputs on the GitHub Actions job.

## Known limits

| Limit | What happens | Status |
| --- | --- | --- |
| Name-field error | The model fills the Name field with "Ada Lovelace" instead of "Ada." The form reports "Application incomplete," and the agent exhausts its 30-turn cap trying to recover. | Not fixed. Happens alone and alongside another agent, at about the same rate. Not a concurrency bug. |
| Fenced JSON | The model gets the facts and the decision right, but wraps its JSON answer in a Markdown code fence. The strict-JSON check does not accept a fence, so the task fails even though the answer is correct. | Not fixed. Seen across Qwen3.5 9B, Qwen3.5 4B, and Qwen3.6 35B A3B. Needs a prompt change or a parser that strips fences. |
| DuckDuckGo headless block | Headless Chrome's default user agent gets "No images found" from DuckDuckGo, even though a normal browser gets results. | Fixed. Set a normal Chrome user agent with `--user-agent`. |
| Snapshot size | The DuckDuckGo Images page snapshot is about 175,000 characters, mostly ad-tracking links. A full run with all of that text overflows a 32K context after about 22 turns. | Fixed for this task. Shorten links over 300 characters and keep only the newest large page view in history. Runs then peak at 14.6K–24.3K prompt tokens. |
| Decide p95 time | The decide summary gives only median times, no p95. | Known debt. `decide.json` keeps every latency (`latencies_ms`, `server_ms`), so a p95 can come later from the raw report. |
| Kernel/driver trap | After a worker reboot to a new kernel, the matching NVIDIA driver module was missing, so the live model container could not start. | Fixed. Install the matching `linux-modules-nvidia` package and load the module. Check `nvidia-smi` after any kernel update, before a benchmark run. |

## Recommendations

- Keep `qwen3.5-9b-q4_k_m` as the **provisional** everyday browser agent. Use
  `qwen3.5-4b-q4_k_m` for short, simple, deterministic workflows. Do not
  switch models between browser actions in the same task.
- For long job-application work, use one 32K Qwen3.5 9B agent and queue
  additional jobs rather than run two agents at once. Two 24K agents are a
  memory-feasible experiment, not a reliable production setting yet.
- Do not carry the 32K result over to other model presets, and do not treat
  it as proof of GPU headroom once a browser also runs on the worker.
- Do not promote Gemma 4 E4B as a fast default. Its API probes pass, but its
  browser task pass rate is too low.
- Granite 4.1 8B is a good structured-output comparison, but not yet a faster
  reliable browser default.
- Run more repeated strict-JSON and long-page trials before naming a final
  winner between Qwen3.5 9B and 4B, and before promoting Qwen3.6 35B A3B or
  choosing between its Q4_K_XL and IQ4_XS quants.

See [docs/browser-agent-validation.md](browser-agent-validation.md) for the
full log, including the earlier Apple Metal checks and every raw-report path.
