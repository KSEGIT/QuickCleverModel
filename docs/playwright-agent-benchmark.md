# Local Playwright agent benchmark

This is an opt-in, end-to-end benchmark for a **dedicated test server**. It uses
real `@playwright/mcp@0.0.82` browser tools against local synthetic pages. It does
not download weights, access employers, or change the inference server. Normal
unit tests do not start Chrome.

The eight tasks are: basic form, three-step form, structured job extraction,
conditional Apply (both eligible and ineligible cases), popup recovery, wrong
browser state, one injected browser-action failure, and a large job page. A
separate synthetic job application tests personal details, authorisation,
dropdown/radio controls, CV upload, and a fact-bound free-text answer. The
ineligible branch and the job application make ten cases per repetition.

The browser fixture is `tests/fixtures/agent-benchmark.html`. Its browser state
is checked by a harness-only oracle after the agent finishes. The model does not
receive `browser_evaluate` or `browser_run_code_unsafe`. Extraction answers must
be strict JSON **after a successful deliberate page read**; a plausible final
answer cannot substitute for completing a form. The conditional Apply case
requires a page read before clicking and a reason that matches the stated
authorisation/Python rule. The simulated failure is injected once at the first
`browser_click` in the tool-failure case; the agent must then inspect the page
before retrying. Each repetition gets a fresh page URL. The candidate and CV
contain synthetic facts only. The application interest answer is an exact safe
sentence from those facts, so invented experience cannot pass the oracle.
The tool broker rejects navigation to anything other than that task's exact
loopback fixture URL. It also rejects file uploads other than the temporary
synthetic CV. The fixture contains no outbound links or navigation actions.

## Run

Start one model on a separate local llama-server and record its command, model
checksum, context, flash-attention and offload settings. Use a single model for
an entire task; do not switch models per browser action. Then run:

```sh
python3 tests/playwright_agent_bench.py \
  --base-url http://127.0.0.1:18080 \
  --model qwen3.5-9b-q4_k_m --context 8192 --reasoning off \
  --repetitions 3 --task-timeout 300 \
  --output run/validation/qwen9-playwright.json
```

Repeat the same command for each model on the RTX 3070 Ti. Run the full eight
tasks plus application first with `--tool-inventory core` (eight relevant
tools). Use `--tool-inventory full` for the larger safe browser-tool set. For
tool-selection comparisons, run `--inventory-probe-only --tool-inventory 1`,
then `5`, then `20`, and finally `full`; each checks both `auto` and `required`
with an actual navigation. A 1- or 5-tool subset cannot complete the full form,
so these modes are probe-only. The JSON report records the exact advertised
inventory. If fewer than 20 safe tools are advertised, the 20-tool probe says
`SKIPPED` instead of silently claiming a 20+ pass. The separate synthetic API
matrix in `runtime_probe.py` tests 24 tool declarations. The browser benchmark does
not expose page JavaScript execution, network route changes, or file-system
tools to the model. Use `--force-first-tool` only for a separate required-choice
comparison; the rest of each task uses `tool_choice=auto`. The default is auto
for every turn.

The benchmark uses the production MCP launcher with `--snapshot-mode none` and
`--image-responses omit`. `browser_snapshot` and `browser_find` are available
for deliberate inspection. It fails if an ordinary tool response contains an
automatic `### Snapshot`, or if any tool returns an inline image. The CV is a
temporary text file in the harness-owned directory. Chrome, Node/npm and the
pinned MCP package must be installed; the first `npx` call may need network.

Each task has a hard 300-second default wall-clock limit, 30 LLM turns and 60
browser calls. Increase these only with a recorded reason. Reports are saved
after every case, so a partially completed run retains its evidence.

## Interpret the report

Rank models by successful **whole-task** time and success rate, not generation
rate alone. The report records total wall time, model inference time, browser
execution time, turns, input/output tokens, first-token latency, server-reported
prompt/generation rates, malformed calls, wrong arguments, unknown tools,
MCP errors, browser-action errors, and recovery. A timed-out inference request
or tool call still contributes its elapsed time. Peak VRAM is sampled from the
whole NVIDIA GPU every 0.5 seconds;
peak system RAM is whole-host used memory from `/proc/meminfo`. These values
include other processes. They are `null` where the host cannot supply them.
Model load time is deliberately `null`: measure it separately from server logs,
without including a download. Report context as a server setting; the
`--context` value here is only a label.

`cache_prompt` is enabled by default so repeated turns can reuse the prefix.
For a no-cache control, repeat the same task sequence with `--no-cache-prompt`.
For a controlled prefix probe, restart the dedicated server and immediately run:

```sh
python3 tests/playwright_agent_bench.py \
  --base-url http://127.0.0.1:18080 --model qwen3.5-9b-q4_k_m \
  --cache-probe-only --output run/validation/qwen9-prefix-cache.json
```

It sends the same several-thousand-token system prompt and browser-tool schemas
four times, changing only the short user suffix. The first three calls enable
`cache_prompt`; the fourth disables it as a control. It records `cache_n`,
`prompt_n`, `prompt_ms`, prompt tokens/sec, first-token time and usage for every
call. The first result is a **true cold** first call only if the server was
restarted and had no previous prompt. Otherwise label it first-call/unknown
cache state. A reused model slot or previous browser task can bias comparison.
Server `timings` are also included per task turn so prompt and generation rates
can be recomputed. The summary median and p95 use only completed tasks;
failures still count against success rate and total wall time.

Run 8192 first, then 12288 and 16384 only when the model fits. Compare Qwen
thinking off/on with the same tasks and settings. Use at least three
repetitions and keep raw JSON reports. The same suite applies to Qwen3.6 35B
A3B, Qwen3.5 9B/4B, Granite 4.1 8B, Gemma 4 E4B and optional Bonsai 1-bit.
Real Codex, Claude Code and OpenCode tests remain separate in
`tests/runtime_clients.py`; this direct browser loop is not a pass for those
clients. On a host without Chrome or the target GPU, mark browser/CUDA rows
`SKIPPED` with the reason instead of inferring a result.

The tool schemas are obtained from the running MCP server's `tools/list`, not
hard-coded. The pinned MCP option and tool behavior were checked against the
[Playwright MCP v0.0.82 documentation](https://github.com/microsoft/playwright-mcp/blob/v0.0.82/README.md)
and the same version's package used by `start-playwright-mcp.sh`.
