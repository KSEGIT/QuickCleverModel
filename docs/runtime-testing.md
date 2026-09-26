# Runtime acceptance tests

Run these tests against a separate test server. Do not replace the working image
until the Bonsai load tests pass. A source-code match or an offline unit test does
not prove that a model loads or that CUDA works.

The live probes use Python's standard library. They do not download models, start
llama-server, change server settings, or load personal client configuration.
Reports contain request bodies, response bodies and client transcripts. Keep them
in `run/validation/`, which Git ignores. Authentication headers are not recorded.
Use fixture prompts only.

## Offline checks

```sh
python3 -m unittest discover -s tests -p 'test_runtime*.py' -v
```

These check stream framing, partial JSON arguments, missing terminal events,
multiple Responses instructions, tool-result IDs, different tools across turns,
client isolation, pinned MCP flags, and timeout cleanup. These tests have no model
and must not be reported as live client or inference passes.

## API contracts and tool use

Set `BONSAI_API_KEY` in the environment if the test server requires it. Pass the
server URL explicitly. A `/v1` suffix is accepted.

```sh
mkdir -p run/validation
python3 tests/runtime_probe.py \
  --base-url http://127.0.0.1:18080 \
  --model bonsai-27b-1bit \
  --runtime 922be44aa6ac81b46f092716351cddff1c1733a7 \
  --context 8192 --label new-f16-fa-off \
  --output run/validation/bonsai-1bit-contracts.json
```

The default run checks `/health`, `/v1/models`, model-specific `/props`, and plain
and streamed Chat Completions, Responses and Messages. It validates IDs, response
envelopes, usage fields and completion events. Responses requests contain both
`instructions` and a separate developer message to exercise the Bonsai Codex
template workaround. Generation wording is not fixed; an empty final message
still fails because clients need usable output.

Add `--tools` to check parsed function names and JSON arguments, both `auto` and
`required`, streaming fragments, tool-result continuation and a second call in
the conversation. Each API also gets a conversation with two different tools:
`lookup_code`, followed by `verify_code`, followed by a final answer that uses
the verification result. Nothing executes model-supplied code; the fixture
returns fixed tool results.

Run each model separately. Copy the model IDs from `/v1/models`; do not substitute
a different quantisation or model family when reporting a result.

```sh
python3 tests/runtime_probe.py \
  --base-url http://127.0.0.1:18080 --model qwen3.5-9b-q4_k_m \
  --runtime 922be44aa6ac81b46f092716351cddff1c1733a7 \
  --context 8192 --tools --reasoning off \
  --output run/validation/qwen9b-tools-off.json
```

The exact model IDs may differ from this example. Discovery failure is a failed
test, not evidence that a model is unsupported.

For Qwen agent stress tests use `--full-matrix`. It covers 48 combinations:

For a focused repeat on a slower model, add `--matrix-indices` with explicit
zero-based case numbers, for example `--matrix-indices 15,20,31,32,39,40,47`.
The report records each selected case. A focused run is not a 48-case pass.

| Variable | Values |
| --- | --- |
| Declared tools | 1, 5, 24 |
| Tool choice | `auto`, `required` |
| System prompt | short; 220 separate instructions |
| Transport | ordinary JSON; SSE |
| Assistant response | tool only; requested prose followed by a tool |

Each successful case continues with a tool result and checks a second tool call.
This is a model-reliability test; it is separate from API contract checks. The long
prompt is several thousand tokens for typical tokenizers. Use the recorded
`usage.prompt_tokens` or server token counts to report its actual length. It can
overflow small contexts when combined with tools or client instructions.

A prose+tool case fails if the model returns only a tool call. That means mixed
output was not exercised; it does not by itself prove a parser defect. Inspect
the saved raw output before classifying a failure. Report `auto` and `required`
separately. Never turn a failed `auto` run into a pass by forcing every call.

The process exits nonzero on any failure. An unavailable server records a failed
health check and skips inference. Reports are saved after every case, so completed
results survive an interrupted long run.

## Reasoning comparison

Repeat the same Qwen tool matrix with `--reasoning off`, `on`, and
`--reasoning bounded --reasoning-budget 128`. These switches apply to all three
generation APIs. `off` sends the boolean template argument
`enable_thinking: false`; `bounded` also sends `reasoning_budget_tokens` for
Chat/Responses or `thinking.budget_tokens` for Messages. The field names and
adapter pass-through were checked against the
[pinned server source](https://github.com/PrismML-Eng/llama.cpp/blob/922be44aa6ac81b46f092716351cddff1c1733a7/tools/server/server-common.cpp).

Use a server with `--reasoning-budget -1` for the enabled comparison. A per-request
budget of `-1` inherits the server default; it does not override a server set to
zero. Save the server command and startup log with each report. Check that the
model's template supports the argument. Granite and Bonsai require their own
template and reasoning checks; Qwen settings must not be copied to them by default.

Compare elapsed time, `first_token_seconds`, generated token counts, tool success
and final fixture success. A request accepting a setting is not proof that the
setting changed reasoning. Inspect `reasoning_content` and server output as well.

## Real clients

```sh
python3 tests/runtime_clients.py \
  --base-url http://127.0.0.1:18080 --model qwen3.5-9b-q4_k_m \
  --context 8192 --client all --timeout 600 \
  --output run/validation/qwen9b-clients.json
```

Repeat for Qwen 4B and Granite 8B. Each installed client gets its own temporary
Git repository and a random token in `input.txt`. It must read that file and write
the same token to `output.txt`. The token is not in the ordinary client prompt.
The harness checks the exact output and that the input was preserved. It records
CLI versions, exit codes, structured events and full transcripts.

Codex uses Responses, an explicit local provider, `workspace-write`, no personal
config or rules, and an ephemeral session. It receives fixture `AGENTS.md` plus
the task prompt. Claude uses `--bare`, an empty MCP configuration, file tools only
and local permission rules. OpenCode uses an isolated configuration with read/edit
permission, denied shell/network tools and denied external directories. On macOS
the harness wraps each client in `sandbox-exec`: only the temporary
fixture tree is writable or readable outside system runtime paths. The Codex
CLI may fail to initialise if it needs its personal state before it can use
`--ephemeral`; report that as a client failure. On hosts without this file
confinement, the harness skips real clients. It removes only its own temporary
files and stops its own process group on timeout.

These are minimal client tests. They do not prove that a client's full default
system prompt and complete tool inventory fit in 8K. Run the full matrix and
measure prompt length before widening permissions or changing context defaults.
Missing CLIs or an unavailable server produce `SKIPPED` with a reason and a
nonzero exit status. A client error or an incorrect fixture produces `FAIL`.

### Claude with Playwright

First run the independent browser transport smoke test:

```sh
python3 tests/smoke_playwright.py
```

It covers navigation, deliberate snapshots, form fill, dropdowns, clicks, result
extraction, omitted images, a failed element lookup and browser recovery. This
test does not use a model and is not a Claude+Playwright pass.

Then add `--claude-playwright` to the client command. This starts a local fixture
HTTP server and a pinned `@playwright/mcp@0.0.82` subprocess with `--isolated`,
`--headless`, `--snapshot-mode none` and `--image-responses omit`. Claude must use
browser tools to fill the form, select Repair, submit it, extract the result and
save it. The harness checks the file and verifies that browser tool calls appear
in the client transcript. It reports `claude-playwright` separately. Node/npm and
Chrome are required; the first run may need network access for the pinned package.

Codex MCP namespaces are not covered by the flat file-tool smoke. Test that
separately before claiming namespace support.

## GPU and performance matrix

Keep the old image and weights until an apples-to-apples baseline exists. For
each useful combination, run the same model, prompt, seed, layer offload, batch
sizes, parallel count and context on old and new builds. Record cold model load
time from server logs separately from warm generation. Do not include a model
download in load time.

Start with 8192 tokens, f16 K/V and flash attention off. Compare flash attention
on without changing anything else. Then compare q8_0 K/V if the model/backend
supports it. Repeat successful settings at 12288 and 16384. Only test more
aggressive cache quantisation after the conservative case passes. Save the exact
server command, image ID/digest, GPU model, driver and model file checksum.

The JSON report records server `timings` and `usage`, total request time, first
SSE event time, and first content/reasoning/tool delta time. The latter is an API
first-token measurement, not a CUDA kernel timing. The probe samples whole-GPU
memory with `nvidia-smi` before and after each request when available. These are
snapshots, not peak VRAM, and they describe the probe host. Run on the GPU server
or sample the remote GPU separately. Never label Apple unified memory as VRAM.

Use continuous GPU sampling on the test server to measure peak VRAM. Preserve
the raw samples. Record model load/offload lines, architecture and tokenizer
recognition from startup logs. API success alone does not prove GPU offload.

For a throughput baseline, the existing `./bench.sh -n 5 -m bonsai-27b-1bit`
reports medians when run on the server host. Retain the same build/configuration
labels used for the API probe. Do not average different prompt lengths together.
Compare only matched requests and record tool correctness beside speed. A fast
run with broken tools is not an acceptable agent preset.

## Report requirements

For every requested model, report load, offload, VRAM, actual context, prompt and
generation speed, Chat Completions, Responses, Messages, `auto`, `required`,
streaming and multi-turn tools. Give each client its own result. Use `PASS`,
`FAIL`, or `SKIPPED — reason`; never infer a pass from source support or another
model's result.

Record unresolved context limits, model-switch delay, parser failures, unsupported
vision input and client incompatibilities. Recommend a default agent preset only
after measurements on the target 8 GB RTX GPU. Native Mac tests can catch API and
template defects but cannot establish CUDA performance or VRAM defaults.

Client setup follows [official Codex provider documentation](https://developers.openai.com/codex/config-advanced/),
[Claude headless documentation](https://code.claude.com/docs/en/headless), and
[OpenCode permissions](https://opencode.ai/docs/permissions/). CLI help from the
installed versions was checked during implementation.
