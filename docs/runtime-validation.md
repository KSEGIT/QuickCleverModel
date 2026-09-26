# Runtime validation record

This branch uses Prism `922be44aa6ac81b46f092716351cddff1c1733a7`.
Live inference below ran on an Apple M5 Metal host with 24 GB unified memory.
Metal buffer sizes are **not RTX 3070 VRAM measurements**. The x86 CUDA Docker
image built under emulation (1,550,294,857 bytes), but CUDA inference, old/new image-size
comparison, 8K/12K/16K fit, flash attention and KV quantisation comparisons on
the target 8 GB GPU remain **SKIPPED — no SSH access to the NVIDIA host**.
Do not deploy this branch as an RTX validated runtime yet.

All test requests used a local server on `127.0.0.1` and synthetic fixture data.
Raw reports and startup logs are in the gitignored `run/` directory of the
test worktree. The probe records exact request and response bodies, usage and
timings, but never authentication headers. `PASS` below means the named test
actually ran and met its assertions; it does not imply agent reliability on a
different GPU or prompt distribution.

## Native model and API checks

| Model | GGUF load / Metal offload | Context | Chat | Responses | Messages | Auto / required tools | Multi-turn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bonsai 27B 1-bit | PASS / PASS, 65/65 layers; projector loaded | 8K | PASS | PASS | PASS at 512-token cap | PASS / PASS | PASS |
| Bonsai 27B ternary, official PQ2_0 | PASS / PASS, 65/65 layers; projector loaded | 8K | PASS | PASS | PASS | PASS / PASS | PASS |
| Bonsai 27B ternary text, official PQ2_0 | PASS / PASS, 65/65 layers; no projector | 8K | PASS | PASS | PASS | PASS / PASS | PASS |
| Qwen3.5 4B Q4_K_M | PASS / PASS, 33/33 layers | 8K | PASS | FAIL, multiple system messages | PASS | PASS / PASS | PASS, except Responses with multiple system messages |
| Qwen3.5 9B Q4_K_M | PASS / PASS, 33/33 layers | 8K | PASS | FAIL, multiple system messages | PASS | PASS / PASS in tested cases | PASS in tested cases, except Responses with multiple system messages |
| Granite 4.1 8B Q4_K_M | PASS / PASS, 41/41 layers | 8K | PASS | PASS | PASS | PASS / PASS | PASS |

All three Bonsai aliases and Granite each passed 20/20 contract checks. These include
health, model discovery, props, plain and streamed responses for all three APIs,
auto and required tool selection, JSON arguments, tool result continuation, and
a two-tool conversation. A stricter 1-bit run with a 256-token cap failed its
streamed Messages text check because the model spent the allowance on thinking;
the final 512-token run passed 20/20. Qwen 4B passed 16/20 contract
checks. Its four failures
were Responses requests that placed a developer message after `instructions`;
the embedded Qwen template raised `System message must be at the beginning`.
A separate Responses conversation with one initial system message and two
different tools passed. This isolates the compatibility issue to message order,
not to all Responses tool parsing. Qwen 9B reproduced the same four Responses
failures and passed the other 12 contract checks.

Qwen 4B passed 44 of 48 agent tool scenarios: 1, 5 and 24 declared tools;
`auto` and `required`; short and 22,403-character system prompts; ordinary and
streaming calls; and tool-result continuation with a second call. Four cases
asked for prose followed by a tool call and produced a pure tool call instead.
Those four do **not** prove a parser failure; they did not exercise mixed
output. No Qwen 4B case answered in prose instead of calling the requested
tool. The full report has 56 PASS and 8 FAIL because it also contains the four
Responses multi-system failures. This is one deterministic trial per case, not
a statistical reliability estimate.

Qwen 9B passed 20 distinct agent tool scenarios tested across an interrupted
matrix run and a focused completion run. The focused run selected case indices
15, 20, 31, 32, 39, 40 and 47 from the documented 48-case matrix; all seven
passed. Together the runs cover 1, 5 and 24 tools; both tool choices; short and
long prompts; streaming; mixed text plus parsed tool calls; and tool-result
continuation. This is **not** a complete 48-case Qwen 9B result. The 24-tool
long case used about 7,200 prompt tokens in the 8K slot, leaving little room
for longer agent turns. Its measured Metal generation speed was about 6.5–7.5
tokens/s during that case, with concurrent Docker build work on this host.

The model startup logs show Qwen 4B architecture `qwen35`, Qwen 9B architecture
`qwen35` and Granite architecture `granite`. The Qwen tokenizer is recognised as
the model's BPE vocabulary, and each uses its embedded Jinja template. Metal
mapped model buffers at 8K were about 2,603.50 MiB for Qwen 4B, 5,406.91 MiB
for Qwen 9B and 5,096.77 MiB for Granite; CPU mapped weights and KV/compute
buffers use additional memory. These figures cannot establish 8 GB CUDA fit.
The official PQ2_0 ternary reported `PQ2_0 - 2.13 bpw (group 128)`, mapped
6,822.71 MiB of model weights on Metal, offloaded all 65 layers and loaded the
original projector. Its custom Bonsai template passed the multi-system
Responses test in both plain and streaming calls. The separate legacy Q2_0
file still fails to load with current Prism; the successful test used only the
pinned official PQ2_0 replacement.
The text-only alias used the same official GGUF and template without an
`mmproj`; it also passed 20/20 checks. Its startup accepted the existing
`ngram-simple`, `cache-reuse`, `cache-ram`, flash-attention and reasoning-budget
flags under current Prism. It mapped the same 6,822.71 MiB Metal model buffer
and loaded without a projector.
The 1-bit model mapped 3,616.77 MiB of weights on Metal, offloaded all 65
layers, loaded its projector and used the same custom template.

## Real client fixtures

Each client got a new temporary Git repository with a random token in
`input.txt`. Success required a real file-tool read and an exact `output.txt`
edit. The macOS harness confines file access to its temporary tree and system
runtime paths. It strips personal service credentials from the child environment.

| Model | Codex CLI | Claude Code | Claude + Playwright MCP | OpenCode |
| --- | --- | --- | --- | --- |
| Qwen3.5 4B | FAIL — Qwen Responses template HTTP 500 in initial run; confined rerun cannot initialise Codex app-server | FAIL — prompt too long at 8K | FAIL — prompt too long at 8K | One initial PASS; confined rerun FAIL, output added a line-number prefix |
| Qwen3.5 9B | FAIL — confined Codex app-server cannot initialise | FAIL — prompt too long at 8K | FAIL — prompt too long at 8K | FAIL — read worked, but file edit copied OpenCode's `1: ` line-number prefix |
| Granite 4.1 8B | FAIL — confined Codex app-server cannot initialise | FAIL — prompt too long at 8K | FAIL — prompt too long at 8K | PASS — exact file read and edit |

Codex `--ignore-user-config`, `--ephemeral` and its local provider still try to
initialise an app-server resource denied by the file sandbox. Relaxing that
restriction would invalidate the isolated-client claim. The observed Claude
error is its actual `Prompt is too long` result before token usage is returned;
the separate `/v1/messages` API passed for Granite. The independent pinned
Playwright MCP browser smoke passed navigation, deliberate element discovery,
form fill, select, click, result extraction, missing-element recovery and omitted
automatic snapshots/images. It does not count as Claude plus browser PASS.

## Performance and configuration

An initial matched native `llama-bench` run on Bonsai 1-bit used 512 prompt
tokens, 128 generated tokens, three repetitions, all Metal layers, flash
attention on, q4_0 K/V and ubatch 256. Old Prism averaged 173.26 prompt
tokens/s and 16.68 generated tokens/s. New Prism averaged 134.61 and 16.36
respectively. This suggests about 22% slower prompt processing on this host,
but the new run overlapped other preparation; repeat both runs without competing
loads before calling it a regression. It is **not a CUDA before/after result**.

Median server-reported generation speed across non-streaming Chat requests in
each native report was 9.79 tokens/s for Bonsai 1-bit, 6.21 for ternary vision,
6.34 for ternary text, 33.42 for Qwen 4B, 7.72 for Qwen 9B and 21.04 for
Granite. These are mixed short/long fixture requests on the Metal host, not
matched throughput benchmarks and not RTX predictions. Qwen 9B's focused
report included several long prompts. Target-GPU throughput remains unmeasured.

The configured remote server answered a read-only `/props` request with
`build_info: b9598-4dd165625`, confirming the old Prism revision still runs
there. Its router listed the legacy ternary text alias as resident. No model
switch or benchmark request was sent to that service, so there is no matched
old/new RTX throughput comparison yet.

The new agent presets start at 8192 tokens, f16 K/V, flash attention `auto`,
one slot and ubatch 256. Qwen starts with reasoning off; Granite uses its
native default. The launcher exposes per-family context, flash attention,
offload and KV controls. No 12K/16K, q8_0 K/V, flash attention off/on or
reasoning on/bounded comparison has yet passed on the target GPU. Keep these
controls as test settings until correctness and memory measurements exist.

A focused **native Metal** Qwen 4B comparison used one declared tool, a short
prompt, and both `auto` and `required` choice. Each case included a tool result
and a second tool call. All six off/on/bounded-128 cases passed. The two-case
wall time was 7.54 seconds with reasoning off, 15.68 seconds on, and 15.88
seconds with a 128-token budget. Each first call generated 26 tokens with
reasoning off and 74 with on or bounded-128. The on and bounded-128 responses
contained the same reasoning text, so 128 did not constrain this short case.
A separate `auto` case with a 32-token budget passed in 6.58 seconds; its first
call generated 59 tokens and cut the reasoning text mid-sentence. This proves
the per-request budget is applied, but does not prove it is useful for longer
agent tasks. One deterministic trial per setting cannot establish reliability.
These results support keeping Qwen reasoning off as the initial repetitive-tool
preset, pending task-level tests on the RTX target.

## Current release decision

Keep the existing Bonsai preset as the default local agent until the official
PQ2_0 aliases, client workflows and RTX 3070 CUDA matrix pass. The new model
presets are available for explicit testing; native API results do not justify
making one the universal default. Qwen native Responses has a multi-system
template failure relevant to Codex. Claude Code's full prompt does not fit the
tested 8K setup. Codex MCP namespaces, multimodal Qwen, target VRAM and model
switch latency have not been verified.
