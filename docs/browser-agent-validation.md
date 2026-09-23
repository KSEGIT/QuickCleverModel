# Browser-agent validation status

This is the evidence available on 2026-09-23. It is **not** an RTX 3070 Ti
comparison. The target GPU host was not accessible from this workspace, so no
CUDA fit, offload, VRAM, or production default is claimed. Raw local reports
are under the ignored `run/` directory; the commands below reproduce them.

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
| Qwen3.6 35B A3B, all quants | CUDA load, Playwright and real clients | SKIPPED — target GPU unavailable | Artifact revisions, exact filenames, sizes and hashes were checked at the model registry; static Prism loader/parser inspection is not a runtime test. |
| Qwen3.5 9B, Granite 4.1 8B, Bonsai 1-bit | New full Playwright comparison | SKIPPED — target GPU unavailable | Existing availability is preserved; no result for this new benchmark is inferred. |

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
