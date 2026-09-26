#!/usr/bin/env python3
"""Long-context needle probe: fill most of the context, ask for one line back.

Builds a prompt of numbered filler lines sized to ~85% of --target-tokens,
asks the model to quote one line by number, and checks the exact line comes
back. Posts a single non-streaming request to /v1/chat/completions with
temperature 0 and thinking off. A 400 (for example: context too small for the
prompt) is reported as FAIL with the server's reason, not raised.

Run from the repository root: python3 tests/long_context_probe.py ...
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

# Measured, not guessed: on the real Qwen3.5-9B server,
#   " ".join(f"Line {i}: the orange crate number {i} holds navel oranges "
#            "from Valencia." for i in range(2700))
# plus a one-line question came to 59,906 prompt tokens (the model correctly
# quoted the requested line back). 59906 / 2700 = 22.187... tokens/line.
# TOKENS_PER_LINE must stay pinned to this measurement (see
# test_long_context_probe.py's line-count pins for ctx=32768/65536) — do not
# "round" it to 22 again, that was the bug that made every probe prompt come
# out 41% over budget and get a 400 back from the server.
TOKENS_PER_LINE = 22.2
TARGET_FRACTION = 0.85


def make_line(i):
    """One filler line, ~22.2 measured tokens: unpadded, so its own token
    count does not drift as the index grows past 6 digits."""
    return f"Line {i}: the orange crate number {i} holds navel oranges from Valencia."


def build_prompt(target_tokens):
    """Numbered filler lines filling ~85% of target_tokens (space-joined, the
    same layout as the measurement above), then a question asking to quote
    the middle line. Returns (prompt, line) where `line` is the exact text
    the model should quote."""
    line_count = max(2, int(TARGET_FRACTION * target_tokens / TOKENS_PER_LINE))
    lines = [make_line(i) for i in range(line_count)]
    quote_i = line_count // 2
    line = lines[quote_i]
    body = " ".join(lines)
    question = f" Quote line {quote_i} exactly, and only that line, with no extra words."
    return body + question, line


def check_answer(answer, line):
    """The model's answer must contain the exact target line's text."""
    return bool(line) and line.strip() in (answer or "")


def request_body(model, prompt):
    return {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 128, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}


def call_server(base_url, model, prompt, key_env, timeout):
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(key_env, "")
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(
        base_url.rstrip("/").removesuffix("/v1") + "/v1/chat/completions",
        data=json.dumps(request_body(model, prompt)).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def run(args):
    started = time.monotonic()
    prompt, line = build_prompt(args.target_tokens)
    report = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "model": args.model, "target_tokens": args.target_tokens,
              "status": "FAIL", "prompt_tokens": None, "answer": None, "reason": None}
    try:
        payload = call_server(args.base_url, args.model, prompt, args.key_env, args.request_timeout)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        error.close()
        report["reason"] = f"HTTP {error.code}: {detail}"
    except (urllib.error.URLError, OSError, ValueError) as error:
        report["reason"] = f"{type(error).__name__}: {error}"
    else:
        usage = payload.get("usage", {}) or {}
        report["prompt_tokens"] = usage.get("prompt_tokens")
        answer = payload["choices"][0]["message"].get("content") or ""
        report["answer"] = answer
        if check_answer(answer, line):
            report["status"] = "PASS"
        else:
            report["reason"] = "the target line was not found in the answer"
    report["seconds"] = round(time.monotonic() - started, 2)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['status']} target_tokens={args.target_tokens} "
          f"prompt_tokens={report['prompt_tokens']} {report['seconds']:.1f}s: {report['reason']}",
          flush=True)
    return 0 if report["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--key-env", default="BONSAI_API_KEY")
    parser.add_argument("--request-timeout", type=float, default=120)
    args = parser.parse_args()
    if args.target_tokens < 2 * TOKENS_PER_LINE:
        parser.error("target-tokens is too small to build a probe prompt")
    args.output = str(Path(args.output).resolve())
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
