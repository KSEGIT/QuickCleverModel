#!/usr/bin/env python3
"""Opt-in live llama-server contracts, tool reliability and timing capture (stdlib)."""
import argparse
import copy
import datetime
import json
import os
import pathlib
import platform
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sse_events(lines):
    """Parse SSE data frames, preserving the terminal marker for validation."""
    data = []
    for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                value = "\n".join(data)
                yield {"_done": True} if value == "[DONE]" else json.loads(value)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    require(not data, "SSE ended in an incomplete frame")


def chat_stream(events):
    message = {"content": "", "tool_calls": []}
    calls = {}
    finished = False
    for event in events:
        require("error" not in event, f"stream error: {event}")
        for choice in event.get("choices", []):
            finished |= choice.get("finish_reason") is not None
            delta = choice.get("delta", {})
            if "role" in delta:
                require(delta["role"] == "assistant", "chat stream role changed")
                message["role"] = delta["role"]
            message["content"] += delta.get("content") or ""
            for fragment in delta.get("tool_calls", []):
                call = calls.setdefault(fragment["index"], {
                    "id": "", "function": {"name": "", "arguments": ""}})
                if "type" in fragment:
                    require(fragment["type"] == "function", "chat stream tool type changed")
                    call["type"] = fragment["type"]
                if fragment.get("id"):
                    call["id"] = fragment["id"]
                for key in ("name", "arguments"):
                    call["function"][key] += fragment.get("function", {}).get(key) or ""
    require(finished, "chat stream missing finish_reason")
    require(message.get("role") == "assistant", "chat stream missing assistant role")
    require(all(call.get("type") == "function" for call in calls.values()),
            "chat stream missing function tool type")
    message["tool_calls"] = [calls[index] for index in sorted(calls)]
    return message


def messages_text(message):
    require(message.get("stop_reason") == "end_turn",
            f"Messages text did not complete: {message.get('stop_reason')}")
    text = "".join(part.get("text", "") for part in message.get("content", [])
                   if part.get("type") == "text")
    require(bool(text.strip()), "Messages has no usable text")
    return text


def tool_arguments(message):
    calls = message.get("tool_calls", [])
    require(bool(calls), "model returned no parsed tool_calls")
    require(len(calls) == 1, f"expected one lookup call, got {len(calls)}")
    call = calls[0]
    require(bool(call.get("id")), "tool call missing id")
    require(call.get("type") == "function", "tool call type changed")
    require(call["function"]["name"] == "lookup_code", "model selected wrong tool")
    args = json.loads(call["function"]["arguments"])
    require(args == {"key": "fixture"}, f"incorrect JSON tool arguments: {args!r}")
    return args


def functions(count):
    schema = {"type": "object", "properties": {"key": {"type": "string"}},
              "required": ["key"], "additionalProperties": False}
    return [{"type": "function", "function": {
        "name": "lookup_code" if i == 0 else f"distractor_{i}",
        "description": "Read the current fixture code." if i == 0 else
                       f"Unrelated inventory operation number {i}; never reads fixture codes.",
        "parameters": schema}} for i in range(count)]


def tool_cases():
    return [{"count": count, "choice": choice, "long": long,
             "stream": stream, "text_first": text_first}
            for count in (1, 5, 24) for choice in ("auto", "required")
            for long in (False, True) for stream in (False, True)
            for text_first in (False, True)]


def matrix_subset(indices):
    """Select stable zero-based cases for a focused, reproducible live run."""
    all_cases = tool_cases()
    if not indices:
        return all_cases
    try:
        selected = [int(part) for part in indices.split(",")]
    except ValueError as error:
        raise ValueError("matrix indices must be comma-separated integers") from error
    if len(selected) != len(set(selected)) or any(i < 0 or i >= len(all_cases) for i in selected):
        raise ValueError("matrix indices must be unique and between 0 and 47")
    return [all_cases[i] for i in selected]


def gpu_memory():
    """Whole-GPU memory on the probe host; remote hosts must sample separately."""
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                                 "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=5, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


class Probe:
    def __init__(self, args):
        self.args = args
        self.base = args.base_url.rstrip("/").removesuffix("/v1")
        self.key = os.environ.get(args.key_env, "")
        self.results = []
        self.requests = []
        self.report = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       "base_url": self.base, "model": args.model, "label": args.label,
                       "runtime": args.runtime, "context_label": args.context,
                       "probe_host": platform.platform(), "probe_arch": platform.machine(),
                       "reasoning": args.reasoning, "results": self.results,
                       "requests": self.requests}

    def request(self, path, body=None):
        body = self.request_body(path, body)
        record = {"path": path, "request": copy.deepcopy(body), "gpu_memory_before_mib": gpu_memory()}
        self.requests.append(record)
        headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        if self.key:
            headers.update({"Authorization": f"Bearer {self.key}", "x-api-key": self.key})
        request = urllib.request.Request(self.base + path, headers=headers,
                                         data=None if body is None else json.dumps(body).encode())
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.args.timeout) as response:
                record["http_status"] = response.status
                if body and body.get("stream"):
                    require("text/event-stream" in response.headers.get("Content-Type", ""),
                            "stream response is not text/event-stream")
                    result = []
                    for event in sse_events(response):
                        if "first_event_seconds" not in record:
                            record["first_event_seconds"] = time.monotonic() - started
                        # First content/reasoning/tool-argument delta, not merely headers.
                        if "first_token_seconds" not in record and (
                            any(c.get("delta", {}).get(k) for c in event.get("choices", [])
                                for k in ("content", "reasoning_content", "tool_calls")) or
                            event.get("type") in ("response.output_text.delta",
                                "response.reasoning_text.delta", "response.reasoning_summary_text.delta",
                                "response.function_call_arguments.delta", "content_block_delta")):
                            record["first_token_seconds"] = time.monotonic() - started
                        result.append(event)
                else:
                    result = json.load(response)
                record["response"] = copy.deepcopy(result)
                # Preserve server-reported token rates separately from wall time.
                samples = result if isinstance(result, list) else [result]
                for sample in samples:
                    if sample.get("timings"):
                        record["timings"] = sample["timings"]
                    if sample.get("usage"):
                        record.setdefault("usage", {}).update(sample["usage"])
                    nested = sample.get("response") or sample.get("message")
                    if isinstance(nested, dict) and nested.get("usage"):
                        record.setdefault("usage", {}).update(nested["usage"])
                return result
        except urllib.error.HTTPError as error:
            record["http_status"] = error.code
            record["error_body"] = error.read().decode("utf-8", errors="replace")
            raise AssertionError(f"{path}: HTTP {error.code}: {record['error_body'][:1000]}") from error
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            record["gpu_memory_after_mib"] = gpu_memory()

    def save(self):
        pathlib.Path(self.args.output).write_text(json.dumps(self.report, indent=2) + "\n")

    def check(self, name, test, category="contract"):
        start = time.monotonic()
        try:
            detail = test()
            result = {"name": name, "category": category, "status": "PASS", "detail": detail}
        except Exception as error:
            result = {"name": name, "category": category, "status": "FAIL",
                      "detail": f"{type(error).__name__}: {error}"}
        result["seconds"] = time.monotonic() - start
        self.results.append(result)
        self.save()
        print(f"{result['status']} {name}: {result['detail'] or ''}", flush=True)

    def generation(self):
        body = {"model": self.args.model, "temperature": 0, "seed": 42,
                "max_tokens": self.args.max_tokens, "cache_prompt": False}
        return self.request_body("/v1/chat/completions", body)

    def request_body(self, path, body):
        body = copy.deepcopy(body)
        if not body or path not in ("/v1/chat/completions", "/v1/responses", "/v1/messages"):
            return body
        if self.args.reasoning != "default":
            body["chat_template_kwargs"] = {"enable_thinking": self.args.reasoning != "off"}
        if self.args.reasoning == "bounded":
            if path == "/v1/messages":
                body["thinking"] = {"type": "enabled", "budget_tokens": self.args.reasoning_budget}
            else:
                body["reasoning_budget_tokens"] = self.args.reasoning_budget
        return body

    def chat(self, messages, **extra):
        return self.request("/v1/chat/completions", {**self.generation(),
                            "messages": messages, **extra})

    def metadata(self, endpoint):
        data = self.request(endpoint)
        require(isinstance(data, dict), f"{endpoint}: expected JSON object")
        if endpoint == "/health":
            require(data.get("status") == "ok", f"health changed: {data}")
        elif endpoint == "/v1/models":
            require(data.get("object") == "list", "models.object must be list")
            require(any(m.get("id") == self.args.model for m in data.get("data", [])),
                    f"model {self.args.model!r} not in discovery response")
        else:
            # Router /props requires the selected model in a query parameter.
            require(isinstance(data.get("default_generation_settings"), dict),
                    "props.default_generation_settings missing")
        return {key: data[key] for key in ("status", "object", "build_info", "model_alias", "model_ftype")
                if key in data}

    def text_chat(self, stream=False):
        data = self.chat([{"role": "system", "content": "Be concise."},
                          {"role": "user", "content": "Say hello."}], stream=stream)
        if stream:
            require(any(e.get("_done") for e in data), "chat SSE missing [DONE]")
            message = chat_stream(data)
        else:
            require(data.get("object") == "chat.completion", "chat object changed")
            require(bool(data.get("id")), "chat id missing")
            require(isinstance(data.get("usage"), dict), "chat usage missing")
            message = data["choices"][0]["message"]
        require(message.get("role") == "assistant", "chat role changed")
        require(bool(message.get("content")), "no normal text generated")

    def responses(self, stream=False, tools=False):
        body = {"model": self.args.model, "instructions": "Be concise. Use tools when requested.",
                "input": [{"role": "developer", "content": "Follow both instruction messages."},
                          {"role": "user", "content": "Use lookup_code with key fixture." if tools else "Say hello."}],
                "max_output_tokens": self.args.max_tokens, "stream": stream}
        if tools:
            body.update(tools=[{"type": "function", **functions(1)[0]["function"]}], tool_choice="required")
        data = self.request("/v1/responses", body)
        if stream:
            require(not any(e.get("type") in ("error", "response.failed", "response.incomplete")
                            or "error" in e for e in data), "Responses stream failed")
            complete = [e for e in data if e.get("type") == "response.completed"]
            require(len(complete) == 1, "Responses stream missing response.completed")
            data = complete[0]["response"]
        require(data.get("object") == "response", "Responses object changed")
        require(data.get("status") == "completed", f"Responses did not complete: {data.get('status')}")
        require(bool(data.get("id")), "Responses id missing")
        require(isinstance(data.get("usage"), dict), "Responses usage missing")
        require(isinstance(data.get("output"), list) and data["output"], "Responses output missing")
        if tools:
            calls = [item for item in data["output"] if item.get("type") == "function_call"]
            require(len(calls) == 1, "Responses did not parse one function_call")
            call = calls[0]
            require(bool(call.get("call_id")), "Responses call_id missing")
            tool_arguments({"tool_calls": [{"id": call["call_id"], "type": "function", "function": call}]})
            body["input"] += data["output"] + [{"type": "function_call_output",
                              "call_id": call["call_id"], "output": '{"code":"violet-731"}'},
                              {"role": "user", "content": "Report the returned code."}]
            body.update(stream=False, tool_choice="none")
            continuation = self.request("/v1/responses", body)
            require(continuation.get("status") == "completed", "Responses continuation incomplete")
            text = "".join(part.get("text", "") for item in continuation.get("output", [])
                           for part in item.get("content", []))
            require("violet-731" in text, "Responses tool result not used")
        else:
            require(any(item.get("type") == "message" and item.get("content")
                        for item in data["output"]), "Responses has no output message")

    def messages(self, stream=False, tools=False):
        body = {"model": self.args.model, "max_tokens": self.args.max_tokens,
                "messages": [{"role": "user", "content": "Use lookup_code with key fixture." if tools else "Say hello."}],
                "stream": stream}
        if tools:
            function = functions(1)[0]["function"]
            body.update(tools=[{"name": function["name"], "description": function["description"],
                                "input_schema": function["parameters"]}], tool_choice={"type": "any"})
        data = self.request("/v1/messages", body)
        if stream:
            require(not any(e.get("type") == "error" or "error" in e for e in data), "Messages stream failed")
            starts = [e for e in data if e.get("type") == "message_start"]
            require(len(starts) == 1, "Messages stream missing message_start")
            require(any(e.get("type") == "message_stop" for e in data), "Messages stream missing message_stop")
            message = starts[0]["message"]
            blocks = {}
            json_parts = {}
            for event in data:
                if event.get("type") == "content_block_start":
                    blocks[event["index"]] = event["content_block"]
                elif event.get("type") == "content_block_delta":
                    delta = event["delta"]
                    index = event["index"]
                    if delta.get("type") == "text_delta":
                        blocks[index]["text"] = blocks[index].get("text", "") + delta["text"]
                    elif delta.get("type") == "input_json_delta":
                        json_parts[index] = json_parts.get(index, "") + delta["partial_json"]
                elif event.get("type") == "message_delta":
                    message.update(event["delta"])
            for index, value in json_parts.items():
                blocks[index]["input"] = json.loads(value)
            message["content"] = [blocks[index] for index in sorted(blocks)]
            data = message
        require(data.get("type") == "message" and data.get("role") == "assistant", "Messages envelope changed")
        require(bool(data.get("id")), "Messages id missing")
        require(isinstance(data.get("usage"), dict), "Messages usage missing")
        require(isinstance(data.get("content"), list) and data["content"], "Messages content missing")
        if tools:
            calls = [part for part in data["content"] if part.get("type") == "tool_use"]
            require(len(calls) == 1, "Messages did not parse one tool_use")
            call = calls[0]
            require(call.get("name") == "lookup_code" and call.get("input") == {"key": "fixture"},
                    "Messages tool name/JSON arguments wrong")
            require(bool(call.get("id")), "Messages tool id missing")
            require(data.get("stop_reason") == "tool_use", "Messages tool stop_reason changed")
            body["messages"] += [{"role": "assistant", "content": data["content"]},
                                 {"role": "user", "content": [{"type": "tool_result",
                                  "tool_use_id": call["id"], "content": '{"code":"violet-731"}'},
                                  {"type": "text", "text": "Report the returned code."}]}]
            body.update(stream=False, tool_choice={"type": "auto"})
            result = self.request("/v1/messages", body)
            require("violet-731" in messages_text(result),
                    "Messages tool result not used")
        else:
            messages_text(data)

    def tool_case(self, case):
        system = "Use lookup_code to obtain unknown fixture codes. Never invent the answer."
        if case["long"]:
            system += "\n" + "\n".join(
                f"Instruction {i}: preserve existing files, check facts, keep answers brief, and use the relevant tool."
                for i in range(220))
        prompt = "Look up the current fixture code using lookup_code with key fixture."
        if case["text_first"]:
            prompt += " First say 'Checking now.' then call the tool in the same response."
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        tools = functions(case["count"])
        data = self.chat(messages, tools=tools, tool_choice=case["choice"], stream=case["stream"])
        if case["stream"]:
            require(any(e.get("_done") for e in data), "tool stream missing [DONE]")
            message = chat_stream(data)
        else:
            message = data["choices"][0]["message"]
        tool_arguments(message)
        if case["text_first"]:
            require(bool(message.get("content")), "text+tool scenario produced pure tool only; mixed parsing not exercised")
        messages += [message, {"role": "tool", "tool_call_id": message["tool_calls"][0]["id"],
                               "content": '{"code":"violet-731"}'}]
        reply = self.chat(messages, tools=tools, tool_choice="none")["choices"][0]["message"]
        require("violet-731" in (reply.get("content") or ""), "tool result not used in continuation")
        # A second call after a tool result catches multi-turn template/parser regressions.
        messages += [reply, {"role": "user", "content": "Look up key fixture once more to check it is unchanged."}]
        second = self.chat(messages, tools=tools, tool_choice=case["choice"])["choices"][0]["message"]
        tool_arguments(second)
        return {"text_with_tool": bool(message.get("content")), "system_characters": len(system)}

    def multi_tool(self, api):
        """Two distinct tools across turns, with independently checked JSON arguments."""
        declarations = [functions(1)[0]["function"], {
            "name": "verify_code", "description": "Check whether a returned code is valid.",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                           "required": ["code"], "additionalProperties": False}}]
        history = [{"role": "user", "content": "Use lookup_code with key fixture. Do not verify it yet."}]
        for step, expected in enumerate(("lookup_code", "verify_code")):
            if api == "chat":
                body = self.chat(history, tools=[{"type": "function", "function": d} for d in declarations],
                                 tool_choice="required")
                assistant = body["choices"][0]["message"]
                calls = assistant.get("tool_calls", [])
                require(len(calls) == 1, "chat multi-tool turn did not produce one call")
                call = calls[0]
                name, args, call_id = call["function"]["name"], json.loads(call["function"]["arguments"]), call["id"]
                history += [assistant]
            elif api == "responses":
                body = self.request("/v1/responses", {"model": self.args.model, "input": history,
                    "tools": [{"type": "function", **d} for d in declarations],
                    "tool_choice": "required", "max_output_tokens": self.args.max_tokens})
                calls = [item for item in body.get("output", []) if item.get("type") == "function_call"]
                require(len(calls) == 1, "Responses multi-tool turn did not produce one call")
                call = calls[0]
                name, args, call_id = call["name"], json.loads(call["arguments"]), call["call_id"]
                history += body["output"]
            else:
                body = self.request("/v1/messages", {"model": self.args.model, "messages": history,
                    "tools": [{"name": d["name"], "description": d["description"], "input_schema": d["parameters"]}
                              for d in declarations], "tool_choice": {"type": "any"},
                    "max_tokens": self.args.max_tokens})
                calls = [item for item in body.get("content", []) if item.get("type") == "tool_use"]
                require(len(calls) == 1, "Messages multi-tool turn did not produce one call")
                call = calls[0]
                name, args, call_id = call["name"], call["input"], call["id"]
                history += [{"role": "assistant", "content": body["content"]}]
            require(name == expected, f"multi-tool expected {expected}, got {name}")
            require(args == ({"key": "fixture"} if step == 0 else {"code": "violet-731"}),
                    f"multi-tool arguments wrong: {args}")
            require(bool(call_id), "multi-tool call id missing")
            output = '{"code":"violet-731"}' if step == 0 else '{"valid":true,"receipt":"checked-492"}'
            next_prompt = ("Now call verify_code with the returned code." if step == 0 else
                           "Report the verification receipt. No further tools are needed.")
            if api == "chat":
                history += [{"role": "tool", "tool_call_id": call_id, "content": output},
                            {"role": "user", "content": next_prompt}]
            elif api == "responses":
                history += [{"type": "function_call_output", "call_id": call_id, "output": output},
                            {"role": "user", "content": next_prompt}]
            else:
                history += [{"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": call_id, "content": output},
                    {"type": "text", "text": next_prompt}]}]
        if api == "chat":
            final = self.chat(history, tools=[{"type": "function", "function": d} for d in declarations],
                              tool_choice="none")["choices"][0]["message"].get("content") or ""
        elif api == "responses":
            final_body = self.request("/v1/responses", {"model": self.args.model, "input": history,
                "tools": [{"type": "function", **d} for d in declarations],
                "tool_choice": "none", "max_output_tokens": self.args.max_tokens})
            final = "".join(p.get("text", "") for item in final_body.get("output", []) for p in item.get("content", []))
        else:
            final_body = self.request("/v1/messages", {"model": self.args.model, "messages": history,
                "tools": [{"name": d["name"], "description": d["description"], "input_schema": d["parameters"]}
                          for d in declarations], "tool_choice": {"type": "auto"}, "max_tokens": self.args.max_tokens})
            final = messages_text(final_body)
        require("checked-492" in final, "model did not use the second tool's result")

    def run(self):
        self.check("health", lambda: self.metadata("/health"))
        if self.results[-1]["status"] != "PASS":
            self.results.append({"name": "remaining live tests", "status": "SKIPPED",
                                 "detail": "health check failed; no inference requests sent"})
            self.save()
            return 1
        self.check("models", lambda: self.metadata("/v1/models"))
        self.check("props", lambda: self.metadata("/props?" + urllib.parse.urlencode({"model": self.args.model})))
        for stream in (False, True):
            self.check(f"chat stream={stream}", lambda stream=stream: self.text_chat(stream))
            self.check(f"responses multiple instructions stream={stream}", lambda stream=stream: self.responses(stream))
            self.check(f"messages stream={stream}", lambda stream=stream: self.messages(stream))
            if self.args.tools:
                self.check(f"responses tool roundtrip stream={stream}",
                           lambda stream=stream: self.responses(stream, True), "model-tool")
                self.check(f"messages tool roundtrip stream={stream}",
                           lambda stream=stream: self.messages(stream, True), "model-tool")
        if self.args.tools:
            for api in ("chat", "responses", "messages"):
                self.check(f"{api} two distinct tools", lambda api=api: self.multi_tool(api), "model-tool")
            cases = matrix_subset(self.args.matrix_indices) if self.args.full_matrix else [
                {"count": 1, "choice": choice, "long": False, "stream": stream, "text_first": False}
                for choice in ("auto", "required") for stream in (False, True)]
            for case in cases:
                self.check("tools " + json.dumps(case, sort_keys=True),
                           lambda case=case: self.tool_case(case), "model-tool")
        return int(any(r["status"] == "FAIL" for r in self.results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Explicit test server URL; never inferred from personal config")
    parser.add_argument("--model", required=True)
    parser.add_argument("--key-env", default="BONSAI_API_KEY")
    parser.add_argument("--output", required=True, help="JSON evidence, including request/response bodies but no auth headers")
    parser.add_argument("--runtime", default="unrecorded", help="Exact image digest/Prism SHA from version.sh")
    parser.add_argument("--label", default="unlabelled", help="e.g. old-f16-fa-off or new-q8-fa-on")
    parser.add_argument("--context", type=int, default=8192, help="Report label only; configure server separately")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--tools", action="store_true")
    parser.add_argument("--full-matrix", action="store_true", help="48 cases, each with tool continuation and second call")
    parser.add_argument("--matrix-indices", default="",
                        help="With --full-matrix, run selected zero-based case indices (0-47), comma-separated")
    parser.add_argument("--reasoning", choices=("default", "off", "on", "bounded"), default="default",
                        help="All generation APIs; verify template supports enable_thinking")
    parser.add_argument("--reasoning-budget", type=int, default=128)
    args = parser.parse_args()
    if args.matrix_indices and not args.full_matrix:
        parser.error("--matrix-indices requires --full-matrix")
    try:
        matrix_subset(args.matrix_indices)
    except ValueError as error:
        parser.error(str(error))
    if args.full_matrix:
        args.tools = True
    raise SystemExit(Probe(args).run())


if __name__ == "__main__":
    main()
