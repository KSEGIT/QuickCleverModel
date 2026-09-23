#!/usr/bin/env python3
"""Opt-in, local-only Chat Completions + real Playwright MCP agent benchmark.

The benchmark never downloads weights. Start a separate llama-server first.
Run from the repository root; see docs/playwright-agent-benchmark.md.
"""
import argparse
from contextlib import contextmanager
import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import signal
import socket
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/agent-benchmark.html"
_PROBE_SPEC = importlib.util.spec_from_file_location("qcm_runtime_probe",
    Path(__file__).with_name("runtime_probe.py"))
_PROBE = importlib.util.module_from_spec(_PROBE_SPEC)
_PROBE_SPEC.loader.exec_module(_PROBE)
chat_stream, sse_events = _PROBE.chat_stream, _PROBE.sse_events
TASKS = {
    "basic_form": "Open the page. Fill Name Ada, Surname Lovelace, Country UK. Click Continue and check the result.",
    "multi_page": "Open the page. Complete all three steps: Name Ada, Country UK, accept terms, then Finish. Check completion.",
    "extraction": "Open the job page. Return only strict JSON with title, company, salary, location, remote (boolean), technologies (array).",
    "conditional_application": "Open the job page. Apply only if the candidate has UK work authorisation and Python experience. Return only strict JSON with apply (boolean) and reason (string). Do not click Apply if ineligible.",
    "popup": "Open the page. Dismiss the cookie/newsletter popup, then fill Name Ada, Surname Lovelace, Country UK and click Continue.",
    "wrong_state": "Open the page. The form is not initially visible. Inspect the state, find the form, then fill Name Ada, Surname Lovelace, Country UK and click Continue.",
    "tool_failure": "Open the page. Fill Name Ada, Surname Lovelace, Country UK and click Continue. A browser action may fail once; inspect its error and recover.",
    "large_page": "Open the large job-results page. Find the Browser Engineer listing at Example Labs. Return only strict JSON with title, company, salary, location, remote (boolean), technologies (array).",
    "job_application": "Open the synthetic job application. Apply with the supplied candidate facts. Upload the supplied synthetic CV file. Answer why you are interested using only supplied facts. Submit and check the receipt.",
}
EXPECTED_JOB = {"title": "Browser Engineer", "company": "Example Labs",
                "salary": "GBP 85000-100000", "location": "London", "remote": True,
                "technologies": ["Python", "Playwright", "TypeScript"]}
ALLOWED_INTEREST = "I am interested in this browser role because I know Python and Playwright."
ALLOWED_TOOLS = {
    "browser_navigate", "browser_snapshot", "browser_find",
    "browser_fill_form", "browser_select_option", "browser_click", "browser_type",
    "browser_press_key", "browser_hover", "browser_wait_for", "browser_file_upload",
    "browser_console_messages", "browser_take_screenshot", "browser_resize",
    "browser_handle_dialog", "browser_emulate_media", "browser_close",
    "browser_network_requests", "browser_network_request", "browser_drag",
}
CORE_TOOLS = {"browser_navigate", "browser_snapshot", "browser_find", "browser_fill_form",
              "browser_select_option", "browser_click", "browser_type", "browser_file_upload"}
INVENTORY_ORDER = ("browser_navigate", "browser_snapshot", "browser_find", "browser_click",
                   "browser_fill_form", "browser_select_option", "browser_type", "browser_file_upload")
CACHE_PROBE_PLAN = (("first_call", True), ("warm_prefix_2", True),
                    ("warm_prefix_3", True), ("no_cache_control", False))


def cache_probe_messages(index):
    """Several-thousand-token stable system prefix with a changing short suffix."""
    system = ("This is a controlled prompt-prefix cache measurement. Do not call browser tools. "
              "Reply with only the requested marker.\n" + "\n".join(
                  f"Rule {i:03d}: preserve the synthetic fixture, use only local facts, "
                  "keep browser actions safe, and do not invent missing candidate history."
                  for i in range(120)))
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"Reply with marker CACHE-{index}."}]


def safe_browser_call(name, arguments, url, cv_path):
    if name not in ALLOWED_TOOLS:
        return False
    if name == "browser_navigate":
        return arguments == {"url": url}
    if name == "browser_file_upload":
        return arguments == {"paths": [cv_path]}
    return True


def strict_json(value):
    def unique_pairs(pairs):
        output = {}
        for key, item in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = item
        return output

    def reject_constant(item):
        raise ValueError(f"non-standard JSON constant: {item}")

    try:
        return json.loads(value.strip(), object_pairs_hook=unique_pairs,
                          parse_constant=reject_constant)
    except (ValueError, TypeError):
        return None


def read_evidence(task, output):
    """Do not count an empty/unrelated find as observing task facts."""
    content = output.lower()
    if task in ("extraction", "large_page"):
        return all(part in content for part in (
            "browser engineer", "example labs", "85000", "playwright"))
    if task == "conditional_application":
        return "uk work authorisation" in content and "python" in content
    if task == "tool_failure":
        return "contact form" in content and "name" in content
    return bool(content.strip()) and "no results found" not in content


def task_passed(task, state, answer, variant="eligible", injected=False,
                page_read=False, inspected_after_failure=False, unsafe_retry=False,
                applied_before_read=False):
    if task in ("extraction", "large_page"):
        return page_read and strict_json(answer) == EXPECTED_JOB
    if task == "conditional_application":
        decision = strict_json(answer)
        expected = variant == "eligible"
        reason = decision.get("reason", "") if isinstance(decision, dict) else ""
        if expected:
            reason_ok = ("python" in reason.lower() and "authoris" in reason.lower()
                         and not re.search(r"\b(no|not|without|lack\w*)\b", reason.lower()))
        else:
            reason_ok = bool(re.search(r"\b(no|not|without|lack\w*)\b.{0,35}authoris", reason.lower()))
        return (isinstance(decision, dict) and decision.get("apply") is expected
                and isinstance(decision.get("reason"), str) and reason_ok and page_read
                and not applied_before_read
                and state.get("applied") is expected)
    if task == "multi_page":
        return state.get("complete") is True and state.get("step") == 3
    if task == "job_application":
        return (state.get("complete") is True and state.get("step") == 4
                and state.get("name") == "Ada" and state.get("surname") == "Lovelace"
                and state.get("country") == "UK" and state.get("authorised") == "Yes"
                and state.get("remote") == "yes" and state.get("cv") == "synthetic-cv.txt"
                and state.get("interest") == ALLOWED_INTEREST)
    if task == "popup":
        return state.get("complete") is True and state.get("popupDismissed") is True
    if task == "tool_failure":
        return (state.get("complete") is True and injected and inspected_after_failure
                and not unsafe_retry)
    return (state.get("complete") is True and state.get("name") == "Ada"
            and state.get("surname") == "Lovelace" and state.get("country") == "UK")


def schema_error(value, schema):
    """Validate the ordinary JSON-schema subset used by MCP tool arguments."""
    if not isinstance(schema, dict):
        return None
    expected = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "integer": int,
             "number": (int, float), "boolean": bool}
    if expected in types and (not isinstance(value, types[expected]) or
                              (expected in ("integer", "number") and isinstance(value, bool))):
        return f"expected {expected}"
    if "enum" in schema and value not in schema["enum"]:
        return "value not in enum"
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                return f"missing {key}"
        for key, item in value.items():
            sub = schema.get("properties", {}).get(key)
            if sub is None and schema.get("additionalProperties") is False:
                return f"unknown {key}"
            if sub is not None:
                error = schema_error(item, sub)
                if error:
                    return f"{key}: {error}"
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            error = schema_error(item, schema["items"])
            if error:
                return f"item {index}: {error}"
    return None


def validate_call(name, raw, names, schemas=None):
    if name not in names:
        return "hallucinated", None
    try:
        args = json.loads(raw)
    except (TypeError, ValueError):
        return "malformed", None
    if not isinstance(args, dict):
        return "malformed", None
    if schemas and name in schemas and schema_error(args, schemas[name]):
        return "wrong_arguments", args
    return "valid", args


def valid_call_ids(calls):
    ids = [call.get("id") for call in calls]
    return all(isinstance(value, str) and value for value in ids) and len(ids) == len(set(ids))


def valid_turn_finish(reason, calls):
    return reason == ("tool_calls" if calls else "stop")


def select_inventory(names, requested):
    available = set(names) & ALLOWED_TOOLS
    if requested == "core":
        return (sorted(available & CORE_TOOLS), "PASS" if CORE_TOOLS <= available else "SKIPPED")
    if requested == "full":
        return sorted(available), "PASS"
    count = int(requested)
    ordered = [name for name in INVENTORY_ORDER if name in available]
    ordered += sorted(available - set(ordered))
    return ordered[:count], "PASS" if len(available) >= count and "browser_navigate" in available else "SKIPPED"


def summarize(results):
    good = sorted(float(r["seconds"]) for r in results if r["status"] == "PASS")
    n = len(results)
    position = (len(good) - 1) * .95 if good else 0
    lower = int(position)
    p95 = good[lower] + (good[min(lower + 1, len(good) - 1)] - good[lower]) * (position - lower) if good else None
    return {"tasks": n, "successful": len(good), "failed": n - len(good),
            "success_pct": round(100 * len(good) / n, 1) if n else 0,
            "total_wall_seconds": round(sum(float(r.get("seconds", 0)) for r in results), 3),
            "median_success_seconds": statistics.median(good) if good else None,
            "p95_success_seconds": p95,
            "tool_errors": sum(r.get("tool_errors", 0) for r in results),
            "mcp_tool_errors": sum(r.get("mcp_tool_errors", 0) for r in results),
            "wrong_arguments": sum(r.get("wrong_arguments", 0) for r in results),
            "malformed_tool_calls": sum(r.get("malformed_tool_calls", 0) for r in results),
            "hallucinated_tools": sum(r.get("hallucinated_tools", 0) for r in results),
            "hallucinated_browser_actions": sum(r.get("hallucinated_browser_actions", 0)
                                                for r in results)}


@contextmanager
def hard_deadline(seconds):
    """Bound all socket reads/tool calls, not only pauses between LLM turns."""
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        raise RuntimeError("hard task timeout requires a Unix main thread")
    old_handler = signal.getsignal(signal.SIGALRM)
    def expired(*_):
        raise TimeoutError("hard task wall-clock limit reached")
    signal.signal(signal.SIGALRM, expired)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
        signal.signal(signal.SIGALRM, old_handler)


def memory_sample():
    sample = {}
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                 "--format=csv,noheader,nounits"], capture_output=True,
                                text=True, timeout=3, check=True)
        sample["vram_mib"] = max(int(line.strip()) for line in result.stdout.splitlines())
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    try:
        fields = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            fields[key] = int(value.strip().split()[0])
        sample["system_ram_used_mib"] = (fields["MemTotal"] - fields["MemAvailable"]) / 1024
    except (OSError, KeyError, ValueError):
        pass
    return sample


class MemorySampler:
    def __init__(self):
        self.stop = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            self.samples.append(memory_sample())
            self.stop.wait(.5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=5)

    def peak(self, key):
        values = [sample[key] for sample in self.samples if key in sample]
        return max(values) if values else None


def mcp_text(result):
    if any(item.get("type") == "image" for item in result.get("content", [])):
        raise AssertionError("MCP returned an image despite --image-responses omit")
    return "\n".join(item.get("text", "") for item in result.get("content", []))


def browser_state(mcp):
    result = mcp.request("tools/call", {"name": "browser_evaluate",
        "arguments": {"function": "() => window.__qcmResult()"}})
    if result.get("isError"):
        return {"oracle_error": mcp_text(result)}
    output = mcp_text(result)
    brace = output.find("{")
    if brace < 0:
        return {"oracle_error": output}
    try:
        return json.JSONDecoder().raw_decode(output[brace:])[0]
    except ValueError:
        return {"oracle_error": output}


class Agent:
    def __init__(self, args, mcp, declarations):
        self.args = args
        self.mcp = mcp
        self.declarations = declarations
        self.names = {tool["function"]["name"] for tool in declarations}
        self.schemas = {tool["function"]["name"]: tool["function"]["parameters"]
                        for tool in declarations}
        self.base = args.base_url.rstrip("/").removesuffix("/v1")
        self.key = os.environ.get(args.key_env, "")

    def chat(self, messages, deadline, required=False, choice="auto", cache_prompt=None, max_tokens=None):
        body = {"model": self.args.model, "messages": messages, "tools": self.declarations,
                "tool_choice": "required" if required else choice, "temperature": 0, "seed": 42,
                "max_tokens": max_tokens or self.args.max_tokens, "stream": True,
                "stream_options": {"include_usage": True},
                "cache_prompt": self.args.cache_prompt if cache_prompt is None else cache_prompt}
        if self.args.reasoning != "default":
            body["chat_template_kwargs"] = {"enable_thinking": self.args.reasoning == "on"}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        request = urllib.request.Request(self.base + "/v1/chat/completions",
                                         data=json.dumps(body).encode(), headers=headers)
        start = time.monotonic()
        timeout = min(self.args.request_timeout, max(1, deadline - start))
        events = []
        first_token = None
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for event in sse_events(response):
                if first_token is None and (any(choice.get("delta", {}).get(field)
                        for choice in event.get("choices", [])
                        for field in ("content", "reasoning_content", "tool_calls"))):
                    first_token = time.monotonic() - start
                events.append(event)
        usage = next((event["usage"] for event in reversed(events) if event.get("usage")), {})
        timings = next((event["timings"] for event in reversed(events) if event.get("timings")), {})
        finishes = [choice["finish_reason"] for event in events for choice in event.get("choices", [])
                    if choice.get("finish_reason") is not None]
        return chat_stream(events), {"seconds": time.monotonic() - start,
                                    "first_token_seconds": first_token,
                                    "usage": usage, "timings": timings,
                                    "finish_reason": finishes[-1] if len(finishes) == 1 else None}

    def cache_probe(self):
        samples = []
        for index, (label, cache) in enumerate(CACHE_PROBE_PLAN, 1):
            deadline = time.monotonic() + self.args.task_timeout
            message, timing = self.chat(cache_probe_messages(index), deadline,
                                        choice="none", cache_prompt=cache, max_tokens=64)
            if not valid_turn_finish(timing["finish_reason"], message.get("tool_calls", [])):
                raise AssertionError("prefix-cache probe did not finish with plain text")
            samples.append({"label": label, "cache_prompt": cache,
                            "first_token_seconds": timing["first_token_seconds"],
                            "request_seconds": timing["seconds"],
                            "cache_n": timing["timings"].get("cache_n"),
                            "prompt_n": timing["timings"].get("prompt_n"),
                            "prompt_ms": timing["timings"].get("prompt_ms"),
                            "prompt_tps": timing["timings"].get("prompt_per_second"),
                            "predicted_n": timing["timings"].get("predicted_n"),
                            "usage": timing["usage"], "answer": message.get("content")})
        return samples

    def inventory_probe(self, url):
        """Check tool selection with 1/5/20/real advertised MCP tool schemas."""
        samples = []
        for choice in ("auto", "required"):
            self.mcp.request("tools/call", {"name": "browser_navigate",
                "arguments": {"url": "about:blank"}})
            messages = [{"role": "system", "content": "Use the browser_navigate tool to open the supplied local fixture URL. Do not invent a URL."},
                        {"role": "user", "content": f"Open {url} using browser_navigate now."}]
            started = time.monotonic()
            sample = {"choice": choice, "status": "FAIL"}
            try:
                message, timing = self.chat(messages, started + self.args.task_timeout, choice=choice)
                calls = message.get("tool_calls", [])
                sample.update(first_token_seconds=timing["first_token_seconds"],
                              timings=timing["timings"], usage=timing["usage"],
                              finish_reason=timing["finish_reason"])
                if not valid_turn_finish(timing["finish_reason"], calls) or not valid_call_ids(calls):
                    raise AssertionError("invalid tool-call finish or IDs")
                if len(calls) != 1:
                    raise AssertionError("expected one navigation tool call")
                call = calls[0]
                name = call["function"]["name"]
                kind, arguments = validate_call(name, call["function"].get("arguments"),
                                                self.names, self.schemas)
                if kind != "valid" or name != "browser_navigate" or not safe_browser_call(
                        name, arguments, url, self.args.cv_path):
                    raise AssertionError(f"wrong tool or arguments: {kind} {name} {arguments}")
                raw = self.mcp.request("tools/call", {"name": name, "arguments": arguments})
                if raw.get("isError") or browser_state(self.mcp).get("task") != "basic_form":
                    raise AssertionError("browser did not reach fixture")
                sample["status"] = "PASS"
            except (AssertionError, urllib.error.URLError, OSError, ValueError) as error:
                sample["reason"] = f"{type(error).__name__}: {error}"
            sample["seconds"] = time.monotonic() - started
            samples.append(sample)
        return samples

    def run_task(self, task, url, variant="eligible"):
        started = time.monotonic()
        deadline = started + self.args.task_timeout
        prompt = TASKS[task] + f" URL: {url}"
        if task == "conditional_application":
            prompt += (" Candidate: Ada has UK work authorisation and Python experience." if variant == "eligible"
                       else " Candidate: Ada has Python experience but NO UK work authorisation.")
        if task == "job_application":
            prompt += (f" Candidate: Ada Lovelace, based in UK, authorised to work in UK,"
                       f" can work remotely, knows Python and Playwright. CV: {self.args.cv_path}."
                       f" For the interest answer, enter exactly: {ALLOWED_INTEREST}"
                       " Do not claim years of experience or other unsupported facts.")
        system = ("You are a local browser agent. Use the supplied browser tools on the local fixture only. "
                  "Inspect the current page before choosing element refs. Request browser_find or browser_snapshot "
                  "only when needed; ordinary operations do not include automatic snapshots. "
                  "Handle tool errors by inspecting state and retrying safely. Do not invent candidate facts. "
                  "When done, answer briefly, or use exact JSON when requested. Do not access other sites.")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        result = {"task": task, "variant": variant, "status": "FAIL", "model": self.args.model,
                  "turns": [], "navigated": False, "page_read": False, "tool_calls": 0,
                  "tool_errors": 0, "mcp_tool_errors": 0, "wrong_arguments": 0,
                  "malformed_tool_calls": 0, "hallucinated_tools": 0,
                  "hallucinated_browser_actions": 0, "unsafe_retry": False,
                  "inspected_after_failure": False, "applied_before_read": False,
                  "browser_seconds": 0.0, "inference_seconds": 0.0,
                  "input_tokens": 0, "output_tokens": 0, "prompt_eval_tokens": 0,
                  "prompt_eval_ms": 0.0, "generation_tokens": 0, "generation_ms": 0.0}
        with MemorySampler() as sampler:
            try:
                with hard_deadline(self.args.task_timeout):
                    self._run_turns(task, messages, result, deadline, variant, url)
            except (TimeoutError, urllib.error.URLError, OSError, ValueError, AssertionError) as error:
                result["reason"] = f"{type(error).__name__}: {error}"
            result["peak_vram_mib"] = sampler.peak("vram_mib")
            result["peak_system_ram_used_mib"] = sampler.peak("system_ram_used_mib")
        result["seconds"] = time.monotonic() - started
        result["answer"] = result.get("answer", "")
        result["llm_turns"] = len(result["turns"])
        result["first_token_seconds"] = result["turns"][0]["first_token_seconds"] if result["turns"] else None
        result["prompt_tps"] = (1000 * result["prompt_eval_tokens"] / result["prompt_eval_ms"]
                                if result["prompt_eval_ms"] else None)
        result["generation_tps"] = (1000 * result["generation_tokens"] / result["generation_ms"]
                                    if result["generation_ms"] else None)
        result["unattributed_seconds"] = max(0.0, result["seconds"] - result["inference_seconds"]
                                              - result["browser_seconds"])
        result["recovery_success"] = (task == "tool_failure" and result.get("injected_failure", False)
                                      and result["status"] == "PASS")
        return result

    def _run_turns(self, task, messages, result, deadline, variant, url):
        for turn in range(self.args.max_turns):
            if time.monotonic() >= deadline:
                raise TimeoutError("task wall-clock limit reached")
            call_start = time.monotonic()
            try:
                message, timing = self.chat(messages, deadline,
                                            required=self.args.force_first_tool and turn == 0)
            finally:
                # A request interrupted by the hard deadline still consumed time.
                result["inference_seconds"] += time.monotonic() - call_start
            usage, server = timing["usage"], timing["timings"]
            result["input_tokens"] += usage.get("prompt_tokens", 0)
            result["output_tokens"] += usage.get("completion_tokens", 0)
            result["prompt_eval_tokens"] += server.get("prompt_n", 0)
            result["prompt_eval_ms"] += server.get("prompt_ms", 0)
            result["generation_tokens"] += server.get("predicted_n", 0)
            result["generation_ms"] += server.get("predicted_ms", 0)
            calls = message.get("tool_calls", [])
            turn_record = {"number": turn + 1, "llm_seconds": timing["seconds"],
                           "first_token_seconds": timing["first_token_seconds"],
                           "finish_reason": timing["finish_reason"],
                           "usage": usage, "timings": server, "content": message.get("content"),
                           "calls": []}
            result["turns"].append(turn_record)
            if not valid_turn_finish(timing["finish_reason"], calls):
                raise AssertionError(f"invalid finish_reason {timing['finish_reason']!r} for tool count {len(calls)}")
            if calls and not valid_call_ids(calls):
                result["malformed_tool_calls"] += 1
                raise AssertionError("tool-call IDs must be nonempty and unique")
            messages.append(message)
            if not calls:
                answer = message.get("content") or ""
                result["answer"] = answer
                if task in ("extraction", "large_page"):
                    state = {}
                else:
                    tool_start = time.monotonic()
                    try:
                        state = browser_state(self.mcp)
                    finally:
                        result["browser_seconds"] += time.monotonic() - tool_start
                result["browser_state"] = state
                result["status"] = "PASS" if (result["navigated"] and task_passed(
                    task, state, answer, variant, result.get("injected_failure", False),
                    page_read=result["page_read"],
                    inspected_after_failure=result["inspected_after_failure"],
                    unsafe_retry=result["unsafe_retry"],
                    applied_before_read=result["applied_before_read"])) else "FAIL"
                result["reason"] = "oracle passed" if result["status"] == "PASS" else "task oracle failed"
                break
            for call in calls:
                if result["tool_calls"] >= self.args.max_tool_calls:
                    raise TimeoutError("tool-call cap reached")
                result["tool_calls"] += 1
                name = call.get("function", {}).get("name")
                kind, arguments = validate_call(name, call.get("function", {}).get("arguments"),
                                                self.names, self.schemas)
                entry = {"name": name, "arguments": arguments, "kind": kind}
                turn_record["calls"].append(entry)
                if kind != "valid":
                    result["tool_errors"] += 1
                    field = {"malformed": "malformed_tool_calls", "hallucinated": "hallucinated_tools",
                             "wrong_arguments": "wrong_arguments"}[kind]
                    result[field] += 1
                    tool_result = f"Error: {kind} tool call"
                elif not safe_browser_call(name, arguments, url, self.args.cv_path):
                    result["tool_errors"] += 1
                    result["hallucinated_browser_actions"] += 1
                    entry["kind"] = "out_of_fixture"
                    tool_result = "Error: tool action is outside the local fixture boundary"
                else:
                    if task == "conditional_application" and name == "browser_click" and not result["page_read"]:
                        result["applied_before_read"] = True
                    if (task == "tool_failure" and result.get("injected_failure", False)
                            and name == "browser_click" and not result["inspected_after_failure"]):
                        result["unsafe_retry"] = True
                    tool_start = time.monotonic()
                    try:
                        if task == "tool_failure" and not result.get("injected_failure", False) and name == "browser_click":
                            result["injected_failure"] = True
                            entry["injected_failure"] = True
                            result["tool_errors"] += 1
                            tool_result = "Error: simulated transient browser action failure. Inspect the page and retry."
                        else:
                            raw = self.mcp.request("tools/call", {"name": name, "arguments": arguments})
                            tool_result = mcp_text(raw)
                            entry["is_error"] = bool(raw.get("isError"))
                            if raw.get("isError"):
                                result["tool_errors"] += 1
                                result["mcp_tool_errors"] += 1
                                if re.search(r"invalid arguments|validation|expected .+|required", tool_result, re.I):
                                    result["wrong_arguments"] += 1
                                elif re.search(r"unknown ref|not found|no element|selector", tool_result, re.I):
                                    result["hallucinated_browser_actions"] += 1
                            else:
                                if name == "browser_navigate":
                                    result["navigated"] = True
                                if (name in ("browser_snapshot", "browser_find")
                                        and result["navigated"] and read_evidence(task, tool_result)):
                                    result["page_read"] = True
                                    if result.get("injected_failure", False):
                                        result["inspected_after_failure"] = True
                    finally:
                        result["browser_seconds"] += time.monotonic() - tool_start
                entry["result"] = tool_result
                if name not in ("browser_snapshot", "browser_find") and "### Snapshot" in tool_result:
                    raise AssertionError("automatic accessibility snapshot leaked into tool response")
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": tool_result})
        else:
            result["reason"] = "LLM turn cap reached"


def run(args):
    from smoke_playwright import MCP
    with tempfile.TemporaryDirectory(prefix="qcm-agent-bench-") as temporary:
        root = Path(temporary)
        args.cv_path = str(root / "synthetic-cv.txt")
        Path(args.cv_path).write_text("Synthetic candidate: Ada Lovelace. UK work authorisation. Python and Playwright.\n")
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        fixture = ThreadingHTTPServer(("127.0.0.1", 0),
            partial(SimpleHTTPRequestHandler, directory=str(FIXTURE.parent)))
        threading.Thread(target=fixture.serve_forever, daemon=True).start()
        env = dict(os.environ, PW_MCP_PORT=str(port), PW_MCP_BROWSER="chrome",
                   PW_MCP_OUTPUT_DIR=str(root / "browser"), PW_MCP_SNAPSHOT="none")
        with open(root / "mcp.log", "w+") as log:
            process = subprocess.Popen(["bash", str(ROOT / "start-playwright-mcp.sh"), "--headless",
                "--port", str(port), "--browser", "chrome", "--isolated", "--snapshot-mode", "none",
                "--image-responses", "omit", "--output-dir", str(root / "browser")],
                cwd=root, env=env, stdout=log, stderr=log, start_new_session=True)
            try:
                mcp = MCP(port)
                startup = time.monotonic() + 60
                while True:
                    try:
                        mcp.request("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                    "clientInfo": {"name": "qcm-agent-bench", "version": "1"}})
                        break
                    except urllib.error.URLError:
                        if process.poll() is not None or time.monotonic() > startup:
                            log.seek(0)
                            raise RuntimeError("Playwright MCP failed to start: " + log.read())
                        time.sleep(.25)
                mcp.request("notifications/initialized", notification=True)
                inventory = mcp.request("tools/list")["tools"]
                selected, inventory_status = select_inventory(
                    [tool["name"] for tool in inventory], args.tool_inventory)
                declarations = [{"type": "function", "function": {"name": t["name"],
                    "description": t.get("description", ""), "parameters": t["inputSchema"]}}
                    for t in inventory if t["name"] in selected]
                names = {t["function"]["name"] for t in declarations}
                if args.tool_inventory in ("core", "full") and not CORE_TOOLS <= names:
                    raise RuntimeError("Pinned MCP is missing required tools: " + repr(CORE_TOOLS - names))
                agent = Agent(args, mcp, declarations)
                report = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          "model": args.model, "base_url": args.base_url,
                          "host": platform.platform(), "context_label": args.context,
                          "reasoning": args.reasoning, "cache_prompt": args.cache_prompt,
                          "tool_inventory": sorted(names), "tool_inventory_requested": args.tool_inventory,
                          "tool_inventory_status": inventory_status,
                          "real_mcp_20plus_status": "PASS" if len(set(t["name"] for t in inventory) & ALLOWED_TOOLS) >= 20 else "SKIPPED",
                          "model_load_seconds": None,
                          "model_load_note": "Measure separately from server startup log; not inferred from task time",
                          "results": []}
                if inventory_status == "SKIPPED":
                    report["reason"] = f"MCP advertised fewer than {args.tool_inventory} safe tools"
                    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
                    print("SKIPPED: " + report["reason"], flush=True)
                    return 0
                if args.cache_probe_only:
                    report["prefix_cache_probe"] = agent.cache_probe()
                    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
                    print("PASS: prefix-cache probe recorded four calls", flush=True)
                    return 0
                if args.inventory_probe_only:
                    url = f"http://127.0.0.1:{fixture.server_port}/{FIXTURE.name}?task=basic_form"
                    report["inventory_probe"] = agent.inventory_probe(url)
                    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
                    for sample in report["inventory_probe"]:
                        print(f"{sample['status']} inventory {args.tool_inventory} choice={sample['choice']}", flush=True)
                    return int(any(sample["status"] != "PASS" for sample in report["inventory_probe"]))
                cases = [(name, variant) for name in args.tasks for variant in
                         (("eligible", "ineligible") if name == "conditional_application" else ("eligible",))]
                for repetition in range(args.repetitions):
                    for task, variant in cases:
                        # No previous task may leave a useful page state behind.
                        mcp.request("tools/call", {"name": "browser_navigate",
                            "arguments": {"url": "about:blank"}})
                        url = (f"http://127.0.0.1:{fixture.server_port}/{FIXTURE.name}?task={task}"
                               f"&variant={variant}&repeat={repetition}")
                        result = agent.run_task(task, url, variant)
                        result["repetition"] = repetition
                        report["results"].append(result)
                        report["summary"] = summarize(report["results"])
                        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
                        print(f"{result['status']} {task} {variant} {result['seconds']:.1f}s: {result.get('reason')}", flush=True)
                return int(any(result["status"] != "PASS" for result in report["results"]))
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                fixture.shutdown()
                fixture.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Dedicated, already-running local llama-server URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--key-env", default="BONSAI_API_KEY")
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--task-timeout", type=float, default=300)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-tool-calls", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--context", type=int, default=8192, help="Report label only; set server context separately")
    parser.add_argument("--reasoning", choices=("default", "off", "on"), default="default")
    parser.add_argument("--tool-inventory", choices=("core", "full", "1", "5", "20"), default="core")
    parser.add_argument("--inventory-probe-only", action="store_true",
                        help="Check navigation tool selection with auto and required; use with 1, 5, 20 or full")
    parser.add_argument("--force-first-tool", action="store_true", help="Required on first turn only; rest use auto")
    parser.add_argument("--no-cache-prompt", dest="cache_prompt", action="store_false")
    parser.add_argument("--cache-probe-only", action="store_true",
                        help="Four calls: first, two warm-prefix, then no-cache control; restart server first for true cold")
    parser.set_defaults(cache_prompt=True)
    args = parser.parse_args()
    if args.repetitions < 1 or args.task_timeout <= 0 or args.max_turns < 1 or args.max_tool_calls < 1:
        parser.error("repetitions, task-timeout, max-turns and max-tool-calls must be positive")
    if args.tool_inventory in ("1", "5", "20") and not args.inventory_probe_only:
        parser.error("1/5/20 tool subsets require --inventory-probe-only; full tasks need core tools")
    args.output = str(Path(args.output).resolve())
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
