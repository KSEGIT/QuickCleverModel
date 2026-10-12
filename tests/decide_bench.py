#!/usr/bin/env python3
"""Fixed-answer decision benchmark: free-text Qwen, schema-constrained Qwen,
and GLiNER2.5-Decide on the same labelled items. See docs/benchmarks.md."""
import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOSTS = {"qwen-free": "rtx", "qwen-schema": "rtx", "gliner": "cpu (CI runner)"}
SYSTEM = ("You classify one input. Answer with exactly one label from this list "
          "and nothing else: {labels}.")


def load_dataset(dir_path):
    items = []
    for path in sorted(Path(dir_path).glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                items.append(json.loads(line))
    return items


def parse_free_label(text, labels):
    found = {label for label in labels
             if re.search(rf"(?<![a-z]){re.escape(label)}(?![a-z])", (text or "").lower())}
    return found.pop() if len(found) == 1 else None


def build_prompt(item):
    return [{"role": "system", "content": SYSTEM.format(labels=", ".join(item["labels"]))},
            {"role": "user", "content": item["text"]}]


def _base_body(model, messages, max_tokens):
    return {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}


def free_body(model, messages, labels):
    return _base_body(model, messages, 16)


def schema_body(model, messages, labels):
    body = _base_body(model, messages, 32)
    body["response_format"] = {"type": "json_schema", "json_schema": {
        "name": "decision", "schema": {
            "type": "object", "properties": {"answer": {"type": "string", "enum": list(labels)}},
            "required": ["answer"], "additionalProperties": False}}}
    return body


def score(items, predictions, latencies_ms):
    stats = {}
    for item, pred, ms in zip(items, predictions, latencies_ms):
        s = stats.setdefault(item["task"], {"correct": 0, "total": 0, "invalid": 0, "latencies_ms": []})
        s["total"] += 1
        s["latencies_ms"].append(ms)
        if pred is None:
            s["invalid"] += 1
        elif pred == item["answer"]:
            s["correct"] += 1
    return stats


def run_runner(name, items, predict):
    """predict(item) -> label or None. Any exception aborts this runner only."""
    predictions, latencies = [], []
    try:
        if items:
            predict(items[0])  # warm-up, excluded (model load / first-call cost)
        for item in items:
            start = time.perf_counter()
            predictions.append(predict(item))
            latencies.append(round((time.perf_counter() - start) * 1000, 1))
    except Exception as error:  # recorded, not swallowed: it reaches the dashboard
        return {"host": HOSTS[name], "error": f"{type(error).__name__}: {error}", "tasks": {}}
    return {"host": HOSTS[name], "error": None, "tasks": score(items, predictions, latencies)}


def _post(base_url, body, key, timeout):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
                                     json.dumps(body).encode(), headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)["choices"][0]["message"].get("content") or ""


def qwen_predictor(kind, args, key):
    def predict(item):
        messages = build_prompt(item)
        if kind == "qwen-free":
            return parse_free_label(_post(args.base_url, free_body(args.model, messages, item["labels"]),
                                          key, args.request_timeout), item["labels"])
        text = _post(args.base_url, schema_body(args.model, messages, item["labels"]), key, args.request_timeout)
        try:
            answer = json.loads(text).get("answer")
        except (ValueError, AttributeError):
            return None
        return answer if answer in item["labels"] else None
    return predict


def gliner_predictor(model_id):
    from gliner2 import AutoExtractor
    model = AutoExtractor.from_pretrained(model_id)

    def predict(item):
        result = model.classify_text(item["text"], {item["task"]: item["labels"]})
        value = result.get(item["task"]) if isinstance(result, dict) else None
        if isinstance(value, dict):  # some versions return {"label": .., "confidence": ..}
            value = value.get("label")
        if isinstance(value, list):  # some versions return a ranked list of (label, score) or dicts
            value = value[0] if value else None
            if isinstance(value, (list, tuple)):
                value = value[0] if value else None
            elif isinstance(value, dict):
                value = value.get("label")
        return value if value in item["labels"] else None
    return predict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--decision-model", default="fastino/GLiNER2.5-Decide")
    parser.add_argument("--runners", nargs="+", default=["qwen-free", "qwen-schema", "gliner"],
                        choices=("qwen-free", "qwen-schema", "gliner"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--key-env", default="BONSAI_API_KEY")
    parser.add_argument("--dataset-dir", default=str(ROOT / "tests" / "fixtures" / "decide"))
    parser.add_argument("--request-timeout", type=float, default=60)
    args = parser.parse_args(argv)
    items = load_dataset(args.dataset_dir)
    key = os.environ.get(args.key_env, "")
    report = {"schema": 1, "model": args.model, "decision_model": args.decision_model, "runners": {}}
    for name in args.runners:
        if name == "gliner":
            if args.decision_model == "none":
                continue
            try:
                predict = gliner_predictor(args.decision_model)
            except Exception as error:
                report["runners"][name] = {"host": HOSTS[name], "tasks": {},
                                           "error": f"{type(error).__name__}: {error}"}
                continue
        else:
            predict = qwen_predictor(name, args, key)
        report["runners"][name] = run_runner(name, items, predict)
        print(f"{name}: {report['runners'][name]['error'] or 'ok'}", flush=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    ran = any(r["tasks"] for r in report["runners"].values())
    return 0 if ran else 1


if __name__ == "__main__":
    sys.exit(main())
