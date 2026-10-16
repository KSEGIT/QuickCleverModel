"""decide_bench.py: dataset contract and the pure scoring/parsing parts."""
import contextlib
import importlib.util
import io
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("decide_bench", ROOT / "tests" / "decide_bench.py")
db = importlib.util.module_from_spec(spec)
spec.loader.exec_module(db)
LABELS = {"route": ["browser", "code", "search", "chat", "none"],
          "apply": ["apply", "skip"],
          "page_state": ["form", "results", "error", "done"]}


class DatasetTest(unittest.TestCase):
    def test_items_are_valid_and_balanced(self):
        items = db.load_dataset(ROOT / "tests" / "fixtures" / "decide")
        self.assertEqual(len(items), 60)
        self.assertEqual(len({i["id"] for i in items}), 60)
        for item in items:
            self.assertEqual(item["labels"], LABELS[item["task"]], item["id"])
            self.assertIn(item["answer"], item["labels"], item["id"])
        for task, labels in LABELS.items():
            for label in labels:
                n = sum(1 for i in items if i["task"] == task and i["answer"] == label)
                self.assertGreaterEqual(n, 3, f"{task}/{label}")


class LabelLeakageTest(unittest.TestCase):
    def test_text_never_contains_its_own_task_label(self):
        """A model could win by echoing a label straight out of the input
        instead of classifying it (e.g. an item whose answer is "error"
        containing the word "Error"). Same word-boundary rule as
        parse_free_label, checked against every label of the item's task."""
        items = db.load_dataset(ROOT / "tests" / "fixtures" / "decide")
        for item in items:
            text_lower = item["text"].lower()
            leaked = [label for label in item["labels"]
                      if re.search(rf"(?<![a-z]){re.escape(label)}(?![a-z])", text_lower)]
            with self.subTest(item=item["id"]):
                self.assertEqual(leaked, [], f"{item['id']}: text leaks label(s) {leaked}")


class ParseFreeLabelTest(unittest.TestCase):
    labels = ["browser", "code", "search", "chat", "none"]

    def test_single_label_any_case_and_punctuation(self):
        self.assertEqual(db.parse_free_label("Browser.", self.labels), "browser")
        self.assertEqual(db.parse_free_label('{"answer": "code"}', self.labels), "code")

    def test_two_different_labels_is_invalid(self):
        self.assertIsNone(db.parse_free_label("search, not browser", self.labels))

    def test_no_label_is_invalid(self):
        self.assertIsNone(db.parse_free_label("I would open a web page", self.labels))

    def test_label_inside_another_word_does_not_count(self):
        self.assertIsNone(db.parse_free_label("barcode", self.labels))


class BodiesTest(unittest.TestCase):
    def test_schema_body_limits_answer_to_labels(self):
        body = db.schema_body("m", [{"role": "user", "content": "x"}], ["apply", "skip"])
        schema = body["response_format"]["json_schema"]["schema"]
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertEqual(schema["properties"]["answer"]["enum"], ["apply", "skip"])
        self.assertEqual(schema["required"], ["answer"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})

    def test_free_body_has_no_response_format(self):
        body = db.free_body("m", [{"role": "user", "content": "x"}], ["a", "b"])
        self.assertNotIn("response_format", body)
        self.assertEqual(body["max_tokens"], 16)

    def test_prompt_lists_labels(self):
        msgs = db.build_prompt({"text": "t", "labels": ["apply", "skip"], "task": "apply"})
        self.assertIn("apply, skip", msgs[0]["content"])
        self.assertEqual(msgs[-1]["content"], "t")


class ScoreTest(unittest.TestCase):
    def test_counts_correct_invalid_and_latency(self):
        items = [{"task": "apply", "answer": "apply"}, {"task": "apply", "answer": "skip"},
                 {"task": "apply", "answer": "skip"}]
        stats = db.score(items, ["apply", None, "apply"], [10.0, 20.0, 30.0])
        self.assertEqual(stats["apply"], {"correct": 1, "total": 3, "invalid": 1,
                                          "latencies_ms": [10.0, 20.0, 30.0]})


class RunRunnerErrorTest(unittest.TestCase):
    def test_runner_error_is_recorded_not_raised(self):
        def boom(item):
            raise OSError("HTTP 400: unknown model")
        result = db.run_runner("qwen-schema", [{"task": "apply", "answer": "apply",
                                                "labels": ["apply", "skip"], "text": "t"}], boom)
        self.assertEqual(result["tasks"], {})
        self.assertIn("HTTP 400", result["error"])
        self.assertEqual(result["host"], "rtx")


class ServerTimingTest(unittest.TestCase):
    def test_server_ms_is_prompt_plus_predicted(self):
        self.assertEqual(db.server_ms_from({"timings": {"prompt_ms": 12.25, "predicted_ms": 30.5}}), 42.8)

    def test_server_ms_is_none_without_timings(self):
        self.assertIsNone(db.server_ms_from({"choices": []}))
        self.assertIsNone(db.server_ms_from({"timings": {"prompt_ms": 1.0}}))

    def test_score_keeps_server_ms_next_to_latencies(self):
        items = [{"task": "apply", "answer": "apply"}, {"task": "apply", "answer": "skip"}]
        stats = db.score(items, ["apply", "skip"], [50.0, 60.0], [20.0, 25.0])
        self.assertEqual(stats["apply"]["server_ms"], [20.0, 25.0])
        self.assertEqual(stats["apply"]["latencies_ms"], [50.0, 60.0])

    def test_score_has_no_server_ms_when_none_known(self):
        items = [{"task": "apply", "answer": "apply"}]
        self.assertNotIn("server_ms", db.score(items, ["apply"], [50.0], [None])["apply"])
        self.assertNotIn("server_ms", db.score(items, ["apply"], [50.0])["apply"])

    def test_run_runner_splits_label_and_server_ms(self):
        items = [{"task": "apply", "answer": "apply", "labels": ["apply", "skip"], "text": "t"}]
        result = db.run_runner("qwen-free", items, lambda item: ("apply", 7.5))
        self.assertEqual(result["tasks"]["apply"]["correct"], 1)
        self.assertEqual(result["tasks"]["apply"]["server_ms"], [7.5])


class QwenOverHttpTest(unittest.TestCase):
    """main() against a fake llama-server: server_ms comes from "timings"."""

    def test_qwen_runners_record_server_ms(self):
        import http.server
        import os
        import tempfile
        import threading

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                labels = body["messages"][0]["content"].split(": ", 1)[1].rstrip(".").split(", ")
                content = (json.dumps({"answer": labels[0]}) if "response_format" in body else labels[0])
                data = json.dumps({"choices": [{"message": {"content": content}}],
                                   "timings": {"prompt_ms": 3.0, "predicted_ms": 4.0}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "decide.json")
            with contextlib.redirect_stdout(io.StringIO()):
                rc = db.main(["--base-url", f"http://127.0.0.1:{server.server_port}", "--model", "m",
                              "--runners", "qwen-free", "qwen-schema", "--output", out])
            report = json.loads(Path(out).read_text())
        self.assertEqual(rc, 0)
        for runner in ("qwen-free", "qwen-schema"):
            tasks = report["runners"][runner]["tasks"]
            self.assertEqual(sum(t["total"] for t in tasks.values()), 60)
            for t in tasks.values():
                self.assertEqual(t["server_ms"], [7.0] * t["total"])
                self.assertEqual(len(t["latencies_ms"]), t["total"])


class RevisionPinTest(unittest.TestCase):
    def test_default_model_has_a_pin(self):
        self.assertRegex(db.pinned_revision("fastino/GLiNER2.5-Decide"), r"^[0-9a-f]{40}$")

    def test_unknown_model_has_no_pin(self):
        self.assertIsNone(db.pinned_revision("someone/else"))

    def test_no_revision_loads_the_repo_id(self):
        self.assertEqual(db.gliner_source("fastino/x", None), "fastino/x")

    def test_revision_loads_the_pinned_snapshot_directory(self):
        import sys
        import types
        calls = []
        fake = types.ModuleType("huggingface_hub")
        fake.snapshot_download = lambda **kw: calls.append(kw) or "/cache/snap/abc"
        saved = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = fake
        try:
            self.assertEqual(db.gliner_source("fastino/x", "abc"), "/cache/snap/abc")
        finally:
            if saved is None:
                del sys.modules["huggingface_hub"]
            else:
                sys.modules["huggingface_hub"] = saved
        self.assertEqual(calls, [{"repo_id": "fastino/x", "revision": "abc"}])

    def test_report_records_the_revision(self):
        """The report records the revision even when model loading fails."""
        import os
        import tempfile
        from unittest import mock

        with mock.patch.object(db, "gliner_predictor", side_effect=RuntimeError("model loading disabled")):
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "decide.json")
                with contextlib.redirect_stdout(io.StringIO()):
                    db.main(["--base-url", "http://127.0.0.1:9", "--model", "m", "--runners", "gliner",
                             "--decision-model", "fastino/GLiNER2.5-Decide", "--output", out,
                             "--decision-model-revision", "deadbeef"])
                report = json.loads(Path(out).read_text())

        self.assertEqual(report["decision_model_revision"], "deadbeef")

    def test_finished_runners_survive_a_later_kill(self):
        """run_suites kills the job at its cap; runners already done must stay on disk."""
        import os
        import tempfile
        from unittest import mock

        done = {"host": "rtx", "error": None,
                "tasks": {"apply": {"correct": 1, "total": 1, "invalid": 0, "latencies_ms": [5.0]}}}
        with mock.patch.object(db, "run_runner", side_effect=[done, KeyboardInterrupt()]):
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "decide.json")
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
                    db.main(["--base-url", "http://127.0.0.1:9", "--model", "m", "--output", out,
                             "--runners", "qwen-free", "qwen-schema", "--decision-model", "none"])
                report = json.loads(Path(out).read_text())
                leftovers = [n for n in os.listdir(tmp) if n != "decide.json"]

        self.assertEqual(report["runners"], {"qwen-free": done})
        self.assertEqual(leftovers, [])


@unittest.skipUnless(importlib.util.find_spec("gliner2"), "gliner2 not installed")
class GlinerImportTest(unittest.TestCase):
    def test_autoextractor_is_importable(self):
        from gliner2 import AutoExtractor  # noqa: F401


if __name__ == "__main__":
    unittest.main()
