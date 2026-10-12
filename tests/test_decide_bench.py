"""decide_bench.py: dataset contract and the pure scoring/parsing parts."""
import importlib.util
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


@unittest.skipUnless(importlib.util.find_spec("gliner2"), "gliner2 not installed")
class GlinerImportTest(unittest.TestCase):
    def test_autoextractor_is_importable(self):
        from gliner2 import AutoExtractor  # noqa: F401


if __name__ == "__main__":
    unittest.main()
