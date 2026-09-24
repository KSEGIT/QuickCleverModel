"""Offline contracts for the opt-in live browser-agent benchmark."""
import importlib.util
import io
import json
import pathlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = pathlib.Path(__file__).with_name("playwright_agent_bench.py")
SPEC = importlib.util.spec_from_file_location("playwright_agent_bench", SCRIPT)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class BenchmarkContractTests(unittest.TestCase):
    def test_agent_prompt_explains_playwright_refs_and_json(self):
        prompt = bench.BROWSER_SYSTEM_PROMPT
        self.assertIn('target "e6"', prompt)
        self.assertIn('not "ref=e6"', prompt)
        self.assertIn('not "[ref=e6]"', prompt)
        self.assertIn('no Markdown fence', prompt)
        self.assertIn('Do not repeat an action that returned an error', prompt)

    def test_eight_tasks_plus_synthetic_job_application(self):
        self.assertEqual(set(bench.TASKS), {
            "basic_form", "multi_page", "extraction", "conditional_application",
            "popup", "wrong_state", "tool_failure", "large_page", "job_application"})
        fixture = pathlib.Path(__file__).with_name("fixtures") / "agent-benchmark.html"
        source = fixture.read_text()
        for task in bench.TASKS:
            self.assertIn(f'"{task}"', source)
        self.assertIn('type="file"', source)

    def test_oracle_requires_browser_state_and_strict_json(self):
        self.assertTrue(bench.task_passed("basic_form", {"complete": True,
            "name": "Ada", "surname": "Lovelace", "country": "UK"}, "Done"))
        self.assertFalse(bench.task_passed("basic_form", {"complete": False}, "Done"))
        answer = '{"title":"Browser Engineer","company":"Example Labs","salary":"GBP 85000-100000","location":"London","remote":true,"technologies":["Python","Playwright","TypeScript"]}'
        self.assertTrue(bench.task_passed("extraction", {}, answer, page_read=True))
        self.assertFalse(bench.task_passed("extraction", {}, answer, page_read=False))
        self.assertFalse(bench.task_passed("extraction", {}, "```json\n" + answer + "\n```", page_read=True))
        self.assertIsNone(bench.strict_json('{"apply":true,"apply":false}'))
        self.assertIsNone(bench.strict_json('{"value":NaN}'))
        self.assertFalse(bench.read_evidence("extraction", "No results found"))
        self.assertFalse(bench.read_evidence("conditional_application", "No results found"))
        self.assertTrue(bench.read_evidence("extraction", "Browser Engineer at Example Labs; salary GBP 85000-100000; Python and Playwright"))
        self.assertTrue(bench.read_evidence("conditional_application", "UK work authorisation and Python experience required"))

    def test_conditional_application_has_both_outcomes(self):
        self.assertTrue(bench.task_passed("conditional_application", {"applied": True},
                                          '{"apply":true,"reason":"UK work authorisation and Python experience confirmed"}', "eligible", page_read=True))
        self.assertTrue(bench.task_passed("conditional_application", {"applied": False},
                                          '{"apply":false,"reason":"No UK work authorisation"}', "ineligible", page_read=True))
        self.assertFalse(bench.task_passed("conditional_application", {"applied": True},
                                           '{"apply":false,"reason":"No UK work authorisation"}', "ineligible", page_read=True))
        self.assertFalse(bench.task_passed("conditional_application", {"applied": True},
                                           '{"apply":true,"reason":"No work authorisation"}', "eligible", page_read=True))
        self.assertFalse(bench.task_passed("conditional_application", {"applied": False},
                                           '{"apply":false,"reason":"No UK work authorisation"}', "ineligible", page_read=False))

    def test_recovery_requires_observed_error_and_success(self):
        self.assertFalse(bench.task_passed("tool_failure", {"complete": True}, "Done", injected=False,
                                           inspected_after_failure=True))
        self.assertFalse(bench.task_passed("tool_failure", {"complete": True}, "Done", injected=True,
                                           inspected_after_failure=False))
        self.assertTrue(bench.task_passed("tool_failure", {"complete": True}, "Done", injected=True,
                                          inspected_after_failure=True))

    def test_tool_rejection_counts_are_distinct(self):
        self.assertEqual(bench.validate_call("browser_click", '{"target":"e1"}',
                                             {"browser_click"}), ("valid", {"target": "e1"}))
        self.assertEqual(bench.validate_call("browser_click", "{oops", {"browser_click"})[0], "malformed")
        self.assertEqual(bench.validate_call("browser_fake", "{}", {"browser_click"})[0], "hallucinated")

    def test_fixture_confinement(self):
        url = "http://127.0.0.1:1234/agent-benchmark.html?task=basic_form"
        cv = "/tmp/synthetic-cv.txt"
        self.assertTrue(bench.safe_browser_call("browser_navigate", {"url": url}, url, cv))
        self.assertFalse(bench.safe_browser_call("browser_navigate", {"url": "https://example.com"}, url, cv))
        self.assertFalse(bench.safe_browser_call("browser_file_upload", {"paths": ["/etc/passwd"]}, url, cv))
        self.assertTrue(bench.safe_browser_call("browser_file_upload", {"paths": [cv]}, url, cv))
        self.assertFalse(bench.safe_browser_call("browser_evaluate", {"function": "() => 1"}, url, cv))

    def test_prefix_cache_probe_has_stable_long_prefix_and_control(self):
        first, second = bench.cache_probe_messages(1), bench.cache_probe_messages(2)
        self.assertEqual(first[0], second[0])
        self.assertGreater(len(first[0]["content"]), 10000)
        self.assertNotEqual(first[1], second[1])
        self.assertEqual(bench.CACHE_PROBE_PLAN, (("first_call", True),
            ("warm_prefix_2", True), ("warm_prefix_3", True), ("no_cache_control", False)))

    def test_job_oracle_checks_every_field_even_if_page_reports_complete(self):
        state = {"complete": True, "step": 4, "name": "Ada", "surname": "Lovelace",
                 "country": "UK", "authorised": "Yes", "remote": "yes",
                 "cv": "synthetic-cv.txt", "interest": bench.ALLOWED_INTEREST}
        self.assertTrue(bench.task_passed("job_application", state, "Done"))
        for field in ("name", "surname", "country", "authorised", "remote", "cv", "interest"):
            broken = dict(state, **{field: "wrong"})
            self.assertFalse(bench.task_passed("job_application", broken, "Done"), field)
        self.assertFalse(bench.task_passed("job_application", dict(state,
            interest=bench.ALLOWED_INTEREST + " I have ten years of experience."), "Done"))

    def test_schema_validation_and_unique_call_ids(self):
        schema = {"browser_click": {"type": "object", "required": ["target"],
            "properties": {"target": {"type": "string"}}}}
        self.assertEqual(bench.validate_call("browser_click", '{}', set(schema), schema)[0], "wrong_arguments")
        self.assertEqual(bench.validate_call("browser_click", '{"target":3}', set(schema), schema)[0], "wrong_arguments")
        self.assertEqual(bench.validate_call("browser_click", '{"target":"e1"}', set(schema), schema)[0], "valid")
        self.assertFalse(bench.valid_call_ids([{"id": "x"}, {"id": "x"}]))
        self.assertFalse(bench.valid_call_ids([{"id": ""}]))
        self.assertTrue(bench.valid_call_ids([{"id": "x"}, {"id": "y"}]))

    def test_finish_reason_must_match_turn(self):
        self.assertTrue(bench.valid_turn_finish("stop", []))
        self.assertTrue(bench.valid_turn_finish("tool_calls", [{"id": "x"}]))
        self.assertFalse(bench.valid_turn_finish("length", []))
        self.assertFalse(bench.valid_turn_finish("stop", [{"id": "x"}]))
        self.assertFalse(bench.valid_turn_finish("tool_calls", []))

    def test_inventory_subsets_and_real_mcp_shortage(self):
        names = ["browser_navigate", "browser_snapshot", "browser_find",
                 "browser_click", "browser_fill_form", "browser_select_option"]
        self.assertEqual(len(bench.select_inventory(names, "1")[0]), 1)
        self.assertEqual(len(bench.select_inventory(names, "5")[0]), 5)
        self.assertEqual(bench.select_inventory(names, "20")[1], "SKIPPED")

    def test_summary_uses_successful_completion_time(self):
        summary = bench.summarize([{"status": "PASS", "seconds": 10},
                                   {"status": "FAIL", "seconds": 1},
                                   {"status": "PASS", "seconds": 30}])
        self.assertEqual(summary["success_pct"], 66.7)
        self.assertEqual(summary["median_success_seconds"], 20)
        self.assertEqual(summary["p95_success_seconds"], 29)

    def test_pinned_prism_stream_usage_and_timings(self):
        args = SimpleNamespace(base_url="http://127.0.0.1:1", model="fixture",
            key_env="QCM_UNUSED_KEY", max_tokens=32, cache_prompt=True,
            reasoning="default", request_timeout=2)
        agent = bench.Agent(args, None, [])
        frames = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "OK"}}]},
            {"choices": [{"index": 0, "finish_reason": "stop", "delta": {}}]},
            {"choices": [], "usage": {"prompt_tokens": 17, "completion_tokens": 2},
             "timings": {"cache_n": 12, "prompt_n": 5, "prompt_ms": 10,
                         "predicted_n": 2, "predicted_ms": 20}},
        ]
        payload = b"".join(b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames)
        payload += b"data: [DONE]\n\n"
        def open_request(request, timeout):
            body = json.loads(request.data)
            self.assertEqual(body["stream_options"], {"include_usage": True})
            self.assertTrue(body["cache_prompt"])
            return io.BytesIO(payload)
        with patch.object(bench.urllib.request, "urlopen", side_effect=open_request):
            reply, timings = agent.chat(bench.cache_probe_messages(1), 1000000000)
        self.assertEqual(reply["content"], "OK")
        self.assertEqual(timings["usage"]["prompt_tokens"], 17)
        self.assertEqual(timings["timings"]["cache_n"], 12)
        self.assertIsNotNone(timings["first_token_seconds"])


if __name__ == "__main__":
    unittest.main()
