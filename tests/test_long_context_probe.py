"""Tests for the long-context needle probe.

stdlib unittest, not pytest — see tests/test_cache_viz.py for the convention.
No network: HTTP calls are mocked at urllib.request.urlopen.
"""
import importlib.util
import io
import json
import pathlib
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "long_context_probe", ROOT / "tests" / "long_context_probe.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class BuildPromptTest(unittest.TestCase):
    def test_line_count_fills_about_85_percent_of_target(self):
        # int(0.85 * 4400 / 22.2) = 168 lines, indices 0..167.
        prompt, _ = probe.build_prompt(4400)
        self.assertIn(probe.make_line(167), prompt)
        self.assertNotIn(probe.make_line(168), prompt)

    def test_quotes_the_middle_line(self):
        prompt, line = probe.build_prompt(4400)  # 168 lines -> quote index 84
        self.assertIn("Quote line 84 exactly", prompt)
        self.assertEqual(line, probe.make_line(84))

    def test_line_text_appears_verbatim_in_the_prompt(self):
        prompt, line = probe.build_prompt(2200)
        self.assertIn(line, prompt)

    def test_small_target_still_returns_at_least_two_lines(self):
        prompt, line = probe.build_prompt(1)
        self.assertIn(line, prompt)
        self.assertTrue(line.startswith("Line "))

    def test_larger_target_yields_more_lines_and_a_higher_quote_number(self):
        _, small_line = probe.build_prompt(2200)
        _, big_line = probe.build_prompt(22000)
        self.assertNotEqual(small_line, big_line)

    def test_lines_are_space_joined_like_the_measured_baseline(self):
        # The 59,906-prompt-token measurement this heuristic is pinned to
        # used " ".join(...), not newline-joined lines. Tokenization is not
        # separator-agnostic, so this must not silently change.
        prompt, _ = probe.build_prompt(2200)
        self.assertNotIn("\n", prompt)


class TokensPerLineConstantTest(unittest.TestCase):
    """Guards the exact measurement this whole heuristic is pinned to."""

    def test_tokens_per_line_matches_the_recorded_measurement(self):
        # 59,906 prompt tokens / 2700 lines on the real Qwen3.5-9B server.
        self.assertAlmostEqual(probe.TOKENS_PER_LINE, 59906 / 2700, places=1)

    def test_target_fraction_is_85_percent(self):
        self.assertEqual(probe.TARGET_FRACTION, 0.85)


class LineCountPinTest(unittest.TestCase):
    """Pins the line count for common context sizes so the heuristic cannot
    silently drift back to overshooting the server's context (as
    TOKENS_PER_LINE = 22 did before this fix)."""

    def test_32768_context_plans_1254_lines_quoting_line_627(self):
        prompt, line = probe.build_prompt(32768)
        self.assertIn(probe.make_line(1253), prompt)
        self.assertNotIn(probe.make_line(1254), prompt)
        self.assertEqual(line, probe.make_line(627))
        self.assertIn("Quote line 627 exactly", prompt)

    def test_65536_context_plans_2509_lines_quoting_line_1254(self):
        prompt, line = probe.build_prompt(65536)
        self.assertIn(probe.make_line(2508), prompt)
        self.assertNotIn(probe.make_line(2509), prompt)
        self.assertEqual(line, probe.make_line(1254))
        self.assertIn("Quote line 1254 exactly", prompt)


class CheckAnswerTest(unittest.TestCase):
    def test_exact_quote_passes(self):
        line = probe.make_line(42)
        self.assertTrue(probe.check_answer(line, line))
        self.assertTrue(probe.check_answer(f"Sure, here it is: {line}", line))

    def test_paraphrase_or_wrong_line_fails(self):
        line = probe.make_line(42)
        other = probe.make_line(43)
        self.assertFalse(probe.check_answer(other, line))
        self.assertFalse(probe.check_answer("I don't know.", line))
        self.assertFalse(probe.check_answer("", line))
        self.assertFalse(probe.check_answer(None, line))

    def test_empty_expected_line_never_passes(self):
        self.assertFalse(probe.check_answer("anything", ""))


class RequestBodyTest(unittest.TestCase):
    def test_temperature_zero_and_thinking_off(self):
        body = probe.request_body("m", "prompt")
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertFalse(body["stream"])


class RunTest(unittest.TestCase):
    def _output_path(self):
        return pathlib.Path(tempfile.mkdtemp()) / "long_context.json"

    def test_pass_when_server_quotes_the_line(self):
        prompt, line = probe.build_prompt(4400)
        payload = {"choices": [{"message": {"content": f"Here: {line}"}}],
                   "usage": {"prompt_tokens": 4123}}

        def fake_urlopen(request, timeout):
            return io.BytesIO(json.dumps(payload).encode())

        output = self._output_path()
        args = type("Args", (), {"base_url": "http://127.0.0.1:1", "model": "m",
                                  "target_tokens": 4400, "output": str(output),
                                  "key_env": "QCM_UNUSED_KEY", "request_timeout": 5})()
        with patch.object(probe.urllib.request, "urlopen", side_effect=fake_urlopen):
            code = probe.run(args)
        self.assertEqual(code, 0)
        report = json.loads(output.read_text())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["prompt_tokens"], 4123)
        self.assertEqual(report["target_tokens"], 4400)
        self.assertIsNone(report["reason"])

    def test_fail_when_answer_misses_the_line(self):
        payload = {"choices": [{"message": {"content": "I have no idea."}}], "usage": {}}

        def fake_urlopen(request, timeout):
            return io.BytesIO(json.dumps(payload).encode())

        output = self._output_path()
        args = type("Args", (), {"base_url": "http://127.0.0.1:1", "model": "m",
                                  "target_tokens": 4400, "output": str(output),
                                  "key_env": "QCM_UNUSED_KEY", "request_timeout": 5})()
        with patch.object(probe.urllib.request, "urlopen", side_effect=fake_urlopen):
            code = probe.run(args)
        self.assertEqual(code, 1)
        report = json.loads(output.read_text())
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("not found", report["reason"])

    def test_http_400_is_reported_as_fail_with_reason_not_raised(self):
        def fake_urlopen(request, timeout):
            raise urllib.error.HTTPError(
                "http://127.0.0.1:1/v1/chat/completions", 400, "Bad Request",
                {}, io.BytesIO(b'{"error":"context too small"}'))

        output = self._output_path()
        args = type("Args", (), {"base_url": "http://127.0.0.1:1", "model": "m",
                                  "target_tokens": 4400, "output": str(output),
                                  "key_env": "QCM_UNUSED_KEY", "request_timeout": 5})()
        with patch.object(probe.urllib.request, "urlopen", side_effect=fake_urlopen):
            code = probe.run(args)
        self.assertEqual(code, 1)
        report = json.loads(output.read_text())
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("HTTP 400", report["reason"])
        self.assertIn("context too small", report["reason"])


if __name__ == "__main__":
    unittest.main()
