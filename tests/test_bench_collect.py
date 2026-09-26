"""Tests for tests/bench/collect.py — stdlib unittest only, no network.

Covers: summarize() on the real-shaped fixture dirs under tests/fixtures/bench/
(pass counts, medians, peaks, absent suites, crashed suites) and index()
(sorting by created_utc, schema, tolerance of a corrupt summary file).
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

from tests.bench import collect

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "tests", "fixtures", "bench")


def fixture_path(*parts):
    return os.path.join(FIXTURES, *parts)


class SummarizeRun32kTest(unittest.TestCase):
    """32K Qwen3.5 9B: fixture job_application PASS 15.7s; live_web 1/2."""

    def setUp(self):
        self.summary = collect.summarize(fixture_path("run-32k"))

    def test_top_level_fields_from_meta(self):
        self.assertEqual(self.summary["schema"], 1)
        self.assertEqual(self.summary["run_id"], "2026092501")
        self.assertEqual(self.summary["created_utc"], "2026-09-25T20:01:34+00:00")
        self.assertEqual(self.summary["commit"], "9f1c2ab")
        self.assertEqual(self.summary["model"], "qwen3.5-9b-q4_k_m")
        self.assertEqual(self.summary["ctx"], 32768)
        self.assertEqual(self.summary["gpu"], "RTX 3070 Ti")

    def test_fixture_suite(self):
        s = self.summary["suites"]["fixture"]
        self.assertEqual(s["passed"], 1)
        self.assertEqual(s["total"], 1)
        self.assertEqual(s["pass_rate"], 100.0)
        self.assertEqual(s["median_seconds"], 15.7)
        self.assertEqual(s["peak_vram_mib"], 6269)
        self.assertNotIn("error", s)

    def test_live_web_suite_one_of_two_passed(self):
        s = self.summary["suites"]["live_web"]
        self.assertEqual(s["passed"], 1)
        self.assertEqual(s["total"], 2)
        self.assertEqual(s["pass_rate"], 50.0)
        self.assertEqual(s["median_seconds"], 49.5)
        self.assertEqual(s["peak_vram_mib"], 6120)

    def test_long_context_absent_when_not_requested(self):
        self.assertNotIn("long_context", self.summary["suites"])
        self.assertNotIn("concurrency_serial", self.summary["suites"])
        self.assertNotIn("concurrency_parallel", self.summary["suites"])


class SummarizeRun2x24kTest(unittest.TestCase):
    """2x24K: concurrency parallel 2/4, serial 3/4, peak 6,823 MiB."""

    def setUp(self):
        self.summary = collect.summarize(fixture_path("run-2x24k"))

    def test_concurrency_serial_three_of_four(self):
        s = self.summary["suites"]["concurrency_serial"]
        self.assertEqual((s["passed"], s["total"]), (3, 4))
        self.assertEqual(s["pass_rate"], 75.0)
        self.assertEqual(s["median_seconds"], 16.3)
        self.assertEqual(s["peak_vram_mib"], 6823)

    def test_concurrency_parallel_two_of_four(self):
        s = self.summary["suites"]["concurrency_parallel"]
        self.assertEqual((s["passed"], s["total"]), (2, 4))
        self.assertEqual(s["pass_rate"], 50.0)
        self.assertEqual(s["median_seconds"], 29.7)
        self.assertEqual(s["peak_vram_mib"], 6823)

    def test_concurrency_phases_share_one_vram_sample(self):
        """run_suites.py samples VRAM once for the whole concurrency suite
        (both phases run back to back under one VramSampler) and writes a
        single vram-concurrency.json — never a per-phase file. Both summary
        entries must read that same peak, not two independent ones."""
        serial = self.summary["suites"]["concurrency_serial"]
        parallel = self.summary["suites"]["concurrency_parallel"]
        self.assertEqual(serial["peak_vram_mib"], 6823)
        self.assertEqual(parallel["peak_vram_mib"], 6823)
        self.assertEqual(serial["peak_vram_mib"], parallel["peak_vram_mib"])

    def test_run_level_peak_is_max_across_suites(self):
        peaks = [s["peak_vram_mib"] for s in self.summary["suites"].values()]
        self.assertEqual(max(peaks), 6823)


class SummarizeRun64kTest(unittest.TestCase):
    """64K: long_context PASS 59,906 tokens 26.1s, peak 7,323 MiB; live_web 2/3."""

    def setUp(self):
        self.summary = collect.summarize(fixture_path("run-64k"))

    def test_long_context_suite(self):
        s = self.summary["suites"]["long_context"]
        self.assertEqual((s["passed"], s["total"]), (1, 1))
        self.assertEqual(s["pass_rate"], 100.0)
        self.assertEqual(s["median_seconds"], 26.1)
        self.assertEqual(s["peak_prompt_tokens"], 59906)
        self.assertEqual(s["peak_vram_mib"], 7323)

    def test_live_web_two_of_three(self):
        s = self.summary["suites"]["live_web"]
        self.assertEqual((s["passed"], s["total"]), (2, 3))
        self.assertAlmostEqual(s["pass_rate"], 66.7)
        self.assertEqual(s["median_seconds"], 69.5)

    def test_run_level_peak_is_long_context(self):
        peaks = [s["peak_vram_mib"] for s in self.summary["suites"].values()]
        self.assertEqual(max(peaks), 7323)


class SummarizeErrorsTest(unittest.TestCase):
    """A crashed suite is recorded with 0 passed and its error; other requested
    suites still summarize normally; suites never requested stay absent."""

    def setUp(self):
        self.summary = collect.summarize(fixture_path("run-errors"))

    def test_crashed_suite_has_error_and_zero_counts(self):
        s = self.summary["suites"]["fixture"]
        self.assertEqual(s["passed"], 0)
        self.assertEqual(s["total"], 0)
        self.assertEqual(s["pass_rate"], 0.0)
        self.assertIsNone(s["median_seconds"])
        self.assertIsNone(s["peak_vram_mib"])
        self.assertIn("500", s["error"])

    def test_other_requested_suite_unaffected(self):
        s = self.summary["suites"]["long_context"]
        self.assertEqual((s["passed"], s["total"]), (1, 1))
        self.assertNotIn("error", s)

    def test_never_requested_suite_is_absent(self):
        self.assertNotIn("live_web", self.summary["suites"])
        self.assertNotIn("concurrency_serial", self.summary["suites"])


class SummarizeMissingFilesTest(unittest.TestCase):
    """collect.py is tolerant of missing files: no meta.json at all still
    produces a summary, with null top-level fields and suites inferred from
    whatever raw report files are actually on disk."""

    def setUp(self):
        self.summary = collect.summarize(fixture_path("run-missing-meta"))

    def test_missing_meta_gives_null_top_fields(self):
        self.assertIsNone(self.summary["run_id"])
        self.assertIsNone(self.summary["created_utc"])
        self.assertIsNone(self.summary["commit"])
        self.assertIsNone(self.summary["model"])
        self.assertIsNone(self.summary["ctx"])
        self.assertIsNone(self.summary["gpu"])
        self.assertEqual(self.summary["schema"], 1)

    def test_suite_inferred_from_files_on_disk(self):
        s = self.summary["suites"]["fixture"]
        self.assertEqual((s["passed"], s["total"]), (1, 1))
        self.assertIsNone(s["peak_vram_mib"])  # no vram-fixture.json present

    def test_completely_empty_directory_has_no_suites(self):
        with tempfile.TemporaryDirectory() as d:
            summary = collect.summarize(d)
            self.assertEqual(summary["suites"], {})
            self.assertEqual(summary["schema"], 1)


class SummarizeCliTest(unittest.TestCase):
    """The `summarize` subcommand writes exactly the JSON collect.summarize()
    would return, invoked the way the workflow invokes it."""

    def test_summarize_subcommand_writes_out_file(self):
        with tempfile.TemporaryDirectory() as d:
            out_path = os.path.join(d, "summary.json")
            result = subprocess.run(
                [sys.executable, os.path.join(ROOT, "tests", "bench", "collect.py"),
                 "summarize", fixture_path("run-32k"), "--out", out_path],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(out_path) as f:
                written = json.load(f)
            self.assertEqual(written, collect.summarize(fixture_path("run-32k")))


class IndexTest(unittest.TestCase):
    def test_index_sorted_ascending_by_created_utc(self):
        index = collect.build_index(fixture_path("runs-index"))
        self.assertEqual(index["schema"], 1)
        run_ids = [r["run_id"] for r in index["runs"]]
        self.assertEqual(run_ids, ["2026092501", "2026092502", "2026092503"])

    def test_index_skips_corrupt_summary_file(self):
        index = collect.build_index(fixture_path("runs-index"))
        # broken.json must not have produced a crash, and must not appear.
        self.assertEqual(len(index["runs"]), 3)

    def test_index_of_empty_directory(self):
        with tempfile.TemporaryDirectory() as d:
            index = collect.build_index(d)
            self.assertEqual(index, {"schema": 1, "runs": []})

    def test_index_subcommand_writes_out_file(self):
        with tempfile.TemporaryDirectory() as d:
            out_path = os.path.join(d, "index.json")
            result = subprocess.run(
                [sys.executable, os.path.join(ROOT, "tests", "bench", "collect.py"),
                 "index", fixture_path("runs-index"), "--out", out_path],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(out_path) as f:
                written = json.load(f)
            self.assertEqual(written, collect.build_index(fixture_path("runs-index")))


class HelperFunctionTest(unittest.TestCase):
    def test_median_of_empty_is_none(self):
        self.assertIsNone(collect.median_or_none([]))

    def test_median_of_values(self):
        self.assertEqual(collect.median_or_none([1.0, 3.0, 2.0]), 2.0)

    def test_pass_rate_zero_total_is_zero(self):
        self.assertEqual(collect.pass_rate(0, 0), 0.0)

    def test_pass_rate_rounds_to_one_decimal(self):
        self.assertEqual(collect.pass_rate(2, 3), 66.7)


if __name__ == "__main__":
    unittest.main()
