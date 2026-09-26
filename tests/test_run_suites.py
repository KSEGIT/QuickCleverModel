"""Tests for tests/bench/run_suites.py.

stdlib unittest, not pytest — see tests/test_cache_viz.py for the convention.
No network: only pure planning/parsing functions and small local subprocesses
(python -c "...", never a real server) are exercised here.
"""
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_suites", ROOT / "tests" / "bench" / "run_suites.py")
run_suites = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_suites)


def make_args(**overrides):
    base = dict(base_url="http://127.0.0.1:1", model="m", key_env="QCM_UNUSED_KEY",
                ctx=32768, parallel=1, suites=["fixture", "long_context", "live_web"],
                repetitions=2, vram_cmd=None, gpu=None, run_id=None)
    base.update(overrides)
    return SimpleNamespace(**base)


class CommandBuilderTest(unittest.TestCase):
    def test_fixture_command_carries_context_and_reasoning_off(self):
        cmd = run_suites.fixture_command(make_args(), pathlib.Path("/tmp/fixture.json"))
        self.assertEqual(cmd[0], sys.executable)
        self.assertTrue(cmd[1].endswith("tests/playwright_agent_bench.py"))
        self.assertIn("--context", cmd)
        self.assertEqual(cmd[cmd.index("--context") + 1], "32768")
        self.assertIn("--reasoning", cmd)
        self.assertEqual(cmd[cmd.index("--reasoning") + 1], "off")
        self.assertIn("--repetitions", cmd)
        self.assertEqual(cmd[cmd.index("--repetitions") + 1], "2")

    def test_long_context_command_uses_ctx_as_target_tokens(self):
        cmd = run_suites.long_context_command(make_args(), pathlib.Path("/tmp/long_context.json"))
        self.assertTrue(cmd[1].endswith("tests/long_context_probe.py"))
        self.assertIn("--target-tokens", cmd)
        self.assertEqual(cmd[cmd.index("--target-tokens") + 1], "32768")
        self.assertNotIn("--reasoning", cmd)  # the probe has no such flag

    def test_live_web_command_passes_image_dir_and_reasoning_off(self):
        cmd = run_suites.live_web_command(make_args(), pathlib.Path("/tmp/live_web-1.json"),
                                          pathlib.Path("/tmp/live-images"))
        self.assertTrue(cmd[1].endswith("tests/live_image_agent.py"))
        self.assertIn("--image-dir", cmd)
        self.assertEqual(cmd[cmd.index("--image-dir") + 1], "/tmp/live-images")
        self.assertEqual(cmd[cmd.index("--reasoning") + 1], "off")

    def test_concurrency_command_uses_exact_recorded_flags(self):
        cmd = run_suites.concurrency_command(make_args(), pathlib.Path("/tmp/concurrency-serial-1.json"))
        self.assertTrue(cmd[1].endswith("tests/playwright_agent_bench.py"))
        for flag, value in (("--tasks", "job_application"), ("--repetitions", "1"),
                            ("--task-timeout", "120"), ("--max-tokens", "512"),
                            ("--context", "24576"), ("--reasoning", "off")):
            self.assertEqual(cmd[cmd.index(flag) + 1], value, flag)


class BuildPlanTest(unittest.TestCase):
    def test_plans_one_batch_each_for_fixture_and_long_context(self):
        out_dir = pathlib.Path("/tmp/out")
        batches, errors = run_suites.build_plan(
            make_args(suites=["fixture", "long_context"]), out_dir)
        self.assertEqual(errors, {})
        self.assertEqual(len(batches), 2)
        self.assertEqual([len(b) for b in batches], [1, 1])
        self.assertEqual(batches[0][0]["suite"], "fixture")
        self.assertEqual(batches[0][0]["output"], out_dir / "fixture.json")
        self.assertEqual(batches[1][0]["suite"], "long_context")
        self.assertEqual(batches[1][0]["output"], out_dir / "long_context.json")

    def test_live_web_makes_one_batch_per_repetition(self):
        out_dir = pathlib.Path("/tmp/out")
        batches, errors = run_suites.build_plan(
            make_args(suites=["live_web"], repetitions=3), out_dir)
        self.assertEqual(errors, {})
        self.assertEqual(len(batches), 3)
        outputs = [b[0]["output"] for b in batches]
        self.assertEqual(outputs, [out_dir / "live_web-1.json", out_dir / "live_web-2.json",
                                    out_dir / "live_web-3.json"])
        # All repetitions share one image directory.
        image_dirs = {b[0]["cmd"][b[0]["cmd"].index("--image-dir") + 1] for b in batches}
        self.assertEqual(image_dirs, {str(out_dir / "live-images")})

    def test_concurrency_needs_parallel_2_else_errors_and_no_batches(self):
        out_dir = pathlib.Path("/tmp/out")
        batches, errors = run_suites.build_plan(
            make_args(suites=["concurrency"], parallel=1), out_dir)
        self.assertEqual(batches, [])
        self.assertIn("concurrency", errors)
        self.assertIn("--parallel 2", errors["concurrency"])

    def test_concurrency_with_parallel_2_makes_4_serial_then_2_pairs(self):
        out_dir = pathlib.Path("/tmp/out")
        batches, errors = run_suites.build_plan(
            make_args(suites=["concurrency"], parallel=2), out_dir)
        self.assertEqual(errors, {})
        sizes = [len(b) for b in batches]
        self.assertEqual(sizes, [1, 1, 1, 1, 2, 2])
        serial_outputs = [b[0]["output"].name for b in batches[:4]]
        self.assertEqual(serial_outputs, [f"concurrency-serial-{n}.json" for n in range(1, 5)])
        parallel_outputs = sorted(job["output"].name for batch in batches[4:] for job in batch)
        self.assertEqual(parallel_outputs, [f"concurrency-parallel-{n}.json" for n in range(1, 5)])
        for job in batches[4]:
            self.assertEqual(job["suite"], "concurrency")

    def test_suites_run_in_canonical_order_regardless_of_request_order(self):
        out_dir = pathlib.Path("/tmp/out")
        batches, _ = run_suites.build_plan(
            make_args(suites=["live_web", "fixture"], repetitions=1), out_dir)
        self.assertEqual([b[0]["suite"] for b in batches], ["fixture", "live_web"])


class VramSampleParsingTest(unittest.TestCase):
    def test_parses_a_bare_integer(self):
        self.assertEqual(run_suites.parse_vram_sample("6269\n"), 6269)

    def test_parses_the_last_line_when_there_is_remote_ssh_banner_noise(self):
        self.assertEqual(run_suites.parse_vram_sample("Warning: banner\n6269"), 6269)

    def test_returns_none_for_empty_or_non_numeric_output(self):
        self.assertIsNone(run_suites.parse_vram_sample(""))
        self.assertIsNone(run_suites.parse_vram_sample(None))
        self.assertIsNone(run_suites.parse_vram_sample("not a number"))


class VramPeakTest(unittest.TestCase):
    def test_peak_of_samples(self):
        self.assertEqual(run_suites.vram_peak([100, 6269, 3000]), 6269)

    def test_ignores_failed_samples(self):
        self.assertEqual(run_suites.vram_peak([100, None, 200]), 200)

    def test_empty_or_all_failed_is_none(self):
        self.assertIsNone(run_suites.vram_peak([]))
        self.assertIsNone(run_suites.vram_peak([None, None]))


class VramSamplerTest(unittest.TestCase):
    def test_samples_with_an_injected_runner_and_reports_peak(self):
        values = iter([100, 200, 150])
        sampler = run_suites.VramSampler("unused", interval=0.01,
                                         runner=lambda: next(values, None))
        with sampler:
            sampler.stop.wait(0.05)
        report = sampler.report()
        self.assertGreaterEqual(len(report["samples_mib"]), 1)
        self.assertEqual(report["peak_mib"], max(report["samples_mib"]))

    def test_no_cmd_means_no_sampling_and_null_peak(self):
        sampler = run_suites.VramSampler(None, interval=0.01)
        with sampler:
            pass
        self.assertEqual(sampler.report(), {"samples_mib": [], "peak_mib": None})


class BuildMetaTest(unittest.TestCase):
    def test_builds_fresh_meta_when_nothing_exists_yet(self):
        args = make_args(suites=["fixture", "long_context"], ctx=32768, parallel=1)
        meta = run_suites.build_meta(None, args, "abc1234", "RTX 3070 Ti", "42", "2026-09-26T00:00:00+00:00")
        self.assertEqual(meta, {"model": "m", "ctx": 32768, "parallel": 1, "gpu": "RTX 3070 Ti",
                                "commit": "abc1234", "run_id": "42",
                                "created_utc": "2026-09-26T00:00:00+00:00",
                                "suites": ["fixture", "long_context"]})

    def test_merging_a_second_invocation_unions_suites_and_keeps_first_created_utc(self):
        existing = {"model": "m", "ctx": 32768, "parallel": 1, "gpu": "RTX 3070 Ti",
                    "commit": "abc1234", "run_id": "42", "created_utc": "2026-09-26T00:00:00+00:00",
                    "suites": ["fixture", "long_context", "live_web"]}
        args = make_args(suites=["concurrency"], ctx=49152, parallel=2)
        meta = run_suites.build_meta(existing, args, "abc1234", "RTX 3070 Ti", "42",
                                     "2026-09-26T01:00:00+00:00")
        self.assertEqual(meta["created_utc"], "2026-09-26T00:00:00+00:00")
        self.assertEqual(meta["suites"], ["fixture", "long_context", "live_web", "concurrency"])
        self.assertEqual(meta["ctx"], 49152)
        self.assertEqual(meta["parallel"], 2)


class RunJobTest(unittest.TestCase):
    def test_success_when_the_subprocess_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = pathlib.Path(tmp) / "out.json"
            job = {"suite": "fixture", "output": output,
                   "cmd": [sys.executable, "-c",
                           f"import json,pathlib; pathlib.Path({str(output)!r}).write_text(json.dumps({{'status': 'PASS'}}))"]}
            self.assertIsNone(run_suites.run_job(job, timeout=10))

    def test_crash_reported_when_no_report_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = pathlib.Path(tmp) / "out.json"
            job = {"suite": "fixture", "output": output,
                   "cmd": [sys.executable, "-c", "import sys; sys.exit(1)"]}
            reason = run_suites.run_job(job, timeout=10)
            self.assertIsNotNone(reason)
            self.assertIn("without writing a report", reason)

    def test_crash_reported_when_the_command_cannot_even_start(self):
        job = {"suite": "fixture", "output": pathlib.Path("/nonexistent/out.json"),
               "cmd": ["/nonexistent/interpreter", "-c", "pass"]}
        reason = run_suites.run_job(job, timeout=10)
        self.assertIsNotNone(reason)

    def test_crash_reported_when_the_report_is_not_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = pathlib.Path(tmp) / "out.json"
            job = {"suite": "fixture", "output": output,
                   "cmd": [sys.executable, "-c",
                           f"import pathlib; pathlib.Path({str(output)!r}).write_text('not json')"]}
            reason = run_suites.run_job(job, timeout=10)
            self.assertIn("invalid report JSON", reason)


class RunBatchTest(unittest.TestCase):
    def test_a_pair_runs_concurrently_and_reports_the_first_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            good_output = pathlib.Path(tmp) / "good.json"
            bad_output = pathlib.Path(tmp) / "bad.json"
            good = {"suite": "concurrency", "output": good_output,
                    "cmd": [sys.executable, "-c",
                            f"import json,pathlib; pathlib.Path({str(good_output)!r}).write_text(json.dumps({{'status': 'PASS'}}))"]}
            bad = {"suite": "concurrency", "output": bad_output,
                   "cmd": [sys.executable, "-c", "import sys; sys.exit(1)"]}
            suite, reason = run_suites.run_batch([good, bad], timeout=10)
            self.assertEqual(suite, "concurrency")
            self.assertIn("without writing a report", reason)
            self.assertTrue(good_output.exists())


if __name__ == "__main__":
    unittest.main()
