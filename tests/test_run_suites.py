"""Tests for tests/bench/run_suites.py.

stdlib unittest, not pytest — see tests/test_cache_viz.py for the convention.
No network: only pure planning/parsing functions and small local subprocesses
(python -c "...", never a real server) are exercised here.
"""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
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
                repetitions=2, vram_cmd=None, gpu=None, run_id=None, job_timeout=None)
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
            # Wait for real samples, not a fixed sleep: a loaded CI host may
            # not schedule the thread within a few milliseconds.
            deadline = time.monotonic() + 10
            while len(sampler.samples) < 3 and time.monotonic() < deadline:
                time.sleep(0.01)
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
                                "suites": ["fixture", "long_context"],
                                "suite_settings": {
                                    "fixture": {"ctx": 32768, "parallel": 1},
                                    "long_context": {"ctx": 32768, "parallel": 1}}})

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

    def test_suite_settings_keep_each_phase_server(self):
        """The concurrency phase runs on a different server: its ctx must not
        overwrite the settings the earlier suites ran with."""
        first = run_suites.build_meta(None, make_args(suites=["fixture"], ctx=32768, parallel=1),
                                      "abc1234", None, "42", "2026-09-26T00:00:00+00:00")
        second = run_suites.build_meta(first, make_args(suites=["concurrency"], ctx=49152, parallel=2),
                                       "abc1234", None, "42", "2026-09-26T01:00:00+00:00")
        self.assertEqual(second["suite_settings"], {
            "fixture": {"ctx": 32768, "parallel": 1},
            "concurrency": {"ctx": 49152, "parallel": 2}})


class TimeBudgetTest(unittest.TestCase):
    """The workflow sizes its step caps and refuses too many repetitions
    from these numbers, so the restore step always fits in the job."""

    ALL_PHASE1 = ["fixture", "long_context", "live_web"]

    def test_batch_count_matches_build_plan(self):
        args = make_args(suites=["fixture", "long_context", "live_web", "concurrency"],
                         parallel=2, repetitions=3)
        batches, _ = run_suites.build_plan(args, pathlib.Path("/tmp/out"))
        for suite in args.suites:
            planned = sum(1 for b in batches if b[0]["suite"] == suite)
            self.assertEqual(run_suites.batch_count(suite, 3), planned, suite)

    def test_step_cap_covers_every_job_timeout(self):
        reps = 2
        args = make_args(repetitions=reps)
        floor = sum(run_suites.batch_count(s, reps) * run_suites.job_timeout_for(s, args)
                    for s in self.ALL_PHASE1)
        self.assertGreater(run_suites.step_minutes(self.ALL_PHASE1, reps) * 60, floor)

    def test_no_suites_means_no_step_time(self):
        self.assertEqual(run_suites.step_minutes([], 5), 0)

    def test_max_repetitions_fits_and_one_more_does_not(self):
        for phase1, phase2 in ((self.ALL_PHASE1, ["concurrency"]), (["fixture"], []),
                               (["live_web"], ["concurrency"])):
            with self.subTest(phase1=phase1, phase2=phase2):
                most = run_suites.max_repetitions(phase1, phase2)
                self.assertGreaterEqual(most, 1)
                self.assertTrue(run_suites.fits_job(phase1, phase2, most))
                self.assertFalse(run_suites.fits_job(phase1, phase2, most + 1))
                total = (run_suites.step_minutes(phase1, most) + run_suites.step_minutes(phase2, most)
                         + run_suites.RESERVED_MINUTES)
                self.assertLessEqual(total, run_suites.JOB_TIMEOUT_MINUTES)

    def test_default_suites_cap_is_two_repetitions(self):
        # Documented in docs/benchmark-action.md; update both together.
        self.assertEqual(run_suites.max_repetitions(self.ALL_PHASE1, ["concurrency"]), 2)
        self.assertEqual(run_suites.max_repetitions(self.ALL_PHASE1, []), 2)


class JobTimeoutTest(unittest.TestCase):
    """Regression tests for the flat 900s default that killed normal fixture
    runs (10 cases x repetitions, each up to 300s) mid-way through."""

    def test_fixture_scales_with_cases_and_repetitions(self):
        t1 = run_suites.job_timeout_for("fixture", make_args(repetitions=1))
        t3 = run_suites.job_timeout_for("fixture", make_args(repetitions=3))
        self.assertGreater(t3, t1)
        self.assertEqual(t1, run_suites.FIXTURE_CASES * 1 * run_suites.DEFAULT_TASK_TIMEOUT
                         + run_suites.STARTUP_MARGIN)
        self.assertEqual(t3, run_suites.FIXTURE_CASES * 3 * run_suites.DEFAULT_TASK_TIMEOUT
                         + run_suites.STARTUP_MARGIN)

    def test_a_flat_900s_budget_would_still_have_killed_a_3_repetition_fixture_run(self):
        self.assertGreater(run_suites.job_timeout_for("fixture", make_args(repetitions=3)), 900)

    def test_long_context_live_web_and_concurrency_use_their_own_scripts_defaults(self):
        self.assertEqual(run_suites.job_timeout_for("long_context", make_args()),
                         run_suites.LONG_CONTEXT_REQUEST_TIMEOUT + run_suites.JOB_STARTUP_MARGIN)
        self.assertEqual(run_suites.job_timeout_for("live_web", make_args()),
                         run_suites.LIVE_WEB_TASK_TIMEOUT + run_suites.JOB_STARTUP_MARGIN)
        self.assertEqual(run_suites.job_timeout_for("concurrency", make_args()),
                         run_suites.CONCURRENCY_TASK_TIMEOUT + run_suites.JOB_STARTUP_MARGIN)

    def test_unknown_suite_raises(self):
        with self.assertRaises(ValueError):
            run_suites.job_timeout_for("nope", make_args())


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

    def test_timeout_without_any_output_is_a_plain_timeout_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = pathlib.Path(tmp) / "missing.json"
            job = {"suite": "fixture", "output": output,
                   "cmd": [sys.executable, "-c", "import time; time.sleep(5)"]}
            reason = run_suites.run_job(job, timeout=0.3)
            self.assertIn("timed out", reason)
            self.assertNotIn("kept", reason)
            self.assertFalse(output.exists())

    def test_timeout_with_a_valid_partial_report_keeps_it_not_discards_it(self):
        """The exact bug from the review: a valid partial fixture.json must
        survive a timeout, not be thrown away as a crash."""
        with tempfile.TemporaryDirectory() as tmp:
            output = pathlib.Path(tmp) / "out.json"
            job = {"suite": "fixture", "output": output,
                   "cmd": [sys.executable, "-c",
                           f"import json,pathlib,time; "
                           f"pathlib.Path({str(output)!r}).write_text(json.dumps({{'results': [1]}})); "
                           f"time.sleep(30)"]}
            # 3 s: the child must start Python and write before the kill,
            # even on a slow, loaded CI host.
            reason = run_suites.run_job(job, timeout=3)
            self.assertIsNotNone(reason)
            self.assertIn("timed out", reason)
            self.assertIn("kept the partial report", reason)
            self.assertEqual(json.loads(output.read_text()), {"results": [1]})

    def test_timeout_kills_the_whole_process_group_not_just_the_direct_child(self):
        """Regression test for the orphaned-Chrome/Playwright-children bug:
        subprocess.run(timeout=...) (or Popen.kill()) only reaches the
        process it spawned directly. A backgrounded grandchild, started in
        the same session via start_new_session=True, must die too."""
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = pathlib.Path(tmp) / "child.pid"
            output = pathlib.Path(tmp) / "out.json"
            job = {"suite": "fixture", "output": output,
                   "cmd": ["bash", "-c", f"sleep 30 & echo $! > {pidfile}; wait"]}
            # 2 s: bash must have started the grandchild before the kill.
            reason = run_suites.run_job(job, timeout=2)
            self.assertIn("timed out", reason)
            deadline = time.monotonic() + 5
            pid = None
            while time.monotonic() < deadline:
                if pidfile.exists() and pidfile.read_text().strip():
                    pid = int(pidfile.read_text().strip())
                    break
                time.sleep(0.05)
            self.assertIsNotNone(pid, "the background grandchild never even started")
            # Give the process-group kill a moment to land, then confirm the
            # grandchild (not just the bash process we spawned) is gone.
            gone = False
            for _ in range(50):
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    gone = True
                    break
                time.sleep(0.1)
            self.assertTrue(gone, f"grandchild pid {pid} (sleep 30) was not killed with the group")


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

    def test_pair_deadline_starts_when_the_pair_starts_not_at_each_wait(self):
        """The bug: `for job in batch: process.wait(timeout=timeout)` gives
        EVERY job in the pair a fresh full `timeout`, so a job checked later
        effectively gets (time already elapsed + a full fresh timeout)
        instead of sharing one deadline that starts when the pair starts.
        Both jobs launch concurrently at t=0; `slow` needs 2.6 s (more than
        the shared 2.0 s deadline, so the fix must kill it) but less than
        (fast's 1.0 s + a fresh 2.0 s = 3.0 s), which is exactly the window
        the bug would have given it. The gaps are whole fractions of a
        second so a loaded CI host (slow Python start-up) cannot blur them."""
        with tempfile.TemporaryDirectory() as tmp:
            fast_output = pathlib.Path(tmp) / "fast.json"
            slow_output = pathlib.Path(tmp) / "slow.json"
            fast = {"suite": "concurrency", "output": fast_output,
                    "cmd": [sys.executable, "-c",
                            f"import json,time,pathlib; time.sleep(1.0); "
                            f"pathlib.Path({str(fast_output)!r}).write_text(json.dumps({{'status':'PASS'}}))"]}
            slow = {"suite": "concurrency", "output": slow_output,
                    "cmd": [sys.executable, "-c",
                            f"import json,time,pathlib; time.sleep(2.6); "
                            f"pathlib.Path({str(slow_output)!r}).write_text(json.dumps({{'status':'PASS'}}))"]}
            started = time.monotonic()
            suite, reason = run_suites.run_batch([fast, slow], timeout=2.0)
            elapsed = time.monotonic() - started
            self.assertIn("timed out", reason)
            self.assertTrue(fast_output.exists())
            self.assertFalse(slow_output.exists(),
                             "slow job got a fresh timeout instead of sharing the pair's deadline")
            self.assertLess(elapsed, 2.5,
                            "ran past the shared deadline towards slow's own natural finish time")


class RunIntegrationTest(unittest.TestCase):
    """run() end-to-end with the command builders swapped for tiny local
    scripts — no server, no browser, no network."""

    def test_writes_the_full_contract_file_set_and_exits_0(self):
        originals = (run_suites.fixture_command, run_suites.long_context_command,
                     run_suites.live_web_command, run_suites.concurrency_command)

        def fake_playwright(args, output, *extra, root=None):
            return [sys.executable, "-c",
                    f"import json,pathlib; pathlib.Path({str(output)!r}).write_text(json.dumps({{'status':'PASS'}}))"]

        run_suites.fixture_command = fake_playwright
        run_suites.long_context_command = fake_playwright
        run_suites.live_web_command = fake_playwright
        run_suites.concurrency_command = fake_playwright
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out_dir = pathlib.Path(tmp) / "out"
                args = make_args(suites=["fixture", "long_context", "live_web", "concurrency"],
                                 parallel=2, repetitions=2, vram_cmd="echo 1234",
                                 run_id="99", job_timeout=10, out_dir=str(out_dir))
                code = run_suites.run(args)
                self.assertEqual(code, 0)
                names = {p.name for p in out_dir.iterdir()}
                expected = {"meta.json", "fixture.json", "long_context.json",
                            "live_web-1.json", "live_web-2.json",
                            "vram-fixture.json", "vram-long_context.json",
                            "vram-live_web.json", "vram-concurrency.json"}
                expected |= {f"concurrency-serial-{n}.json" for n in range(1, 5)}
                expected |= {f"concurrency-parallel-{n}.json" for n in range(1, 5)}
                self.assertEqual(expected, expected & names)
                self.assertNotIn("errors.json", names)
                meta = json.loads((out_dir / "meta.json").read_text())
                self.assertEqual(meta["run_id"], "99")
                self.assertEqual(meta["suites"], ["fixture", "long_context", "live_web", "concurrency"])
        finally:
            (run_suites.fixture_command, run_suites.long_context_command,
             run_suites.live_web_command, run_suites.concurrency_command) = originals

    def test_exits_1_and_writes_errors_json_when_concurrency_needs_parallel_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = pathlib.Path(tmp) / "out"
            args = make_args(suites=["concurrency"], parallel=1, repetitions=1,
                             vram_cmd=None, job_timeout=10, out_dir=str(out_dir))
            code = run_suites.run(args)
            self.assertEqual(code, 1)
            errors = json.loads((out_dir / "errors.json").read_text())
            self.assertIn("concurrency", errors)
            self.assertFalse((out_dir / "vram-concurrency.json").exists())


if __name__ == "__main__":
    unittest.main()
