#!/usr/bin/env python3
"""Run the RTX-worker benchmark suites against an already-running server.

Runs each requested suite as a subprocess of the existing test scripts
(tests/playwright_agent_bench.py, tests/long_context_probe.py,
tests/live_image_agent.py), samples VRAM every 2 seconds through an external
command (typically `tests/bench/worker.sh vram`, run over SSH by the
caller), and writes the raw report files described in
.superpowers/sdd/2026-09-26-benchmark-action/contracts.md to --out-dir.

Run from the repository root: python3 tests/bench/run_suites.py ...
"""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SUITE_ORDER = ("fixture", "long_context", "live_web", "concurrency")
# The exact flags recorded from the validated concurrency runs: one
# job_application task, no repeats, a 24,576-token per-slot budget (half of
# the 49,152-token, --parallel 2 context the worker is restarted with).
CONCURRENCY_FLAGS = ("--tasks", "job_application", "--repetitions", "1",
                      "--task-timeout", "120", "--max-tokens", "512",
                      "--context", "24576", "--reasoning", "off")

# Per-suite job timeouts scale with the work each suite actually schedules
# instead of one flat number: a flat 900s killed normal fixture runs (10
# cases per repetition x up to 300s each is already 3,000s for a single
# repetition) while leaving a genuinely hung single-request suite waiting
# needlessly long. These mirror the *other* scripts' own defaults, which
# run_suites.py does not override for fixture/long_context/live_web.
FIXTURE_CASES = 10  # playwright_agent_bench.py's default 9 tasks, +1 extra
                     # variant because conditional_application runs twice.
DEFAULT_TASK_TIMEOUT = 300  # playwright_agent_bench.py --task-timeout default
LONG_CONTEXT_REQUEST_TIMEOUT = 120  # long_context_probe.py --request-timeout default
LIVE_WEB_TASK_TIMEOUT = 300  # live_image_agent.py --task-timeout default
CONCURRENCY_TASK_TIMEOUT = 120  # matches --task-timeout in CONCURRENCY_FLAGS
STARTUP_MARGIN = 120  # model/MCP/browser start-up overhead, multi-case suites
JOB_STARTUP_MARGIN = 60  # same, for suites that are a single request/task
# The worker.sh `remote()` helper uses `ssh -o ConnectTimeout=15`; the VRAM
# sample's own subprocess timeout must exceed that or it kills the SSH
# connection before it even finishes connecting.
VRAM_SAMPLE_TIMEOUT = 20
# The concurrency suite: four one-at-a-time runs, then two concurrent pairs.
CONCURRENCY_SERIAL_RUNS = 4
CONCURRENCY_PAIRS = 2
# Worst-case extra time per batch past its job timeout: _kill_process_group
# waits up to 10 s + 5 s per process, and a pair kills two in turn.
BATCH_KILL_GRACE = 30

# Workflow time budget (.github/workflows/benchmark.yml). The job has
# JOB_TIMEOUT_MINUTES in total. RESERVED_MINUTES covers everything that is
# not a suite step: set-up and tailnet join (~10), both `worker.sh up` steps
# (10 each, their own step caps), merge/summarize/uploads (~5), and the
# `worker.sh down` restore step (20, its own step cap). A suite step's cap is
# its worst case plus STEP_MARGIN_MINUTES, so the step cap only fires if
# run_suites.py itself hangs.
JOB_TIMEOUT_MINUTES = 240
RESERVED_MINUTES = 60
STEP_MARGIN_MINUTES = 5


def job_timeout_for(suite, args):
    """Pure: the wall-clock cap for one job of `suite`, given how much work
    run_suites.py itself asked that job to do (repetitions, in particular)."""
    if suite == "fixture":
        return FIXTURE_CASES * args.repetitions * DEFAULT_TASK_TIMEOUT + STARTUP_MARGIN
    if suite == "long_context":
        return LONG_CONTEXT_REQUEST_TIMEOUT + JOB_STARTUP_MARGIN
    if suite == "live_web":
        return LIVE_WEB_TASK_TIMEOUT + JOB_STARTUP_MARGIN
    if suite == "concurrency":
        return CONCURRENCY_TASK_TIMEOUT + JOB_STARTUP_MARGIN
    raise ValueError(f"unknown suite: {suite}")


def batch_count(suite, repetitions):
    """Pure: how many batches build_plan() schedules for `suite`."""
    if suite in ("fixture", "long_context"):
        return 1
    if suite == "live_web":
        return repetitions
    if suite == "concurrency":
        return CONCURRENCY_SERIAL_RUNS + CONCURRENCY_PAIRS
    raise ValueError(f"unknown suite: {suite}")


def worst_case_seconds(suites, repetitions):
    """Pure: the longest one run_suites.py invocation for `suites` can take
    before every job has hit its own timeout and been killed."""
    args = SimpleNamespace(repetitions=repetitions)
    return sum(batch_count(suite, repetitions) * (job_timeout_for(suite, args) + BATCH_KILL_GRACE)
               for suite in suites)


def step_minutes(suites, repetitions):
    """Pure: the workflow step cap (whole minutes) for one invocation; 0 when
    there are no suites (the step is skipped)."""
    if not suites:
        return 0
    return math.ceil(worst_case_seconds(suites, repetitions) / 60) + STEP_MARGIN_MINUTES


def fits_job(phase1, phase2, repetitions):
    """Pure: whether both suite steps plus the reserve fit in the job."""
    return (step_minutes(phase1, repetitions) + step_minutes(phase2, repetitions)
            + RESERVED_MINUTES) <= JOB_TIMEOUT_MINUTES


def max_repetitions(phase1, phase2, limit=1000):
    """Pure: the largest repetitions value that still fits the job (0 when
    even 1 does not)."""
    best = 0
    for repetitions in range(1, limit + 1):
        if not fits_job(phase1, phase2, repetitions):
            break
        best = repetitions
    return best


def _kill_process_group(process):
    """Kill a whole process group, not just the direct child: bench scripts
    launch Playwright MCP/Chrome as their own children, and Popen.kill()
    (or subprocess.run(timeout=...)'s internal kill) only reaches the
    process we spawned directly, orphaning the rest. Requires the process to
    have been started with start_new_session=True."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except OSError:
        pass
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def fixture_command(args, output, root=ROOT):
    return [sys.executable, str(root / "tests" / "playwright_agent_bench.py"),
            "--base-url", args.base_url, "--model", args.model, "--output", str(output),
            "--key-env", args.key_env, "--repetitions", str(args.repetitions),
            "--context", str(args.ctx), "--reasoning", "off"]


def long_context_command(args, output, root=ROOT):
    return [sys.executable, str(root / "tests" / "long_context_probe.py"),
            "--base-url", args.base_url, "--model", args.model,
            "--target-tokens", str(args.ctx), "--output", str(output),
            "--key-env", args.key_env]


def live_web_command(args, output, image_dir, root=ROOT):
    return [sys.executable, str(root / "tests" / "live_image_agent.py"),
            "--base-url", args.base_url, "--model", args.model, "--output", str(output),
            "--key-env", args.key_env, "--image-dir", str(image_dir), "--reasoning", "off"]


def concurrency_command(args, output, root=ROOT):
    return [sys.executable, str(root / "tests" / "playwright_agent_bench.py"),
            "--base-url", args.base_url, "--model", args.model, "--output", str(output),
            "--key-env", args.key_env, *CONCURRENCY_FLAGS]


def build_plan(args, out_dir, root=ROOT):
    """Pure: which subprocess commands to run for which requested suites.

    Returns (batches, errors). Each batch is a list of one or more job dicts
    ({"suite", "output", "cmd"}) meant to be launched together (more than
    one job in a batch means they run concurrently). `errors` maps a suite
    name to why it could not even be planned (currently: concurrency without
    --parallel 2).
    """
    batches = []
    errors = {}
    requested = set(args.suites)
    if "fixture" in requested:
        output = out_dir / "fixture.json"
        batches.append([{"suite": "fixture", "output": output,
                         "cmd": fixture_command(args, output, root)}])
    if "long_context" in requested:
        output = out_dir / "long_context.json"
        batches.append([{"suite": "long_context", "output": output,
                         "cmd": long_context_command(args, output, root)}])
    if "live_web" in requested:
        image_dir = out_dir / "live-images"
        for n in range(1, args.repetitions + 1):
            output = out_dir / f"live_web-{n}.json"
            batches.append([{"suite": "live_web", "output": output,
                             "cmd": live_web_command(args, output, image_dir, root)}])
    if "concurrency" in requested:
        if args.parallel != 2:
            errors["concurrency"] = f"concurrency requires --parallel 2 (got {args.parallel})"
        else:
            for n in range(1, CONCURRENCY_SERIAL_RUNS + 1):
                output = out_dir / f"concurrency-serial-{n}.json"
                batches.append([{"suite": "concurrency", "output": output,
                                 "cmd": concurrency_command(args, output, root)}])
            for pair in range(CONCURRENCY_PAIRS):
                group = []
                for slot in range(2):
                    n = pair * 2 + slot + 1
                    output = out_dir / f"concurrency-parallel-{n}.json"
                    group.append({"suite": "concurrency", "output": output,
                                  "cmd": concurrency_command(args, output, root)})
                batches.append(group)
    return batches, errors


def parse_vram_sample(text):
    """Pure: the worker prints the MiB integer, possibly after SSH banner
    noise on earlier lines. None means the sample failed or was unreadable."""
    if not text:
        return None
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return None
    try:
        return int(lines[-1])
    except ValueError:
        return None


def vram_peak(samples):
    """Pure: the maximum of the samples that parsed; None if none did."""
    values = [value for value in samples if isinstance(value, int)]
    return max(values) if values else None


def build_meta(existing, args, commit, gpu, run_id, now):
    """Pure: meta.json for this out-dir. A second run_suites.py invocation
    against the same --out-dir (the workflow's non-concurrency then
    concurrency phases) merges in rather than starting over: the suites list
    is unioned and created_utc is kept from the first invocation.
    suite_settings records the server ctx/parallel each suite actually ran
    with, since the two phases use different servers."""
    existing = existing or {}
    suites = list(dict.fromkeys((existing.get("suites") or []) + list(args.suites)))
    suite_settings = dict(existing.get("suite_settings") or {})
    for suite in args.suites:
        suite_settings[suite] = {"ctx": args.ctx, "parallel": args.parallel}
    return {"model": args.model, "ctx": args.ctx, "parallel": args.parallel,
            "gpu": gpu, "commit": commit, "run_id": run_id,
            "created_utc": existing.get("created_utc") or now, "suites": suites,
            "suite_settings": suite_settings}


class VramSampler:
    """Samples an external command every `interval` seconds in a background
    thread. `cmd` is a shell command string (run via the shell, typically
    over SSH); pass a `runner` to inject a fake sampler for tests."""

    def __init__(self, cmd, interval=2.0, runner=None):
        self.cmd = cmd
        self.interval = interval
        self.runner = runner or self._shell_runner
        self.stop = threading.Event()
        self.samples = []
        self._frozen_count = None
        self._process_lock = threading.Lock()
        self._process = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _shell_runner(self):
        try:
            process = subprocess.Popen(self.cmd, shell=True, start_new_session=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                       text=True)
        except OSError:
            return None
        with self._process_lock:
            self._process = process
        try:
            stdout, _ = process.communicate(timeout=VRAM_SAMPLE_TIMEOUT)
        except subprocess.TimeoutExpired:
            _kill_process_group(process)
            return None
        finally:
            with self._process_lock:
                self._process = None
        return parse_vram_sample(stdout)

    def _run(self):
        while not self.stop.is_set():
            self.samples.append(self.runner())
            self.stop.wait(self.interval)

    def __enter__(self):
        if self.cmd is not None:
            self.thread.start()
        return self

    def __exit__(self, *exc_info):
        self.stop.set()
        # An in-flight sample can block for up to VRAM_SAMPLE_TIMEOUT; do not
        # wait that long here — kill it so shutdown stays bounded.
        with self._process_lock:
            process = self._process
        if process is not None:
            _kill_process_group(process)
        if self.thread.is_alive():
            self.thread.join(timeout=5)
        # Freeze the sample count now: a sample that still lands after this
        # (the thread refusing to die, or a scheduling fluke) belongs to
        # whatever runs next, not to the report we are about to write.
        self._frozen_count = len(self.samples)

    def report(self):
        count = self._frozen_count if self._frozen_count is not None else len(self.samples)
        values = [value for value in self.samples[:count] if isinstance(value, int)]
        return {"samples_mib": values, "peak_mib": vram_peak(values)}


def _finalize(job, timed_out, timeout):
    """After a subprocess has been waited on (and killed if it timed out),
    decide whether its output counts as a kept result or a crash. A timeout
    is not by itself a reason to discard a report: playwright_agent_bench.py
    writes fixture.json after every completed case, so a job killed near its
    deadline can still have a fully valid, merely incomplete, report — keep
    it, and just note that it timed out."""
    if job["output"].exists():
        try:
            json.loads(job["output"].read_text())
        except (OSError, ValueError) as error:
            return f"invalid report JSON: {error}"
        if timed_out:
            return f"job timed out after {timeout:.0f}s; kept the partial report"
        return None
    if timed_out:
        return f"job timed out after {timeout:.0f}s"
    return "subprocess exited without writing a report"


def run_job(job, timeout=None):
    """Run one subprocess to completion in its own process group, so a
    timeout can kill Playwright/Chrome children too, not just the Python
    process. Returns None on success (including when the benchmark itself
    recorded a FAIL — that is a normal result, not a crash) or a short
    reason string when the job crashed or timed out."""
    try:
        process = subprocess.Popen(job["cmd"], start_new_session=True)
    except OSError as error:
        return f"{type(error).__name__}: {error}"
    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
        timed_out = True
    return _finalize(job, timed_out, timeout)


def run_batch(batch, timeout=None):
    """Run every job in a batch (concurrently when there is more than one),
    each in its own process group. The batch's deadline starts when the
    batch starts, not afresh at each job's wait() — otherwise two jobs in a
    pair could together run for up to 2x timeout instead of sharing it, the
    same way `for job in batch: job.wait(timeout=timeout)` would. Returns
    (suite, reason) — reason is the first crash/timeout found, or None."""
    suite = batch[0]["suite"]
    if len(batch) == 1:
        return suite, run_job(batch[0], timeout)
    processes = []
    reason = None
    for job in batch:
        try:
            processes.append((job, subprocess.Popen(job["cmd"], start_new_session=True)))
        except OSError as error:
            reason = reason or f"{type(error).__name__}: {error}"
    deadline = None if timeout is None else time.monotonic() + timeout
    for job, process in processes:
        remaining = None if deadline is None else max(0, deadline - time.monotonic())
        timed_out = False
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            _kill_process_group(process)
            timed_out = True
        job_reason = _finalize(job, timed_out, timeout)
        if job_reason and reason is None:
            reason = job_reason
    return suite, reason


def get_commit(root):
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                                capture_output=True, text=True, timeout=5, check=True)
        commit = result.stdout.strip()
        if commit:
            return commit
    except (OSError, subprocess.SubprocessError):
        pass
    return os.environ.get("GITHUB_SHA", "")[:7] or None


def run(args):
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    batches, errors = build_plan(args, out_dir)
    ran = 0
    for suite in SUITE_ORDER:
        if suite not in args.suites or suite in errors:
            continue
        suite_batches = [batch for batch in batches if batch[0]["suite"] == suite]
        timeout = args.job_timeout if args.job_timeout else job_timeout_for(suite, args)
        sampler = VramSampler(args.vram_cmd)
        crash = None
        with sampler:
            for batch in suite_batches:
                ran += len(batch)
                _, reason = run_batch(batch, timeout=timeout)
                if reason and crash is None:
                    crash = reason
                    break
        (out_dir / f"vram-{suite}.json").write_text(json.dumps(sampler.report(), indent=2) + "\n")
        if crash:
            errors[suite] = crash
    if errors:
        (out_dir / "errors.json").write_text(json.dumps(errors, indent=2) + "\n")
    meta_path = out_dir / "meta.json"
    existing = None
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            existing = None
    gpu = args.gpu or os.environ.get("BENCH_GPU_NAME") or None
    run_id = args.run_id or os.environ.get("GITHUB_RUN_ID") or str(int(time.time()))
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    meta = build_meta(existing, args, get_commit(ROOT), gpu, run_id, now)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"run_suites: ran {ran} job(s); errors: {sorted(errors) or 'none'}", flush=True)
    return 0 if ran else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ctx", type=int, required=True)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--suites", nargs="+", required=True,
                        choices=("fixture", "long_context", "live_web", "concurrency"))
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vram-cmd", default=None,
                        help="Shell command that prints used VRAM MiB; sampled every 2s")
    parser.add_argument("--key-env", default="BONSAI_API_KEY")
    parser.add_argument("--gpu", default=None, help="GPU name for meta.json; falls back to $BENCH_GPU_NAME")
    parser.add_argument("--run-id", default=None, help="Run id for meta.json; falls back to $GITHUB_RUN_ID")
    parser.add_argument("--job-timeout", type=float, default=None,
                        help="Override the per-suite wall-clock cap (seconds); by "
                             "default it scales with the suite's own work, see "
                             "job_timeout_for()")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
