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
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
SUITE_ORDER = ("fixture", "long_context", "live_web", "concurrency")
# The exact flags recorded from the validated concurrency runs: one
# job_application task, no repeats, a 24,576-token per-slot budget (half of
# the 49,152-token, --parallel 2 context the worker is restarted with).
CONCURRENCY_FLAGS = ("--tasks", "job_application", "--repetitions", "1",
                      "--task-timeout", "120", "--max-tokens", "512",
                      "--context", "24576", "--reasoning", "off")


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
            for n in range(1, 5):
                output = out_dir / f"concurrency-serial-{n}.json"
                batches.append([{"suite": "concurrency", "output": output,
                                 "cmd": concurrency_command(args, output, root)}])
            for pair in range(2):
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
    is unioned and created_utc is kept from the first invocation."""
    existing = existing or {}
    suites = list(dict.fromkeys((existing.get("suites") or []) + list(args.suites)))
    return {"model": args.model, "ctx": args.ctx, "parallel": args.parallel,
            "gpu": gpu, "commit": commit, "run_id": run_id,
            "created_utc": existing.get("created_utc") or now, "suites": suites}


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
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _shell_runner(self):
        try:
            result = subprocess.run(self.cmd, shell=True, capture_output=True,
                                    text=True, timeout=max(10, self.interval * 2))
        except (OSError, subprocess.SubprocessError):
            return None
        return parse_vram_sample(result.stdout)

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
        if self.thread.is_alive():
            self.thread.join(timeout=5)

    def report(self):
        values = [value for value in self.samples if isinstance(value, int)]
        return {"samples_mib": values, "peak_mib": vram_peak(values)}


def run_job(job, timeout=None):
    """Run one subprocess to completion. Returns None on success (including
    when the benchmark itself recorded a FAIL — that is a normal result, not
    a crash) or a short reason string when the job crashed."""
    try:
        subprocess.run(job["cmd"], timeout=timeout)
    except (OSError, subprocess.SubprocessError) as error:
        return f"{type(error).__name__}: {error}"
    if not job["output"].exists():
        return "subprocess exited without writing a report"
    try:
        json.loads(job["output"].read_text())
    except (OSError, ValueError) as error:
        return f"invalid report JSON: {error}"
    return None


def run_batch(batch, timeout=None):
    """Run every job in a batch (concurrently when there is more than one).
    Returns (suite, reason) — reason is the first crash found, or None."""
    suite = batch[0]["suite"]
    if len(batch) == 1:
        return suite, run_job(batch[0], timeout)
    processes = []
    reason = None
    for job in batch:
        try:
            processes.append((job, subprocess.Popen(job["cmd"])))
        except OSError as error:
            reason = reason or f"{type(error).__name__}: {error}"
    for job, process in processes:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            reason = reason or "subprocess timed out"
            continue
        if not job["output"].exists():
            reason = reason or "subprocess exited without writing a report"
            continue
        try:
            json.loads(job["output"].read_text())
        except (OSError, ValueError) as error:
            reason = reason or f"invalid report JSON: {error}"
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
        sampler = VramSampler(args.vram_cmd)
        crash = None
        with sampler:
            for batch in suite_batches:
                ran += len(batch)
                _, reason = run_batch(batch, timeout=args.job_timeout)
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
    gpu = args.gpu or os.environ.get("BENCH_GPU_NAME")
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
    parser.add_argument("--job-timeout", type=float, default=900,
                        help="Hard wall-clock cap per subprocess, in seconds")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
