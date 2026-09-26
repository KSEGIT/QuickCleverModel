#!/usr/bin/env python3
"""Turn a raw benchmark report directory into a run summary, and roll a
directory of run summaries into an index — both per contracts.md.

Stdlib only. Every read is tolerant of a missing or corrupt file: a suite
that never wrote its report simply stays out of the summary (or, if the
run's meta.json asked for it or errors.json explains why it is missing, it
appears with zero counts and an "error"), and a corrupt run-summary file is
skipped when building the index rather than failing the whole build.

    collect.py summarize RAW_DIR --out SUMMARY.json
    collect.py index RUNS_DIR --out INDEX.json
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys

SCHEMA = 1

# Suite keys as they appear in a run summary's "suites" map.
SUITE_KEYS = ("fixture", "long_context", "live_web",
              "concurrency_serial", "concurrency_parallel")

# meta.json's "suites" (requested) and errors.json's keys name the suite as
# run_suites.py sees it, where "concurrency" covers both concurrency phases.
_ALIASES = {"concurrency": ("concurrency_serial", "concurrency_parallel")}


def _expand_aliases(names):
    expanded = set()
    for name in names:
        expanded.update(_ALIASES.get(name, (name,)))
    return expanded


def load_json(path):
    """Parse path as JSON; return None if it is missing, unreadable, or not
    valid JSON. Never raises."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _natural_key(path):
    """Sort .../live_web-2.json before .../live_web-10.json."""
    match = re.search(r"(\d+)(?=\.json$)", path)
    return (int(match.group(1)) if match else -1, path)


def _glob_reports(dir_path, pattern):
    """Load every file matching pattern inside dir_path, in numeric-suffix
    order, dropping any file that is missing or fails to parse."""
    paths = sorted(glob.glob(os.path.join(dir_path, pattern)), key=_natural_key)
    reports = []
    for path in paths:
        data = load_json(path)
        if data is not None:
            reports.append(data)
    return reports


def median_or_none(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return statistics.median(values)


def pass_rate(passed, total):
    if not total:
        return 0.0
    return round(passed / total * 100, 1)


def _round_or_none(value, digits=1):
    return None if value is None else round(value, digits)


def _int_or_none(value):
    return None if value is None else int(value)


def _vram_peak(dir_path, *candidate_names):
    """peak_mib from the first vram-*.json in candidate_names that exists."""
    for name in candidate_names:
        data = load_json(os.path.join(dir_path, name))
        if data is not None:
            return _int_or_none(data.get("peak_mib"))
    return None


def _oracle_stats(results):
    """passed/total/median_seconds/peak_prompt_tokens/peak_vram_mib for a
    flat list of playwright_agent_bench.py-shaped result dicts (fixture.json
    and the concurrency-*.json reports share this shape): status is an exact
    "PASS"/"FAIL" oracle verdict, tokens live at turns[].usage.prompt_tokens.
    """
    passed = sum(1 for r in results if r.get("status") == "PASS")
    total = len(results)
    seconds = median_or_none([r.get("seconds") for r in results])
    tokens = [
        t.get("usage", {}).get("prompt_tokens")
        for r in results
        for t in r.get("turns", [])
    ]
    tokens = [t for t in tokens if t is not None]
    peak_tokens = max(tokens) if tokens else None
    vram_from_results = [r.get("peak_vram_mib") for r in results if r.get("peak_vram_mib") is not None]
    peak_vram = max(vram_from_results) if vram_from_results else None
    return passed, total, seconds, peak_tokens, peak_vram


def _live_web_stats(reports):
    """live_image_agent.py reports: status starts with "PASS" when an image
    was saved, tokens live flat at turns[].prompt_tokens."""
    passed = sum(1 for r in reports if str(r.get("status", "")).startswith("PASS"))
    total = len(reports)
    seconds = median_or_none([r.get("seconds") for r in reports])
    tokens = [
        t.get("prompt_tokens")
        for r in reports
        for t in r.get("turns", [])
    ]
    tokens = [t for t in tokens if t is not None]
    peak_tokens = max(tokens) if tokens else None
    return passed, total, seconds, peak_tokens


def _suite_dict(passed, total, seconds, peak_tokens, peak_vram, error=None):
    suite = {
        "passed": passed,
        "total": total,
        "pass_rate": pass_rate(passed, total),
        "median_seconds": _round_or_none(seconds),
        "peak_vram_mib": _int_or_none(peak_vram),
        "peak_prompt_tokens": _int_or_none(peak_tokens),
    }
    if error:
        suite["error"] = error
    return suite


def summarize(dir_path):
    """Build the run-summary dict for the raw report directory dir_path."""
    meta = load_json(os.path.join(dir_path, "meta.json")) or {}
    errors = load_json(os.path.join(dir_path, "errors.json")) or {}

    requested = _expand_aliases(meta.get("suites") or [])
    erroring = _expand_aliases(errors.keys())

    def error_for(suite_key, top_level_key):
        return errors.get(suite_key) or errors.get(top_level_key)

    suites = {}

    # fixture
    fixture_report = load_json(os.path.join(dir_path, "fixture.json"))
    fixture_present = fixture_report is not None
    if "fixture" in requested or fixture_present or "fixture" in erroring:
        if fixture_present:
            passed, total, seconds, peak_tokens, peak_vram = _oracle_stats(
                fixture_report.get("results", []))
        else:
            passed, total, seconds, peak_tokens, peak_vram = 0, 0, None, None, None
        if peak_vram is None:
            peak_vram = _vram_peak(dir_path, "vram-fixture.json")
        suites["fixture"] = _suite_dict(
            passed, total, seconds, peak_tokens, peak_vram,
            error=error_for("fixture", "fixture"))

    # long_context
    lc_report = load_json(os.path.join(dir_path, "long_context.json"))
    lc_present = lc_report is not None
    if "long_context" in requested or lc_present or "long_context" in erroring:
        if lc_present:
            passed = 1 if lc_report.get("status") == "PASS" else 0
            total = 1
            seconds = lc_report.get("seconds")
            peak_tokens = lc_report.get("prompt_tokens")
        else:
            passed, total, seconds, peak_tokens = 0, 0, None, None
        peak_vram = _vram_peak(dir_path, "vram-long_context.json")
        suites["long_context"] = _suite_dict(
            passed, total, seconds, peak_tokens, peak_vram,
            error=error_for("long_context", "long_context"))

    # live_web (one or more live_web-<n>.json repetitions)
    live_reports = _glob_reports(dir_path, "live_web-*.json")
    if "live_web" in requested or live_reports or "live_web" in erroring:
        passed, total, seconds, peak_tokens = _live_web_stats(live_reports)
        peak_vram = _vram_peak(dir_path, "vram-live_web.json")
        suites["live_web"] = _suite_dict(
            passed, total, seconds, peak_tokens, peak_vram,
            error=error_for("live_web", "live_web"))

    # concurrency: serial and parallel phases, each its own suite key
    for phase, pattern, vram_name in (
        ("concurrency_serial", "concurrency-serial-*.json", "vram-concurrency_serial.json"),
        ("concurrency_parallel", "concurrency-parallel-*.json", "vram-concurrency_parallel.json"),
    ):
        phase_reports = _glob_reports(dir_path, pattern)
        results = [r for report in phase_reports for r in report.get("results", [])]
        if phase in requested or phase_reports or phase in erroring:
            if phase_reports:
                passed, total, seconds, peak_tokens, peak_vram = _oracle_stats(results)
            else:
                passed, total, seconds, peak_tokens, peak_vram = 0, 0, None, None, None
            if peak_vram is None:
                peak_vram = _vram_peak(dir_path, vram_name, "vram-concurrency.json")
            suites[phase] = _suite_dict(
                passed, total, seconds, peak_tokens, peak_vram,
                error=error_for(phase, "concurrency"))

    return {
        "schema": SCHEMA,
        "run_id": meta.get("run_id"),
        "created_utc": meta.get("created_utc"),
        "commit": meta.get("commit"),
        "model": meta.get("model"),
        "ctx": meta.get("ctx"),
        "gpu": meta.get("gpu"),
        "suites": suites,
    }


def build_index(runs_dir):
    """Roll every *.json run-summary file in runs_dir into the index."""
    runs = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        data = load_json(path)
        if isinstance(data, dict) and "suites" in data:
            runs.append(data)
    runs.sort(key=lambda r: r.get("created_utc") or "")
    return {"schema": SCHEMA, "runs": runs}


def _write_json(obj, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="collect.py", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_summarize = sub.add_parser("summarize", help="summarize one raw report directory")
    p_summarize.add_argument("dir", help="raw report directory (run_suites.py --out-dir)")
    p_summarize.add_argument("--out", required=True, help="path to write the run summary JSON")

    p_index = sub.add_parser("index", help="rebuild the index from a directory of run summaries")
    p_index.add_argument("dir", help="directory of run-summary JSON files")
    p_index.add_argument("--out", required=True, help="path to write index.json")

    args = parser.parse_args(argv)

    if args.command == "summarize":
        _write_json(summarize(args.dir), args.out)
    elif args.command == "index":
        _write_json(build_index(args.dir), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
