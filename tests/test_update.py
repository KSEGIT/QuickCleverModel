"""Tests for update.sh — the Linux/Docker updater and its smoke test.

stdlib unittest, not pytest — see tests/test_cache_viz.py for the convention.

No test here starts the stack, pulls an image or talks to Docker: the suite
runs on a bare CI runner with no weights and no GPU. These assert the shape
of the contract instead, by extracting a shell function out of update.sh and
running it in isolation — the same technique tests/test_launchd_path.py uses
on stack.sh, so the real code is exercised rather than a paraphrase of it.

Every JSON fixture below is a VERBATIM response captured from the box
(firesand-worker) on 2026-09-10, during and after the stale-bind-mount outage
that motivated this script. They are not invented.
"""
import json
import os
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPDATE_SH = os.path.join(ROOT, "update.sh")

# --- verbatim captures from firesand-worker, 2026-09-10 ----------------------

# What a model load looked like while /app/models was an empty directory.
# Note the container was "Up 29 hours (healthy)" and /health said
# {"status":"ok"} the whole time this was happening.
FAILED_LOAD = (
    '{"error":{"code":500,"message":"model name=bonsai-27b-1bit failed to '
    'load","type":"server_error"}}'
)

# A GOOD reply from bonsai-27b-ternary after the mount was fixed. This is the
# fixture that matters most: "content" is the empty string and the actual text
# is in "reasoning_content", because Bonsai is a thinking model and this reply
# hit max_tokens while still inside its reasoning block. A smoke test that
# only looks at "content" scores this healthy model as broken and triggers a
# needless rollback.
GOOD_REASONING_ONLY = json.dumps({
    "choices": [{
        "finish_reason": "length",
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Here's a thinking process:\n\n1. **Analyze"
                                 " User Request:**\n   - **Topic:** Roman"
                                 " Empire",
        },
    }],
    "model": "bonsai-27b-ternary",
    "object": "chat.completion",
})

# An ordinary reply that does fill "content".
GOOD_CONTENT = json.dumps({
    "choices": [{
        "finish_reason": "stop",
        "index": 0,
        "message": {"role": "assistant", "content": "OK"},
    }],
    "model": "bonsai-27b-1bit",
    "object": "chat.completion",
})

# Trimmed from the real /v1/models on the box: three aliases, one router.
MODELS_LIST = json.dumps({
    "data": [
        {"id": "bonsai-27b-1bit", "object": "model", "owned_by": "llamacpp"},
        {"id": "bonsai-27b-ternary", "object": "model", "owned_by": "llamacpp"},
        {"id": "bonsai-27b-ternary-text", "object": "model",
         "owned_by": "llamacpp"},
    ]
})


def run_fn(fn_name, stdin="", args=()):
    """Extract one function from update.sh and run only that function.

    update.sh is sourced with BONSAI_UPDATE_LIB=1, which makes it define its
    functions and return instead of executing an update. That keeps the test
    honest (it runs the shipped code) without needing Docker on the runner.
    """
    quoted = " ".join(f"'{a}'" for a in args)
    script = (
        f'set -uo pipefail; export BONSAI_UPDATE_LIB=1; '
        f'source "{UPDATE_SH}"; {fn_name} {quoted}'
    )
    r = subprocess.run(["bash", "-c", script], input=stdin,
                       capture_output=True, text=True)
    # 127 means the function was never defined — update.sh is missing or failed
    # to source. Without this guard every "must fail" assertion below would
    # pass for the wrong reason and the suite would go green on no script.
    if r.returncode == 127:
        raise AssertionError(f"{fn_name} not defined by update.sh: {r.stderr}")
    return r


class SmokeTestContract(unittest.TestCase):
    """smoke_ok decides whether the stack gets rolled back. Both directions
    of a wrong answer are expensive: a false pass ships a dark stack, a false
    fail rolls back a working one."""

    def test_failed_load_is_not_ok(self):
        r = run_fn("smoke_ok", stdin=FAILED_LOAD)
        self.assertNotEqual(r.returncode, 0,
                            "a 500 'failed to load' must fail the smoke test")

    def test_reasoning_only_reply_is_ok(self):
        r = run_fn("smoke_ok", stdin=GOOD_REASONING_ONLY)
        self.assertEqual(r.returncode, 0,
                         "a reply with only reasoning_content is a real "
                         "generation and must pass, or every thinking model "
                         "triggers a spurious rollback")

    def test_ordinary_reply_is_ok(self):
        r = run_fn("smoke_ok", stdin=GOOD_CONTENT)
        self.assertEqual(r.returncode, 0)

    def test_empty_reply_is_not_ok(self):
        r = run_fn("smoke_ok", stdin='{"choices":[]}')
        self.assertNotEqual(r.returncode, 0)

    def test_garbage_is_not_ok(self):
        """curl writes nothing on a connection failure; that must not pass."""
        r = run_fn("smoke_ok", stdin="")
        self.assertNotEqual(r.returncode, 0)


class ModelListParsing(unittest.TestCase):
    def test_every_model_id_is_listed(self):
        r = run_fn("parse_model_ids", stdin=MODELS_LIST)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.split(),
                         ["bonsai-27b-1bit", "bonsai-27b-ternary",
                          "bonsai-27b-ternary-text"],
                         "the smoke test loads every model, so every alias "
                         "must come out of the list")


class PreflightGuardsTheOutage(unittest.TestCase):
    """The regression test for 2026-09-10.

    The repo was moved with `mv QuickCleverModel/ ../firesandbrain1/` while
    the stack ran. Docker recreated the now-missing bind-mount source as an
    empty root-owned directory, so /app/models held no weights and every
    model load failed for 29 hours. An updater that restarts into that state
    turns one outage into a rollback loop, so it must refuse to start.
    """

    def test_empty_models_dir_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r = run_fn("models_present", args=(d,))
            self.assertNotEqual(r.returncode, 0,
                                "an empty models dir is exactly the outage; "
                                "preflight must reject it")

    def test_missing_models_dir_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r = run_fn("models_present", args=(os.path.join(d, "nope"),))
            self.assertNotEqual(r.returncode, 0)

    def test_directory_of_weights_is_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "Ternary-Bonsai-27B-gguf")
            os.makedirs(sub)
            open(os.path.join(sub, "Ternary-Bonsai-27B-Q2_0.gguf"), "w").close()
            r = run_fn("models_present", args=(d,))
            self.assertEqual(r.returncode, 0, r.stderr)

    def test_dir_with_only_a_cache_subdir_is_rejected(self):
        """fetch-models.sh leaves a .cache/ behind. A models dir holding only
        that is still a dir with no weights in it."""
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "Bonsai-27B-gguf", ".cache"))
            r = run_fn("models_present", args=(d,))
            self.assertNotEqual(r.returncode, 0)


class RunsFromACopy(unittest.TestCase):
    """update.sh rewrites its own file during `git pull` and `git reset --hard`.

    Bash reads a script lazily, by byte offset. If the file changes while it
    runs, bash resumes at a stale offset and executes whatever bytes now sit
    there. The damage lands hardest on rollback — the one path whose job is to
    rescue a broken stack. So the script copies itself to a temp file and
    exec's that, leaving bash reading a file no git command will touch.
    """

    def test_root_follows_the_reexec_override(self):
        """After re-exec the script lives in a temp dir, so dirname($0) no
        longer finds the repo. ROOT must come from the exported override or
        every path in the script points at /tmp."""
        script = (
            'set -uo pipefail; export BONSAI_UPDATE_LIB=1; '
            'export BONSAI_UPDATE_ROOT=/tmp/pretend-repo; '
            f'source "{UPDATE_SH}"; echo "$ROOT"'
        )
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "/tmp/pretend-repo", r.stderr)

    def test_root_defaults_to_the_script_directory(self):
        script = (
            'set -uo pipefail; export BONSAI_UPDATE_LIB=1; '
            f'source "{UPDATE_SH}"; echo "$ROOT"'
        )
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), ROOT, r.stderr)

    def test_the_copy_deletes_itself(self):
        """A weekly timer that leaks a file per run fills /tmp slowly enough
        that nobody connects the two."""
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, TMPDIR=tmp)
            r = subprocess.run(["bash", UPDATE_SH, "--help"],
                               capture_output=True, text=True, cwd=ROOT, env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            leftovers = [f for f in os.listdir(tmp) if f.startswith("bonsai-update.")]
            self.assertEqual(leftovers, [],
                             f"the re-exec copy was left behind: {leftovers}")

    def test_help_still_works_through_the_copy(self):
        r = subprocess.run(["bash", UPDATE_SH, "--help"],
                           capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--rollback", r.stdout,
                      "usage must survive being printed from the temp copy")


class UsageContract(unittest.TestCase):
    def test_the_three_documented_modes_are_in_the_header(self):
        with open(UPDATE_SH) as fh:
            head = "".join(fh.readlines()[:20])
        for flag in ("--check", "--rollback"):
            self.assertIn(flag, head,
                          f"{flag} is part of the interface and must be in "
                          "the header the usage message prints")

    def test_unknown_flag_exits_nonzero(self):
        r = subprocess.run(["bash", UPDATE_SH, "--wat"],
                           capture_output=True, text=True, cwd=ROOT)
        self.assertNotEqual(r.returncode, 0,
                            "an unknown flag must not be treated as a request "
                            "to update")


class SystemdUnits(unittest.TestCase):
    """The units ship in the repo so the box installs a reviewed file rather
    than one pasted together at the terminal."""

    SERVICE = os.path.join(ROOT, "docker", "bonsai-update.service")
    TIMER = os.path.join(ROOT, "docker", "bonsai-update.timer")

    def test_units_exist(self):
        for p in (self.SERVICE, self.TIMER):
            self.assertTrue(os.path.isfile(p), f"{p} is missing")

    def test_service_is_oneshot_and_runs_update_sh(self):
        with open(self.SERVICE) as fh:
            body = fh.read()
        self.assertIn("Type=oneshot", body)
        self.assertIn("update.sh", body)

    def test_timer_is_weekly_and_spreads_the_load(self):
        with open(self.TIMER) as fh:
            body = fh.read()
        self.assertIn("OnCalendar=weekly", body)
        self.assertIn("RandomizedDelaySec", body,
                      "without a randomized delay every box that ever copies "
                      "this hits GHCR at the same second")
        self.assertIn("Persistent=true", body,
                      "a box that was off on the scheduled day should still "
                      "update when it comes back")

    def test_timer_is_wanted_by_timers_target(self):
        with open(self.TIMER) as fh:
            body = fh.read()
        self.assertIn("WantedBy=timers.target", body,
                      "without an [Install] section `systemctl enable` fails")
