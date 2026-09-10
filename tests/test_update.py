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
import shlex
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


class RollbackIsSticky(unittest.TestCase):
    """A rolled-back branch is a plain ancestor of origin, so the next run
    fast-forwards straight back onto the commit that just failed. Without a
    pin the box goes dark every week on the same commit, and the only signal
    is a line in the journal."""

    def _pinned_bad(self, state_body):
        with tempfile.TemporaryDirectory() as d:
            state = os.path.join(d, "update-state")
            with open(state, "w") as fh:
                fh.write(state_body)
            script = (
                'set -uo pipefail; export BONSAI_UPDATE_LIB=1; '
                f'source "{UPDATE_SH}"; STATE="{state}"; pinned_bad'
            )
            r = subprocess.run(["bash", "-c", script], capture_output=True,
                               text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            return r.stdout.strip()

    def test_reads_the_pinned_commit(self):
        self.assertEqual(
            self._pinned_bad("sha=aaa\nwebui=bbb\nverified=1\nbad=deadbeef\n"),
            "deadbeef")

    def test_is_empty_when_nothing_is_pinned(self):
        self.assertEqual(self._pinned_bad("sha=aaa\nwebui=bbb\nverified=1\n"), "")

    def test_the_newest_pin_wins(self):
        """do_rollback appends, so an old pin must not shadow a newer one."""
        self.assertEqual(
            self._pinned_bad("sha=aaa\nbad=oldbad\nbad=newbad\n"), "newbad")

    def test_update_refuses_a_commit_that_already_failed(self):
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertIn('bad="$(pinned_bad)"', body,
                      "cmd_update must consult the pin before pulling")
        self.assertIn('already failed its smoke test', body,
                      "and must say why it is refusing")


class RollbackReportsHonestly(unittest.TestCase):
    def test_the_reset_is_not_silenced(self):
        """`git reset --hard ... >/dev/null 2>&1` discards the exit status, so
        a reset onto a commit that no longer exists left the broken code
        checked out while the run still logged success."""
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertNotIn('reset --hard "$sha" >/dev/null 2>&1', body)
        self.assertIn("STILL checked out", body,
                      "a failed restore has to be stated, not swallowed")

    def test_answering_is_not_reported_as_restored(self):
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertIn("was NOT fully restored", body,
                      "a stack that answers with the bad code still checked "
                      "out is not a successful rollback")


class CheckActuallyCompares(unittest.TestCase):
    def test_manifest_bytes_are_not_used_as_a_digest(self):
        """The old code took `docker manifest inspect | head -c 40`, which is
        the first 40 bytes of the manifest JSON, and never compared it to
        anything. --check reported "up to date" on a six-month-old image."""
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertNotIn("head -c 40", body)
        self.assertIn("RepoDigests", body,
                      "the local side must be a repo digest, not .Id, which "
                      "is a local config hash that matches nothing upstream")

    def test_no_dependency_on_the_buildx_plugin(self):
        """buildx is a CLI plugin the distro docker.io package does not ship —
        confirmed absent on the box this runs on, where `docker buildx` is an
        unknown command. Depending on it made --check permanently 'unknown'."""
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertNotIn("buildx imagetools inspect \"$1\"", body)

    def test_unknown_drift_is_not_reported_as_up_to_date(self):
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertIn("image drift unknown", body)


class PreflightChecksEveryTool(unittest.TestCase):
    def test_the_smoke_paths_interpreter_is_required(self):
        """smoke_ok and parse_model_ids are python3. A missing interpreter
        exits 127, which reads exactly like a dead model — so it would roll
        back a healthy stack and then fail its own rollback the same way."""
        with open(UPDATE_SH) as fh:
            body = fh.read()
        line = [l for l in body.splitlines() if l.strip().startswith("for tool in")]
        self.assertEqual(len(line), 1, "expected one required-tool list")
        for tool in ("docker", "git", "curl", "python3"):
            self.assertIn(tool, line[0], f"{tool} is used but never checked")


class SmokeHasThreeOutcomes(unittest.TestCase):
    """smoke() decides whether a stack gets torn down, so "the model is dead"
    and "I could not ask" must not collapse into one answer.

    Drives the real smoke() with a stubbed ask(), so no server is needed.
    """

    @staticmethod
    def _ask_stub(models=(200, MODELS_LIST), chat=(200, GOOD_CONTENT)):
        mcode, mbody = models
        ccode, cbody = chat
        return (
            'ask() { case "$1" in '
            f'*models) printf "{mcode}\\n%s" {shlex.quote(mbody)};; '
            f'*) printf "{ccode}\\n%s" {shlex.quote(cbody)};; '
            'esac; }'
        )

    def _smoke(self, stub):
        script = (
            'set -uo pipefail; export BONSAI_UPDATE_LIB=1; '
            f'source "{UPDATE_SH}"; API=http://stub; BONSAI_API_KEY=k; '
            f'{stub}; smoke; echo "rc=$?"'
        )
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        return r.stdout + r.stderr

    def test_every_model_answering_is_success(self):
        self.assertIn("rc=0", self._smoke(self._ask_stub()))

    def test_a_dead_model_is_a_real_failure(self):
        """A 500 'failed to load' is the outage. rc=1 means roll back."""
        out = self._smoke(self._ask_stub(chat=(500, FAILED_LOAD)))
        self.assertIn("rc=1", out)
        self.assertIn("failed to load", out,
                      "the error body is the whole diagnostic — curl -f used "
                      "to throw it away and log an empty reason")

    def test_a_rotated_key_is_not_a_dead_model(self):
        """401 means our credentials are stale, not that the stack is broken.
        Rolling back here would turn our own problem into a real outage."""
        out = self._smoke(self._ask_stub(models=(401, '{"error":"unauthorized"}')))
        self.assertIn("rc=2", out)

    def test_a_401_on_one_model_still_stops_the_run(self):
        out = self._smoke(self._ask_stub(chat=(401, '{"error":"nope"}')))
        self.assertIn("rc=2", out)

    def test_curl_failing_outright_is_not_a_dead_model(self):
        self.assertIn("rc=2", self._smoke("ask() { return 7; }"))

    def test_a_reasoning_only_reply_passes_end_to_end(self):
        """The thinking-model fixture, through the real smoke() this time."""
        out = self._smoke(self._ask_stub(chat=(200, GOOD_REASONING_ONLY)))
        self.assertIn("rc=0", out)

    def test_ask_does_not_use_curl_f(self):
        """curl -f prints nothing on HTTP >= 400 and returns 22, so the error
        body — the only explanation the operator ever gets — is discarded."""
        with open(UPDATE_SH) as fh:
            body = fh.read()
        fn = body[body.index("ask() {"):body.index("smoke() {")]
        self.assertNotIn(" -f", fn, "curl -f would discard the error body")


class PinningIsNarrow(unittest.TestCase):
    """The refusal message says a commit 'already failed its smoke test'. That
    has to be true, so only the smoke-failure path may pin."""

    def test_only_the_smoke_failure_path_pins(self):
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertEqual(body.count("do_rollback pin"), 1,
                         "exactly one call site may pin a commit")
        # build / compose up / health failures are infrastructure, not proof
        # of a bad commit — compose.linux.yaml warns that a bind can lose a
        # race with tailscaled at boot.
        for site in ('warn "build failed"; do_rollback;',
                     'warn "compose up failed"; do_rollback;',
                     'warn "never became healthy"; do_rollback;'):
            self.assertIn(site, body, f"{site!r} must roll back WITHOUT pinning")

    def test_manual_rollback_does_not_pin(self):
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertIn("--rollback) preflight; do_rollback;;", body,
                      "a deliberate rollback must not blacklist the commit")


class SignalHandling(unittest.TestCase):
    def test_the_signal_trap_exits(self):
        """bash RESUMES after a non-EXIT trap handler returns. A handler that
        only cleaned up would let the run keep pulling and building past
        systemd's TimeoutStartSec, until SIGKILL."""
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertIn("exit 143", body)
        self.assertRegex(body, r"trap '[^']*exit 143' INT TERM")

    def test_exec_failure_has_a_fallback(self):
        """/tmp mounted noexec is ordinary hardening; bash can still read a
        script it may not execute."""
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertIn("shopt -s execfail", body)
        self.assertIn('exec bash "$copy" "$@"', body)


class UsageContract(unittest.TestCase):
    def test_the_three_documented_modes_are_in_the_header(self):
        with open(UPDATE_SH) as fh:
            head = "".join(fh.readlines()[:20])
        for flag in ("--check", "--rollback"):
            self.assertIn(flag, head,
                          f"{flag} is part of the interface and must be in "
                          "the header the usage message prints")

    def test_help_is_a_complete_message(self):
        """`sed -n '2,10p'` printed through the first line of the rationale
        paragraph, so --help ended on a dangling fragment, and any edit to the
        header block silently changed the output."""
        r = subprocess.run(["bash", UPDATE_SH, "--help"],
                           capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("Why this smoke-tests", r.stdout,
                         "--help is spilling into the rationale comment")
        self.assertTrue(r.stdout.rstrip().endswith("see stack.sh."),
                        f"--help ends mid-sentence: {r.stdout.rstrip()[-60:]!r}")
        for mode in ("--check", "--rollback"):
            self.assertIn(mode, r.stdout)

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
        # Spelled out, not `weekly`: systemd normalizes `weekly` to Mon
        # 00:00, so with RandomizedDelaySec the real window was Mon
        # 00:00-04:00 while the comment claimed 04:00-08:00.
        self.assertIn("OnCalendar=Mon *-*-* 04:00:00", body)
        self.assertNotIn("OnCalendar=weekly", body)
        self.assertIn("RandomizedDelaySec", body,
                      "without a randomized delay every box that ever copies "
                      "this hits GHCR at the same second")
        self.assertIn("Persistent=true", body,
                      "a box that was off on the scheduled day should still "
                      "update when it comes back")

    def test_service_has_no_install_section(self):
        """A timer-activated oneshot with [Install] can be enabled on its own,
        and then it runs a full pull, rebuild and smoke sweep on every boot —
        while the GPU stack is still coming up."""
        with open(self.SERVICE) as fh:
            lines = [l.strip() for l in fh]
        self.assertNotIn("[Install]", lines,
                         "enable the timer, not the service")

    def test_timer_is_wanted_by_timers_target(self):
        with open(self.TIMER) as fh:
            body = fh.read()
        self.assertIn("WantedBy=timers.target", body,
                      "without an [Install] section `systemctl enable` fails")
