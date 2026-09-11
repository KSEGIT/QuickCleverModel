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
import shutil
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

MODELS_LIST_ONE = json.dumps({"data": [{"id": "bonsai-27b-1bit"}]})

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
        # The override is trusted only from something that really looks like
        # one of our copies: the variable must name the running file AND that
        # file must carry mktemp's bonsai-update.XXXXXX name. Naming the
        # repo's own update.sh must not qualify — that is what let the script
        # delete itself.
        with tempfile.TemporaryDirectory() as d:
            copy = os.path.join(d, "bonsai-update.ABC123")
            shutil.copy(UPDATE_SH, copy)
            script = (
                'set -uo pipefail; export BONSAI_UPDATE_LIB=1; '
                'export BONSAI_UPDATE_ROOT=/tmp/pretend-repo; '
                f'export BONSAI_UPDATE_REEXEC="{copy}"; '
                f'source "{copy}"; echo "$ROOT"'
            )
            r = subprocess.run(["bash", "-c", script], capture_output=True,
                               text=True)
            self.assertEqual(r.stdout.strip(), "/tmp/pretend-repo", r.stderr)

    def test_the_repo_script_never_counts_as_a_copy(self):
        """Naming update.sh itself in the variable must not grant it copy
        status — that combination skipped the self-copy AND armed the delete
        trap, and removed the real script from disk."""
        script = (
            'set -uo pipefail; export BONSAI_UPDATE_LIB=1; '
            'export BONSAI_UPDATE_ROOT=/tmp/pretend-repo; '
            f'export BONSAI_UPDATE_REEXEC="{UPDATE_SH}"; '
            f'source "{UPDATE_SH}"; echo "$ROOT"'
        )
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), ROOT, r.stderr)

    def test_a_root_override_without_a_matching_reexec_is_ignored(self):
        """A leaked BONSAI_UPDATE_ROOT alone must not redirect the script."""
        script = (
            'set -uo pipefail; export BONSAI_UPDATE_LIB=1; '
            'export BONSAI_UPDATE_ROOT=/tmp/pretend-repo; '
            f'source "{UPDATE_SH}"; echo "$ROOT"'
        )
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), ROOT, r.stderr)

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
        # ask() sets globals rather than printing: re-printing "code\\nbody"
        # loses the separator when the body is empty, and the body silently
        # became the status code.
        return (
            'ask() { case "$1" in '
            f'*models) ASK_CODE={mcode}; ASK_BODY={shlex.quote(mbody)};; '
            f'*) ASK_CODE={ccode}; ASK_BODY={shlex.quote(cbody)};; '
            'esac; return 0; }'
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

    def test_an_empty_error_body_is_not_reported_as_the_status_code(self):
        """A 500 with no body used to log "FAILED (HTTP 500) — 500": command
        substitution stripped the trailing newline, so the caller's cut found
        no separator and handed back the code as the body."""
        out = self._smoke(self._ask_stub(chat=(500, "")))
        self.assertIn("rc=1", out)
        self.assertIn("<empty body>", out)
        self.assertNotIn("— 500", out)

    def test_ask_parses_curls_own_layout(self):
        """curl -w appends the code AFTER the body, which survives an empty
        body and a multi-line body alike."""
        script = (
            'set -uo pipefail; export BONSAI_UPDATE_LIB=1; '
            f'source "{UPDATE_SH}"; '
            'out="$(printf \'a\\nb\\n500\')"; '
            'echo "code=[${out##*$\'\\n\'}] body=[${out%$\'\\n\'*}]"'
        )
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertIn("code=[500]", r.stdout)
        self.assertIn("body=[a\nb]", r.stdout)

    def test_ask_does_not_use_curl_f(self):
        """curl -f prints nothing on HTTP >= 400 and returns 22, so the error
        body — the only explanation the operator ever gets — is discarded."""
        with open(UPDATE_SH) as fh:
            body = fh.read()
        fn = body[body.index("ask() {"):body.index("smoke() {")]
        self.assertNotIn(" -f", fn, "curl -f would discard the error body")


class NeverDeletesTheRealScript(unittest.TestCase):
    """The delete-on-exit trap must only ever fire for the temp copy.

    The trap used to arm on a bare BONSAI_UPDATE_REEXEC flag, so anything
    that put that name in the environment — a hand-export, a stale value
    inherited from a parent, or falling through a failed re-exec — armed it
    while BASH_SOURCE[0] was still the repo's own update.sh. Exiting then
    deleted the real script off disk. Reproduced before the fix: after one
    `BONSAI_UPDATE_REEXEC=1 bash update.sh --help`, the file was gone.
    """

    def test_a_stale_reexec_variable_does_not_delete_the_script(self):
        with tempfile.TemporaryDirectory() as d:
            copy = os.path.join(d, "update.sh")
            shutil.copy(UPDATE_SH, copy)
            os.chmod(copy, 0o755)
            env = dict(os.environ, BONSAI_UPDATE_REEXEC="1")
            subprocess.run(["bash", copy, "--help"], capture_output=True,
                           text=True, env=env, cwd=d)
            self.assertTrue(os.path.isfile(copy),
                            "update.sh deleted itself — the EXIT trap fired "
                            "for the repo's own file, not the temp copy")

    def test_a_mismatched_reexec_path_does_not_delete_the_script(self):
        """The variable now carries the copy's path; a value naming some other
        file must not license deleting the one that is running."""
        with tempfile.TemporaryDirectory() as d:
            copy = os.path.join(d, "update.sh")
            shutil.copy(UPDATE_SH, copy)
            os.chmod(copy, 0o755)
            env = dict(os.environ,
                       BONSAI_UPDATE_REEXEC=os.path.join(d, "somewhere-else"))
            subprocess.run(["bash", copy, "--help"], capture_output=True,
                           text=True, env=env, cwd=d)
            self.assertTrue(os.path.isfile(copy))

    def test_the_normal_run_still_cleans_up_its_copy(self):
        """The guard must not be so tight that the real copy leaks."""
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, TMPDIR=tmp)
            env.pop("BONSAI_UPDATE_REEXEC", None)
            r = subprocess.run(["bash", UPDATE_SH, "--help"],
                               capture_output=True, text=True, cwd=ROOT, env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(
                [f for f in os.listdir(tmp) if f.startswith("bonsai-update.")], [])

    def test_a_failed_reexec_does_not_fall_through(self):
        """Falling out of the re-exec block would run on from the very file
        git is about to rewrite — what the copy exists to prevent."""
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertIn("could not run the copy at", body,
                      "the fallback must die, not continue")


class LeakedReexecStillCopies(unittest.TestCase):
    """Keying the guard on the mere PRESENCE of BONSAI_UPDATE_REEXEC meant any
    leaked value — a systemd Environment= line, an inherited export — skipped
    the self-copy, and `git pull`/`git reset --hard` then rewrote the very
    file bash was reading by byte offset.

    Observed through a read-only TMPDIR: if the script still tries to copy
    itself it dies in mktemp, which a presence-based guard would have skipped
    entirely.
    """

    def _run(self, reexec):
        with tempfile.TemporaryDirectory() as d:
            ro = os.path.join(d, "ro")
            os.makedirs(ro)
            os.chmod(ro, 0o500)
            try:
                env = dict(os.environ, TMPDIR=ro)
                if reexec is None:
                    env.pop("BONSAI_UPDATE_REEXEC", None)
                else:
                    env["BONSAI_UPDATE_REEXEC"] = reexec
                return subprocess.run(["bash", UPDATE_SH, "--help"],
                                      capture_output=True, text=True,
                                      cwd=ROOT, env=env)
            finally:
                os.chmod(ro, 0o700)

    def test_a_leaked_flag_value_does_not_skip_the_copy(self):
        r = self._run("1")
        self.assertNotEqual(r.returncode, 0,
                            "a stale value skipped the self-copy entirely")
        self.assertIn("mktemp failed", r.stderr)

    def test_a_mismatched_path_does_not_skip_the_copy(self):
        r = self._run("/some/other/update.sh")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("mktemp failed", r.stderr)

    def test_pointing_the_variable_at_the_repo_script_does_not_disarm_it(self):
        """Setting the variable to the repo's OWN path used to satisfy the
        path comparison, so the script treated the real update.sh as its temp
        copy: it skipped the self-copy AND armed the delete trap. Writing this
        very test deleted update.sh from the worktree. The name pattern is
        what closes it — mktemp copies are bonsai-update.XXXXXX."""
        r = self._run(UPDATE_SH)
        self.assertTrue(os.path.isfile(UPDATE_SH),
                        "update.sh deleted itself")
        self.assertNotEqual(r.returncode, 0,
                            "the repo script must not pass as a temp copy")
        self.assertIn("mktemp failed", r.stderr)


class ReadOnlyCommandsSurviveADarkStack(unittest.TestCase):
    """--check and --rollback are what you reach for when the stack is down,
    so the weights gate must not block them."""

    def setUp(self):
        with open(UPDATE_SH) as fh:
            self.body = fh.read()

    def test_check_returns_before_the_weights_gate(self):
        self.assertIn('[[ "$mode" == "check" ]] && return 0', self.body)

    def test_rollback_skips_the_weights_gate(self):
        self.assertIn('[[ "$mode" == "rollback" ]] || models_present', self.body,
                      "a rollback restores code and images, not weights")

    def test_each_entry_point_declares_its_mode(self):
        for site in ("preflight check", "preflight update", "preflight rollback"):
            self.assertIn(site, self.body)


class StateSurvivesAFailedWrite(unittest.TestCase):
    def test_the_state_file_is_replaced_not_truncated(self):
        """`> "$STATE"` truncates when the redirect is set up, i.e. before
        printf can fail — so a full disk destroyed the only way back and then
        announced it was refusing to proceed without one."""
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertIn('mv -f "$new" "$STATE"', body)
        self.assertNotIn('"$verified" > "$STATE"', body)


class RollbackVerdictIsThreeWay(unittest.TestCase):
    def test_could_not_ask_is_not_a_failed_rollback(self):
        """Every restore step succeeding but a rotated key blocking the probe
        printed "ROLLBACK DID NOT RECOVER THE STACK" for a rollback that
        worked."""
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertIn("could not verify it serves", body)
        section = body[body.index("do_rollback() {"):body.index("cmd_check() {")]
        self.assertIn("case \"$verdict\" in", section)


class GuardsDoNotDependOnTheProjectName(unittest.TestCase):
    def test_the_running_stack_is_found_by_image(self):
        """COMPOSE_PROJECT_NAME in .env renames the container, and a
        name-based lookup then reads as "nothing is running" — letting the
        update restart into the stale bind mount."""
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertIn('docker ps --filter "ancestor=$LLAMA_IMAGE"', body)


class PullCannotOutrunThePin(unittest.TestCase):
    def test_it_merges_the_checked_commit(self):
        """`git pull` runs its own fetch, which can land on a commit newer
        than the one the pin was checked against — including the pinned bad
        one."""
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertIn('git -C "$ROOT" merge --ff-only "$target"', body)
        self.assertNotIn('git -C "$ROOT" pull --ff-only', body,
                         "pull runs its own fetch and can outrun the pin")


class WeightsMayBeSymlinked(unittest.TestCase):
    """Weights routinely live on a second drive with models/ symlinked in.
    Docker resolves that bind mount fine, so a find that refuses to follow
    symlinks reports "no weights" for a stack that works — and preflight then
    blocks update, --check and --rollback alike."""

    def _models_present(self, path):
        script = (
            'set -uo pipefail; export BONSAI_UPDATE_LIB=1; '
            f'source "{UPDATE_SH}"; models_present {shlex.quote(path)}'
        )
        return subprocess.run(["bash", "-c", script], capture_output=True,
                              text=True).returncode

    def test_a_symlinked_models_dir_is_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            real = os.path.join(d, "on-the-big-disk", "Bonsai-27B-gguf")
            os.makedirs(real)
            open(os.path.join(real, "Bonsai-27B-Q1_0.gguf"), "w").close()
            link = os.path.join(d, "models")
            os.symlink(os.path.join(d, "on-the-big-disk"), link)
            self.assertEqual(self._models_present(link), 0,
                             "a symlinked models dir must still count")

    def test_a_symlinked_model_subdir_is_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            models = os.path.join(d, "models")
            os.makedirs(models)
            real = os.path.join(d, "elsewhere")
            os.makedirs(real)
            open(os.path.join(real, "Ternary-Bonsai-27B-Q2_0.gguf"), "w").close()
            os.symlink(real, os.path.join(models, "Ternary-Bonsai-27B-gguf"))
            self.assertEqual(self._models_present(models), 0)

    def test_an_empty_symlink_target_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "empty"))
            link = os.path.join(d, "models")
            os.symlink(os.path.join(d, "empty"), link)
            self.assertNotEqual(self._models_present(link), 0)


class RestoreIsAllOrNothing(unittest.TestCase):
    """A rollback that only half happened must not log "rollback verified"."""

    def setUp(self):
        with open(UPDATE_SH) as fh:
            self.body = fh.read()

    def test_an_unrecorded_llama_image_blocks_the_restore(self):
        self.assertIn("no llama image id recorded", self.body,
                      "an empty recorded id used to skip the guard entirely "
                      "and tag whatever an earlier run left under :rollback")

    def test_a_pruned_webui_image_is_not_silently_skipped(self):
        self.assertIn("is gone (pruned?)", self.body)

    def test_every_restore_failure_clears_restored(self):
        """Each warn in the restore section must also drop `restored`."""
        start = self.body.index("do_rollback() {")
        end = self.body.index("cmd_check() {")
        section = self.body[start:end]
        self.assertGreaterEqual(section.count("restored=0"), 6,
                                "a restore step warns without clearing the flag")


class CheckNeverGuesses(unittest.TestCase):
    def test_an_unresolvable_upstream_is_not_up_to_date(self):
        """`rev-list ... || echo 0` reported "code up to date" on a detached
        HEAD or a branch with no tracking branch."""
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertNotIn("rev-list --count 'HEAD..@{u}' 2>/dev/null || echo 0", body)
        self.assertIn("no upstream for HEAD", body)


class BusyIsNotBroken(unittest.TestCase):
    """The router holds ONE model at a time (--models-max defaults to 1), so a
    request for a second alias while the first is busy is refused. Observed on
    the box by firing two aliases at once:

        500 {"error":{"code":500,"message":"model limit reached, try again later"}}

    It is HTTP 500, not 503, so a status-code allowlist alone misses it. Left
    unhandled, a person chatting while the weekly timer runs makes smoke score
    a healthy model as dead — which rolls the stack back AND pins a good
    commit, blocking every future update.
    """

    # Verbatim from firesand-worker, 2026-09-11.
    BUSY = ('{"error":{"code":500,"message":"model limit reached, try again '
            'later","type":"server_error"}}')

    def _smoke(self, chat_bodies):
        """chat_bodies: list of (code, body) handed out one call at a time."""
        cases = "".join(
            f'{i}) ASK_CODE={c}; ASK_BODY={shlex.quote(b)};; '
            for i, (c, b) in enumerate(chat_bodies, start=1))
        stub = (
            'N=0; ask() { case "$1" in '
            f'*models) ASK_CODE=200; ASK_BODY={shlex.quote(MODELS_LIST_ONE)};; '
            f'*) N=$((N+1)); case "$N" in {cases} esac;; '
            'esac; return 0; }'
        )
        script = (
            'set -uo pipefail; export BONSAI_UPDATE_LIB=1; '
            'export BONSAI_SMOKE_RETRY_SLEEP=0; '
            f'source "{UPDATE_SH}"; API=http://stub; BONSAI_API_KEY=k; '
            f'{stub}; smoke; echo "rc=$?"'
        )
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        return r.stdout + r.stderr

    def test_a_busy_router_is_retried_and_then_passes(self):
        out = self._smoke([(500, self.BUSY), (200, GOOD_CONTENT)])
        self.assertIn("rc=0", out, out)
        self.assertIn("is busy", out, "the retry should be announced")

    def test_a_persistently_busy_router_cannot_be_judged(self):
        """rc=2, never rc=1: busy says nothing about the commit, and rc=1
        would roll back and pin it."""
        out = self._smoke([(500, self.BUSY)] * 3)
        self.assertIn("rc=2", out, out)
        self.assertNotIn("rc=1", out)

    def test_a_loading_model_is_not_a_dead_model(self):
        loading = '{"error":{"code":503,"message":"Loading model","type":"unavailable_error"}}'
        out = self._smoke([(503, loading), (200, GOOD_CONTENT)])
        self.assertIn("rc=0", out, out)

    def test_a_real_load_failure_is_still_a_failure(self):
        """The outage this script exists for must still score as rc=1."""
        out = self._smoke([(500, FAILED_LOAD)] * 3)
        self.assertIn("rc=1", out, out)
        self.assertIn("failed to load", out)


class AnyPinCounts(unittest.TestCase):
    def _is_pinned(self, state_body, target):
        with tempfile.TemporaryDirectory() as d:
            state = os.path.join(d, "s")
            with open(state, "w") as fh:
                fh.write(state_body)
            r = run_real(f'STATE={shlex.quote(state)}', f'is_pinned_bad {target}')
            return "rc=0" in r.stdout

    def test_an_older_pin_still_blocks(self):
        """Pins accumulate. Checking only the newest let a maintainer
        rewinding origin to an older already-failed commit walk the box
        straight back onto it."""
        body = "sha=x\nbad=oldbad\nbad=newbad\n"
        self.assertTrue(self._is_pinned(body, "oldbad"))
        self.assertTrue(self._is_pinned(body, "newbad"))

    def test_an_unpinned_commit_is_allowed(self):
        self.assertFalse(self._is_pinned("sha=x\nbad=oldbad\n", "somethingelse"))

    def test_no_pins_allows_everything(self):
        self.assertFalse(self._is_pinned("sha=x\n", "anything"))


class NothingToDoIsCheap(unittest.TestCase):
    """A fetch that fails only warns, so target ends up equal to HEAD and the
    merge is a no-op — but the run would still do a pre-update smoke sweep, a
    rebuild, a restart and a post-update sweep. With --models-max 1 and three
    aliases that is six cold 27B loads to prove nothing changed."""

    def test_it_returns_early_without_building_or_smoking(self):
        setup = (
            'preflight() { :; }; take_lock() { :; }; '
            'git() { case "$*" in *"rev-parse HEAD"*) echo SAMECOMMIT;; '
            '*rev-parse*) echo SAMECOMMIT;; esac; return 0; }; '
            'is_pinned_bad() { return 1; }; pinned_bad() { printf ""; }; '
            'local_image_digest() { echo sha256:same; }; '
            'remote_image_digest() { echo sha256:same; }; '
            'smoke() { echo "SMOKE RAN"; return 0; }; '
            'compose() { echo "COMPOSE RAN"; return 0; }; '
            'wait_for_health() { echo "HEALTH RAN"; return 0; }'
        )
        r = run_real(setup, 'cmd_update')
        out = r.stdout + r.stderr
        self.assertIn("rc=0", out, out)
        self.assertIn("already up to date", out)
        for ran in ("SMOKE RAN", "COMPOSE RAN", "HEALTH RAN"):
            self.assertNotIn(ran, out, f"{ran} — the expensive path was entered")

    def test_an_image_behind_still_triggers_work(self):
        setup = (
            'preflight() { :; }; take_lock() { :; }; '
            'git() { case "$*" in *"rev-parse HEAD"*) echo SAMECOMMIT;; '
            '*rev-parse*) echo SAMECOMMIT;; esac; return 0; }; '
            'is_pinned_bad() { return 1; }; pinned_bad() { printf ""; }; '
            'local_image_digest() { echo sha256:old; }; '
            'remote_image_digest() { echo sha256:new; }; '
            'wait_for_health() { return 0; }; smoke() { echo "SMOKE RAN"; return 0; }; '
            'record_rollback_point() { :; }; '
            'compose() { echo "COMPOSE RAN"; return 0; }'
        )
        r = run_real(setup, 'cmd_update')
        out = r.stdout + r.stderr
        self.assertIn("SMOKE RAN", out,
                      "a stale image must still get an update run")


class PinningIsNarrow(unittest.TestCase):
    """The refusal message says a commit 'already failed its smoke test'. That
    has to be true, so only the smoke-failure path may pin."""

    def test_only_evidence_of_a_bad_commit_pins(self):
        """Two paths pin, and both are evidence about the CODE.

        A failed smoke test is a model that will not answer. A failed health
        wait comes after a clean build and a clean compose up, and
        llama-server binds /health before loading any model — so 180s of
        silence is the commit, not a transient. Without that second pin, a
        commit that crash-loops the container is re-applied every week,
        which is what README and docs/updating.md already promised it would
        not do.
        """
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertEqual(body.count("do_rollback pin"), 2,
                         "smoke failure and health failure pin; nothing else")
        self.assertIn('warn "never became healthy"; do_rollback pin;', body)

    def test_infrastructure_failures_do_not_pin(self):
        """A build that lost the network, or a compose up that lost a race
        with tailscaled bringing up LLAMA_BIND2 (compose.linux.yaml warns
        about exactly this), says nothing about the commit — and a wrong pin
        blocks every future update until a human clears it."""
        with open(UPDATE_SH) as fh:
            body = fh.read()
        for site in ('warn "build failed"; do_rollback;',
                     'warn "compose up failed"; do_rollback;'):
            self.assertIn(site, body, f"{site!r} must roll back WITHOUT pinning")

    def test_manual_rollback_does_not_pin(self):
        with open(UPDATE_SH) as fh:
            body = fh.read()
        # The dispatcher's rollback call must not pass `pin`: a deliberate
        # rollback of a working commit would otherwise blacklist it, and the
        # next update would refuse it claiming a smoke test that never ran.
        line = [l for l in body.splitlines() if l.strip().startswith("--rollback)")]
        self.assertEqual(len(line), 1, "expected one --rollback dispatch")
        self.assertIn("do_rollback;;", line[0])
        self.assertNotIn("do_rollback pin", line[0])


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


# Stubs for the two commands preflight probes for but CI need not have.
# Functions shadow builtins in bash, so this intercepts `command -v`.
_TOOLS_PRESENT = (
    'command() { case "$2" in docker|git|curl|python3) return 0;; '
    '*) builtin command "$@";; esac; }'
)


def run_real(setup, call):
    """Source update.sh with every function defined, then run one for real.

    The library guard used to sit ABOVE preflight, record_rollback_point,
    do_rollback, cmd_check and cmd_update, so sourcing returned before they
    existed and their tests could only grep the source text. A group-command
    exit-status bug that aborted every single update passed a green suite
    that way. These drive the shipped functions.
    """
    script = (
        f'set -uo pipefail; export BONSAI_UPDATE_LIB=1; source "{UPDATE_SH}"; '
        f'{setup}; {call}; echo "rc=$?"'
    )
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


class RecordRollbackPointRuns(unittest.TestCase):
    """Executed, not grepped. This is the test that was missing when a
    group-command exit status aborted every update on a healthy box."""

    STUBS = (
        'git() { case "$*" in *"rev-parse HEAD"*) echo deadbeefcafe;; esac; return 0; }; '
        'docker() { case "$1" in image) echo "sha256:imageid";; esac; return 0; }'
    )

    def _record(self, existing=None, verified="1"):
        d = tempfile.mkdtemp()
        state = os.path.join(d, "update-state")
        if existing is not None:
            with open(state, "w") as fh:
                fh.write(existing)
        r = run_real(f'{self.STUBS}; STATE={shlex.quote(state)}',
                     f'record_rollback_point {verified}')
        body = ""
        if os.path.isfile(state):
            with open(state) as fh:
                body = fh.read()
        shutil.rmtree(d, ignore_errors=True)
        return r, body

    def test_it_succeeds_on_a_box_with_no_pins(self):
        """The normal case. `{ printf; [[ -n "$pins" ]] && printf; } > f` exits
        with the status of its LAST command, so an empty pins list returned 1,
        the group returned 1, and die fired though the file was written
        correctly — aborting every update and every weekly timer run."""
        r, body = self._record()
        self.assertIn("rc=0", r.stdout,
                      f"record_rollback_point failed with no pins: {r.stderr}")
        self.assertNotIn("refusing to update", r.stderr)
        self.assertIn("sha=deadbeefcafe", body)
        self.assertIn("verified=1", body)

    def test_it_carries_existing_pins_across(self):
        r, body = self._record(existing="sha=old\nbad=deadcommit\n")
        self.assertIn("rc=0", r.stdout, r.stderr)
        self.assertIn("bad=deadcommit", body,
                      "a rewrite must not drop the bad-commit pins")
        self.assertIn("sha=deadbeefcafe", body)

    def test_it_records_an_unverified_point_as_such(self):
        r, body = self._record(verified="0")
        self.assertIn("rc=0", r.stdout, r.stderr)
        self.assertIn("verified=0", body)


class VerifiedPointIsNotTradedAway(unittest.TestCase):
    """A run whose smoke returns "could not ask" leaves the update applied and
    deliberately does not roll back — so HEAD can sit on a broken commit while
    the state still names the last good one. Recording unconditionally on the
    next run overwrote that good sha and retagged :rollback to the broken
    image, and --rollback would then faithfully restore the breakage."""

    STUBS = (
        'git() { case "$*" in *"rev-parse HEAD"*) echo newbrokencommit;; esac; return 0; }; '
        'docker() { case "$1" in image) echo "sha256:newimage";; esac; return 0; }'
    )

    def _record(self, existing, verified):
        d = tempfile.mkdtemp()
        state = os.path.join(d, "update-state")
        with open(state, "w") as fh:
            fh.write(existing)
        r = run_real(f'{self.STUBS}; STATE={shlex.quote(state)}',
                     f'record_rollback_point {verified}')
        with open(state) as fh:
            body = fh.read()
        shutil.rmtree(d, ignore_errors=True)
        return r, body

    def test_an_unverified_run_keeps_the_proven_point(self):
        r, body = self._record("sha=goodcommit\nllama=sha256:goodimage\nverified=1\n", 0)
        self.assertIn("rc=0", r.stdout, r.stderr)
        self.assertIn("sha=goodcommit", body,
                      "the proven rollback point was overwritten with an "
                      "unproven one — the only way back is gone")
        self.assertNotIn("newbrokencommit", body)

    def test_a_verified_run_does_replace_the_point(self):
        r, body = self._record("sha=goodcommit\nverified=1\n", 1)
        self.assertIn("rc=0", r.stdout, r.stderr)
        self.assertIn("sha=newbrokencommit", body,
                      "a proven new point should advance the rollback target")

    def test_an_unverified_run_replaces_an_unverified_point(self):
        """Nothing to protect, so keep the most recent."""
        r, body = self._record("sha=oldcommit\nverified=0\n", 0)
        self.assertIn("rc=0", r.stdout, r.stderr)
        self.assertIn("sha=newbrokencommit", body)


class EnvIsReadBeforeItsSettingsAreUsed(unittest.TestCase):
    def test_smoke_timeout_from_env_takes_effect(self):
        """CHAT_TIMEOUT used to be read ~200 lines above the .env block, so
        BONSAI_SMOKE_TIMEOUT in .env — which .env.example calls the one source
        of truth, and which the timeout warning tells the operator to raise —
        was silently ignored. LLAMA_BIND worked only by sitting below it."""
        with tempfile.TemporaryDirectory() as d:
            repo = os.path.join(d, "repo")
            os.makedirs(repo)
            with open(os.path.join(repo, ".env"), "w") as fh:
                fh.write("BONSAI_SMOKE_TIMEOUT=1234\nLLAMA_BIND=10.1.2.3\n")
            copy = os.path.join(d, "bonsai-update.TEST01")
            shutil.copy(UPDATE_SH, copy)
            script = (
                'set -uo pipefail; export BONSAI_UPDATE_LIB=1; '
                f'export BONSAI_UPDATE_REEXEC={shlex.quote(copy)}; '
                f'export BONSAI_UPDATE_ROOT={shlex.quote(repo)}; '
                f'source {shlex.quote(copy)}; echo "T=$CHAT_TIMEOUT A=$API"'
            )
            r = subprocess.run(["bash", "-c", script], capture_output=True,
                               text=True)
            self.assertIn("T=1234", r.stdout, r.stderr)
            # The bind address already worked; prove it still does.
            self.assertIn("A=http://10.1.2.3:8080", r.stdout)


class ApiKeyStaysOutOfTheProcessTable(unittest.TestCase):
    """`curl -H "Authorization: Bearer $KEY"` puts the key in argv, readable
    by any local user with `ps` for as long as the request runs — up to
    CHAT_TIMEOUT per model, per alias, weekly. The box has several accounts."""

    def test_ask_does_not_pass_the_key_as_an_argument(self):
        with open(UPDATE_SH) as fh:
            body = fh.read()
        fn = body[body.index("ask() {"):body.index("# Exit status, and it has")]
        self.assertNotIn("Bearer $BONSAI_API_KEY", fn,
                         "the key must not appear in curl's argv")
        self.assertIn('--config "$CURL_CONF"', fn)

    def test_the_config_file_is_private_and_cleaned_up(self):
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertIn('chmod 600 "$CURL_CONF"', body)
        self.assertIn('rm -f -- "$CURL_CONF"', body)
        self.assertIn("cleanup", body)

    def test_sourcing_and_exiting_leaves_no_key_file_behind(self):
        """Library mode installs no other traps, so a sourced smoke() used to
        leave a key-bearing file in /tmp permanently."""
        with tempfile.TemporaryDirectory() as d:
            script = (
                f'set -uo pipefail; export BONSAI_UPDATE_LIB=1; '
                f'source "{UPDATE_SH}"; BONSAI_API_KEY=secret; '
                f'TMPDIR={shlex.quote(d)}; curl_conf; echo made'
            )
            r = subprocess.run(["bash", "-c", script], capture_output=True,
                               text=True)
            self.assertIn("made", r.stdout, r.stderr)
            left = [f for f in os.listdir(d) if f.startswith("bonsai-curlcfg.")]
            self.assertEqual(left, [],
                             f"a file holding the API key survived exit: {left}")

    def test_the_config_file_really_gets_mode_600(self):
        d = tempfile.mkdtemp()
        # Inspect it from inside the run: the EXIT trap now removes it.
        r = run_real(f'BONSAI_API_KEY=secret; TMPDIR={shlex.quote(d)}',
                     'curl_conf && { stat -c "%a" "$CURL_CONF" 2>/dev/null '
                     '|| stat -f "%Lp" "$CURL_CONF"; cat "$CURL_CONF"; }')
        self.assertIn("600", r.stdout, r.stderr)
        self.assertIn("Bearer secret", r.stdout)
        shutil.rmtree(d, ignore_errors=True)


class AFailedWriteDoesNotEatTheRollbackPoint(unittest.TestCase):
    """A group command exits with the status of its LAST command. Both obvious
    spellings get this wrong:

        { printf; [[ -n "$pins" ]] && printf; }      -> 1 when pins is empty
        { printf; if [[ -n "$pins" ]]; then ...; fi } -> 0 even if printf failed

    The first aborted every update on a healthy box. The second masked a
    write that failed on a full disk, and `mv -f` then replaced a good
    rollback point with a truncated one — leaving the update to proceed with
    no way back.
    """

    STUBS = (
        'git() { case "$*" in *"rev-parse HEAD"*) echo newcommit;; esac; return 0; }; '
        'docker() { case "$1" in image) echo "sha256:img";; esac; return 0; }'
    )

    def test_enospc_does_not_replace_a_good_state_file(self):
        if not os.path.exists("/dev/full"):
            self.skipTest("/dev/full is Linux-only")
        d = tempfile.mkdtemp()
        try:
            state = os.path.join(d, "update-state")
            with open(state, "w") as fh:
                fh.write("sha=goodcommit\nllama=sha256:goodimage\nverified=1\n")
            # The script writes to "$STATE.new"; point that at a device that
            # accepts the open and then fails every write.
            os.symlink("/dev/full", state + ".new")
            r = run_real(f'{self.STUBS}; STATE={shlex.quote(state)}',
                         'record_rollback_point 1')
            self.assertNotIn("rc=0", r.stdout,
                             "a write that failed was reported as success")
            self.assertIn("refusing to update", r.stderr)
            with open(state) as fh:
                self.assertIn("sha=goodcommit", fh.read(),
                              "the good rollback point was destroyed")
        finally:
            shutil.rmtree(d, ignore_errors=True)


class PreflightModesRun(unittest.TestCase):
    """Also executed. --check and --rollback are the commands you reach for
    when the stack is dark, so their gates matter most when things are broken."""

    def _preflight(self, mode, with_key=True, with_weights=False):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "docker"))
        compose = os.path.join(d, "docker", "compose.linux.yaml")
        env = os.path.join(d, ".env")
        open(compose, "w").close()
        open(env, "w").close()
        if with_weights:
            sub = os.path.join(d, "models", "Bonsai-27B-gguf")
            os.makedirs(sub)
            open(os.path.join(sub, "w.gguf"), "w").close()
        key = 'BONSAI_API_KEY=k' if with_key else 'unset BONSAI_API_KEY'
        setup = (
            f'{_TOOLS_PRESENT}; running_working_dir() {{ printf ""; }}; '
            f'ROOT={shlex.quote(d)}; COMPOSE_FILE={shlex.quote(compose)}; '
            f'ENV_FILE={shlex.quote(env)}; {key}'
        )
        r = run_real(setup, f'preflight {mode}')
        shutil.rmtree(d, ignore_errors=True)
        return r

    def test_check_does_not_need_a_dotenv_at_all(self):
        """A fresh clone has no .env yet. The file checks used to sit above
        the check early-return, so the documented read-only command died."""
        d = tempfile.mkdtemp()
        setup = (
            f'{_TOOLS_PRESENT}; running_working_dir() {{ printf ""; }}; '
            f'ROOT={shlex.quote(d)}; COMPOSE_FILE={shlex.quote(d)}/nope.yaml; '
            f'ENV_FILE={shlex.quote(d)}/nope.env; unset BONSAI_API_KEY'
        )
        r = run_real(setup, 'preflight check')
        shutil.rmtree(d, ignore_errors=True)
        self.assertIn("rc=0", r.stdout, r.stderr)

    def test_update_still_needs_a_dotenv(self):
        d = tempfile.mkdtemp()
        setup = (
            f'{_TOOLS_PRESENT}; running_working_dir() {{ printf ""; }}; '
            f'ROOT={shlex.quote(d)}; COMPOSE_FILE={shlex.quote(d)}/nope.yaml; '
            f'ENV_FILE={shlex.quote(d)}/nope.env'
        )
        r = run_real(setup, 'preflight update')
        shutil.rmtree(d, ignore_errors=True)
        self.assertNotIn("rc=0", r.stdout)

    def test_check_does_not_need_an_api_key(self):
        """cmd_check runs git fetch, rev-list and a registry HEAD. It never
        touches $API, so demanding a key broke the documented read-only
        command on a fresh clone."""
        r = self._preflight("check", with_key=False)
        self.assertIn("rc=0", r.stdout, r.stderr)

    def test_check_does_not_need_weights(self):
        r = self._preflight("check", with_weights=False)
        self.assertIn("rc=0", r.stdout, r.stderr)

    def test_rollback_does_not_need_weights(self):
        """A rollback restores code and images, not weights."""
        r = self._preflight("rollback", with_weights=False)
        self.assertIn("rc=0", r.stdout, r.stderr)

    def test_update_does_need_weights(self):
        r = self._preflight("update", with_weights=False)
        self.assertNotIn("rc=0", r.stdout)
        self.assertIn("no *.gguf", r.stderr)

    def test_update_accepts_a_stocked_models_dir(self):
        r = self._preflight("update", with_weights=True)
        self.assertIn("rc=0", r.stdout, r.stderr)

    def test_update_still_needs_an_api_key(self):
        r = self._preflight("update", with_key=False, with_weights=True)
        self.assertNotIn("rc=0", r.stdout)
        self.assertIn("BONSAI_API_KEY", r.stderr)


class OneWriterAtATime(unittest.TestCase):
    """The weekly timer and a human running --rollback share one checkout."""

    def test_a_second_run_is_refused_while_the_lock_is_held(self):
        if not shutil.which("flock"):
            self.skipTest("flock not available (BSD/macOS); Linux-only path")
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "run"))
            state = os.path.join(d, "run", "update-state")
            lock = os.path.join(d, "run", "update.lock")
            # Hold the lock, then try to take it.
            script = (
                f'set -uo pipefail; export BONSAI_UPDATE_LIB=1; '
                f'source "{UPDATE_SH}"; ROOT={shlex.quote(d)}; '
                f'STATE={shlex.quote(state)}; '
                f'exec 8>{shlex.quote(lock)}; flock -n 8 || exit 9; '
                f'( take_lock ) ; echo "rc=$?"'
            )
            r = subprocess.run(["bash", "-c", script], capture_output=True,
                               text=True)
            self.assertIn("already running", r.stderr,
                          f"a concurrent run was not refused: {r.stdout}{r.stderr}")

    def test_the_lock_is_taken_on_both_writing_paths(self):
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertIn("preflight update\n  take_lock", body)
        self.assertIn("preflight rollback; take_lock", body)
        self.assertNotIn("preflight check; take_lock", body,
                         "--check writes nothing and must not block on a lock")


class LibraryGuardSitsBelowTheDefinitions(unittest.TestCase):
    def test_every_command_function_is_reachable_when_sourced(self):
        """Structural regression: with the guard higher up, the tests for
        these could only grep the file."""
        names = ["preflight", "record_rollback_point", "do_rollback",
                 "cmd_check", "cmd_update", "running_working_dir", "smoke"]
        checks = "; ".join(f'declare -F {n} >/dev/null || echo "MISSING {n}"'
                           for n in names)
        script = (f'export BONSAI_UPDATE_LIB=1; source "{UPDATE_SH}"; {checks}')
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "", r.stdout)

    def test_sourcing_does_not_re_exec_or_arm_the_delete_trap(self):
        """Library mode must have no side effects."""
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, TMPDIR=tmp, BONSAI_UPDATE_LIB="1")
            env.pop("BONSAI_UPDATE_REEXEC", None)
            r = subprocess.run(["bash", "-c", f'source "{UPDATE_SH}"; echo sourced'],
                               capture_output=True, text=True, env=env)
            self.assertIn("sourced", r.stdout)
            self.assertEqual(
                [f for f in os.listdir(tmp) if f.startswith("bonsai-update.")], [],
                "sourcing copied itself")
            self.assertTrue(os.path.isfile(UPDATE_SH))


class StoppedContainersAreNotRunningStacks(unittest.TestCase):
    def test_the_name_fallback_checks_running_state(self):
        """docker inspect resolves STOPPED containers, so one left behind by
        the very move this script detects would make preflight refuse both
        update and rollback, citing a directory that no longer exists."""
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertIn("{{if .State.Running}}", body)


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
