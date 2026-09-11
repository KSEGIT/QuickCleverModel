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
import re
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
    is a line in the journal.

    The pin VALUES are covered by AnyPinCounts, which drives is_pinned_bad —
    the real gate. pinned_bad() was removed: it was assigned and never read,
    and a grep-based test was the only thing holding it in place.
    """

    def test_update_consults_the_pin_before_moving(self):
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertIn('if is_pinned_bad "$target"; then', body,
                      "cmd_update must consult the pin before merging")
        self.assertIn("already failed its smoke test", body,
                      "and must say why it is refusing")
        self.assertNotIn("pinned_bad()", body.replace("is_pinned_bad()", ""),
                         "pinned_bad was dead code")


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
        # Slice to the NEXT function, not to smoke(): webui_ok sits between
        # them and uses curl -f on purpose — it is a status-only health poll
        # with no body worth keeping.
        fn = body[body.index("ask() {"):body.index("smoke_body() {")]
        # Match the FLAG, not the substring: the function also contains
        # `! -f "$CURL_ERR"`, which is a file test, not curl's --fail.
        self.assertNotIn("curl -f", fn, "curl -f would discard the error body")
        self.assertNotIn("-fsS", fn, "curl -f would discard the error body")
        self.assertIn("curl -sS", fn)

    def test_curl_stderr_is_kept(self):
        """-S exists so curl explains the failure despite -s. Sending stderr
        to /dev/null threw that away, and every transport failure logged a
        bare "curl failed (exit 7)" with no reason."""
        with open(UPDATE_SH) as fh:
            body = fh.read()
        fn = body[body.index("ask() {"):body.index("smoke_body() {")]
        self.assertIn('2>"$CURL_ERR"', fn)
        self.assertIn("ASK_ERR=", fn)
        self.assertIn("${ASK_ERR:-curl failed}", body,
                      "the captured message must reach the warning")


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


class PinsOnlyWhatThisRunMoved(unittest.TestCase):
    """record_rollback_point deliberately keeps an older proven commit A when a
    run could not prove itself, so HEAD can already be B when the next run
    starts. If that run then fails because of a NEW open-webui image rather
    than the code, pinning would blacklist B — a commit that was already
    running and that this update never introduced — and every later run would
    refuse it. docs/updating.md promises only the code gets blamed."""

    BASE = (
        'preflight() { :; }; take_lock() { :; }; '
        'docker() { return 0; }; is_pinned_bad() { return 1; }; '
        'record_rollback_point() { :; }; wait_for_health() { return 0; }; '
        'webui_ok() { return 0; }; compose() { return 0; }; '
        'do_rollback() { echo "ROLLBACK pin=[${1:-}]"; return 0; }; '
        'N=0; smoke() { N=$((N+1)); [ "$N" = 1 ] && return 0; return 1; }'
    )

    def _run(self, head, target, have="sha256:old", want="sha256:new"):
        setup = (
            f'{self.BASE}; '
            # merge-base --is-ancestor is what decides "did the code
            # move", so the stub has to answer it: ancestor (0) when HEAD
            # already carries target, not-an-ancestor (1) on a real move.
            f'git() {{ case "$*" in '
            f'*"merge-base --is-ancestor"*) [ "{head}" = "{target}" ]; return $?;; '
            f'*"rev-parse HEAD"*) echo {head};; '
            f'*rev-parse*) echo {target};; *merge*) return 0;; esac; return 0; }}; '
            f'local_image_digest() {{ echo {have}; }}; '
            f'remote_image_digest() {{ echo {want}; }}'
        )
        r = run_real(setup, 'cmd_update')
        return r.stdout + r.stderr

    def test_an_image_only_update_does_not_pin_the_running_commit(self):
        out = self._run(head="SAMECOMMIT", target="SAMECOMMIT")
        self.assertIn("ROLLBACK pin=[]", out,
                      "the code never moved, so the failure is not its fault")
        self.assertIn("updating images only", out)

    def test_a_real_code_move_does_pin(self):
        out = self._run(head="OLDCOMMIT", target="NEWCOMMIT")
        self.assertIn("ROLLBACK pin=[pin]", out,
                      "this run introduced the commit, so it may be blamed")


class AlreadyBrokenDoesNotBlameTheCommit(unittest.TestCase):
    """cmd_update computes `verified` from the pre-update sweep but used not
    to let it gate the pin. A driver upgrade, an OOM, or weights that moved —
    every cause the script's own comment names — would blacklist an innocent
    commit, and is_pinned_bad would then refuse every later run until
    something newer landed, while the real outage went untouched."""

    def _run(self, pre_ok, post_ok, head="OLD", target="NEW"):
        pre = pre_ok if isinstance(pre_ok, str) else ("0" if pre_ok else "1")
        post = post_ok if isinstance(post_ok, str) else ("0" if post_ok else "1")
        setup = (
            'preflight() { :; }; take_lock() { :; }; docker() { return 0; }; '
            'is_pinned_bad() { return 1; }; record_rollback_point() { :; }; '
            'wait_for_health() { return 0; }; webui_ok() { return 0; }; compose() { return 0; }; '
            'do_rollback() { echo "ROLLBACK pin=[${1:-}]"; return 0; }; '
            f'N=0; smoke() {{ N=$((N+1)); [ "$N" = 1 ] && return {pre}; return {post}; }}; '
            f'git() {{ case "$*" in '
            f'*"merge-base --is-ancestor"*) [ "{head}" = "{target}" ]; return $?;; '
            f'*"rev-parse HEAD"*) echo {head};; '
            f'*rev-parse*) echo {target};; esac; return 0; }}; '
            'local_image_digest() { echo a; }; remote_image_digest() { echo b; }'
        )
        r = run_real(setup, 'cmd_update')
        return r.stdout + r.stderr

    def test_dark_before_and_after_does_not_pin(self):
        out = self._run(pre_ok=False, post_ok=False)
        self.assertIn("ROLLBACK pin=[]", out,
                      "a stack already failing before the update must not "
                      "blame the new commit")
        self.assertIn("already failing before this update", out)

    def test_healthy_before_and_broken_after_does_pin(self):
        out = self._run(pre_ok=True, post_ok=False)
        self.assertIn("ROLLBACK pin=[pin]", out,
                      "the update genuinely broke a working stack")

    def test_could_not_judge_beforehand_still_pins(self):
        """rc=2 is "could not ask", not "proven broken", and folding it in
        recreated the weekly-outage loop: the timer fires at 04:00 while
        someone is chatting, the router answers "model limit reached" to every
        alias with --models-max 1, smoke returns 2 — and a genuinely
        crash-looping commit was then rolled back UNPINNED, so the box walked
        onto it again the next week, and the next."""
        out = self._run(pre_ok="2", post_ok=False)
        self.assertIn("ROLLBACK pin=[pin]", out,
                      "an unjudgeable pre-check must not disarm the pin")
        self.assertIn("could not judge the models before", out,
                      "and must not claim the stack was not serving")

    def test_the_message_matches_what_was_actually_observed(self):
        out = self._run(pre_ok=False, post_ok=False)
        self.assertIn("are NOT serving before", out)
        self.assertNotIn("could not judge the models before", out)


class MovedAndPinnableAreDifferentQuestions(unittest.TestCase):
    """`pin_arg` gets cleared when the stack was already broken beforehand.
    Reusing it as "did this run move code" meant that on an already-broken
    stack the merge DID move HEAD, pin_arg was empty, and the build-failure
    handler announced "nothing to undo: this run moved no code" — false —
    skipped the stash and the reset, and left the checkout on a new, unbuilt
    commit."""

    def _run(self, pre_verdict, marker):
        # A marker FILE, not an echo: the real code runs the reset with
        # >/dev/null 2>&1, so a stub that prints is invisible.
        setup = (
            'preflight() { :; }; take_lock() { :; }; docker() { return 0; }; '
            'is_pinned_bad() { return 1; }; record_rollback_point() { :; }; '
            'wait_for_health() { return 0; }; webui_ok() { return 0; }; '
            'stash_local_edits() { echo "STASHED"; }; '
            'do_rollback() { echo "ROLLBACK pin=[${1:-}]"; return 0; }; '
            f'smoke() {{ return {pre_verdict}; }}; '
            'local_image_digest() { echo a; }; remote_image_digest() { echo a; }; '
            'compose() { case "$1" in build) echo "BUILD FAILED"; return 1;; esac; return 0; }; '
            'git() { case "$*" in '
            '*"merge-base --is-ancestor"*) return 1;; '          # a real move
            '*"rev-parse HEAD"*) echo OLDCOMMIT;; '
            '*rev-parse*) echo NEWCOMMIT;; '
            f'*"reset --hard"*) touch {shlex.quote(marker)};; '
            'esac; return 0; }'
        )
        r = run_real(setup, 'cmd_update')
        return r.stdout + r.stderr

    def test_a_broken_stack_still_gets_its_merge_undone(self):
        with tempfile.TemporaryDirectory() as d:
            marker = os.path.join(d, "reset-ran")
            out = self._run(pre_verdict=1, marker=marker)   # pin disarmed
            self.assertIn("undoing the code move", out,
                          "the merge moved HEAD, so a failed build must undo it")
            self.assertIn("STASHED", out)
            self.assertTrue(os.path.exists(marker),
                            "the checkout was left on a new, unbuilt commit")
            self.assertNotIn("nothing to undo", out)

    def test_a_healthy_stack_also_gets_its_merge_undone(self):
        with tempfile.TemporaryDirectory() as d:
            marker = os.path.join(d, "reset-ran")
            self._run(pre_verdict=0, marker=marker)
            self.assertTrue(os.path.exists(marker))

    def test_an_image_only_run_has_nothing_to_undo(self):
        d = tempfile.mkdtemp()
        marker = os.path.join(d, "reset-ran")
        setup = (
            'preflight() { :; }; take_lock() { :; }; docker() { return 0; }; '
            'is_pinned_bad() { return 1; }; record_rollback_point() { :; }; '
            'wait_for_health() { return 0; }; webui_ok() { return 0; }; '
            'stash_local_edits() { echo "STASHED"; }; smoke() { return 0; }; '
            'do_rollback() { echo "ROLLBACK"; return 0; }; '
            'local_image_digest() { echo old; }; remote_image_digest() { echo new; }; '
            'compose() { case "$1" in build) return 1;; esac; return 0; }; '
            'git() { case "$*" in '
            '*"merge-base --is-ancestor"*) return 0;; '          # no move
            '*"rev-parse HEAD"*) echo SAME;; *rev-parse*) echo SAME;; '
            f'*"reset --hard"*) touch {shlex.quote(marker)};; esac; return 0; }}'
        )
        out = run_real(setup, 'cmd_update')
        out = out.stdout + out.stderr
        self.assertIn("nothing to undo", out)
        self.assertFalse(os.path.exists(marker),
                         "an image-only run must not touch the checkout")


class LeakedLibraryFlagIsIgnored(unittest.TestCase):
    """BONSAI_UPDATE_LIB alone used to switch off the self-copy and the signal
    traps, and the guard at the bottom is a `return` — an ERROR at the top
    level of an executed script, not an exit. Bash printed "can only `return'
    from a function or sourced script" and carried straight on into
    cmd_update. A leaked export therefore made a real run execute `git merge`
    and `git reset --hard` against the very file bash was reading by byte
    offset, with no traps installed either."""

    def test_an_executed_script_ignores_the_flag(self):
        with tempfile.TemporaryDirectory() as d:
            copy = os.path.join(d, "update.sh")
            shutil.copy(UPDATE_SH, copy)
            os.chmod(copy, 0o755)
            env = dict(os.environ, BONSAI_UPDATE_LIB="1", TMPDIR=d)
            r = subprocess.run(["bash", copy, "--help"], capture_output=True,
                               text=True, cwd=d, env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("ignoring it", r.stderr)
            self.assertNotIn("can only `return'", r.stderr,
                             "the library guard fell through into the dispatcher")
            self.assertIn("--rollback", r.stdout)

    def test_the_warning_is_said_once_not_twice(self):
        with tempfile.TemporaryDirectory() as d:
            copy = os.path.join(d, "update.sh")
            shutil.copy(UPDATE_SH, copy)
            os.chmod(copy, 0o755)
            env = dict(os.environ, BONSAI_UPDATE_LIB="1", TMPDIR=d)
            r = subprocess.run(["bash", copy, "--help"], capture_output=True,
                               text=True, cwd=d, env=env)
            self.assertEqual(r.stderr.count("ignoring it"), 1,
                             "the re-exec'd copy repeated the warning")

    def test_sourcing_still_works(self):
        r = subprocess.run(
            ["bash", "-c",
             f'export BONSAI_UPDATE_LIB=1; source "{UPDATE_SH}"; '
             'declare -F cmd_update >/dev/null && echo DEFINED'],
            capture_output=True, text=True)
        self.assertIn("DEFINED", r.stdout, r.stderr)
        self.assertNotIn("ignoring it", r.stderr)


class HealthDeadlineMatchesTheInstaller(unittest.TestCase):
    def test_the_default_matches_install_sh(self):
        """install.sh waits 900s for this exact endpoint on this exact stack
        ("model load on first request is genuinely slow"). This is the one
        path that PINS on failure, so a shorter default would blacklist good
        commits on any box slower than the one 180 was picked on."""
        with open(os.path.join(ROOT, "install.sh")) as fh:
            installer = fh.read()
        m = re.search(r'wait_for "http://[^"]*:8080/health"\s+(\d+)', installer)
        self.assertIsNotNone(m, "install.sh no longer waits on :8080/health")
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertIn(f"BONSAI_HEALTH_TIMEOUT:-{m.group(1)}", body,
                      "update.sh disagrees with install.sh about the same "
                      "endpoint, and this one pins on failure")


class AFailedStashStopsTheRewind(unittest.TestCase):
    """`git stash push` writes commits, so it needs a committer identity — on
    a service account with no user.email it simply fails. Returning 0 let the
    caller hard-reset anyway and delete the work, with a WARN line as the only
    trace, while docs/updating.md promises "Nothing is deleted"."""

    def _repo(self, d):
        subprocess.run(["git", "init", "-q", d], check=True, capture_output=True)
        for k, v in (("user.email", "t@t"), ("user.name", "t")):
            subprocess.run(["git", "-C", d, "config", k, v], check=True,
                           capture_output=True)
        with open(os.path.join(d, "f.txt"), "w") as fh:
            fh.write("committed\n")
        subprocess.run(["git", "-C", d, "add", "-A"], check=True, capture_output=True)
        subprocess.run(["git", "-C", d, "commit", "-qm", "one"], check=True,
                       capture_output=True)
        first = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip()
        with open(os.path.join(d, "f.txt"), "w") as fh:
            fh.write("two\n")
        subprocess.run(["git", "-C", d, "add", "-A"], check=True, capture_output=True)
        subprocess.run(["git", "-C", d, "commit", "-qm", "two"], check=True,
                       capture_output=True)
        return first

    def test_the_checkout_is_left_alone_when_the_stash_fails(self):
        with tempfile.TemporaryDirectory() as d:
            first = self._repo(d)
            with open(os.path.join(d, "f.txt"), "w") as fh:
                fh.write("MY UNCOMMITTED WORK\n")
            state = os.path.join(d, "update-state")
            with open(state, "w") as fh:
                fh.write(f"sha={first}\nwebui=\nllama=\nverified=1\n")
            setup = (
                f'ROOT={shlex.quote(d)}; STATE={shlex.quote(state)}; '
                'docker() { return 0; }; compose() { return 0; }; '
                'wait_for_health() { return 0; }; webui_ok() { return 0; }; '
                'smoke() { return 0; }; '
                # the failure mode: stash refuses
                'stash_local_edits() { warn "could not stash"; return 1; }'
            )
            r = run_real(setup, 'do_rollback')
            with open(os.path.join(d, "f.txt")) as fh:
                self.assertEqual(fh.read(), "MY UNCOMMITTED WORK\n",
                                 f"the rewind deleted the work: {r.stderr}")
            self.assertIn("preserve uncommitted work", r.stderr)

    def test_stash_failure_returns_nonzero(self):
        with open(UPDATE_SH) as fh:
            body = fh.read()
        fn = body[body.index("stash_local_edits() {"):body.index("do_rollback() {")]
        self.assertIn("return 1", fn,
                      "a failed stash must stop the caller rewinding")


class ABadImagePinAppliesWhenTheCodeMovesToo(unittest.TestCase):
    """The pin was only consulted on the code-is-current path. A week that
    brought BOTH a new commit and a broken image looped forever: the commit is
    not pinned (the image is to blame), the rollback rewinds the code, so next
    week the commit is "new" again, the early exit never runs, and the same
    broken image is pulled, fails and force-recreates every Monday."""

    def _run(self, pinned):
        d = tempfile.mkdtemp()
        try:
            state = os.path.join(d, "s")
            with open(state, "w") as fh:
                fh.write("sha=OLD\n" + ("badimg=sha256:broken\n" if pinned else ""))
            setup = (
                f'STATE={shlex.quote(state)}; '
                'preflight() { :; }; take_lock() { :; }; docker() { return 0; }; '
                'is_pinned_bad() { return 1; }; record_rollback_point() { :; }; '
                'wait_for_health() { return 0; }; webui_ok() { return 0; }; '
                'smoke() { return 0; }; stash_local_edits() { :; }; '
                'compose() { case "$1" in pull) echo "PULLED";; esac; return 0; }; '
                'local_image_digest() { echo sha256:old; }; '
                'remote_image_digest() { echo sha256:broken; }; '
                'git() { case "$*" in *"merge-base --is-ancestor"*) return 1;; '
                '*"rev-parse HEAD"*) echo OLD;; *rev-parse*) echo NEW;; esac; return 0; }'
            )
            r = run_real(setup, 'cmd_update')
            return r.stdout + r.stderr
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_a_pinned_image_is_not_pulled_even_on_a_code_move(self):
        out = self._run(pinned=True)
        self.assertNotIn("PULLED", out,
                         "the known-broken image was pulled again")
        self.assertIn("skipping the open-webui pull", out)

    def test_an_unpinned_image_is_still_pulled(self):
        out = self._run(pinned=False)
        self.assertIn("PULLED", out)


class ImagePinsAreNotDuplicated(unittest.TestCase):
    def test_the_same_digest_is_pinned_once(self):
        """A repeating weekly failure appended the same digest every run."""
        with open(UPDATE_SH) as fh:
            body = fh.read()
        fn = body[body.index("chat_ui_failed() {"):body.index("cmd_update() {")]
        self.assertIn('! is_pinned_bad_image "$webui_after"', fn,
                      "nothing stops the same pin being appended weekly")


class EveryDeadlineKnobHonoursTheEnvironment(unittest.TestCase):
    """The save/restore block exists so an explicit `KNOB=x ./update.sh` beats
    .env. PRECHECK_TIMEOUT was read after .env but left out of the list, so
    for that one knob the inversion was still live."""

    KNOBS = ("BONSAI_SMOKE_TIMEOUT", "BONSAI_SMOKE_SWEEP_MAX",
             "BONSAI_HEALTH_TIMEOUT", "BONSAI_WEBUI_TIMEOUT",
             "BONSAI_SMOKE_RETRY_SLEEP", "BONSAI_PRECHECK_TIMEOUT")

    def test_precheck_beats_dot_env(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "docker"))
            copy = os.path.join(d, "bonsai-update.TESTY")
            shutil.copy(UPDATE_SH, copy)
            with open(os.path.join(d, ".env"), "w") as fh:
                fh.write("BONSAI_API_KEY=k\nBONSAI_PRECHECK_TIMEOUT=77\n")
            script = (
                'set -uo pipefail; export BONSAI_UPDATE_LIB=1; '
                f'export BONSAI_UPDATE_REEXEC={shlex.quote(copy)}; '
                f'export BONSAI_UPDATE_ROOT={shlex.quote(d)}; '
                'export BONSAI_PRECHECK_TIMEOUT=5; '
                f'source {shlex.quote(copy)}; echo "P=$PRECHECK_TIMEOUT"'
            )
            r = subprocess.run(["bash", "-c", script], capture_output=True,
                               text=True)
            self.assertIn("P=5", r.stdout, r.stderr)

    def test_every_knob_is_saved_and_restored(self):
        with open(UPDATE_SH) as fh:
            body = fh.read()
        for knob in self.KNOBS:
            with self.subTest(knob=knob):
                self.assertEqual(body.count(f'_pre_'), body.count('_pre_'))
                self.assertIn(f'{knob}="$_pre', body,
                              f"{knob} is not restored over .env")

    def test_every_knob_is_documented(self):
        with open(os.path.join(ROOT, ".env.example")) as fh:
            env_example = fh.read()
        with open(os.path.join(ROOT, "docs", "updating.md")) as fh:
            docs = fh.read()
        for knob in self.KNOBS:
            with self.subTest(knob=knob):
                self.assertIn(knob, env_example)
                self.assertIn(knob, docs)


class AHealthTimeoutIsNotProofOfDeath(unittest.TestCase):
    """The pre-check deadline is 30s, and /health silence at 30s is ambiguous:
    a box that rebooted shortly before the timer is still doing a cold 27B
    load. Mapping it to verdict 1 ("proven dead") cleared pin_arg, so a
    genuinely crash-looping commit was rolled back UNPINNED and fast-forwarded
    onto again the next week, forever."""

    def _run(self, health_ok):
        setup = (
            'preflight() { :; }; take_lock() { :; }; docker() { return 0; }; '
            'is_pinned_bad() { return 1; }; record_rollback_point() { :; }; '
            f'wait_for_health() {{ return {0 if health_ok else 1}; }}; '
            'webui_ok() { return 0; }; compose() { return 0; }; '
            'stash_local_edits() { :; }; '
            'do_rollback() { echo "ROLLBACK pin=[${1:-}]"; return 0; }; '
            'smoke() { return 1; }; '
            'local_image_digest() { echo same; }; remote_image_digest() { echo other; }; '
            'git() { case "$*" in *"merge-base --is-ancestor"*) return 1;; '
            '*"rev-parse HEAD"*) echo OLD;; *rev-parse*) echo NEW;; esac; return 0; }'
        )
        r = run_real(setup, 'cmd_update')
        return r.stdout + r.stderr

    def test_a_precheck_health_timeout_does_not_disarm_the_pin(self):
        out = self._run(health_ok=False)
        self.assertIn("ROLLBACK pin=[pin]", out,
                      "a 30s health timeout was treated as proof the models "
                      "were dead, so a crash-looping commit escaped the pin")
        self.assertIn("not as proof they are dead", out)

    def test_models_answering_with_errors_still_disarms_it(self):
        """smoke returning 1 IS proof, and must still disarm."""
        out = self._run(health_ok=True)
        self.assertIn("ROLLBACK pin=[]", out)


class APinnedImageWithNoLocalCopyStops(unittest.TestCase):
    """`compose up -d` pulls whatever is missing, so skipping the explicit
    pull is not enough: after a `docker system prune -a` the known-broken
    image gets pulled anyway, fails the UI check, force-recreates, and repeats
    every week."""

    def _run(self, have):
        d = tempfile.mkdtemp()
        try:
            state = os.path.join(d, "s")
            with open(state, "w") as fh:
                fh.write("sha=SAME\nbadimg=sha256:broken\n")
            setup = (
                f'STATE={shlex.quote(state)}; '
                'preflight() { :; }; take_lock() { :; }; docker() { return 0; }; '
                'is_pinned_bad() { return 1; }; record_rollback_point() { :; }; '
                'wait_for_health() { return 0; }; webui_ok() { return 0; }; '
                'smoke() { return 0; }; compose() { echo "COMPOSE RAN"; return 0; }; '
                f'local_image_digest() {{ printf "%s" {shlex.quote(have)}; }}; '
                'remote_image_digest() { echo sha256:broken; }; '
                'git() { case "$*" in *"merge-base --is-ancestor"*) return 0;; '
                '*rev-parse*) echo SAME;; esac; return 0; }'
            )
            r = run_real(setup, 'cmd_update')
            return r.stdout + r.stderr
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_no_local_copy_refuses_rather_than_pulling_it_again(self):
        out = self._run(have="")
        self.assertNotIn("COMPOSE RAN", out,
                         "compose up would have pulled the known-broken image")
        self.assertIn("no local copy to fall back on", out)
        self.assertNotIn("rc=0", out)

    def test_a_local_copy_means_nothing_to_do(self):
        out = self._run(have="sha256:old")
        self.assertIn("rc=0", out, out)
        self.assertNotIn("COMPOSE RAN", out)


class UnknownDigestsPinNothing(unittest.TestCase):
    def test_it_does_not_claim_a_new_image_arrived(self):
        """The else branch asserted "a new image was pulled" even when both
        digests were unknown — logging a wrong reason and pinning nothing, so
        a commit that genuinely broke the UI escaped."""
        with open(UPDATE_SH) as fh:
            body = fh.read()
        fn = body[body.index("chat_ui_failed() {"):body.index("cmd_update() {")]
        self.assertIn('if [[ -z "$webui_before" || -z "$webui_after" ]]; then', fn)
        self.assertIn("cannot tell", fn)


class CheckRespectsThePins(unittest.TestCase):
    """--check reported drift and exited 1 forever after a pin, while
    ./update.sh said "nothing to do" and exited 0 — so a monitoring cron on
    --check alerted permanently on a settled state."""

    def _check(self, pinned_commit=False, pinned_image=False):
        d = tempfile.mkdtemp()
        try:
            state = os.path.join(d, "s")
            with open(state, "w") as fh:
                if pinned_commit:
                    fh.write("bad=UPSTREAM\n")
                if pinned_image:
                    fh.write("badimg=sha256:new\n")
            setup = (
                f'{_TOOLS_PRESENT}; STATE={shlex.quote(state)}; ROOT=/tmp; '
                'docker() { return 0; }; '
                'local_image_digest() { echo sha256:old; }; '
                'remote_image_digest() { echo sha256:new; }; '
                'git() { case "$*" in *rev-list*) echo 3;; '
                '*rev-parse*) echo UPSTREAM;; esac; return 0; }'
            )
            r = run_real(setup, 'cmd_check')
            return r.stdout + r.stderr
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_a_pinned_commit_is_not_drift(self):
        out = self._check(pinned_commit=True, pinned_image=True)
        self.assertIn("rc=0", out, out)
        self.assertIn("is pinned", out)

    def test_unpinned_drift_is_still_reported(self):
        out = self._check()
        self.assertNotIn("rc=0", out)
        self.assertIn("behind origin by 3", out)


class RollbackDoesNotRecreateForNothing(unittest.TestCase):
    def test_nothing_restored_means_no_force_recreate(self):
        """Force-recreating when every restore step was a no-op is a
        multi-minute outage and an eviction of the resident model, in exchange
        for nothing."""
        with tempfile.TemporaryDirectory() as d:
            state = os.path.join(d, "s")
            with open(state, "w") as fh:
                fh.write("sha=\nwebui=\nllama=\nverified=1\nui=1\n")
            setup = (
                f'ROOT={shlex.quote(d)}; STATE={shlex.quote(state)}; '
                'docker() { return 1; }; '
                'compose() { echo "COMPOSE RAN"; return 0; }; '
                'wait_for_health() { return 0; }; smoke() { return 0; }; '
                'webui_ok() { return 0; }'
            )
            r = run_real(setup, 'do_rollback')
            out = r.stdout + r.stderr
            self.assertNotIn("COMPOSE RAN", out,
                             "recreated the stack having restored nothing")
            self.assertIn("nothing was actually restored", out)


class PreflightAsksTheDaemon(unittest.TestCase):
    """preflight checked that the docker BINARY exists, not that the daemon
    answers. With dockerd down, record_rollback_point still ran: `docker image
    inspect` returned empty for both images and wrote blank ids, silently
    disarming the image half of --rollback even though bonsai-llama:rollback
    was still valid on disk. The run then died at `compose build` anyway."""

    def _preflight(self, mode, daemon_ok):
        d = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(d, "docker"))
            compose = os.path.join(d, "docker", "compose.linux.yaml")
            env = os.path.join(d, ".env")
            open(compose, "w").close()
            open(env, "w").close()
            sub = os.path.join(d, "models", "g")
            os.makedirs(sub)
            open(os.path.join(sub, "w.gguf"), "w").close()
            rc = 0 if daemon_ok else 1
            setup = (
                f'{_TOOLS_PRESENT}; running_working_dir() {{ printf ""; }}; '
                f'docker() {{ case "$*" in *version*) return {rc};; esac; return 0; }}; '
                f'ROOT={shlex.quote(d)}; COMPOSE_FILE={shlex.quote(compose)}; '
                f'ENV_FILE={shlex.quote(env)}; BONSAI_API_KEY=k'
            )
            return run_real(setup, f'preflight {mode}')
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_update_refuses_when_the_daemon_is_down(self):
        r = self._preflight("update", daemon_ok=False)
        self.assertNotIn("rc=0", r.stdout)
        self.assertIn("cannot reach the docker daemon", r.stderr)

    def test_rollback_refuses_when_the_daemon_is_down(self):
        r = self._preflight("rollback", daemon_ok=False)
        self.assertNotIn("rc=0", r.stdout)

    def test_update_proceeds_when_the_daemon_answers(self):
        r = self._preflight("update", daemon_ok=True)
        self.assertIn("rc=0", r.stdout, r.stderr)


class ThePreCheckDoesNotWaitOutARestart(unittest.TestCase):
    def test_it_uses_the_short_deadline(self):
        """The pre-update probes ask what the stack is doing NOW. Using the
        restart deadlines meant a run against a deliberately-stopped stack
        idled ~30 minutes before doing anything."""
        with open(UPDATE_SH) as fh:
            body = fh.read()
        # The CALL, not the definition: the definition sits above this.
        pre = body[body.index('log "checking the current stack before'):
                   body.index('record_rollback_point "$verified"')]
        self.assertIn('wait_for_health "$PRECHECK_TIMEOUT"', pre)
        self.assertIn('webui_ok "$PRECHECK_TIMEOUT"', pre)

    def test_the_post_restart_waits_keep_the_long_deadline(self):
        with open(UPDATE_SH) as fh:
            body = fh.read()
        # Just the health wait: the chat-UI deadline below it deliberately
        # drops to the short one when the UI was already down, since no amount
        # of waiting starts a stopped container.
        post = body[body.index('log "waiting for health"'):
                    body.index('log "checking the chat UI"')]
        self.assertIn("wait_for_health ||", post,
                      "the post-restart wait must keep the full deadline")
        self.assertNotIn("PRECHECK_TIMEOUT", post)

    def test_a_ui_already_down_is_not_waited_for(self):
        with open(UPDATE_SH) as fh:
            body = fh.read()
        ui = body[body.index('log "checking the chat UI"'):
                  body.index('log "smoke testing every model"')]
        self.assertIn('(( ui_before )) || ui_deadline="$PRECHECK_TIMEOUT"', ui,
                      "15 minutes spent on a verdict already known")


class ABadImageIsPinnedToo(unittest.TestCase):
    """Only code was ever pinned, so a permanently broken upstream :main
    looped forever: the rollback retags the old image, so next week the local
    digest differs from the registry again, the "nothing to do" exit is
    skipped, and the run rebuilds, pulls the same broken image, fails, and
    force-recreates — evicting the resident 27B model and forcing a second
    cold smoke sweep. Every Monday, with nothing saying it is a loop."""

    def _pinned(self, state_body, digest):
        with tempfile.TemporaryDirectory() as d:
            state = os.path.join(d, "s")
            with open(state, "w") as fh:
                fh.write(state_body)
            r = run_real(f'STATE={shlex.quote(state)}',
                         f'is_pinned_bad_image {shlex.quote(digest)}')
            return "rc=0" in r.stdout

    def test_a_pinned_digest_is_recognised(self):
        self.assertTrue(self._pinned("sha=x\nbadimg=sha256:abc\n", "sha256:abc"))

    def test_an_unpinned_digest_is_not(self):
        self.assertFalse(self._pinned("sha=x\nbadimg=sha256:abc\n", "sha256:zzz"))

    def test_a_code_pin_is_not_an_image_pin(self):
        """bad= and badimg= must not be confused for one another."""
        self.assertFalse(self._pinned("sha=x\nbad=sha256:abc\n", "sha256:abc"))
        with tempfile.TemporaryDirectory() as d:
            state = os.path.join(d, "s")
            with open(state, "w") as fh:
                fh.write("sha=x\nbadimg=deadbeef\n")
            r = run_real(f'STATE={shlex.quote(state)}', 'is_pinned_bad deadbeef')
            self.assertNotIn("rc=0", r.stdout,
                             "an image pin was read as a commit pin")

    def test_both_pin_kinds_survive_a_state_rewrite(self):
        d = tempfile.mkdtemp()
        try:
            state = os.path.join(d, "s")
            with open(state, "w") as fh:
                fh.write("sha=old\nbad=commit1\nbadimg=sha256:img1\n")
            stubs = (
                'git() { case "$*" in *"rev-parse HEAD"*) echo new;; esac; return 0; }; '
                'docker() { case "$1" in image) echo "sha256:i";; esac; return 0; }'
            )
            run_real(f'{stubs}; STATE={shlex.quote(state)}',
                     'record_rollback_point 1')
            with open(state) as fh:
                body = fh.read()
            self.assertIn("bad=commit1", body)
            self.assertIn("badimg=sha256:img1", body,
                          "the image pin was dropped by the rewrite")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_a_pinned_image_stops_the_weekly_rebuild(self):
        with tempfile.TemporaryDirectory() as d:
            state = os.path.join(d, "s")
            with open(state, "w") as fh:
                fh.write("sha=SAME\nbadimg=sha256:broken\n")
            setup = (
                f'STATE={shlex.quote(state)}; '
                'preflight() { :; }; take_lock() { :; }; docker() { return 0; }; '
                'is_pinned_bad() { return 1; }; record_rollback_point() { :; }; '
                'wait_for_health() { return 0; }; webui_ok() { return 0; }; '
                'smoke() { return 0; }; compose() { echo "COMPOSE RAN"; return 0; }; '
                'local_image_digest() { echo sha256:old; }; '
                'remote_image_digest() { echo sha256:broken; }; '
                'git() { case "$*" in *"merge-base --is-ancestor"*) return 0;; '
                '*rev-parse*) echo SAME;; esac; return 0; }'
            )
            r = run_real(setup, 'cmd_update')
            out = r.stdout + r.stderr
            self.assertIn("rc=0", out, out)
            self.assertIn("already broke the chat UI here", out)
            self.assertNotIn("COMPOSE RAN", out,
                             "it rebuilt and pulled the known-broken image again")


class TheStateFileRecordsTheUiSeparately(unittest.TestCase):
    """verified tracks the models, because that is what a rollback restores.
    The UI answer is kept beside it so the diagnosis is not lost."""

    STUBS = (
        'git() { case "$*" in *"rev-parse HEAD"*) echo abc123;; esac; return 0; }; '
        'docker() { case "$1" in image) echo "sha256:img";; esac; return 0; }'
    )

    def _record(self, ui_was_up):
        d = tempfile.mkdtemp()
        try:
            state = os.path.join(d, "s")
            run_real(f'{self.STUBS}; STATE={shlex.quote(state)}; '
                     f'UI_WAS_UP={ui_was_up}', 'record_rollback_point 1')
            with open(state) as fh:
                return fh.read()
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_a_live_ui_is_recorded(self):
        self.assertIn("ui=1", self._record(1))

    def test_a_dead_ui_is_recorded(self):
        body = self._record(0)
        self.assertIn("ui=0", body)
        self.assertIn("verified=1", body,
                      "the models were proven; the UI is a separate fact")


class ABigRewindIsAnnounced(unittest.TestCase):
    def test_rolling_back_more_than_one_commit_says_so(self):
        """The recorded point can predate this run's own change, and a
        multi-week rewind logged as "rollback verified" is the kind of success
        nobody wants to discover later."""
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(["git", "init", "-q", d], check=True, capture_output=True)
            for k, v in (("user.email", "t@t"), ("user.name", "t")):
                subprocess.run(["git", "-C", d, "config", k, v], check=True,
                               capture_output=True)
            shas = []
            for i in range(4):
                with open(os.path.join(d, "f.txt"), "w") as fh:
                    fh.write(f"{i}\n")
                subprocess.run(["git", "-C", d, "add", "-A"], check=True,
                               capture_output=True)
                subprocess.run(["git", "-C", d, "commit", "-qm", f"c{i}"],
                               check=True, capture_output=True)
                shas.append(subprocess.run(["git", "-C", d, "rev-parse", "HEAD"],
                                           capture_output=True, text=True).stdout.strip())
            state = os.path.join(d, "s")
            with open(state, "w") as fh:
                fh.write(f"sha={shas[0]}\nwebui=\nllama=\nverified=1\nui=1\n")
            setup = (
                f'ROOT={shlex.quote(d)}; STATE={shlex.quote(state)}; '
                'docker() { return 0; }; compose() { return 0; }; '
                'wait_for_health() { return 0; }; smoke() { return 0; }; '
                'webui_ok() { return 0; }; stash_local_edits() { return 0; }'
            )
            r = run_real(setup, 'do_rollback')
            self.assertIn("rewinds 3 commits", r.stderr,
                          f"a 3-commit rewind went unannounced: {r.stderr}")


class AUiAlreadyDownIsNotTheCommitsFault(unittest.TestCase):
    """webui_ok was probed only when pre_verdict == 0, so a run where smoke
    returned 2 never looked at the UI — and the post-update check then blamed
    the commit for a UI that had been dead since last week's reboot.

    Scenario: timer fires at 04:30 while someone is chatting, the router
    answers "model limit reached" to every alias, smoke returns 2, the UI is
    never probed. A good commit is merged and built, llama comes up, webui_ok
    fails for the pre-existing reason, no new image was pulled, and the good
    commit is blacklisted until a human edits run/update-state.
    """

    def _run(self, ui_before_ok):
        first = "0" if ui_before_ok else "1"
        setup = (
            'preflight() { :; }; take_lock() { :; }; docker() { return 0; }; '
            'is_pinned_bad() { return 1; }; record_rollback_point() { :; }; '
            'wait_for_health() { return 0; }; smoke() { return 2; }; '
            'compose() { return 0; }; stash_local_edits() { :; }; '
            'do_rollback() { echo "ROLLBACK pin=[${1:-}]"; return 0; }; '
            f'W=0; webui_ok() {{ W=$((W+1)); [ "$W" = 1 ] && return {first}; return 1; }}; '
            'local_image_digest() { echo same; }; remote_image_digest() { echo other; }; '
            'git() { case "$*" in *"merge-base --is-ancestor"*) return 1;; '
            '*"rev-parse HEAD"*) echo OLD;; *rev-parse*) echo NEW;; esac; return 0; }'
        )
        r = run_real(setup, 'cmd_update')
        return r.stdout + r.stderr

    def test_a_ui_already_down_does_not_roll_back_at_all(self):
        """Rolling back cannot start a container that was already stopped —
        and aborting there skipped the model smoke test, letting a commit that
        killed llama escape unpinned behind the stopped UI."""
        out = self._run(ui_before_ok=False)
        self.assertIn("not rolling back for it", out)
        self.assertIn("smoke testing every model", out,
                      "the run must carry on to the models")

    def test_a_ui_that_was_up_and_then_died_does_pin(self):
        out = self._run(ui_before_ok=True)
        self.assertIn("ROLLBACK pin=[pin]", out,
                      "the UI worked before this run and does not now")

    def test_the_ui_is_probed_even_when_smoke_cannot_judge(self):
        out = self._run(ui_before_ok=False)
        self.assertIn("is not answering either", out,
                      "the UI must be probed regardless of smoke's verdict")


class ADeadUiDoesNotShieldABadCommit(unittest.TestCase):
    """The pre-check used to overload one variable: a chat UI that was merely
    stopped forced the "already failing" branch, which then disarmed the pin
    for a genuine MODEL outage. A commit that killed llama was rolled back
    UNPINNED and walked onto again every Monday — until someone happened to
    start the UI container."""

    def _run(self, ui_up, models_pre, models_post):
        setup = (
            'preflight() { :; }; take_lock() { :; }; docker() { return 0; }; '
            'is_pinned_bad() { return 1; }; record_rollback_point() { :; }; '
            'wait_for_health() { return 0; }; compose() { return 0; }; '
            'stash_local_edits() { :; }; '
            'do_rollback() { echo "ROLLBACK pin=[${1:-}]"; return 0; }; '
            f'webui_ok() {{ return {0 if ui_up else 1}; }}; '
            f'N=0; smoke() {{ N=$((N+1)); [ "$N" = 1 ] && return {models_pre}; '
            f'return {models_post}; }}; '
            'local_image_digest() { echo same; }; remote_image_digest() { echo other; }; '
            'git() { case "$*" in *"merge-base --is-ancestor"*) return 1;; '
            '*"rev-parse HEAD"*) echo OLD;; *rev-parse*) echo NEW;; esac; return 0; }'
        )
        r = run_real(setup, 'cmd_update')
        return r.stdout + r.stderr

    def test_a_stopped_ui_does_not_disarm_the_model_pin(self):
        out = self._run(ui_up=False, models_pre=0, models_post=1)
        self.assertIn("ROLLBACK pin=[pin]", out,
                      "a stopped chat UI shielded a commit that killed llama")

    def test_models_proven_dead_beforehand_still_disarms_it(self):
        out = self._run(ui_up=True, models_pre=1, models_post=1)
        self.assertIn("ROLLBACK pin=[]", out)

    def test_a_healthy_stack_broken_by_the_update_pins(self):
        out = self._run(ui_up=True, models_pre=0, models_post=1)
        self.assertIn("ROLLBACK pin=[pin]", out)


class ADefinitelyDeadUiFailsTheRun(unittest.TestCase):
    """On the nothing-to-do path, models_verdict 2 returns 0 — but the UI
    answer is never ambiguous. With smoke unable to judge AND the UI down, the
    weekly unit succeeded and nobody learned the chat interface was dead."""

    def _run(self, ui_up, models):
        setup = (
            'preflight() { :; }; take_lock() { :; }; docker() { return 0; }; '
            'is_pinned_bad() { return 1; }; record_rollback_point() { :; }; '
            'wait_for_health() { return 0; }; compose() { return 0; }; '
            f'webui_ok() {{ return {0 if ui_up else 1}; }}; '
            f'smoke() {{ return {models}; }}; '
            'local_image_digest() { echo same; }; remote_image_digest() { echo same; }; '
            'git() { case "$*" in *"merge-base --is-ancestor"*) return 0;; '
            '*rev-parse*) echo SAME;; esac; return 0; }'
        )
        r = run_real(setup, 'cmd_update')
        return r.stdout + r.stderr

    def test_unjudgeable_models_plus_a_dead_ui_fails(self):
        out = self._run(ui_up=False, models=2)
        self.assertNotIn("rc=0", out,
                         "the unit succeeded while the chat UI was down")
        self.assertIn("chat UI", out)

    def test_unjudgeable_models_with_a_live_ui_still_passes(self):
        out = self._run(ui_up=True, models=2)
        self.assertIn("rc=0", out, out)

    def test_everything_healthy_passes(self):
        out = self._run(ui_up=True, models=0)
        self.assertIn("rc=0", out, out)
        self.assertIn("already up to date", out)


class ABadRequestIsNotADeadModel(unittest.TestCase):
    """400 is by definition client-side — a llama.cpp release tightening
    validation, or an alias the router will not serve for this body. Falling
    through to smoke_ok scored it as a dead model: rollback AND pin."""

    def _smoke(self, code):
        one = json.dumps({"data": [{"id": "a"}]})
        body = '{"error":{"message":"invalid request"}}'
        stub = (
            'ask() { case "$1" in '
            f'*models) ASK_CODE=200; ASK_BODY={shlex.quote(one)};; '
            f'*) ASK_CODE={code}; ASK_BODY={shlex.quote(body)};; '
            'esac; return 0; }'
        )
        r = run_real(f'API=http://stub; BONSAI_API_KEY=k; {stub}', 'smoke')
        return r.stdout + r.stderr

    def test_400_cannot_be_judged(self):
        self.assertIn("rc=2", self._smoke(400),
                      "a bad request must not blacklist a commit")

    def test_401_is_still_cannot_judge(self):
        self.assertIn("rc=2", self._smoke(401))

    def test_500_is_still_a_real_failure(self):
        self.assertIn("rc=1", self._smoke(500))


class ManualRollbackRunsWithoutPreSeeding(unittest.TestCase):
    """UI_WAS_UP was only ever assigned inside cmd_update, so
    `./update.sh --rollback` hit `(( UI_WAS_UP ))` with the variable unset —
    and under `set -u` that aborts the shell mid-rollback. The operator got a
    raw "unbound variable" instead of the verdict the script exists to
    produce, with neither the restore summary nor the final line printed.

    Every existing test injected UI_WAS_UP itself, which is exactly why the
    suite was green while the real entry point was broken.
    """

    def _rollback_without_seeding(self, ui_answers):
        with tempfile.TemporaryDirectory() as d:
            state = os.path.join(d, "s")
            with open(state, "w") as fh:
                fh.write("sha=\nwebui=\nllama=\nverified=1\n")
            setup = (
                f'ROOT={shlex.quote(d)}; STATE={shlex.quote(state)}; '
                'docker() { return 0; }; compose() { return 0; }; '
                'wait_for_health() { return 0; }; smoke() { return 0; }; '
                f'webui_ok() {{ return {0 if ui_answers else 1}; }}'
            )
            return run_real(setup, 'do_rollback')

    def test_it_does_not_abort_on_an_unbound_variable(self):
        r = self._rollback_without_seeding(ui_answers=False)
        out = r.stdout + r.stderr
        self.assertNotIn("unbound variable", out,
                         "the shell died partway through the rollback")
        self.assertIn("rc=", r.stdout, "do_rollback never returned")

    def test_it_still_reaches_its_verdict(self):
        r = self._rollback_without_seeding(ui_answers=True)
        self.assertIn("rc=", r.stdout, r.stderr)

    def test_the_default_is_defined_at_global_scope(self):
        with open(UPDATE_SH) as fh:
            lines = fh.read().splitlines()
        # Unindented means global scope — inside a function it would not help
        # the --rollback path, which is the one that was aborting.
        self.assertIn("UI_WAS_UP=0", lines,
                      "only cmd_update set it, so --rollback ran with it unset")

    def test_the_manual_path_probes_the_ui_first(self):
        """So the verdict can tell "the rollback did not bring it back" from
        "it was already down" — with one attempt, not the polling window."""
        with open(UPDATE_SH) as fh:
            body = fh.read()
        arm = body[body.index("  --rollback)"):body.index("  -h|--help)")]
        self.assertIn("webui_answers_now && UI_WAS_UP=1", arm)
        self.assertLess(arm.index("UI_WAS_UP=1"), arm.index("do_rollback"),
                        "the probe must happen before the rewind")


class ContentPartsDoNotLookLikeADeadModel(unittest.TestCase):
    def _ok(self, body):
        r = subprocess.run(
            ["bash", "-c",
             f'export BONSAI_UPDATE_LIB=1; source "{UPDATE_SH}"; smoke_ok'],
            input=body, capture_output=True, text=True)
        return r.returncode == 0, r.stderr

    def test_a_list_content_is_a_real_generation(self):
        """The OpenAI content-parts shape. `list + str` raises TypeError,
        which exits non-zero and reads as a dead model: rollback AND pin."""
        body = json.dumps({"choices": [{"message": {
            "content": [{"type": "text", "text": "OK"}]}}]})
        ok, err = self._ok(body)
        self.assertTrue(ok, f"a healthy reply scored as dead: {err}")
        self.assertNotIn("TypeError", err)

    def test_a_null_content_with_reasoning_still_passes(self):
        body = json.dumps({"choices": [{"message": {
            "content": None, "reasoning_content": "thinking"}}]})
        ok, err = self._ok(body)
        self.assertTrue(ok, err)

    def test_an_empty_reply_is_still_a_failure(self):
        ok, _ = self._ok(json.dumps({"choices": [{"message": {"content": ""}}]}))
        self.assertFalse(ok)

    def test_structurally_empty_content_is_not_a_generation(self):
        """The str() fallback that fixed the list crash turned [] into "[]",
        {} into "{}", 0 into "0" and False into "False" — all non-empty, so a
        reply with no content at all scored as a real generation. The one
        check this whole script exists to perform would have reported a dead
        model as healthy."""
        for empty in ([], {}, 0, False):
            body = json.dumps({"choices": [{"message": {"content": empty}}]})
            with self.subTest(content=empty):
                ok, err = self._ok(body)
                self.assertFalse(ok, f"{empty!r} scored as a real generation")
                self.assertNotIn("Traceback", err)

    def test_content_parts_with_no_text_is_not_a_generation(self):
        body = json.dumps({"choices": [{"message": {
            "content": [{"type": "image_url", "image_url": {"url": "x"}}]}}]})
        ok, _ = self._ok(body)
        self.assertFalse(ok, "a reply with no text scored as a generation")

    def test_multiple_content_parts_are_joined(self):
        body = json.dumps({"choices": [{"message": {"content": [
            {"type": "text", "text": "OK"},
            {"type": "text", "text": " then"}]}}]})
        ok, err = self._ok(body)
        self.assertTrue(ok, err)


class RollbackDoesNotBlameItselfForAStoppedUi(unittest.TestCase):
    def test_a_ui_down_beforehand_is_not_a_failed_rollback(self):
        """Saying "ROLLBACK DID NOT RECOVER THE STACK" contradicted the same
        run's own "not blaming the commit for the UI", and pointed the
        operator at the rollback path instead of the stopped container."""
        with tempfile.TemporaryDirectory() as d:
            state = os.path.join(d, "s")
            with open(state, "w") as fh:
                fh.write("sha=\nwebui=\nllama=\nverified=1\n")
            setup = (
                f'ROOT={shlex.quote(d)}; STATE={shlex.quote(state)}; '
                'UI_WAS_UP=0; docker() { return 0; }; compose() { return 0; }; '
                'wait_for_health() { return 0; }; smoke() { return 0; }; '
                'webui_ok() { return 1; }'
            )
            r = run_real(setup, 'do_rollback')
            out = r.stdout + r.stderr
            self.assertIn("still down, as it was", out)
            self.assertNotIn("DID NOT RECOVER", out)

    def test_a_ui_that_was_up_and_did_not_come_back_is_a_failure(self):
        with tempfile.TemporaryDirectory() as d:
            state = os.path.join(d, "s")
            with open(state, "w") as fh:
                fh.write("sha=\nwebui=\nllama=\nverified=1\n")
            setup = (
                f'ROOT={shlex.quote(d)}; STATE={shlex.quote(state)}; '
                'UI_WAS_UP=1; docker() { return 0; }; compose() { return 0; }; '
                'wait_for_health() { return 0; }; smoke() { return 0; }; '
                'webui_ok() { return 1; }'
            )
            r = run_real(setup, 'do_rollback')
            self.assertIn("does not", r.stdout + r.stderr)


class EnvironmentBeatsDotEnv(unittest.TestCase):
    """`set -a; . .env` assigns unconditionally, so .env silently beat an
    explicit `BONSAI_HEALTH_TIMEOUT=60 ./update.sh`. It also defeated this
    harness's short deadlines on any machine with a real .env — measured:
    a 5s suite became 30s with one knob set, and would poll for 15 minutes
    per unstubbed call at the value the docs recommend."""

    def _knobs(self, env_file_values, overrides):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "docker"))
            copy = os.path.join(d, "bonsai-update.TESTX")
            shutil.copy(UPDATE_SH, copy)
            with open(os.path.join(d, ".env"), "w") as fh:
                fh.write("BONSAI_API_KEY=k\n" + env_file_values)
            pre = "".join(f'export {k}={v}; ' for k, v in overrides.items())
            script = (
                'set -uo pipefail; export BONSAI_UPDATE_LIB=1; '
                f'export BONSAI_UPDATE_REEXEC={shlex.quote(copy)}; '
                f'export BONSAI_UPDATE_ROOT={shlex.quote(d)}; {pre}'
                f'source {shlex.quote(copy)}; '
                'echo "H=$HEALTH_TIMEOUT W=$WEBUI_TIMEOUT C=$CHAT_TIMEOUT"'
            )
            r = subprocess.run(["bash", "-c", script], capture_output=True,
                               text=True)
            return r.stdout.strip(), r.stderr

    def test_dot_env_applies_when_nothing_is_exported(self):
        out, err = self._knobs("BONSAI_HEALTH_TIMEOUT=25\n", {})
        self.assertIn("H=25", out, err)

    def test_an_explicit_environment_wins(self):
        out, err = self._knobs("BONSAI_HEALTH_TIMEOUT=25\nBONSAI_WEBUI_TIMEOUT=26\n",
                               {"BONSAI_HEALTH_TIMEOUT": "7",
                                "BONSAI_WEBUI_TIMEOUT": "8"})
        self.assertIn("H=7", out, err)
        self.assertIn("W=8", out, err)

    def test_the_harness_pins_every_deadline(self):
        """A missing stub must fail fast, not poll."""
        with open(__file__) as fh:
            harness = fh.read()
        for knob in ("BONSAI_HEALTH_TIMEOUT=1", "BONSAI_WEBUI_TIMEOUT=1",
                     "BONSAI_SMOKE_RETRY_SLEEP=0"):
            self.assertIn(knob, harness)
        self.assertIn("WEBUI_URL=http://127.0.0.1:9", harness,
                      "tests must not be able to reach a live box")


class MalformedRepliesDoNotBlacklist(unittest.TestCase):
    """smoke_ok is the one that can pin a commit, so a shape it did not expect
    must not read as a dead model."""

    def _ok(self, body):
        r = subprocess.run(
            ["bash", "-c",
             f'export BONSAI_UPDATE_LIB=1; source "{UPDATE_SH}"; smoke_ok'],
            input=body, capture_output=True, text=True)
        return r.returncode == 0, r.stderr

    def test_a_non_dict_choice_does_not_raise(self):
        ok, err = self._ok('{"choices": ["not a dict"]}')
        self.assertFalse(ok)
        self.assertNotIn("AttributeError", err,
                         "a traceback in the journal instead of a verdict")

    def test_a_non_dict_message_does_not_raise(self):
        ok, err = self._ok('{"choices": [{"message": "not a dict"}]}')
        self.assertFalse(ok)
        self.assertNotIn("AttributeError", err)

    def test_a_non_dict_model_entry_does_not_raise(self):
        r = subprocess.run(
            ["bash", "-c",
             f'export BONSAI_UPDATE_LIB=1; source "{UPDATE_SH}"; parse_model_ids'],
            input='{"data": ["bare string", {"id": "real"}]}',
            capture_output=True, text=True)
        self.assertNotIn("AttributeError", r.stderr)
        self.assertIn("real", r.stdout)

    def test_a_good_reply_still_passes(self):
        ok, _ = self._ok(GOOD_CONTENT)
        self.assertTrue(ok)


class TheWeeklyCheckCoversTheUi(unittest.TestCase):
    """webui_ok was only wired to the POST-update path, so the common case —
    code current, images current — returned 0 having proved only that the
    models answer. A UI stopped or crash-looping since a host reboot went
    unnoticed every Monday, and record_rollback_point wrote verified=1 from a
    models-only proof."""

    BASE = (
        'preflight() { :; }; take_lock() { :; }; docker() { return 0; }; '
        'is_pinned_bad() { return 1; }; wait_for_health() { return 0; }; '
        'compose() { echo "COMPOSE RAN"; return 0; }; '
        'record_rollback_point() { echo "RECORD verified=$1"; }; '
        'local_image_digest() { echo same; }; remote_image_digest() { echo same; }; '
        'git() { case "$*" in *"merge-base --is-ancestor"*) return 0;; '
        '*rev-parse*) echo SAME;; esac; return 0; }'
    )

    def test_a_dead_ui_is_noticed_even_with_nothing_to_update(self):
        r = run_real(f'{self.BASE}; smoke() {{ return 0; }}; '
                     'webui_ok() { echo "WEBUI CHECKED"; return 1; }', 'cmd_update')
        out = r.stdout + r.stderr
        self.assertIn("WEBUI CHECKED", out,
                      "the weekly run never looked at the chat UI")
        self.assertNotIn("rc=0", out, "a dead UI must not report success")
        # verified reflects the MODELS, which is what a rollback can restore.
        # Requiring the UI too froze the rollback target on a box where Open
        # WebUI is deliberately stopped: it never became 1 again, the
        # keep-the-proven-point guard held the old commit, and updates kept
        # applying — so a failure weeks later rewound every commit since. The
        # UI is recorded separately, as ui=.
        self.assertIn("RECORD verified=1", out)

    def test_a_healthy_box_still_reports_nothing_to_do(self):
        r = run_real(f'{self.BASE}; smoke() {{ return 0; }}; '
                     'webui_ok() { return 0; }', 'cmd_update')
        out = r.stdout + r.stderr
        self.assertIn("rc=0", out, out)
        self.assertIn("already up to date", out)
        self.assertIn("RECORD verified=1", out)

    def test_could_not_judge_is_not_a_weekly_false_alarm(self):
        """Someone chatting inside the 04:00-08:00 window makes the router
        answer "model limit reached" to every alias, so smoke returns 2. The
        unit used to fail every week asserting an outage nobody observed."""
        r = run_real(f'{self.BASE}; smoke() {{ return 2; }}; '
                     'webui_ok() { return 0; }', 'cmd_update')
        out = r.stdout + r.stderr
        self.assertIn("rc=0", out, "an unjudgeable run must not fail the unit")
        self.assertIn("could not be judged", out)
        self.assertNotIn("did NOT pass their check", out,
                         "that asserts a failure that was never observed")

    def test_a_genuinely_dead_stack_still_fails(self):
        r = run_real(f'{self.BASE}; smoke() {{ return 1; }}; '
                     'webui_ok() { return 0; }', 'cmd_update')
        out = r.stdout + r.stderr
        self.assertNotIn("rc=0", out)
        self.assertIn("did NOT pass their check", out)


class AChatUiBrokenByTheCommitDoesPin(unittest.TestCase):
    """docker/compose.linux.yaml defines the open-webui service — its
    environment, ports and volumes — and is version-controlled. A commit
    adding a bad WEBUI_* variable or a colliding port breaks the UI with no
    new image involved. Refusing to pin there let that commit be fetched,
    rebuilt (~45 min of CUDA) and rolled back every week, forever."""

    def _run(self, image_changed):
        # A marker FILE, not a counter: local_image_digest is called inside
        # command substitution, so a shell variable bumped there lives in a
        # subshell and never advances. The compose stub drops the marker on
        # `pull`, which is exactly when a new image would arrive.
        d = tempfile.mkdtemp()
        try:
            marker = shlex.quote(os.path.join(d, "pulled"))
            pull = f'touch {marker}' if image_changed else ':'
            setup = (
                'preflight() { :; }; take_lock() { :; }; docker() { return 0; }; '
                'is_pinned_bad() { return 1; }; record_rollback_point() { :; }; '
                'wait_for_health() { return 0; }; smoke() { return 0; }; '
                f'compose() {{ case "$1" in pull) {pull};; esac; return 0; }}; '
                'stash_local_edits() { :; }; '
                'do_rollback() { echo "ROLLBACK pin=[${1:-}]"; return 0; }; '
                # passes the pre-check, fails after the update
                'W=0; webui_ok() { W=$((W+1)); [ "$W" = 1 ] && return 0; return 1; }; '
                f'local_image_digest() {{ if [ -e {marker} ]; then echo new; '
                'else echo same; fi; }; '
                'remote_image_digest() { echo other; }; '
                'git() { case "$*" in *"merge-base --is-ancestor"*) return 1;; '
                '*"rev-parse HEAD"*) echo OLD;; *rev-parse*) echo NEW;; esac; return 0; }'
            )
            r = run_real(setup, 'cmd_update')
            return r.stdout + r.stderr
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_no_new_image_means_the_commit_is_to_blame(self):
        out = self._run(image_changed=False)
        self.assertIn("ROLLBACK pin=[pin]", out,
                      "a UI broken with no new image is the commit's doing")
        self.assertIn("no new image was", out)

    def test_a_new_image_is_not_the_commits_fault(self):
        out = self._run(image_changed=True)
        self.assertIn("ROLLBACK pin=[]", out,
                      ":main moved under us; do not blacklist the commit")
        self.assertIn("after a new image was", out)


class TheUiHasItsOwnDeadline(unittest.TestCase):
    def test_webui_uses_its_own_timeout(self):
        """Open WebUI's first start after a :main pull runs DB migrations,
        which outlast a llama restart. Borrowing BONSAI_HEALTH_TIMEOUT left
        the operator with no documented knob for a rollback it caused."""
        with open(UPDATE_SH) as fh:
            body = fh.read()
        fn = body[body.index("webui_ok() {"):body.index("# Exit status, and it has")]
        self.assertIn("WEBUI_TIMEOUT", fn)
        self.assertNotIn("HEALTH_TIMEOUT", fn)

    def test_the_knob_is_documented(self):
        for rel in ((".env.example",), ("docs", "updating.md")):
            with open(os.path.join(ROOT, *rel)) as fh:
                self.assertIn("BONSAI_WEBUI_TIMEOUT", fh.read(),
                              f"{'/'.join(rel)} does not mention the knob")


class CheckFailsLoudWhenItCannotFetch(unittest.TestCase):
    def test_a_failed_fetch_is_not_up_to_date(self):
        """rev-list counts against the stale ref and returns 0, so --check
        printed "code up to date" and exited 0 during a DNS or GHCR outage —
        while the branches either side of it refuse to report from a
        comparison that did not happen."""
        setup = (
            f'{_TOOLS_PRESENT}; ROOT=/tmp; '
            'docker() { case "$*" in *version*) return 0;; esac; return 0; }; '
            'local_image_digest() { echo same; }; remote_image_digest() { echo same; }; '
            'git() { case "$*" in *fetch*) return 1;; '
            '*rev-list*) echo 0;; esac; return 0; }'
        )
        r = run_real(setup, 'cmd_check')
        out = r.stdout + r.stderr
        self.assertNotIn("rc=0", out, "a failed fetch must not report success")
        self.assertIn("commit drift unknown", out)
        self.assertNotIn("code up to date", out)

    def test_a_good_fetch_with_no_drift_is_up_to_date(self):
        setup = (
            f'{_TOOLS_PRESENT}; ROOT=/tmp; '
            'docker() { return 0; }; '
            'local_image_digest() { echo same; }; remote_image_digest() { echo same; }; '
            'git() { case "$*" in *rev-list*) echo 0;; esac; return 0; }'
        )
        r = run_real(setup, 'cmd_check')
        out = r.stdout + r.stderr
        self.assertIn("rc=0", out, out)
        self.assertIn("code up to date", out)


class TheChatUiIsHalfTheStack(unittest.TestCase):
    """compose pull replaces a moving :main tag and compose up recreates the
    container, but every check aimed at llama alone — so an open-webui image
    that crash-loops left compose returning 0, every model answering, and the
    run logging "update complete and verified" while the UI was down."""

    BASE = (
        'preflight() { :; }; take_lock() { :; }; docker() { return 0; }; '
        'is_pinned_bad() { return 1; }; record_rollback_point() { :; }; '
        'wait_for_health() { return 0; }; smoke() { return 0; }; '
        'compose() { return 0; }; '
        'do_rollback() { echo "ROLLBACK pin=[${1:-}]"; return 0; }; '
        'local_image_digest() { echo old; }; remote_image_digest() { echo new; }; '
        'git() { case "$*" in *"merge-base --is-ancestor"*) return 0;; '
        '*rev-parse*) echo SAME;; esac; return 0; }'
    )

    def test_a_ui_down_throughout_does_not_report_success(self):
        """It was down before the run, so rolling back cannot start it — but
        "verified" would be a lie while half the stack is dark, and exiting 0
        tells the timer everything is fine."""
        r = run_real(f'{self.BASE}; webui_ok() {{ return 1; }}', 'cmd_update')
        out = r.stdout + r.stderr
        self.assertNotIn("update complete and verified", out)
        self.assertNotIn("rc=0", out, "the unit succeeded with the UI down")
        self.assertIn("still down", out)

    def test_a_ui_down_throughout_does_not_roll_back(self):
        """Rolling back cannot start a container that was already stopped, and
        aborting there would skip the model smoke test."""
        r = run_real(f'{self.BASE}; webui_ok() {{ return 1; }}', 'cmd_update')
        out = r.stdout + r.stderr
        self.assertNotIn("ROLLBACK", out)
        self.assertIn("smoke testing every model", out)

    def test_a_live_ui_and_live_models_pass(self):
        r = run_real(f'{self.BASE}; webui_ok() {{ return 0; }}', 'cmd_update')
        out = r.stdout + r.stderr
        self.assertIn("update complete and verified", out, out)

    def test_the_rollback_also_proves_the_ui(self):
        with open(UPDATE_SH) as fh:
            body = fh.read()
        section = body[body.index("do_rollback() {"):body.index("cmd_check() {")]
        self.assertIn("webui_ok", section,
                      "a rollback that restores a broken UI is not verified")


class HeadAheadOfUpstreamIsCurrent(unittest.TestCase):
    """Comparing shas said "not current" whenever the checkout carried a local
    commit on top of @{u}. The run then merged an ANCESTOR (git prints
    "Already up to date", HEAD does not move), rebuilt, pulled, force-restarted
    and swept twice — every Monday, to change nothing — while logging "moving
    to <upstream-sha>" and arming the pin on a run that moved no code."""

    def _current(self, ancestor_rc):
        setup = (
            f'ROOT=/tmp; git() {{ case "$*" in '
            f'*"merge-base --is-ancestor"*) return {ancestor_rc};; '
            'esac; return 0; }'
        )
        r = run_real(setup, 'code_already_current abc123')
        return "rc=0" in r.stdout

    def test_head_containing_the_target_counts_as_current(self):
        self.assertTrue(self._current(0))

    def test_a_target_not_in_head_is_a_real_move(self):
        self.assertFalse(self._current(1))

    def test_it_uses_ancestry_not_sha_equality(self):
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertIn("merge-base --is-ancestor", body)
        self.assertIn("code_already_current", body)
        self.assertNotIn('[[ "$target" == "$head_sha" ]]', body)
        self.assertNotIn('[[ "$target" != "$head_sha" ]]', body)


class BuildFailureStashesToo(unittest.TestCase):
    def test_the_merge_path_stashes_before_resetting(self):
        """do_rollback stashes before its reset for exactly this reason; the
        merge path had been left out. `git merge --ff-only` refuses only on
        COLLIDING paths, so an uncommitted edit to an untouched file reaches
        the build failure alive and would be erased without a word."""
        with open(UPDATE_SH) as fh:
            body = fh.read()
        section = body[body.index("if ! compose build llama; then"):
                       body.index('log "pulling open-webui"')]
        self.assertIn("stash_local_edits", section)
        self.assertLess(section.index("stash_local_edits"),
                        section.index("reset --hard"),
                        "the stash must happen before the reset")


class CheckDoesNotGuessAboutTheDaemon(unittest.TestCase):
    def test_an_unreachable_daemon_is_reported_as_unknown(self):
        """preflight only checks the docker BINARY. With the daemon stopped,
        or the user not in the docker group, local_image_digest returns empty
        and --check claimed the image was missing — sending the operator to
        pull something already on disk."""
        with open(UPDATE_SH) as fh:
            body = fh.read()
        section = body[body.index("cmd_check() {"):body.index("cmd_update() {")]
        self.assertIn("cannot reach the docker daemon", section)
        self.assertLess(section.index("cannot reach the docker daemon"),
                        section.index("not present locally"),
                        "the daemon check must come first")


class DocsQuoteTheRealDefaults(unittest.TestCase):
    def test_the_sweep_default_matches_the_code(self):
        """An operator sizing TimeoutStartSec from the table under-budgeted by
        900s per sweep, and that unit exists to avoid SIGTERM mid-rollback."""
        with open(os.path.join(ROOT, "docs", "updating.md")) as fh:
            docs = fh.read()
        with open(UPDATE_SH) as fh:
            body = fh.read()
        multiplier = "4" if "CHAT_TIMEOUT * 4" in body else "3"
        self.assertIn(f"| `BONSAI_SMOKE_SWEEP_MAX` | {multiplier}x the above |",
                      docs, "the docs table disagrees with the code default")


class RollbackDoesNotEatLocalEdits(unittest.TestCase):
    """The same data loss commit 4a2d9a8 fixed on the build path. An
    image-only run whose new open-webui image fails calls do_rollback with a
    recorded sha equal to HEAD — a reset that undoes nothing and can only
    delete."""

    def _repo(self, d):
        def git(*a):
            subprocess.run(["git", "-C", d, *a], check=True,
                           capture_output=True)
        subprocess.run(["git", "init", "-q", d], check=True,
                       capture_output=True)
        git("config", "user.email", "t@t"); git("config", "user.name", "t")
        with open(os.path.join(d, "models.ini.in"), "w") as fh:
            fh.write("committed\n")
        git("add", "-A"); git("commit", "-qm", "base")
        head = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        return head

    def _rollback(self, d, sha):
        state = os.path.join(d, "update-state")
        with open(state, "w") as fh:
            fh.write(f"sha={sha}\nwebui=\nllama=\nverified=1\n")
        setup = (
            f'ROOT={shlex.quote(d)}; STATE={shlex.quote(state)}; '
            'docker() { return 0; }; compose() { return 0; }; webui_ok() { return 0; }; '
            'wait_for_health() { return 0; }; webui_ok() { return 0; }; smoke() { return 0; }'
        )
        return run_real(setup, 'do_rollback')

    def test_an_uncommitted_edit_survives_a_no_op_rollback(self):
        with tempfile.TemporaryDirectory() as d:
            head = self._repo(d)
            with open(os.path.join(d, "models.ini.in"), "w") as fh:
                fh.write("MY UNCOMMITTED WORK\n")
            r = self._rollback(d, head)          # sha == HEAD: nothing to undo
            with open(os.path.join(d, "models.ini.in")) as fh:
                self.assertEqual(fh.read(), "MY UNCOMMITTED WORK\n",
                                 "a rollback that moved nothing deleted the "
                                 f"operator's work: {r.stderr}")
            self.assertIn("leaving it alone", r.stdout + r.stderr)

    def test_a_real_rewind_stashes_instead_of_discarding(self):
        with tempfile.TemporaryDirectory() as d:
            first = self._repo(d)
            with open(os.path.join(d, "other.txt"), "w") as fh:
                fh.write("second commit\n")
            subprocess.run(["git", "-C", d, "add", "-A"], check=True,
                           capture_output=True)
            subprocess.run(["git", "-C", d, "commit", "-qm", "second"],
                           check=True, capture_output=True)
            with open(os.path.join(d, "models.ini.in"), "w") as fh:
                fh.write("MY UNCOMMITTED WORK\n")
            r = self._rollback(d, first)          # a genuine rewind
            stash = subprocess.run(["git", "-C", d, "stash", "list"],
                                   capture_output=True, text=True).stdout
            self.assertIn("update.sh rollback", stash,
                          f"local edits were discarded, not stashed: {r.stderr}")


class RollbackAvoidsNoOpRecreate(unittest.TestCase):
    """A fully restored state must not evict the resident model just to retag it."""

    def test_matching_checkout_and_image_tags_leave_the_stack_running(self):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(["git", "init", "-q", d], check=True,
                           capture_output=True)
            subprocess.run(["git", "-C", d, "config", "user.email", "t@t"],
                           check=True, capture_output=True)
            subprocess.run(["git", "-C", d, "config", "user.name", "t"],
                           check=True, capture_output=True)
            with open(os.path.join(d, "tracked"), "w") as fh:
                fh.write("base\n")
            subprocess.run(["git", "-C", d, "add", "-A"], check=True,
                           capture_output=True)
            subprocess.run(["git", "-C", d, "commit", "-qm", "base"],
                           check=True, capture_output=True)
            head = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"],
                                  check=True, capture_output=True, text=True).stdout.strip()
            state = os.path.join(d, "update-state")
            with open(state, "w") as fh:
                fh.write(f"sha={head}\nwebui=sha256:webui\n"
                         "llama=sha256:llama\nverified=1\n")

            setup = (
                f'ROOT={shlex.quote(d)}; STATE={shlex.quote(state)}; '
                'docker() { case "$1:$3" in '
                'image:bonsai-llama:rollback) echo sha256:llama;; '
                'image:bonsai-llama:cuda) echo sha256:llama;; '
                'image:sha256:webui) return 0;; '
                'image:ghcr.io/open-webui/open-webui:main) echo sha256:webui;; '
                'tag:*) echo "TAG RAN";; esac; }; '
                'compose() { echo "COMPOSE RAN"; }; '
                'wait_for_health() { return 0; }; webui_ok() { return 0; }; '
                'smoke() { return 0; }'
            )
            r = run_real(setup, 'do_rollback')

            self.assertIn("rc=0", r.stdout, r.stderr)
            self.assertNotIn("TAG RAN", r.stdout + r.stderr)
            self.assertNotIn("COMPOSE RAN", r.stdout + r.stderr)


class SweepDeadlineBoundsRetries(unittest.TestCase):
    def test_the_ceiling_is_checked_per_attempt(self):
        """Checked only between aliases, one alias's retry loop could run
        3 x CHAT_TIMEOUT past the ceiling the systemd budget assumes."""
        with open(UPDATE_SH) as fh:
            body = fh.read()
        loop = body[body.index("for attempt in 1 2 3; do"):
                    body.index("(( busy )) || break")]
        self.assertIn("SECONDS > sweep_deadline", loop,
                      "the retry loop must respect the sweep ceiling")

    def test_the_default_ceiling_leaves_headroom(self):
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertIn("CHAT_TIMEOUT * 4", body,
                      "3x leaves nothing for /v1/models or backoff with "
                      "three aliases, so the last one is skipped")


class DeadOutranksUnknown(unittest.TestCase):
    """A model proven dead must outrank one that could not be asked. Returning
    2 because a LATER alias hit a rotated key threw away the evidence, and
    cmd_update then left the outage in production instead of rolling back."""

    def _smoke(self, replies):
        cases = "".join(
            f'{i}) ASK_CODE={c}; ASK_BODY={shlex.quote(b)};; '
            for i, (c, b) in enumerate(replies, start=1))
        two = json.dumps({"data": [{"id": "a"}, {"id": "b"}]})
        stub = (
            'N=0; ask() { case "$1" in '
            f'*models) ASK_CODE=200; ASK_BODY={shlex.quote(two)};; '
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

    def test_a_dead_model_then_a_rotated_key_still_rolls_back(self):
        out = self._smoke([(500, FAILED_LOAD), (401, '{"error":"nope"}')])
        self.assertIn("rc=1", out,
                      "the proven dead model was discarded in favour of "
                      "'cannot judge', leaving the outage in production")

    def test_only_unknowns_is_still_cannot_judge(self):
        out = self._smoke([(401, '{"error":"nope"}'), (401, '{"error":"nope"}')])
        self.assertIn("rc=2", out)

    def test_one_unknown_does_not_stop_the_others_being_judged(self):
        """The per-alias flag: a 401 on the first must not skip judging the
        second."""
        out = self._smoke([(401, '{"error":"nope"}'), (500, FAILED_LOAD)])
        self.assertIn("rc=1", out)


class TransientWordsInAGoodReply(unittest.TestCase):
    def test_a_200_saying_try_again_later_is_not_busy(self):
        """transient_reply used to run for every status, so a genuine
        completion whose TEXT contained "try again later" was retried three
        times and the whole run returned 'cannot judge'."""
        body = json.dumps({"choices": [{"message":
                          {"content": "Sure — try again later if it fails."}}]})
        one = json.dumps({"data": [{"id": "a"}]})
        stub = (
            'ask() { case "$1" in '
            f'*models) ASK_CODE=200; ASK_BODY={shlex.quote(one)};; '
            f'*) ASK_CODE=200; ASK_BODY={shlex.quote(body)};; '
            'esac; return 0; }'
        )
        script = (
            'set -uo pipefail; export BONSAI_UPDATE_LIB=1; '
            'export BONSAI_SMOKE_RETRY_SLEEP=0; '
            f'source "{UPDATE_SH}"; API=http://stub; BONSAI_API_KEY=k; '
            f'{stub}; smoke; echo "rc=$?"'
        )
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        out = r.stdout + r.stderr
        self.assertIn("rc=0", out, out)
        self.assertNotIn("is busy", out)


class KeyEncodingSurvivesOddCharacters(unittest.TestCase):
    """The API key goes into a curl config file, which has its own escaping.

    Two ways to get this wrong, both measured against the live box:
      quoted, unescaped -> curl ends the value at the first " and sends
        `Authorization: Bearer ab` for the key ab"cd\\ef;
      unquoted          -> curl sends NO Authorization header at all (401
        against the API, zero authorization lines under --trace-ascii).
    So: quoted AND escaped. This drives the real curl rather than asserting
    on the file's text, because the file's text was never the question.
    """

    ODD_KEY = 'ab"cd\\ef'

    def test_the_key_arrives_intact_through_curl(self):
        if not shutil.which("curl"):
            self.skipTest("curl not available")
        import http.server
        import threading

        received = {}

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                received["auth"] = self.headers.get("Authorization", "")
                self.send_response(200)
                self.end_headers()

            def log_message(self, *a):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.handle_request, daemon=True)
        t.start()
        try:
            d = tempfile.mkdtemp()
            try:
                run_real(
                    f'BONSAI_API_KEY={shlex.quote(self.ODD_KEY)}; '
                    f'TMPDIR={shlex.quote(d)}',
                    'curl_conf && curl -sS --config "$CURL_CONF" -o /dev/null '
                    f'-m 5 http://127.0.0.1:{port}/')
            finally:
                shutil.rmtree(d, ignore_errors=True)
        finally:
            t.join(timeout=5)
            srv.server_close()

        self.assertEqual(received.get("auth"), f"Bearer {self.ODD_KEY}",
                         "the key was mangled by the curl config encoding")

    def test_the_value_is_quoted_and_escaped(self):
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertIn('header = "Authorization: Bearer %s"', body,
                      "an unquoted value makes curl send no header at all")
        self.assertIn('esc="${BONSAI_API_KEY//', body, "the key must be escaped")


class AwkwardAliasesDoNotFakeAnOutage(unittest.TestCase):
    """Aliases come from /v1/models and, per models.ini.in, are ini section
    names — free-form operator-editable text. Interpolated straight into a
    JSON string, one containing a quote or a backslash produced malformed
    JSON, the server answered 400, and smoke_ok scored that as a dead model:
    roll back AND pin the commit."""

    def _body(self, alias):
        # ':' not '': run_real joins setup and call with ';', and an empty
        # setup would leave a leading ';' — a bash syntax error.
        r = run_real(':', f'smoke_body {shlex.quote(alias)}')
        self.assertIn("rc=0", r.stdout, r.stderr)
        # run_real appends an "rc=" line; the JSON is everything before it.
        return r.stdout.rsplit("rc=", 1)[0].strip()

    def test_a_quote_in_an_alias_still_yields_valid_json(self):
        alias = 'we"ird'
        parsed = json.loads(self._body(alias))
        self.assertEqual(parsed["model"], alias)

    def test_a_backslash_in_an_alias_still_yields_valid_json(self):
        alias = "back\\slash"
        parsed = json.loads(self._body(alias))
        self.assertEqual(parsed["model"], alias)

    def test_an_ordinary_alias_is_unchanged(self):
        parsed = json.loads(self._body("bonsai-27b-ternary"))
        self.assertEqual(parsed["model"], "bonsai-27b-ternary")
        self.assertEqual(parsed["max_tokens"], 8)

    def test_the_body_is_not_hand_interpolated(self):
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertIn('-d "$(smoke_body "$id")"', body)
        self.assertNotIn('\\"model\\":\\"$id\\"', body)


class BuildFailureUndoesOnlyItsOwnMove(unittest.TestCase):
    """`git merge --ff-only` refuses only when local edits collide with the
    merged files, so uncommitted edits to untouched files survive the merge.
    An unconditional `git reset --hard` after a failed build then destroyed
    them — on image-only runs it reset to the commit already checked out, so
    it could only ever delete work, never undo anything."""

    def test_the_reset_is_guarded_by_an_actual_move(self):
        with open(UPDATE_SH) as fh:
            body = fh.read()
        section = body[body.index("if ! compose build llama; then"):
                       body.index("log \"pulling open-webui\"")]
        # `moved`, not `pin_arg`. They answer different questions: pin_arg
        # also gets cleared when the stack was already broken beforehand, and
        # reusing it here meant that on an already-broken stack the merge DID
        # move HEAD while the handler announced "nothing to undo", skipped the
        # stash and the reset, and left the checkout on a new, unbuilt commit.
        self.assertIn("if (( moved )); then", section,
                      "the reset must run whenever this run actually merged")
        self.assertNotIn('if [[ -n "$pin_arg" ]]; then', section,
                         "pin_arg is cleared for reasons unrelated to moving")
        self.assertIn("nothing to undo", section,
                      "and must say so plainly when it did not")
        reset_at = section.index("reset --hard")
        guard_at = section.index("if (( moved ))")
        self.assertLess(guard_at, reset_at, "the guard must precede the reset")


class DocsMatchTheCode(unittest.TestCase):
    def test_the_pin_check_is_documented_after_the_stack_test(self):
        """The doc used to list the pin refusal before the pre-update check.
        An operator reading it would conclude a pinned box gets no weekly
        check — the exact thing the code was changed to avoid."""
        with open(os.path.join(ROOT, "docs", "updating.md")) as fh:
            body = fh.read()
        tests_at = body.index("**Tests the stack before changing it.**")
        pin_at = body.index("**Refuses a commit that already failed.**")
        self.assertLess(tests_at, pin_at,
                        "docs list the pin check before the stack test, but "
                        "cmd_update runs it after")


class DocumentedKnobsExist(unittest.TestCase):
    """The failure messages tell the operator to raise these in .env, so they
    have to be findable there."""

    def test_every_tunable_is_in_env_example(self):
        path = os.path.join(ROOT, ".env.example")
        with open(path) as fh:
            body = fh.read()
        for knob in ("BONSAI_SMOKE_TIMEOUT", "BONSAI_HEALTH_TIMEOUT",
                     "BONSAI_SMOKE_SWEEP_MAX", "BONSAI_SMOKE_RETRY_SLEEP"):
            self.assertIn(knob, body, f"{knob} is named in a warning but "
                                      "documented nowhere")

    def test_a_sweep_has_a_ceiling(self):
        with open(UPDATE_SH) as fh:
            body = fh.read()
        self.assertIn("sweep_deadline", body,
                      "without a ceiling the retry loop triples the worst "
                      "case and systemd SIGTERMs a rollback midway")


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
    merge is a no-op — but the run would still rebuild, restart and sweep
    again. With --models-max 1 and three aliases that is six cold 27B loads to
    prove nothing changed.

    The check and the rollback-point write still happen first, though: on a
    box that never drifts they are the only thing that would ever notice a
    dark stack, and without them --rollback would have nothing to restore.
    """

    STUBS = (
        'preflight() { :; }; take_lock() { :; }; '
        'git() { case "$*" in '
        '*"merge-base --is-ancestor"*) return 0;; '
        '*"rev-parse HEAD"*) echo SAMECOMMIT;; '
        '*rev-parse*) echo SAMECOMMIT;; esac; return 0; }; '
        'docker() { return 0; }; '
        'is_pinned_bad() { return 1; }; '
        'record_rollback_point() { echo "RECORDED verified=$1"; }; '
        'wait_for_health() { return 0; }; webui_ok() { return 0; }; '
        'compose() { echo "COMPOSE RAN"; return 0; }'
    )

    def test_it_checks_the_stack_then_returns_without_rebuilding(self):
        setup = (
            f'{self.STUBS}; smoke() {{ echo "SMOKE RAN"; return 0; }}; '
            'local_image_digest() { echo sha256:same; }; '
            'remote_image_digest() { echo sha256:same; }'
        )
        r = run_real(setup, 'cmd_update')
        out = r.stdout + r.stderr
        self.assertIn("rc=0", out, out)
        self.assertIn("already up to date", out)
        self.assertIn("SMOKE RAN", out,
                      "the weekly check must still run — it is the only thing "
                      "that would notice a dark stack on a box that never drifts")
        self.assertIn("RECORDED verified=1", out,
                      "--rollback needs a point even on a box that never drifts")
        self.assertNotIn("COMPOSE RAN", out,
                         "the expensive path was entered anyway")

    def test_a_dark_stack_is_reported_even_with_nothing_to_update(self):
        """The failure mode this guards: a driver upgrade or an OOM kills every
        model while HEAD equals origin, and the timer says 'nothing to do'."""
        setup = (
            f'{self.STUBS}; smoke() {{ return 1; }}; '
            'local_image_digest() { echo sha256:same; }; '
            'remote_image_digest() { echo sha256:same; }'
        )
        r = run_real(setup, 'cmd_update')
        out = r.stdout + r.stderr
        self.assertNotIn("rc=0", out, "a dark stack must not exit 0")
        self.assertIn("did NOT pass their check", out)

    def test_an_image_behind_still_triggers_work(self):
        setup = (
            f'{self.STUBS}; smoke() {{ echo "SMOKE RAN"; return 0; }}; '
            'local_image_digest() { echo sha256:old; }; '
            'remote_image_digest() { echo sha256:new; }'
        )
        r = run_real(setup, 'cmd_update')
        out = r.stdout + r.stderr
        self.assertIn("COMPOSE RAN", out,
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
        # Both failure paths pass $pin_arg, which is "pin" only when this
        # run actually moved the checkout — see PinsOnlyWhatThisRunMoved.
        self.assertEqual(body.count('do_rollback "$pin_arg"'), 2,
                         "smoke failure and health failure pin; nothing else")
        self.assertNotIn("do_rollback pin", body,
                         "pinning must be conditional, never unconditional")
        self.assertIn("never became healthy within", body)

    def test_infrastructure_failures_do_not_pin(self):
        """A build that lost the network, or a compose up that lost a race
        with tailscaled bringing up LLAMA_BIND2 (compose.linux.yaml warns
        about exactly this), says nothing about the commit — and a wrong pin
        blocks every future update until a human clears it."""
        with open(UPDATE_SH) as fh:
            body = fh.read()
        # compose up failure still rolls back (the stack WAS touched) but
        # must not pin.
        self.assertIn('warn "compose up failed"; do_rollback;', body,
                      "compose up failure must roll back WITHOUT pinning")
        # A failed build touched nothing, so it must not go near do_rollback:
        # force-recreating would evict a resident 27B model over a transient
        # build error, and resetting to the recorded sha could rewind past
        # commits this run never introduced.
        self.assertIn("the running stack was not touched", body)
        self.assertNotIn('warn "build failed"; do_rollback', body)

    def test_manual_rollback_does_not_pin(self):
        with open(UPDATE_SH) as fh:
            body = fh.read()
        # The dispatcher's rollback call must not pass `pin`: a deliberate
        # rollback of a working commit would otherwise blacklist it, and the
        # next update would refuse it claiming a smoke test that never ran.
        arm = body[body.index("  --rollback)"):body.index("  -h|--help)")]
        self.assertIn("do_rollback;;", arm)
        self.assertNotIn("do_rollback pin", arm)
        self.assertNotIn('do_rollback "$pin_arg"', arm)


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
        # Short deadlines so an UNSTUBBED wait_for_health or webui_ok fails in
        # a second instead of polling for minutes. A missing stub should show
        # up as a failure, not as a suite that takes six minutes.
        #
        # BONSAI_WEBUI_TIMEOUT belongs here too: without it an unstubbed
        # webui_ok polled for its full window, and on a dev box actually
        # running Open WebUI on 9090 it returned 0 and passed for the wrong
        # reason. These only bite because .env used to override the process
        # environment; that is fixed in update.sh, and this belt-and-braces
        # stays.
        'set -uo pipefail; export BONSAI_UPDATE_LIB=1; '
        'export BONSAI_HEALTH_TIMEOUT=1 BONSAI_WEBUI_TIMEOUT=1 '
        'BONSAI_SMOKE_RETRY_SLEEP=0; '
        f'source "{UPDATE_SH}"; '
        # Aim at a dead port unless the test says otherwise, so nothing can
        # reach the operator's live box through LLAMA_BIND/WEBUI_BIND in .env.
        'API=http://127.0.0.1:9; WEBUI_URL=http://127.0.0.1:9; '
        # And isolate the state file. Without this, any test that reached
        # record_rollback_point or a pin wrote to the REPO's own
        # run/update-state — so pins leaked from one test into the next and
        # into the working tree. RUN_TMP is created by update.sh and removed
        # by its EXIT trap, so this cleans itself up.
        'STATE="${RUN_TMP:-/tmp}/update-state"; '
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

    @staticmethod
    def _leftovers(d):
        found = []
        for root, _dirs, files in os.walk(d):
            found.extend(os.path.join(root, f) for f in files)
        return found

    def test_sourcing_and_exiting_leaves_no_key_file_behind(self):
        """Library mode installs no other traps, so a sourced smoke() used to
        leave a key-bearing file in /tmp permanently. Checks for ANY leftover
        file, not a filename prefix — the prefix moved once already."""
        with tempfile.TemporaryDirectory() as d:
            script = (
                f'set -uo pipefail; export BONSAI_UPDATE_LIB=1; '
                f'TMPDIR={shlex.quote(d)}; export TMPDIR; '
                f'source "{UPDATE_SH}"; BONSAI_API_KEY=secret; '
                f'curl_conf; echo made'
            )
            r = subprocess.run(["bash", "-c", script], capture_output=True,
                               text=True)
            self.assertIn("made", r.stdout, r.stderr)
            left = self._leftovers(d)
            self.assertEqual(left, [],
                             f"a file holding the API key survived exit: {left}")

    def test_the_real_script_leaves_no_run_directory(self):
        """`exec` replaces the process WITHOUT running its EXIT trap, so a
        temp directory created before the re-exec leaked on every single run.
        Measured on the box: one bonsai-run.* left per invocation."""
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, TMPDIR=d)
            env.pop("BONSAI_UPDATE_REEXEC", None)
            r = subprocess.run(["bash", UPDATE_SH, "--help"],
                               capture_output=True, text=True, cwd=ROOT, env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            left = sorted(os.listdir(d))
            self.assertEqual(left, [], f"left behind in TMPDIR: {left}")

    def test_no_key_file_survives_when_ask_runs_in_a_pipeline(self):
        """The path is assigned lazily, so reaching `ask` through a pipeline
        assigned it in a SUBSHELL — invisible to the parent's cleanup trap,
        and the key file survived the run. Measured on the box before the
        fix: one curl.conf and one curl.err left in /tmp."""
        with tempfile.TemporaryDirectory() as d:
            script = (
                f'set -uo pipefail; export BONSAI_UPDATE_LIB=1; '
                f'TMPDIR={shlex.quote(d)}; export TMPDIR; '
                f'source "{UPDATE_SH}"; API=http://127.0.0.1:9; '
                f'BONSAI_API_KEY=secret; smoke 2>&1 | head -1'
            )
            subprocess.run(["bash", "-c", script], capture_output=True,
                           text=True)
            left = self._leftovers(d)
            self.assertEqual(left, [],
                             f"key material left behind: {left}")

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
            # preflight now probes the DAEMON, not just the binary; without
            # a stub the real docker CLI blocks for minutes on a machine
            # with none running.
            f'{_TOOLS_PRESENT}; running_working_dir() {{ printf ""; }}; '
            'docker() { return 0; }; '
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
            # preflight now probes the DAEMON, not just the binary; without
            # a stub the real docker CLI blocks for minutes on a machine
            # with none running.
            f'{_TOOLS_PRESENT}; running_working_dir() {{ printf ""; }}; '
            'docker() { return 0; }; '
            f'ROOT={shlex.quote(d)}; COMPOSE_FILE={shlex.quote(d)}/nope.yaml; '
            f'ENV_FILE={shlex.quote(d)}/nope.env; unset BONSAI_API_KEY'
        )
        r = run_real(setup, 'preflight check')
        shutil.rmtree(d, ignore_errors=True)
        self.assertIn("rc=0", r.stdout, r.stderr)

    def test_update_still_needs_a_dotenv(self):
        d = tempfile.mkdtemp()
        setup = (
            # preflight now probes the DAEMON, not just the binary; without
            # a stub the real docker CLI blocks for minutes on a machine
            # with none running.
            f'{_TOOLS_PRESENT}; running_working_dir() {{ printf ""; }}; '
            'docker() { return 0; }; '
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
        # The --rollback arm is multi-line now; read the whole case block.
        arm = body[body.index("  --rollback)"):body.index("  -h|--help)")]
        self.assertIn("preflight rollback", arm)
        self.assertIn("take_lock", arm)
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
