"""Tests for setup-opencode.sh: connect-target resolution and drift detection.

stdlib unittest, not pytest — see tests/test_cache_viz.py for the convention.

Run: python3 -m unittest discover -s tests -v

Background, because it is the whole point of these tests: BONSAI_HOST is the
address llama-server BINDS (start-server.sh: --host "$HOST"). It is NOT where
a client connects. Conflating the two put a remote tailnet IP where a bind
address was expected — a machine cannot bind an address it does not own — and
in the other direction rewrote a working remote client config to 127.0.0.1.
Connect and bind now have separate names, and these tests pin that.

Every test runs against a fixture .env (BONSAI_ENV_FILE) and a throwaway
config dir (XDG_CONFIG_HOME), so the real .env and ~/.config are untouched.
"""
import json
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "setup-opencode.sh")
LOCAL_KEY = "bonsai-localkey000000"
REMOTE_KEY = "bonsai-remotekey11111"


class Harness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="opencode-test-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.env_file = os.path.join(self.tmp, "env")
        self.cfg_home = os.path.join(self.tmp, "cfg")
        self.cfg = os.path.join(self.cfg_home, "opencode", "opencode.json")

        # Stub `opencode` when the real one is absent. Without this the render
        # tests skip on a machine that has no opencode — precisely the machine
        # this script exists to set up — and the suite reports OK having
        # verified nothing.
        self.bin_dir = os.path.join(self.tmp, "bin")
        os.makedirs(self.bin_dir, exist_ok=True)
        if not shutil.which("opencode"):
            stub = os.path.join(self.bin_dir, "opencode")
            with open(stub, "w") as fh:
                fh.write('#!/bin/sh\n[ "$1" = "--version" ] && echo "stub" || exit 0\n')
            os.chmod(stub, 0o755)

    def write_env(self, _probe=False, **pairs):
        """Write a fixture .env.

        BONSAI_CTX is set by default so the render skips its /props probe:
        fixture URLs point at unreachable hosts, and waiting out the timeout
        on every render turned a 3s suite into a 64s one. Tests that are
        specifically about context resolution pass _probe=True to leave it
        unset and exercise the real path.
        """
        if not _probe:
            pairs.setdefault("BONSAI_CTX", "8192")
        with open(self.env_file, "w") as fh:
            for k, v in pairs.items():
                fh.write(f"{k}={v}\n")

    def run_script(self, *args):
        # Strip every BONSAI_* from the inherited environment. The script
        # sources .env with `set -a`, but a var already exported in the
        # developer's shell survives and silently overrides the fixture —
        # in a repo whose whole premise is exporting BONSAI_*, that is a
        # near-certain flake rather than a theoretical one.
        env = {k: v for k, v in os.environ.items() if not k.startswith("BONSAI_")}
        env["BONSAI_ENV_FILE"] = self.env_file
        env["XDG_CONFIG_HOME"] = self.cfg_home
        env["PATH"] = self.bin_dir + os.pathsep + env.get("PATH", "")
        return subprocess.run([SCRIPT, *args], capture_output=True, text=True,
                              env=env, cwd=ROOT, stdin=subprocess.DEVNULL)

    def rendered(self):
        with open(self.cfg) as fh:
            return json.load(fh)["provider"]["bonsai"]["options"]


class ConnectTargetTest(Harness):
    def test_defaults_to_localhost_not_a_bind_wildcard(self):
        """Nothing configured must mean this machine, never 0.0.0.0."""
        self.write_env(BONSAI_API_KEY=LOCAL_KEY)
        self.assertEqual(self.run_script().returncode, 0)
        self.assertEqual(self.rendered()["baseURL"], "http://127.0.0.1:8080/v1")

    def test_bind_wildcard_is_never_used_as_a_connect_address(self):
        """BONSAI_HOST=0.0.0.0 is a bind wildcard; connecting to it is wrong."""
        self.write_env(BONSAI_API_KEY=LOCAL_KEY, BONSAI_HOST="0.0.0.0")
        self.assertEqual(self.run_script().returncode, 0)
        self.assertEqual(self.rendered()["baseURL"], "http://127.0.0.1:8080/v1")

    def test_server_url_wins_over_host(self):
        self.write_env(BONSAI_API_KEY=LOCAL_KEY, BONSAI_HOST="0.0.0.0",
                       BONSAI_SERVER_URL="http://10.1.2.3:9999/v1")
        self.assertEqual(self.run_script().returncode, 0)
        self.assertEqual(self.rendered()["baseURL"], "http://10.1.2.3:9999/v1")

    def test_port_is_honoured(self):
        """A client pinned to 8080 breaks the moment BONSAI_PORT moves."""
        self.write_env(BONSAI_API_KEY=LOCAL_KEY, BONSAI_PORT="18080")
        self.assertEqual(self.run_script().returncode, 0)
        self.assertEqual(self.rendered()["baseURL"], "http://127.0.0.1:18080/v1")

    def test_remote_key_wins_over_local_key(self):
        """Sending this machine's key to another box just yields 401."""
        self.write_env(BONSAI_API_KEY=LOCAL_KEY, BONSAI_SERVER_KEY=REMOTE_KEY,
                       BONSAI_SERVER_URL="http://10.1.2.3:8080/v1")
        self.assertEqual(self.run_script().returncode, 0)
        self.assertEqual(self.rendered()["apiKey"], REMOTE_KEY)

    def test_local_key_used_when_no_server_key(self):
        self.write_env(BONSAI_API_KEY=LOCAL_KEY)
        self.assertEqual(self.run_script().returncode, 0)
        self.assertEqual(self.rendered()["apiKey"], LOCAL_KEY)


class DriftCheckTest(Harness):
    """No opencode binary needed: --check skips the install gate entirely, and
    the render path only writes JSON. Gating these on the binary meant a fresh
    box — exactly what this script exists to set up — ran none of them."""

    def render_then(self, **changes):
        """Render a config, then change .env so the two disagree."""
        base = dict(BONSAI_API_KEY=LOCAL_KEY,
                    BONSAI_SERVER_URL="http://10.1.2.3:8080/v1")
        self.write_env(**base)
        self.assertEqual(self.run_script().returncode, 0)
        base.update(changes)
        self.write_env(**base)
        return self.run_script("--check")

    def test_passes_when_in_sync(self):
        self.write_env(BONSAI_API_KEY=LOCAL_KEY,
                       BONSAI_SERVER_URL="http://10.1.2.3:8080/v1")
        self.assertEqual(self.run_script().returncode, 0)
        proc = self.run_script("--check")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_detects_key_drift(self):
        proc = self.render_then(BONSAI_API_KEY="bonsai-somethingelse0")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("apiKey", proc.stderr)

    def test_detects_url_drift(self):
        proc = self.render_then(BONSAI_SERVER_URL="http://10.9.9.9:8080/v1")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("baseURL", proc.stderr)

    def test_check_never_writes(self):
        self.write_env(BONSAI_API_KEY=LOCAL_KEY,
                       BONSAI_SERVER_URL="http://10.1.2.3:8080/v1")
        self.assertEqual(self.run_script().returncode, 0)
        before = open(self.cfg).read()
        self.write_env(BONSAI_API_KEY="bonsai-somethingelse0",
                       BONSAI_SERVER_URL="http://10.9.9.9:8080/v1")
        self.run_script("--check")
        self.assertEqual(open(self.cfg).read(), before, "--check modified the config")

    def test_context_mismatch_is_a_note_when_env_is_silent(self):
        """A remote server's context is unknowable from here — do not fail on a guess."""
        self.write_env(BONSAI_API_KEY=LOCAL_KEY, BONSAI_CTX="4096",
                       BONSAI_SERVER_URL="http://10.1.2.3:8080/v1")
        self.assertEqual(self.run_script().returncode, 0)   # renders ctx 4096
        self.write_env(_probe=True, BONSAI_API_KEY=LOCAL_KEY,  # silent on ctx
                       BONSAI_SERVER_URL="http://10.1.2.3:8080/v1")
        proc = self.run_script("--check")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("note", proc.stdout)

    def test_context_mismatch_fails_when_env_states_it(self):
        proc = self.render_then(BONSAI_CTX="99999")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("context", proc.stderr)
