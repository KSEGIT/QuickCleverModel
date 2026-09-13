"""Tests for claude-bonsai.sh: the environment it hands to Claude Code.

stdlib unittest, not pytest — see tests/test_cache_viz.py for the convention.

Run: python3 -m unittest discover -s tests -v

Background: Claude Code talks to this server directly, because llama-server
answers /v1/messages as well as the OpenAI routes. No proxy. What it does need
is four environment variables set correctly, and three of them fail quietly
when they are wrong:

- ANTHROPIC_BASE_URL must NOT end in /v1. Claude Code appends it, and
  /v1/v1/messages is a 404 that reads like the server being down.
- ANTHROPIC_DEFAULT_HAIKU_MODEL, left unset, makes Claude Code ask this server
  for a Haiku model it has never heard of.
- CLAUDE_CODE_MAX_CONTEXT_TOKENS, left unset, makes Claude Code assume 200k for
  a model it does not recognise. It then compacts too late and the server
  truncates the oldest turns silently.

Every test runs the script with a stub `claude` on PATH that prints its
environment, so nothing here contacts a server or starts an agent.
"""
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "claude-bonsai.sh")
KEY = "bonsai-testkey0000000"


class Harness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="claude-bonsai-test-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.env_file = os.path.join(self.tmp, "env")

        # A stub that reports the environment instead of launching an agent.
        self.bin_dir = os.path.join(self.tmp, "bin")
        os.makedirs(self.bin_dir)
        stub = os.path.join(self.bin_dir, "claude")
        with open(stub, "w") as fh:
            fh.write(
                "#!/bin/sh\n"
                'echo "BASE=$ANTHROPIC_BASE_URL"\n'
                'echo "TOKEN=$ANTHROPIC_AUTH_TOKEN"\n'
                'echo "MODEL=$ANTHROPIC_MODEL"\n'
                'echo "HAIKU=$ANTHROPIC_DEFAULT_HAIKU_MODEL"\n'
                'echo "CTX=$CLAUDE_CODE_MAX_CONTEXT_TOKENS"\n'
                'echo "ARGS=$*"\n'
            )
        os.chmod(stub, 0o755)

    def write_env(self, **pairs):
        with open(self.env_file, "w") as fh:
            for k, v in pairs.items():
                fh.write(f"{k}={v}\n")

    def run_script(self, *args):
        # Strip inherited BONSAI_*: this repo's whole premise is exporting
        # those, so a developer's shell would otherwise override the fixture.
        env = {k: v for k, v in os.environ.items() if not k.startswith("BONSAI_")}
        env["BONSAI_ENV_FILE"] = self.env_file
        env["PATH"] = self.bin_dir + os.pathsep + env.get("PATH", "")
        return subprocess.run([SCRIPT, *args], capture_output=True, text=True,
                              env=env, cwd=ROOT, stdin=subprocess.DEVNULL)

    def handed(self, *args):
        result = self.run_script(*args)
        self.assertEqual(0, result.returncode, result.stderr)
        return dict(
            line.split("=", 1)
            for line in result.stdout.splitlines()
            if "=" in line
        )


class BaseUrlTest(Harness):
    def test_never_ends_in_v1(self):
        """Claude Code appends /v1 — a /v1 base gives /v1/v1/messages."""
        self.write_env(BONSAI_API_KEY=KEY,
                       BONSAI_SERVER_URL="http://10.1.2.3:8080/v1")
        self.assertEqual("http://10.1.2.3:8080", self.handed()["BASE"])

    def test_defaults_to_localhost(self):
        self.write_env(BONSAI_API_KEY=KEY)
        self.assertEqual("http://127.0.0.1:8080", self.handed()["BASE"])

    def test_bind_wildcard_is_never_a_connect_address(self):
        """BONSAI_HOST is where the server binds; nobody dials 0.0.0.0."""
        self.write_env(BONSAI_API_KEY=KEY, BONSAI_HOST="0.0.0.0")
        self.assertEqual("http://127.0.0.1:8080", self.handed()["BASE"])

    def test_port_is_honoured(self):
        self.write_env(BONSAI_API_KEY=KEY, BONSAI_PORT="18080")
        self.assertEqual("http://127.0.0.1:18080", self.handed()["BASE"])


class CredentialTest(Harness):
    def test_remote_key_wins(self):
        """This machine's key against another box is a guaranteed 401."""
        self.write_env(BONSAI_API_KEY=KEY, BONSAI_SERVER_KEY="bonsai-remote111111")
        self.assertEqual("bonsai-remote111111", self.handed()["TOKEN"])

    def test_no_key_is_an_error_not_a_401(self):
        self.write_env(BONSAI_PORT="8080")
        result = self.run_script()
        self.assertEqual(1, result.returncode)
        self.assertIn("BONSAI_API_KEY", result.stderr)

    def test_missing_env_file_is_reported(self):
        result = self.run_script()  # never wrote one
        self.assertEqual(1, result.returncode)
        self.assertIn(".env", result.stderr)


class ModelAndContextTest(Harness):
    def test_defaults_to_the_text_preset(self):
        """No mmproj, so the prompt cache survives — see models.ini.in."""
        self.write_env(BONSAI_API_KEY=KEY)
        handed = self.handed()
        self.assertEqual("bonsai-27b-ternary-text", handed["MODEL"])

    def test_background_jobs_get_a_model_that_exists(self):
        """Unset, Claude Code asks this server for a Haiku model."""
        self.write_env(BONSAI_API_KEY=KEY)
        handed = self.handed()
        self.assertEqual(handed["MODEL"], handed["HAIKU"])

    def test_context_prefers_the_text_specific_value(self):
        self.write_env(BONSAI_API_KEY=KEY, BONSAI_CTX="65536",
                       BONSAI_CTX_TEXT="131072")
        self.assertEqual("131072", self.handed()["CTX"])

    def test_context_falls_back_to_the_shared_value(self):
        self.write_env(BONSAI_API_KEY=KEY, BONSAI_CTX="65536")
        self.assertEqual("65536", self.handed()["CTX"])

    def test_context_has_a_default(self):
        """Unset must not mean 'let Claude Code assume 200k'."""
        self.write_env(BONSAI_API_KEY=KEY)
        self.assertEqual("131072", self.handed()["CTX"])

    def test_model_is_overridable(self):
        self.write_env(BONSAI_API_KEY=KEY, BONSAI_CLAUDE_MODEL="bonsai-27b-1bit")
        self.assertEqual("bonsai-27b-1bit", self.handed()["MODEL"])


class PassthroughTest(Harness):
    def test_arguments_reach_claude(self):
        self.write_env(BONSAI_API_KEY=KEY)
        handed = self.handed("-p", "hello", "--mcp-config", "mcp.json")
        self.assertEqual("-p hello --mcp-config mcp.json", handed["ARGS"])


if __name__ == "__main__":
    unittest.main()
