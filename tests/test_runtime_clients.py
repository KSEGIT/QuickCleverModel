"""Client smoke fixtures must not reuse cloud routing or broad permissions."""
import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "runtime_clients", pathlib.Path(__file__).with_name("runtime_clients.py"))
clients = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(clients)


class ClientIsolationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.args = argparse.Namespace(base_url="http://127.0.0.1:18080/v1/", model="fixture-model",
                                       key_env="QCM_ISOLATION_TEST_KEY", context=8192)

    def test_cloud_credentials_and_routing_are_not_inherited(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "private", "ANTHROPIC_BASE_URL": "cloud",
                                         "AWS_SECRET_ACCESS_KEY": "private", "OPENCODE_CONFIG": "private"}):
            env = clients.child_environment(self.root)
        for name in ("OPENAI_API_KEY", "ANTHROPIC_BASE_URL", "AWS_SECRET_ACCESS_KEY", "OPENCODE_CONFIG"):
            self.assertNotIn(name, env)

    def test_no_os_confinement_means_skip_not_unconfined_execution(self):
        with mock.patch.object(clients.platform, "system", return_value="unsupported"):
            with self.assertRaisesRegex(RuntimeError, "confinement"):
                clients.confined_command(["codex", "exec"], self.root)

    def test_macos_profile_denies_personal_repositories_and_allows_only_fixture_writes(self):
        with mock.patch.object(clients.platform, "system", return_value="Darwin"), \
             mock.patch.object(clients.shutil, "which", return_value="/usr/bin/sandbox-exec"):
            argv = clients.confined_command(["codex", "exec"], self.root)
        self.assertEqual(argv[0], "/usr/bin/sandbox-exec")
        profile = argv[2]
        self.assertIn("(deny file-read* file-write*)", profile)
        self.assertIn(str(self.root.resolve()), profile)
        self.assertNotIn('(subpath "/Users")', profile)
        self.assertNotIn('(subpath "/Volumes")', profile)

    @unittest.skipUnless(os.environ.get("QCM_TEST_CONFINEMENT") == "1", "opt-in OS sandbox check")
    def test_os_blocks_real_reads_outside_fixture(self):
        inside = self.root / "inside.txt"
        inside.write_text("fixture")
        with tempfile.TemporaryDirectory() as outside:
            target = pathlib.Path(outside) / "private.txt"
            target.write_text("must not be readable")
            allowed = subprocess.run(clients.confined_command(["/bin/cat", str(inside)], self.root),
                                     capture_output=True, text=True)
            denied = subprocess.run(clients.confined_command(["/bin/cat", str(target)], self.root),
                                    capture_output=True, text=True)
        self.assertEqual(allowed.stdout, "fixture", allowed.stderr)
        self.assertNotEqual(denied.returncode, 0)
        self.assertNotIn("must not be readable", denied.stdout)

    def test_codex_uses_sandbox_and_explicit_provider(self):
        env = clients.child_environment(self.root)
        argv = clients.command("codex", self.args, self.root, self.root, env, "fixture task")
        self.assertIn("--ignore-user-config", argv)
        self.assertIn("--ignore-rules", argv)
        self.assertIn("--ephemeral", argv)
        self.assertIn("workspace-write", argv)
        self.assertIn('model_providers.qcm_test.base_url="http://127.0.0.1:18080/v1"', argv)
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", argv)
        self.assertNotIn("CODEX_HOME", env)

    def test_claude_limits_tools_and_does_not_load_personal_mcp(self):
        env = clients.child_environment(self.root)
        argv = clients.command("claude", self.args, self.root, self.root, env, "fixture task")
        self.assertIn("--bare", argv)
        self.assertIn("--strict-mcp-config", argv)
        self.assertIn("Read,Write,Edit", argv)
        self.assertIn("dontAsk", argv)
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:18080")
        self.assertEqual(env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "8192")
        self.assertEqual(env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"], "2048")
        self.assertEqual(env["DISABLE_COMPACT"], "1")
        self.assertEqual(env["MAX_THINKING_TOKENS"], "0")
        settings = json.loads((self.root / "claude-settings.json").read_text())
        self.assertIn("Bash", settings["permissions"]["deny"])
        self.assertNotIn("--dangerously-skip-permissions", argv)

    def test_opencode_config_keeps_key_out_of_file_and_denies_external_access(self):
        env = clients.child_environment(self.root)
        with mock.patch.dict(os.environ, {"QCM_ISOLATION_TEST_KEY": "private-key"}):
            argv = clients.command("opencode", self.args, self.root, self.root, env, "fixture task")
        raw = (self.root / "opencode.json").read_text()
        self.assertNotIn("private-key", raw)
        self.assertEqual(json.loads(raw)["permission"]["external_directory"], "deny")
        self.assertEqual(json.loads(raw)["permission"]["*"], "deny")
        self.assertIn("--pure", argv)

    def test_browser_uses_pinned_mcp_with_explicit_snapshots(self):
        env = clients.child_environment(self.root)
        argv = clients.command("claude", self.args, self.root, self.root, env, "fixture task", browser=True)
        config = json.loads(argv[argv.index("--mcp-config") + 1])
        mcp_args = config["mcpServers"]["playwright"]["args"]
        self.assertIn("@playwright/mcp@0.0.82", mcp_args)
        self.assertEqual(mcp_args[mcp_args.index("--snapshot-mode") + 1], "none")
        self.assertEqual(mcp_args[mcp_args.index("--image-responses") + 1], "omit")

    def test_process_transcript_survives_timeout(self):
        env = clients.child_environment(self.root)
        with self.assertRaises(subprocess.TimeoutExpired) as caught:
            clients.run_process([sys.executable, "-u", "-c",
                "import time; print('started'); time.sleep(10)"], env, self.root, 0.25)
        self.assertIn("started", caught.exception.stdout)


if __name__ == "__main__":
    unittest.main()
