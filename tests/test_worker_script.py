"""tests/bench/worker.sh drives the RTX worker over SSH.

stdlib unittest, not pytest — see tests/test_cache_viz.py for the convention.

No test touches a real worker: a fake `ssh` on PATH logs its argv (one JSON
list per line) and answers the few remote commands the script sends. The
answers are steered with FAKE_* environment variables.
"""
import json
import os
import stat
import subprocess
import tempfile
import textwrap
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "tests", "bench", "worker.sh")

FAKE_SSH = textwrap.dedent("""\
    #!/usr/bin/env python3
    import json, os, sys
    with open(os.environ["FAKE_SSH_LOG"], "a") as log:
        log.write(json.dumps(sys.argv[1:]) + "\\n")
    cmd = sys.argv[-1]
    if "docker inspect" in cmd:
        rc = int(os.environ.get("FAKE_TEST_RC", "0"))
        if rc == 0:
            print('{"status":"ok"}')
        sys.exit(rc)
    if "http_code" in cmd:
        print(os.environ.get("FAKE_LIVE_CODE", "200"), end="")
        sys.exit(0)
    if "nvidia-smi" in cmd:
        print(os.environ.get("FAKE_VRAM", "6269"))
        sys.exit(0)
    if "/health" in cmd:
        print('{"status":"ok"}', end="")
        sys.exit(0)
    sys.exit(0)
    """)

UP_ENV = {
    "MODELS_DIR": "/srv/models",
    "TEMPLATE_FILE": "/srv/templates/qwen3.5.jinja",
    "MODEL_FILE": "/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf",
    "MODEL_ALIAS": "qwen3.5-9b-q4_k_m",
}


class WorkerScriptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        bindir = os.path.join(self.tmp.name, "bin")
        os.mkdir(bindir)
        fake = os.path.join(bindir, "ssh")
        with open(fake, "w") as f:
            f.write(FAKE_SSH)
        os.chmod(fake, os.stat(fake).st_mode | stat.S_IXUSR)
        self.log = os.path.join(self.tmp.name, "ssh.log")
        self.env = {
            "PATH": bindir + os.pathsep + os.environ.get("PATH", ""),
            "HOME": self.tmp.name,
            "FAKE_SSH_LOG": self.log,
            "WORKER_SSH": "bench@rtx-worker",
            "POLL_INTERVAL": "0",
            "UP_TIMEOUT": "1",
            "DOWN_TIMEOUT": "1",
        }

    def run_script(self, *args, **env):
        full = dict(self.env)
        full.update(env)
        return subprocess.run(["bash", SCRIPT, *args], capture_output=True,
                              text=True, env=full, timeout=60)

    def calls(self):
        if not os.path.exists(self.log):
            return []
        with open(self.log) as f:
            return [json.loads(line) for line in f]

    def commands(self):
        return [argv[-1] for argv in self.calls()]

    def index_of(self, needle):
        for i, cmd in enumerate(self.commands()):
            if needle in cmd:
                return i
        self.fail(f"no remote command contains {needle!r}: {self.commands()}")

    def test_missing_worker_ssh_fails_without_calling_ssh(self):
        for sub in ("up", "down", "vram", "health"):
            with self.subTest(sub=sub):
                out = self.run_script(sub, WORKER_SSH="", **UP_ENV)
                self.assertNotEqual(out.returncode, 0)
                self.assertIn("WORKER_SSH", out.stderr)
        self.assertEqual(self.calls(), [])

    def test_unknown_subcommand_is_rejected(self):
        out = self.run_script("reboot")
        self.assertNotEqual(out.returncode, 0)
        self.assertEqual(self.calls(), [])

    def test_up_stops_live_before_running_test_container(self):
        out = self.run_script("up", **UP_ENV)
        self.assertEqual(out.returncode, 0, out.stderr)
        stop_live = self.index_of("docker stop bonsai-llama-1")
        run_test = self.index_of("docker run")
        self.assertLess(stop_live, run_test)
        # It waits for health after starting, not before.
        self.assertGreater(self.index_of("docker inspect"), run_test)

    def test_up_reproduces_the_recorded_server_command(self):
        out = self.run_script("up", CTX="49152", PARALLEL="2", **UP_ENV)
        self.assertEqual(out.returncode, 0, out.stderr)
        cmd = self.commands()[self.index_of("docker run")]
        for part in (
            "docker run -d --rm --gpus all --name qcm-bench",
            "-p 127.0.0.1:18080:8080",
            "-v /srv/models:/models:ro",
            "-v /srv/templates/qwen3.5.jinja:/template.jinja:ro",
            "--entrypoint /app/src/llama.cpp-prism/build/bin/llama-server",
            "qcm-rtx-validation:922be44",
            "--host 0.0.0.0 --port 8080",
            "--model /models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf",
            "--alias qwen3.5-9b-q4_k_m",
            "--chat-template-file /template.jinja",
            "--ctx-size 49152 --n-gpu-layers 99 --flash-attn auto",
            "--cache-type-k f16 --cache-type-v f16 --reasoning off",
            "--parallel 2",
        ):
            self.assertIn(part, cmd)
        self.assertNotIn("--jinja", cmd.replace("/template.jinja", ""))

    def test_up_uses_strict_host_key_checking(self):
        self.run_script("up", **UP_ENV)
        for argv in self.calls():
            self.assertIn("StrictHostKeyChecking=yes", argv)
            self.assertIn("BatchMode=yes", argv)
            self.assertIn("bench@rtx-worker", argv)

    def test_up_requires_model_settings(self):
        env = dict(UP_ENV)
        del env["MODEL_FILE"]
        out = self.run_script("up", **env)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("MODEL_FILE", out.stderr)
        self.assertEqual(self.calls(), [])

    def test_up_rejects_non_numeric_ctx(self):
        out = self.run_script("up", CTX="32768; rm -rf /", **UP_ENV)
        self.assertNotEqual(out.returncode, 0)
        self.assertEqual(self.calls(), [])

    def test_up_fails_fast_when_test_container_dies(self):
        out = self.run_script("up", FAKE_TEST_RC="3", UP_TIMEOUT="30", **UP_ENV)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("exited", out.stderr)

    def test_up_fails_when_test_server_never_healthy(self):
        out = self.run_script("up", FAKE_TEST_RC="22", **UP_ENV)
        self.assertNotEqual(out.returncode, 0)

    def test_down_stops_test_then_starts_live(self):
        out = self.run_script("down")
        self.assertEqual(out.returncode, 0, out.stderr)
        stop_test = self.index_of("docker stop qcm-bench")
        start_live = self.index_of("docker start bonsai-llama-1")
        self.assertLess(stop_test, start_live)
        self.assertGreater(self.index_of("http_code"), start_live)

    def test_down_fails_when_live_is_unhealthy(self):
        out = self.run_script("down", FAKE_LIVE_CODE="401")
        self.assertEqual(out.returncode, 1)
        self.assertIn("not healthy", out.stderr)

    def test_down_reads_key_on_worker_and_never_sends_it(self):
        out = self.run_script("down", LIVE_ENV_FILE="/srv/bonsai/.env",
                              LIVE_HEALTH_URL="http://100.64.0.5:8080/health",
                              BONSAI_API_KEY="local-secret-must-not-leak")
        self.assertEqual(out.returncode, 0, out.stderr)
        cmd = self.commands()[self.index_of("http_code")]
        self.assertIn("/srv/bonsai/.env", cmd)
        self.assertIn("BONSAI_API_KEY=", cmd)
        self.assertIn("-H @-", cmd)
        self.assertIn("http://100.64.0.5:8080/health", cmd)
        for argv in self.calls():
            self.assertNotIn("local-secret-must-not-leak", " ".join(argv))
        self.assertNotIn("local-secret-must-not-leak", out.stdout + out.stderr)

    def test_vram_prints_integer(self):
        out = self.run_script("vram", FAKE_VRAM=" 6269 ")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "6269")

    def test_vram_rejects_garbage(self):
        out = self.run_script("vram", FAKE_VRAM="N/A")
        self.assertNotEqual(out.returncode, 0)

    def test_health_prints_server_json(self):
        out = self.run_script("health")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(json.loads(out.stdout), {"status": "ok"})
        self.assertIn("127.0.0.1:18080/health", self.commands()[0])


if __name__ == "__main__":
    unittest.main()
