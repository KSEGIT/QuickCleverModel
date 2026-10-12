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
    with open(os.environ["FAKE_SSH_LOG"]) as log:
        n_calls = sum(1 for _ in log)
    # Keys travel on stdin (curl -H @-); record what the worker would read.
    if os.environ.get("FAKE_SSH_STDIN_LOG") and "-H @-" in sys.argv[-1]:
        with open(os.environ["FAKE_SSH_STDIN_LOG"], "a") as f:
            f.write(sys.stdin.read())
    # FAKE_SSH_FAIL_FIRST=N: the first N connections fail like a dropped link.
    if n_calls <= int(os.environ.get("FAKE_SSH_FAIL_FIRST", "0")):
        print("ssh: connect to host rtx-worker port 22: Connection timed out",
              file=sys.stderr)
        sys.exit(255)
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
    if "/props" in cmd:
        print(os.environ.get("FAKE_PROPS",
              '{"default_generation_settings":{"id":0,"n_ctx":8192}}'), end="")
        sys.exit(int(os.environ.get("FAKE_PROPS_RC", "0")))
    if "chat/completions" in cmd:
        sys.exit(int(os.environ.get("FAKE_WARM_RC", "0")))
    if "/health" in cmd:
        print('{"status":"ok"}', end="")
        sys.exit(0)
    sys.exit(0)
    """)

# Direct mode: one llama-server with --parallel (the concurrency suite).
UP_ENV = {
    "MODE": "direct",
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
            "RETRY_SLEEP": "0",
        }

    def run_script(self, *args, **env):
        full = dict(self.env)
        full.update(env)
        return subprocess.run(["bash", SCRIPT, *args], capture_output=True,
                              stdin=subprocess.DEVNULL, text=True, env=full,
                              timeout=60)

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

    def run_worker_raw(self, *args, **env):
        return self.run_script(*args, **env)

    def run_worker(self, *args, **env):
        """Run, require success, return the remote commands sent."""
        out = self.run_script(*args, **env)
        self.assertEqual(out.returncode, 0, out.stderr)
        return self.commands()

    def test_missing_worker_ssh_fails_without_calling_ssh(self):
        for sub in ("up", "down", "vram", "health", "ctx"):
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

    def test_down_survives_ssh_failures_and_still_starts_live(self):
        # Calls 1-3 fail: stop test (1), start live (2), stop test (3).
        # Attempt 2's live start (call 4) succeeds, then health passes.
        out = self.run_script("down", FAKE_SSH_FAIL_FIRST="3")
        self.assertEqual(out.returncode, 0, out.stderr)
        cmds = self.commands()
        starts = [i for i, c in enumerate(cmds) if "docker start bonsai-llama-1" in c]
        self.assertEqual(starts, [1, 3])
        self.assertIn("http_code", cmds[4])
        self.assertIn("attempt 2/5", out.stderr)

    def test_down_gives_up_after_all_start_attempts(self):
        out = self.run_script("down", FAKE_SSH_FAIL_FIRST="100", START_ATTEMPTS="3")
        self.assertEqual(out.returncode, 1)
        starts = [c for c in self.commands() if "docker start bonsai-llama-1" in c]
        self.assertEqual(len(starts), 3)
        self.assertIn("after 3 attempts", out.stderr)

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

    # --- router mode: the image's own entrypoint (start-server.sh) ---

    def test_router_up_runs_image_entrypoint_with_key_and_models(self):
        log = self.run_worker("up", MODE="router", MODEL_ALIAS="gemma4-e4b",
                              MODELS_DIR="/srv/models", BENCH_API_KEY="k-123", CTX="")
        docker_run = next(l for l in log if "docker run" in l)
        self.assertNotIn("--entrypoint", docker_run)
        self.assertIn("/srv/models:/app/models:ro", docker_run)
        self.assertIn("BONSAI_API_KEY=k-123", docker_run)
        self.assertIn("BONSAI_PORT=8080", docker_run)
        self.assertNotIn("QCM_AGENT_CTX", docker_run)
        self.assertNotIn("BONSAI_CTX", docker_run)

    def test_router_ctx_sets_every_preset_ctx_variable(self):
        log = self.run_worker("up", MODE="router", MODEL_ALIAS="qwen3.5-9b-q4_k_m",
                              MODELS_DIR="/srv/models", BENCH_API_KEY="k", CTX="32768")
        docker_run = next(l for l in log if "docker run" in l)
        for var in ("QCM_AGENT_CTX=32768", "BONSAI_CTX=32768", "BONSAI_CTX_TEXT=32768"):
            self.assertIn(var, docker_run)

    def test_router_up_warms_the_chosen_model(self):
        log = self.run_worker("up", MODE="router", MODEL_ALIAS="granite-4.1-8b-q4_k_m",
                              MODELS_DIR="/srv/models", BENCH_API_KEY="k", CTX="")
        self.assertTrue(any("chat/completions" in l and "granite-4.1-8b-q4_k_m" in l for l in log))

    def test_router_mode_needs_a_key(self):
        result = self.run_worker_raw("up", MODE="router", MODEL_ALIAS="x", MODELS_DIR="/m", BENCH_API_KEY="")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("BENCH_API_KEY", result.stderr)

    def test_direct_mode_is_unchanged(self):
        log = self.run_worker("up", MODE="direct", MODEL_ALIAS="qwen3.5-9b-q4_k_m",
                              MODEL_FILE="/models/q.gguf", TEMPLATE_FILE="/t.jinja",
                              MODELS_DIR="/srv/models", CTX="49152", PARALLEL="2")
        docker_run = next(l for l in log if "docker run" in l)
        self.assertIn("--entrypoint", docker_run)
        self.assertIn("--parallel 2", docker_run)

    def test_router_is_the_default_mode(self):
        log = self.run_worker("up", MODEL_ALIAS="gemma4-e4b", MODELS_DIR="/srv/models",
                              BENCH_API_KEY="k")
        docker_run = next(l for l in log if "docker run" in l)
        self.assertNotIn("--entrypoint", docker_run)

    def test_router_up_stops_live_then_runs_then_warms(self):
        self.run_worker("up", MODE="router", MODEL_ALIAS="gemma4-e4b",
                        MODELS_DIR="/srv/models", BENCH_API_KEY="k")
        stop_live = self.index_of("docker stop bonsai-llama-1")
        run_test = self.index_of("docker run")
        ready = self.index_of("docker inspect")
        warm = self.index_of("chat/completions")
        self.assertLess(stop_live, run_test)
        self.assertLess(run_test, ready)
        self.assertLess(ready, warm)

    def test_router_warm_up_sends_key_on_stdin_not_argv(self):
        stdin_log = os.path.join(self.tmp.name, "stdin.log")
        self.run_worker("up", MODE="router", MODEL_ALIAS="gemma4-e4b",
                        MODELS_DIR="/srv/models", BENCH_API_KEY="warm-secret",
                        FAKE_SSH_STDIN_LOG=stdin_log)
        warm = self.commands()[self.index_of("chat/completions")]
        self.assertNotIn("warm-secret", warm)
        self.assertIn("-H @-", warm)
        with open(stdin_log) as f:
            self.assertIn("Authorization: Bearer warm-secret", f.read())

    def test_router_up_warns_but_succeeds_when_the_model_does_not_load(self):
        # Spec: a preset missing from the image is a suite error, not a crash.
        out = self.run_worker_raw("up", MODE="router", MODEL_ALIAS="no-such-preset",
                                  MODELS_DIR="/srv/models", BENCH_API_KEY="k",
                                  FAKE_WARM_RC="22")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("::warning::", out.stderr)
        self.assertIn("no-such-preset", out.stderr)

    def test_router_up_still_dies_when_the_router_is_unhealthy(self):
        out = self.run_worker_raw("up", MODE="router", MODEL_ALIAS="gemma4-e4b",
                                  MODELS_DIR="/srv/models", BENCH_API_KEY="k",
                                  FAKE_TEST_RC="22")
        self.assertNotEqual(out.returncode, 0)
        self.assertFalse(any("chat/completions" in c for c in self.commands()))

    def test_router_reads_live_env_file_before_its_own_env(self):
        log = self.run_worker("up", MODE="router", MODEL_ALIAS="gemma4-e4b",
                              MODELS_DIR="/srv/models", BENCH_API_KEY="k-123",
                              LIVE_ENV_FILE="/srv/bonsai/.env", CTX="16384")
        docker_run = next(l for l in log if "docker run" in l)
        self.assertIn("--env-file /srv/bonsai/.env", docker_run)
        self.assertLess(docker_run.index("--env-file"), docker_run.index(" -e "))
        self.assertIn("-e BONSAI_API_KEY=k-123", docker_run)
        self.assertIn("-e QCM_AGENT_CTX=16384", docker_run)
        # The image name stays last.
        self.assertTrue(docker_run.rstrip().endswith("qcm-rtx-validation:922be44 >/dev/null"))

    def test_router_without_live_env_file_omits_the_flag(self):
        log = self.run_worker("up", MODE="router", MODEL_ALIAS="gemma4-e4b",
                              MODELS_DIR="/srv/models", BENCH_API_KEY="k", LIVE_ENV_FILE="")
        docker_run = next(l for l in log if "docker run" in l)
        self.assertNotIn("--env-file", docker_run)

    def test_direct_mode_ignores_live_env_file(self):
        log = self.run_worker("up", LIVE_ENV_FILE="/srv/bonsai/.env", **UP_ENV)
        docker_run = next(l for l in log if "docker run" in l)
        self.assertNotIn("--env-file", docker_run)

    def test_router_rejects_non_numeric_ctx(self):
        out = self.run_worker_raw("up", MODE="router", MODEL_ALIAS="x", MODELS_DIR="/m",
                                  BENCH_API_KEY="k", CTX="1; rm -rf /")
        self.assertNotEqual(out.returncode, 0)
        self.assertEqual(self.calls(), [])

    def test_unknown_mode_is_rejected(self):
        out = self.run_worker_raw("up", MODE="hybrid", MODEL_ALIAS="x", MODELS_DIR="/m",
                                  BENCH_API_KEY="k")
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("MODE", out.stderr)
        self.assertEqual(self.calls(), [])

    def test_router_key_never_printed(self):
        out = self.run_worker_raw("up", MODE="router", MODEL_ALIAS="gemma4-e4b",
                                  MODELS_DIR="/srv/models", BENCH_API_KEY="print-secret")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertNotIn("print-secret", out.stdout + out.stderr)

    # --- ctx: the context size the router really gave the preset ---

    def test_ctx_prints_the_preset_context(self):
        out = self.run_worker_raw("ctx", MODEL_ALIAS="gemma4-e4b", BENCH_API_KEY="k",
                                  FAKE_PROPS='{"build_info":"b1","default_generation_settings":{"n_ctx":16384}}')
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "16384")
        cmd = self.commands()[self.index_of("/props")]
        self.assertIn("model=gemma4-e4b", cmd)
        self.assertNotIn("Bearer", cmd)

    def test_ctx_rejects_a_missing_value(self):
        out = self.run_worker_raw("ctx", MODEL_ALIAS="gemma4-e4b", BENCH_API_KEY="k",
                                  FAKE_PROPS='{"error":"unauthorized"}')
        self.assertNotEqual(out.returncode, 0)
        self.assertEqual(out.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
