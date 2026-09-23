"""Exercise the launcher with a fixture binary and no inference weights."""
import configparser
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ServerStartup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "R&D models"
        self.root.mkdir()
        for file in ("start-server.sh", "models.ini.in"):
            shutil.copyfile(ROOT / file, self.root / file)
        binary = self.root / "src/llama.cpp-prism/build/bin/llama-server"
        binary.parent.mkdir(parents=True)
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(("BONSAI_", "QCM_"))}
        self.env["BONSAI_API_KEY"] = "fixture-only"

    def run_launcher(self, **overrides):
        return subprocess.run(["bash", str(self.root / "start-server.sh")],
                              env=dict(self.env, **overrides), text=True, capture_output=True)

    def presets(self):
        parser = configparser.ConfigParser(interpolation=None)
        parser.read(self.root / "run/models.ini")
        return parser

    def test_missing_agent_downloads_are_not_advertised(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(self.presets().sections()),
                         {"*", "bonsai-27b-ternary", "bonsai-27b-1bit", "bonsai-27b-ternary-text"})

    def test_paths_and_agent_defaults_are_rendered(self):
        model = self.root / "models/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf"
        model.parent.mkdir(parents=True)
        model.touch()
        result = self.run_launcher(BONSAI_CTX="65536", BONSAI_CTK="q4_0", QCM_QWEN4_CTX="12288")
        self.assertEqual(result.returncode, 0, result.stderr)
        preset = self.presets()["qwen3.5-4b-q4_k_m"]
        self.assertEqual(preset["model"], str(model))
        self.assertEqual(preset["c"], "12288")
        self.assertEqual(preset["ctk"], "f16")
        self.assertEqual(preset["reasoning"], "off")
        self.assertNotIn("@", (self.root / "run/models.ini").read_text())

    def test_bad_context_fails_before_start(self):
        result = self.run_launcher(QCM_AGENT_CTX="huge")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("QCM_AGENT_CTX", result.stderr)

    def test_bad_cache_type_fails_before_start(self):
        result = self.run_launcher(QCM_QWEN_CTK="q1_0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("QCM_QWEN_CTK", result.stderr)

    def test_legacy_model_requires_separate_current_artifact(self):
        old = self.root / "models/Ternary-Bonsai-27B-gguf/Ternary-Bonsai-27B-Q2_0.gguf"
        old.parent.mkdir(parents=True)
        old.write_bytes(b"legacy remains untouched")
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fetch-models.sh bonsai", result.stderr)
        self.assertEqual(old.read_bytes(), b"legacy remains untouched")

    def test_stale_binary_is_rejected(self):
        shutil.copyfile(ROOT / "runtime-versions.env", self.root / "runtime-versions.env")
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match", result.stderr)


if __name__ == "__main__":
    unittest.main()
