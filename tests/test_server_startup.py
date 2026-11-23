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
        # Records what the launcher hands the server: argv one per line, env.
        binary.write_text('#!/bin/sh\n'
                          'dir=$(dirname "$0")\n'
                          'printf "%s\\n" "$@" > "$dir/args"\n'
                          'env > "$dir/env"\n'
                          'exit 0\n')
        binary.chmod(0o755)
        self.bin_dir = binary.parent
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
        self.assertEqual(preset["cache-reuse"], "256")
        self.assertEqual(preset["reasoning"], "off")
        self.assertNotIn("@", (self.root / "run/models.ini").read_text())

    def test_truenas_preset_uses_existing_renderer_and_download_filter(self):
        shutil.copyfile(ROOT / "docker/models.truenas.ini.in", self.root / "models.ini.in")
        model_files = {
            "qwen3.5-9b-q4_k_m": "Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf",
            "qwen3.5-4b-q4_k_m": "Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf",
            "bonsai-27b-1bit": "Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf",
            "granite-4.1-8b-q4_k_m": "granite-4.1-8b-GGUF/granite-4.1-8b-Q4_K_M.gguf",
        }
        for relative_path in model_files.values():
            model = self.root / "models" / relative_path
            model.parent.mkdir(parents=True, exist_ok=True)
            model.touch()
        projector = self.root / "models/Bonsai-27B-gguf/Bonsai-27B-mmproj-Q8_0.gguf"
        projector.touch()
        result = self.run_launcher(BONSAI_UBATCH="256", QCM_AGENT_UBATCH="256")
        self.assertEqual(result.returncode, 0, result.stderr)
        presets = self.presets()
        self.assertEqual(set(presets.sections()), {"*", *model_files})
        self.assertEqual([name for name in model_files
                          if presets.getboolean(name, "load-on-startup")],
                         ["qwen3.5-9b-q4_k_m"])
        for name, relative_path in model_files.items():
            with self.subTest(model=name):
                self.assertEqual(presets[name]["model"],
                                 str(self.root / "models" / relative_path))
                self.assertEqual(presets[name]["ubatch-size"], "256")
        self.assertEqual(presets["bonsai-27b-1bit"]["mmproj"], str(projector))
        self.assertNotIn("@", (self.root / "run/models.ini").read_text())

        # A stale mounted preset must not advertise an absent optional weight.
        (self.root / "models" / model_files["granite-4.1-8b-q4_k_m"]).unlink()
        result = self.run_launcher(BONSAI_UBATCH="256", QCM_AGENT_UBATCH="256")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(self.presets().sections()),
                         {"*", *model_files} - {"granite-4.1-8b-q4_k_m"})

    def server_args(self):
        return (self.bin_dir / "args").read_text().splitlines()

    def test_key_reaches_server_in_a_private_file_not_argv_or_env(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.server_args()
        self.assertNotIn("fixture-only", "\n".join(args))
        self.assertNotIn("--api-key", args)
        key_file = Path(args[args.index("--api-key-file") + 1])
        self.assertEqual(key_file, self.root / "run/api-key")
        self.assertEqual(key_file.read_text(), "fixture-only\n")
        self.assertEqual(key_file.stat().st_mode & 0o777, 0o400)
        self.assertNotIn("fixture-only", (self.bin_dir / "env").read_text())

    def test_key_file_is_replaced_on_restart_without_leftovers(self):
        self.assertEqual(self.run_launcher().returncode, 0)
        old_inode = (self.root / "run/api-key").stat().st_ino
        result = self.run_launcher(BONSAI_API_KEY="fixture-rotated")
        self.assertEqual(result.returncode, 0, result.stderr)
        key_file = self.root / "run/api-key"
        self.assertEqual(key_file.read_text(), "fixture-rotated\n")
        self.assertEqual(key_file.stat().st_mode & 0o777, 0o400)
        # Renamed over, not rewritten in place: a new inode, no temp files.
        self.assertNotEqual(key_file.stat().st_ino, old_inode)
        self.assertEqual(sorted(p.name for p in (self.root / "run").glob("api-key*")),
                         ["api-key"])

    def test_comma_separated_keys_become_one_key_per_line(self):
        result = self.run_launcher(BONSAI_API_KEY="fixture-one,fixture-two")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "run/api-key").read_text(),
                         "fixture-one\nfixture-two\n")
        self.assertNotIn("fixture-two", "\n".join(self.server_args()))

    def test_key_the_file_format_would_change_fails_before_start(self):
        # '#' lines are comments in --api-key-file: with no other key the
        # server would run with no auth. A line break would add a key. An
        # empty comma item is almost surely a typo.
        for key in ("#fixture", "fixture,#second", "fixture\nsecond", "fixture\rx",
                    ",fixture", "fixture,", "fixture,,second"):
            with self.subTest(key=key):
                result = self.run_launcher(BONSAI_API_KEY=key)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("BONSAI_API_KEY", result.stderr)
                self.assertNotIn("fixture", result.stderr)
                self.assertFalse((self.bin_dir / "args").exists())
                self.assertFalse((self.root / "run/api-key").exists())

    BONSAI_WEIGHTS = {
        "bonsai-27b-ternary": "Ternary-Bonsai-27B-gguf/Ternary-Bonsai-27B-PQ2_0.gguf",
        "bonsai-27b-1bit": "Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf",
    }

    def touch_weights(self, *names):
        for name in names:
            model = self.root / "models" / self.BONSAI_WEIGHTS[name]
            model.parent.mkdir(parents=True, exist_ok=True)
            model.touch()

    def test_model_allowlist_drops_every_other_preset(self):
        self.touch_weights("bonsai-27b-ternary", "bonsai-27b-1bit")
        result = self.run_launcher(QCM_MODELS="bonsai-27b-ternary")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(self.presets().sections()), {"*", "bonsai-27b-ternary"})
        self.assertEqual(self.presets()["*"]["jinja"], "true")
        self.assertIn("models : bonsai-27b-ternary\n", result.stdout)
        self.assertNotIn("WARNING", result.stderr)

    def test_model_allowlist_takes_a_comma_list_with_spaces(self):
        self.touch_weights("bonsai-27b-1bit")
        model = self.root / "models/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf"
        model.parent.mkdir(parents=True)
        model.touch()
        result = self.run_launcher(QCM_MODELS=" bonsai-27b-1bit ,\tqwen3.5-4b-q4_k_m,")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(self.presets().sections()),
                         {"*", "bonsai-27b-1bit", "qwen3.5-4b-q4_k_m"})

    def test_model_allowlist_rejects_unknown_or_absent_models(self):
        self.touch_weights("bonsai-27b-ternary")
        # bonsai-27b-1bit and qwen3.5-9b-q4_k_m are real presets, but their
        # weights are absent here. Spaces or line breaks inside are errors.
        for value in ("bonsai-27b-ternry", "bonsai-27b-1bit", "qwen3.5-9b-q4_k_m",
                      "bonsai-27b-ternary,bonsai-27b-1bit", "*", "a]b", ",",
                      "bonsai-27b ternary", "bonsai-27b-ternary\nbonsai-27b-1bit",
                      "bonsai-27b-ternary\n"):
            with self.subTest(value=value):
                result = self.run_launcher(QCM_MODELS=value)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("QCM_MODELS", result.stderr)
                self.assertFalse((self.bin_dir / "args").exists())

    def test_model_allowlist_rejects_bonsai_without_weights(self):
        # The weight filter never drops Bonsai sections; the allow-list check
        # must still see that the file is missing.
        result = self.run_launcher(QCM_MODELS="bonsai-27b-ternary")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("weights are missing", result.stderr)
        self.assertIn("Ternary-Bonsai-27B-PQ2_0.gguf", result.stderr)

    def test_model_allowlist_empties_the_download_cache(self):
        self.touch_weights("bonsai-27b-ternary")
        cache = self.root / "run/empty-model-cache"
        cache.mkdir(parents=True)
        (cache / "models--someone--other-GGUF").mkdir()
        result = self.run_launcher(QCM_MODELS="bonsai-27b-ternary",
                                   LLAMA_CACHE="/somewhere/else")
        self.assertEqual(result.returncode, 0, result.stderr)
        env = (self.bin_dir / "env").read_text().splitlines()
        self.assertIn(f"LLAMA_CACHE={cache}", env)
        self.assertEqual(list(cache.iterdir()), [])

    def test_no_allowlist_leaves_the_download_cache_alone(self):
        result = self.run_launcher(LLAMA_CACHE="/somewhere/else")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LLAMA_CACHE=/somewhere/else",
                      (self.bin_dir / "env").read_text().splitlines())

    def test_model_allowlist_warns_when_nothing_loads_at_startup(self):
        self.touch_weights("bonsai-27b-1bit")
        result = self.run_launcher(QCM_MODELS="bonsai-27b-1bit")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("load-on-startup", result.stderr)

    def test_model_allowlist_warns_about_default_models_it_drops(self):
        self.touch_weights("bonsai-27b-ternary")
        result = self.run_launcher(QCM_MODELS="bonsai-27b-ternary",
                                   DEFAULT_MODELS="bonsai-27b-ternary, qwen3.5-9b-q4_k_m")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DEFAULT_MODELS names qwen3.5-9b-q4_k_m", result.stderr)
        self.assertNotIn("names bonsai-27b-ternary", result.stderr)

    def test_bad_context_fails_before_start(self):
        result = self.run_launcher(QCM_AGENT_CTX="huge")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("QCM_AGENT_CTX", result.stderr)

    def test_bad_cache_type_fails_before_start(self):
        result = self.run_launcher(QCM_QWEN_CTK="q1_0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("QCM_QWEN_CTK", result.stderr)

    def test_new_models_are_hidden_until_selected_artifact_exists(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("qwen3.6-35b-a3b", self.presets())
        self.assertNotIn("gemma4-e4b", self.presets())

    def test_qwen36_fit_uses_headroom_and_optional_explicit_layer_limit(self):
        model = self.root / "models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
        model.parent.mkdir(parents=True)
        model.touch()
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        preset = self.presets()["qwen3.6-35b-a3b"]
        self.assertEqual(preset["model"], str(model))
        self.assertEqual(preset["c"], "8192")
        self.assertEqual(preset["fit-target"], "1536")
        self.assertNotIn("ngl", preset)
        self.assertEqual(preset["reasoning"], "off")
        result = self.run_launcher(QCM_QWEN36_NGL="12", QCM_QWEN36_FIT_TARGET="2048")
        self.assertEqual(result.returncode, 0, result.stderr)
        preset = self.presets()["qwen3.6-35b-a3b"]
        self.assertEqual(preset["ngl"], "12")
        self.assertEqual(preset["fit-target"], "2048")

    def test_alternative_quants_change_only_selected_new_model(self):
        qwen = self.root / "models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
        gemma = self.root / "models/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q5_K_M.gguf"
        for model in (qwen, gemma):
            model.parent.mkdir(parents=True, exist_ok=True)
            model.touch()
        result = self.run_launcher(QCM_QWEN36_QUANT="IQ4_XS", QCM_GEMMA4_QUANT="Q5_K_M")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.presets()["qwen3.6-35b-a3b"]["model"], str(qwen))
        self.assertEqual(self.presets()["gemma4-e4b"]["model"], str(gemma))

    def test_invalid_quant_fails_before_server_start(self):
        result = self.run_launcher(QCM_QWEN36_QUANT="Q2_K")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("QCM_QWEN36_QUANT", result.stderr)

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
