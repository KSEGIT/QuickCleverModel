"""Exercise the launcher with a fixture binary and no inference weights."""
import configparser
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

# A test key holds this marker, and the marker is in nothing else the launcher
# prints or runs. If it shows up in stdout, stderr or argv, a key leaked.
MARKER = "leakmarker-5b1d9e"
TEST_KEY = f"fixture-{MARKER}"


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
                          'ps -o args= -p $$ > "$dir/argv"\n'
                          'env > "$dir/env"\n'
                          'exit 0\n')
        binary.chmod(0o755)
        self.bin_dir = binary.parent
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(("BONSAI_", "QCM_"))}
        self.env["BONSAI_API_KEY"] = TEST_KEY

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

    def key_file(self):
        args = self.server_args()
        return Path(args[args.index("--api-key-file") + 1])

    def key_files(self):
        return sorted(p.name for p in (self.root / "run").glob("*api-key*"))

    def test_key_reaches_server_in_a_private_file_not_argv_or_env(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.server_args()
        self.assertNotIn(MARKER, "\n".join(args))
        self.assertNotIn(MARKER, (self.bin_dir / "argv").read_text())
        self.assertNotIn(MARKER, result.stdout + result.stderr)
        self.assertNotIn("--api-key", args)
        key_file = self.key_file()
        self.assertEqual(key_file.parent, self.root / "run")
        self.assertRegex(key_file.name, r"^api-key\.[0-9]+$")
        self.assertEqual(key_file.read_text(), TEST_KEY + "\n")
        self.assertEqual(key_file.stat().st_mode & 0o777, 0o400)
        self.assertNotIn(MARKER, (self.bin_dir / "env").read_text())
        self.assertEqual(self.key_files(), [key_file.name])

    def test_second_run_leaves_a_live_server_key_file_alone(self):
        # This test process stands in for a running server.
        live = self.root / f"run/api-key.{os.getpid()}"
        live.parent.mkdir(parents=True)
        live.write_text("fixture-live\n")
        live.chmod(0o400)
        result = self.run_launcher(BONSAI_API_KEY="fixture-new")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(live.read_text(), "fixture-live\n")
        self.assertNotEqual(self.key_file(), live)
        self.assertEqual(self.key_file().read_text(), "fixture-new\n")

    def test_key_files_of_gone_servers_are_removed(self):
        gone = subprocess.Popen(["true"])
        gone.wait()
        run = self.root / "run"
        run.mkdir()
        stale = run / f"api-key.{gone.pid}"
        stale.write_text("fixture-stale\n")
        stale.chmod(0o400)
        other = run / "api-key.backup"
        other.write_text("not ours to remove\n")
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(stale.exists())
        self.assertTrue(other.exists())
        # No temporary file is left behind.
        self.assertEqual(self.key_files(), sorted([self.key_file().name, other.name]))

    def test_comma_separated_keys_become_one_key_per_line(self):
        result = self.run_launcher(BONSAI_API_KEY="fixture-one,fixture-two")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.key_file().read_text(), "fixture-one\nfixture-two\n")
        self.assertNotIn("fixture-two", "\n".join(self.server_args()))
        self.assertNotIn("fixture-two", (self.bin_dir / "argv").read_text())

    def test_key_the_file_format_would_change_fails_before_start(self):
        # '#' lines are comments in --api-key-file: with no other key the
        # server would run with no auth. A line break would add a key. An
        # empty comma item is almost surely a typo. A double quote was CSV
        # quoting for --api-key, but is literal in the file.
        m = MARKER
        for key in (f"#{m}", f"{m},#second", f"{m}\nsecond", f"{m}\rx",
                    f",{m}", f"{m},", f"{m},,second",
                    f'"{m}"', f'{m}"x', f'{m},"second,third"'):
            with self.subTest(key=key):
                result = self.run_launcher(BONSAI_API_KEY=key)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("BONSAI_API_KEY", result.stderr)
                self.assertNotIn(MARKER, result.stdout + result.stderr)
                self.assertFalse((self.bin_dir / "args").exists())
                self.assertEqual(self.key_files(), [])

    def test_double_quote_error_names_the_cause(self):
        result = self.run_launcher(BONSAI_API_KEY=f'{MARKER}"x')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("double quote", result.stderr)
        self.assertNotIn(MARKER, result.stdout + result.stderr)

    def test_xtrace_never_prints_the_key(self):
        # `bash -x` and SHELLOPTS=xtrace trace every expansion, key included.
        for how in ("flag", "shellopts"):
            for key in (TEST_KEY, f"{MARKER}-one,{MARKER}-two", f'{MARKER}"x', f"#{MARKER}"):
                with self.subTest(how=how, key=key):
                    env = dict(self.env, BONSAI_API_KEY=key)
                    cmd = ["bash", str(self.root / "start-server.sh")]
                    if how == "flag":
                        cmd.insert(1, "-x")
                    else:
                        env["SHELLOPTS"] = "xtrace"
                    result = subprocess.run(cmd, env=env, text=True, capture_output=True)
                    self.assertNotIn(MARKER, result.stdout + result.stderr)
                    if key == TEST_KEY:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        # Tracing is back on after the key code.
                        self.assertIn("+ exec ", result.stderr)

    def test_stale_temp_files_and_the_legacy_key_file_are_removed(self):
        run = self.root / "run"
        run.mkdir()
        old = run / ".api-key-tmp.OLDOLD"
        old.write_text("fixture-old\n")
        os.utime(old, (1, 1))
        legacy = run / "api-key"
        legacy.write_text("fixture-legacy\n")
        legacy.chmod(0o400)
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(old.exists())
        self.assertFalse(legacy.exists())
        self.assertEqual(self.key_files(), [self.key_file().name])

    def test_a_fresh_temp_file_may_belong_to_a_running_start(self):
        run = self.root / "run"
        run.mkdir()
        fresh = run / ".api-key-tmp.FRESH1"
        fresh.write_text("fixture-fresh\n")
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(fresh.exists())

    def test_cleanup_never_follows_or_removes_symlinks(self):
        run = self.root / "run"
        run.mkdir()
        target = self.root / "precious"
        target.write_text("keep me\n")
        old_tmp = run / ".api-key-tmp.LINK01"
        old_tmp.symlink_to(target)
        os.utime(old_tmp, (1, 1), follow_symlinks=False)
        legacy = run / "api-key"
        legacy.symlink_to(target)
        dead = subprocess.Popen(["true"])
        dead.wait()
        stale_link = run / f"api-key.{dead.pid}"
        stale_link.symlink_to(target)
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(target.read_text(), "keep me\n")
        for link in (old_tmp, legacy, stale_link):
            self.assertTrue(link.is_symlink(), link)

    def test_a_pid_that_exists_for_another_user_keeps_its_key_file(self):
        # PID 1 always exists. For a normal user, `kill -0 1` fails with
        # "Operation not permitted" (EPERM), not "No such process". As root it
        # succeeds. Either way the file must stay.
        run = self.root / "run"
        run.mkdir()
        other = run / "api-key.1"
        other.write_text("fixture-other-user\n")
        other.chmod(0o400)
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(other.read_text(), "fixture-other-user\n")

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
        result = self.run_launcher(QCM_MODELS=" bonsai-27b-1bit ,\tqwen3.5-4b-q4_k_m ")
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
                      "bonsai-27b-ternary\n", "bonsai-27b-ternary,",
                      ",bonsai-27b-ternary", "bonsai-27b-ternary,,bonsai-27b-ternary",
                      "bonsai-27b-ternary, ,bonsai-27b-ternary"):
            with self.subTest(value=value):
                result = self.run_launcher(QCM_MODELS=value)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("QCM_MODELS", result.stderr)
                self.assertFalse((self.bin_dir / "args").exists())

    def write_preset(self, text):
        (self.root / "models.ini.in").write_bytes(text.encode())

    def test_model_allowlist_reads_crlf_presets(self):
        self.touch_weights("bonsai-27b-ternary")
        template = (ROOT / "models.ini.in").read_text().replace("\n", "\r\n")
        self.write_preset(template)
        result = self.run_launcher(QCM_MODELS="bonsai-27b-ternary")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(self.presets().sections()), {"*", "bonsai-27b-ternary"})
        result = self.run_launcher(QCM_MODELS="bonsai-27b-1bit")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("weights that are missing", result.stderr)

    def test_model_allowlist_reads_model_lines_without_spaces(self):
        for name in ("alpha", "gamma"):
            model = self.root / f"models/{name}.gguf"
            model.parent.mkdir(parents=True, exist_ok=True)
            model.touch()
        self.write_preset("[*]\njinja = true\n\n"
                          "[alpha]\nmodel=@ROOT@/models/alpha.gguf\n\n"
                          "[beta]\nmodel =@ROOT@/models/beta.gguf\n\n"
                          "[gamma]\n  model=  @ROOT@/models/gamma.gguf  \r\n")
        result = self.run_launcher(QCM_MODELS="alpha,gamma")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(self.presets().sections()), {"*", "alpha", "gamma"})
        result = self.run_launcher(QCM_MODELS="beta")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("beta.gguf", result.stderr)

    def preset_text(self):
        return (self.root / "run/models.ini").read_text()

    def test_model_allowlist_reads_a_section_name_followed_by_a_comment(self):
        for name in ("alpha", "beta"):
            model = self.root / f"models/{name}.gguf"
            model.parent.mkdir(parents=True, exist_ok=True)
            model.touch()
        self.write_preset("[*]\njinja = true\n\n"
                          "[alpha] ;a comment\nmodel = @ROOT@/models/alpha.gguf\n\n"
                          "[beta]   # another\nmodel = @ROOT@/models/beta.gguf\n")
        result = self.run_launcher(QCM_MODELS="alpha")
        self.assertEqual(result.returncode, 0, result.stderr)
        text = self.preset_text()
        self.assertIn("alpha.gguf", text)
        self.assertNotIn("beta", text)

    def test_model_allowlist_with_a_duplicate_section_keeps_or_drops_both(self):
        for name in ("alpha", "beta"):
            model = self.root / f"models/{name}.gguf"
            model.parent.mkdir(parents=True, exist_ok=True)
            model.touch()
        self.write_preset("[*]\njinja = true\n\n"
                          "[alpha]\nmodel = @ROOT@/models/alpha.gguf\n\n"
                          "[beta]\nmodel = @ROOT@/models/beta.gguf\n\n"
                          "[alpha]\nc = 4096\n\n"
                          "[beta]\nc = 1024\n")
        result = self.run_launcher(QCM_MODELS="alpha")
        self.assertEqual(result.returncode, 0, result.stderr)
        text = self.preset_text()
        self.assertEqual(text.count("[alpha]"), 2)
        self.assertIn("c = 4096", text)
        self.assertNotIn("beta", text)
        self.assertNotIn("c = 1024", text)

    def test_model_allowlist_with_a_duplicate_section_checks_every_model_line(self):
        # The second [alpha] names weights that do not exist. If the check
        # looked only at the first section, the router could start with a
        # preset whose model is missing.
        model = self.root / "models/alpha.gguf"
        model.parent.mkdir(parents=True)
        model.touch()
        self.write_preset("[*]\njinja = true\n\n"
                          "[alpha]\nmodel = @ROOT@/models/alpha.gguf\n\n"
                          "[alpha]\nmodel = @ROOT@/models/missing.gguf\n")
        result = self.run_launcher(QCM_MODELS="alpha")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing.gguf", result.stderr)

    def test_model_allowlist_rejects_bonsai_without_weights(self):
        # The weight filter never drops Bonsai sections; the allow-list check
        # must still see that the file is missing.
        result = self.run_launcher(QCM_MODELS="bonsai-27b-ternary")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("weights that are missing", result.stderr)
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
