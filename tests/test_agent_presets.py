"""Agent presets must not inherit Bonsai-specific generation settings."""
import configparser
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
AGENTS = ("qwen3.5-9b-q4_k_m", "granite-4.1-8b-q4_k_m", "qwen3.5-4b-q4_k_m",
          "qwen3.6-35b-a3b", "gemma4-e4b")


class AgentPresets(unittest.TestCase):
    def setUp(self):
        self.presets = configparser.ConfigParser(interpolation=None)
        self.presets.read(ROOT / "models.ini.in")

    def test_all_eight_aliases_are_declared(self):
        self.assertEqual(set(self.presets.sections()), {
            "*", "bonsai-27b-ternary", "bonsai-27b-1bit", "bonsai-27b-ternary-text", *AGENTS,
        })

    def test_agent_presets_have_family_templates_and_separate_controls(self):
        for model in AGENTS:
            with self.subTest(model=model):
                effective = dict(self.presets["*"])
                effective.update(self.presets[model])
                if model.startswith("qwen3.5-"):
                    self.assertEqual("@ROOT@/qwen3.5-chat-template.jinja",
                                     effective.get("chat-template-file"))
                else:
                    self.assertNotIn("chat-template-file", effective)
                self.assertNotIn("mmproj", effective)
                self.assertNotEqual("@CTX@", effective["c"])
                self.assertNotEqual("@CTK@", effective["ctk"])
                self.assertNotEqual("@CTV@", effective["ctv"])
                self.assertNotEqual("@REASONING_BUDGET@", effective.get("reasoning-budget"))
                self.assertEqual("none", effective["spec-type"])
                self.assertEqual("false", effective["load-on-startup"])

    def test_granite_is_official_instruct_not_guardian(self):
        model = self.presets["granite-4.1-8b-q4_k_m"]["model"]
        self.assertTrue(model.endswith("granite-4.1-8b-Q4_K_M.gguf"))
        self.assertNotIn("guardian", model.lower())

    def test_new_browser_models_use_native_templates_and_text_only_weights(self):
        qwen = self.presets["qwen3.6-35b-a3b"]
        gemma = self.presets["gemma4-e4b"]
        self.assertEqual(qwen["model"], "@QWEN36_MODEL@")
        self.assertEqual(gemma["model"], "@GEMMA4_MODEL@")
        self.assertEqual(qwen["fit"], "on")
        self.assertEqual(qwen["fit-target"], "@QWEN36_FIT_TARGET@")
        self.assertEqual(qwen["ngl"], "@QWEN36_NGL@")
        self.assertEqual(qwen["c"], "@QWEN36_CTX@")
        self.assertEqual(gemma["c"], "@GEMMA4_CTX@")
        self.assertEqual(gemma["ngl"], "@GEMMA4_NGL@")
        for section in (qwen, gemma):
            self.assertNotIn("chat-template-file", section)
            self.assertNotIn("mmproj", section)
            self.assertNotIn("cache-reuse", section)
            self.assertEqual(section["parallel"], "1")
            self.assertEqual(section["load-on-startup"], "false")

    def test_global_preset_does_not_force_model_specific_offload_or_sampling(self):
        global_settings = self.presets["*"]
        for key in ("ngl", "temp", "top-p", "top-k", "min-p", "spec-type",
                    "cache-reuse", "ctk", "ctv", "reasoning-budget", "no-kv-offload"):
            self.assertNotIn(key, global_settings)

    def test_every_placeholder_has_a_startup_substitution(self):
        import re
        script = (ROOT / "start-server.sh").read_text()
        placeholders = set(re.findall(r"@[A-Z0-9_]+@", (ROOT / "models.ini.in").read_text()))
        for token in placeholders:
            self.assertIn(token + "|", script, token)


if __name__ == "__main__":
    unittest.main()
