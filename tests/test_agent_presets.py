"""Agent presets must not inherit Bonsai-specific generation settings."""
import configparser
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
AGENTS = ("qwen3.5-9b-q4_k_m", "granite-4.1-8b-q4_k_m", "qwen3.5-4b-q4_k_m")


class AgentPresets(unittest.TestCase):
    def setUp(self):
        self.presets = configparser.ConfigParser(interpolation=None)
        self.presets.read(ROOT / "models.ini.in")

    def test_all_six_aliases_are_declared(self):
        self.assertEqual(set(self.presets.sections()), {
            "*", "bonsai-27b-ternary", "bonsai-27b-1bit", "bonsai-27b-ternary-text", *AGENTS,
        })

    def test_agent_presets_have_native_templates_and_separate_controls(self):
        for model in AGENTS:
            with self.subTest(model=model):
                effective = dict(self.presets["*"])
                effective.update(self.presets[model])
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

    def test_every_placeholder_has_a_startup_substitution(self):
        import re
        script = (ROOT / "start-server.sh").read_text()
        placeholders = set(re.findall(r"@[A-Z0-9_]+@", (ROOT / "models.ini.in").read_text()))
        for token in placeholders:
            self.assertIn(token + "|", script, token)


if __name__ == "__main__":
    unittest.main()
