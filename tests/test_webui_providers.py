"""Tests for how start-webui.sh composes Open WebUI's provider connections.

stdlib unittest, not pytest — see tests/test_cache_viz.py for the convention.

Run: python3 -m unittest discover -s tests -v

start-webui.sh returns early when sourced rather than executed, so these
tests can source it and inspect the exported environment without launching a
server. The composition is worth pinning because it is easy to get subtly
wrong and impossible to notice: Open WebUI reads the PLURAL forms as
semicolon-separated lists, and a mismatched url/key pairing would silently
send the local key to Hetzner (or the reverse).
"""
import os
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "start-webui.sh")
HETZNER_DEFAULT = "https://inference.hetzner.com/api/v1"


def env_defines(var):
    """True when .env sets `var` — it is sourced with `set -a` and would win."""
    path = os.path.join(ROOT, ".env")
    if not os.path.isfile(path):
        return False
    with open(path) as fh:
        return any(line.strip().startswith(var + "=") for line in fh)


def source_and_dump(extra_env=None):
    """Source the script (it self-terminates) and return its exported env."""
    env = dict(os.environ)
    # Provide BONSAI_API_KEY if not already set (required by start-webui.sh)
    if "BONSAI_API_KEY" not in env:
        env["BONSAI_API_KEY"] = "test-bonsai-key"
    env.update(extra_env or {})
    proc = subprocess.run(
        ["bash", "-c", 'source "$1"; env', "bash", SCRIPT],
        capture_output=True, text=True, env=env, cwd=ROOT)
    if proc.returncode != 0:
        raise AssertionError(f"sourcing failed:\n{proc.stdout}\n{proc.stderr}")
    out = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k] = v
    return out


@unittest.skipIf(env_defines("HETZNER_API_KEY"),
                 ".env sets HETZNER_API_KEY, which overrides the process env")
class ProviderCompositionTest(unittest.TestCase):
    def test_local_only_by_default(self):
        """No Hetzner key means nothing leaves the machine."""
        env = source_and_dump({"HETZNER_API_KEY": ""})
        self.assertNotIn("HETZNER", env.get("OPENAI_API_BASE_URLS", ""))
        self.assertTrue(env["OPENAI_API_BASE_URL"].startswith("http://127.0.0.1:"))
        self.assertTrue(env["OPENAI_API_BASE_URL"].endswith("/v1"))

    def test_hetzner_appended_when_key_present(self):
        env = source_and_dump({"HETZNER_API_KEY": "hz-test-key"})
        urls = env["OPENAI_API_BASE_URLS"].split(";")
        self.assertEqual(len(urls), 2, env["OPENAI_API_BASE_URLS"])
        self.assertTrue(urls[0].startswith("http://127.0.0.1:"))
        self.assertEqual(urls[1], HETZNER_DEFAULT)

    def test_keys_line_up_with_urls(self):
        """A mismatch here would send the local key to Hetzner."""
        env = source_and_dump({"HETZNER_API_KEY": "hz-test-key"})
        urls = env["OPENAI_API_BASE_URLS"].split(";")
        keys = env["OPENAI_API_KEYS"].split(";")
        self.assertEqual(len(urls), len(keys))
        self.assertEqual(keys[1], "hz-test-key")
        self.assertNotEqual(keys[0], "hz-test-key")

    def test_custom_hetzner_base_is_honoured(self):
        env = source_and_dump({"HETZNER_API_KEY": "hz-test-key",
                               "HETZNER_API_BASE": "https://example.invalid/v1"})
        self.assertTrue(env["OPENAI_API_BASE_URLS"].endswith(
            ";https://example.invalid/v1"))

    def test_persistent_config_is_disabled(self):
        """.env must be the only source of truth; the DB must not win.

        This is the setting whose absence let a stale key survive in
        .webui-data and fail every request with "Invalid API Key".
        """
        for extra in ({"HETZNER_API_KEY": ""}, {"HETZNER_API_KEY": "hz-test-key"}):
            with self.subTest(extra=extra):
                self.assertEqual(
                    source_and_dump(extra)["ENABLE_PERSISTENT_CONFIG"], "False")

    def test_stale_plural_env_cleared_in_local_only_mode(self):
        """Regression: plural forms inherited from the parent must be unset.

        If OPENAI_API_BASE_URLS or OPENAI_API_KEYS are set in the parent
        environment but HETZNER_API_KEY is not, the script must unset them
        before exporting the singular forms — otherwise the plural forms take
        precedence in Open WebUI and the local-only config is silently ignored.
        """
        env = source_and_dump({
            "HETZNER_API_KEY": "",
            "OPENAI_API_BASE_URLS": "http://stale.example/v1",
            "OPENAI_API_KEYS": "stale-key"
        })
        self.assertNotIn("OPENAI_API_BASE_URLS", env)
        self.assertNotIn("OPENAI_API_KEYS", env)
        self.assertTrue(env["OPENAI_API_BASE_URL"].startswith("http://127.0.0.1:"))
