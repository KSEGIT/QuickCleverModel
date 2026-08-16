"""Tests for `./stack.sh status --json`.

stdlib unittest, not pytest — matching tests/test_cache_viz.py, which states
the convention and the reason.

Run: python3 -m unittest discover -s tests -v

These tests deliberately do NOT fabricate running services. Standing up a
process whose command line matches sig_of() well enough to fool `listening`
is fragile and would test the fake more than the code. Instead they assert
the contract's shape, its internal consistency, and that it agrees with the
human-readable table.
"""
import json
import os
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STACK = os.path.join(ROOT, "stack.sh")
SERVICES = ["llama", "playwright", "webui"]


def run_stack(args, extra_env=None):
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    return subprocess.run([STACK] + args, capture_output=True, text=True,
                          env=env, cwd=ROOT)


def env_defines(var):
    """True when .env sets `var`, which would override the process env."""
    path = os.path.join(ROOT, ".env")
    if not os.path.isfile(path):
        return False
    with open(path) as fh:
        return any(line.strip().startswith(var + "=") for line in fh)


class StatusJsonTest(unittest.TestCase):
    def setUp(self):
        proc = run_stack(["status", "--json"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.raw = proc.stdout

    def test_emits_valid_json(self):
        json.loads(self.raw)

    def test_one_entry_per_service_in_order(self):
        entries = json.loads(self.raw)
        self.assertEqual([e["service"] for e in entries], SERVICES)

    def test_keys_and_types(self):
        for entry in json.loads(self.raw):
            self.assertIsInstance(entry["service"], str)
            self.assertIsInstance(entry["port"], int)
            self.assertIn(entry["state"], ("up", "down"))

    def test_pid_present_exactly_when_up(self):
        for entry in json.loads(self.raw):
            if entry["state"] == "up":
                self.assertIsInstance(entry["pid"], int, entry)
            else:
                self.assertIsNone(entry["pid"], entry)

    def test_ports_agree_with_human_table(self):
        """The two renderings must never disagree about a port."""
        table = run_stack(["status"]).stdout
        table_ports = {}
        for line in table.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] in SERVICES:
                table_ports[parts[0]] = int(parts[1])
        json_ports = {e["service"]: e["port"] for e in json.loads(self.raw)}
        self.assertEqual(json_ports, table_ports)

    @unittest.skipIf(env_defines("BONSAI_PORT"),
                     ".env sets BONSAI_PORT, which overrides the process env")
    def test_port_honours_env_override(self):
        proc = run_stack(["status", "--json"], {"BONSAI_PORT": "18080"})
        entries = json.loads(proc.stdout)
        llama = next(e for e in entries if e["service"] == "llama")
        self.assertEqual(llama["port"], 18080)

    def test_human_table_still_works(self):
        """--json must not disturb the existing no-flag behavior."""
        out = run_stack(["status"]).stdout
        self.assertIn("SERVICE", out)
        self.assertIn("llama", out)
