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


@unittest.skipIf(env_defines("BONSAI_PORT") or env_defines("PW_MCP_PORT")
                  or env_defines("WEBUI_PORT"),
                 ".env pins a port var, which would override the process env "
                 "below and defeat the port-blackholing this class relies on")
class StatusJsonDownBranchTest(unittest.TestCase):
    """Every assertion in StatusJsonTest runs against a live, all-up stack,
    so cmd_status_json's `down` branch (state="down", pid=null) was never
    exercised — `test_pid_present_exactly_when_up` passed vacuously for it.
    Proof of the gap this closes: changing `${pid:-null}` to `${pid:-}` in
    stack.sh emits `"pid":}` (invalid JSON) for a down service, and every
    test in this file still passed before this class was added.

    No process faking needed: pointing every port var at a port nothing
    listens on makes `listening()` return empty for all three services,
    which is exactly the `down` path.
    """

    DEAD_PORTS = {"BONSAI_PORT": "19999", "PW_MCP_PORT": "19998",
                  "WEBUI_PORT": "19997"}

    def setUp(self):
        proc = run_stack(["status", "--json"], self.DEAD_PORTS)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.raw = proc.stdout

    def test_emits_valid_json_when_all_down(self):
        json.loads(self.raw)  # the crux of the gap: this must not raise

    def test_every_service_reports_down_with_null_pid(self):
        entries = json.loads(self.raw)
        self.assertEqual([e["service"] for e in entries], SERVICES)
        for entry in entries:
            self.assertEqual(entry["state"], "down", entry)
            self.assertIsNone(entry["pid"], entry)
