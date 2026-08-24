"""Every script the stack executes must keep its executable bit.

stdlib unittest, not pytest — see tests/test_cache_viz.py for the convention.

This exists because the bit has been silently stripped twice by automated
commits. It is not cosmetic: stack.sh launches services with

    nohup "$(script_of "$svc")" > "$LOGS/$svc.log" 2>&1 &

so a non-executable launcher means `make up` reports the service down with
nothing in its log but a permission error, and `./setup-opencode.sh` fails
with exit 126. Git tracks the bit in the index, so a mode-only change can
ride along in a diff that otherwise looks like a pure content edit.
"""
import os
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Everything invoked as a command: by stack.sh, by the Makefile, or by a human
# following the docs.
EXECUTABLE = [
    "stack.sh",
    "start-server.sh",
    "start-webui.sh",
    "start-playwright-mcp.sh",
    "fetch-models.sh",
    "bench.sh",
    "install.sh",
    "setup-nvidia.sh",
    "setup-opencode.sh",
    "cache-viz.py",
]


class ScriptPermissionTest(unittest.TestCase):
    def test_scripts_are_executable_on_disk(self):
        for name in EXECUTABLE:
            path = os.path.join(ROOT, name)
            with self.subTest(script=name):
                self.assertTrue(os.path.isfile(path), f"{name} is missing")
                self.assertTrue(os.access(path, os.X_OK),
                                f"{name} is not executable — `./{name}` exits 126")

    def test_executable_bit_is_recorded_in_git(self):
        """On-disk mode is not enough: git stores it separately, and a fresh
        clone gets whatever git recorded. Reads the index rather than HEAD so
        a staged `git update-index --chmod=+x` is seen before it is committed
        — otherwise the fix for this very failure cannot be verified until
        after it lands."""
        out = subprocess.run(["git", "ls-files", "-s", "--", *EXECUTABLE],
                             capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(out.returncode, 0, out.stderr)
        modes = {}
        for line in out.stdout.splitlines():
            meta, _, name = line.partition("\t")
            modes[name.strip()] = meta.split()[0]
        for name in EXECUTABLE:
            with self.subTest(script=name):
                self.assertEqual(modes.get(name), "100755",
                                 f"{name} is {modes.get(name)} in git, expected 100755")
