"""Regression test for C1: launchd's minimal PATH breaks stack.sh.

stdlib unittest, not pytest — see tests/test_cache_viz.py.

Measured via `launchctl print gui/$(id -u)/com.bonsai.menubar`: a
LaunchAgent-started process inherits PATH=/usr/bin:/bin:/usr/sbin:/sbin only
— no Homebrew. `npx` (needed by start-playwright-mcp.sh, which the `playwright`
service execs) lives at /opt/homebrew/bin/npx and does NOT resolve under that
PATH. Without a fix, clicking Restart in the menu bar app (which runs stack.sh
from the installed LaunchAgent) stops all three services, restarts llama fine,
then fails to restart playwright with "npx: command not found" — and until
stack.sh's early-bail grep also catches that message, the failure isn't
noticed for the full 120x2s readiness loop.

stack.sh's fix is a PATH normalization exported near the top of the file.
This test reads that line out of stack.sh itself (rather than hand-copying a
second version of it here) and evaluates it inside a shell that starts from
launchd's exact bare-minimum PATH, so it exercises the real fix rather than a
paraphrase of it. It is written to PASS with the fix in place and FAIL if the
PATH export is ever removed from stack.sh — that is what makes it a
regression test for C1, not just a description of it.
"""
import os
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STACK = os.path.join(ROOT, "stack.sh")

# launchd's own environment for this LaunchAgent — see the docstring above.
MINIMAL_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"

# External commands stack.sh and its three launcher scripts (start-server.sh,
# start-webui.sh, start-playwright-mcp.sh) resolve via PATH rather than an
# absolute path. `npx` is the one that actually depends on the fix (it's
# Homebrew-only); the rest are asserted too because the finding calls for
# "every external command ... needs", not just the one that currently fails.
REQUIRED_COMMANDS = [
    "npx",    # start-playwright-mcp.sh:53 `exec npx ...` — the C1 failure
    "lsof",   # stack.sh port_pid()
    "curl",   # stack.sh ready()
    "ps",     # stack.sh listening(), stop_one()
    "kill",   # stack.sh stop_one()
    "pgrep",  # stack.sh cmd_reap()
    "pkill",  # stack.sh cmd_reap()
    "open",   # stack.sh `open` verb
    "sed",    # start-server.sh preset rendering
]


def _path_export_line():
    """The literal `export PATH=...` line stack.sh adds right after ROOT= to
    work around launchd's minimal environment. Returns None if no such line
    exists (e.g. someone reverted the fix)."""
    with open(STACK) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("export PATH="):
                return line
    return None


class LaunchdPathTest(unittest.TestCase):
    def test_stack_sh_exports_a_path_normalization(self):
        self.assertIsNotNone(
            _path_export_line(),
            "stack.sh no longer exports a PATH normalization near the top — "
            "this is the C1 regression: launchd's minimal PATH "
            f"({MINIMAL_PATH}) will not include Homebrew, and npx (needed by "
            "start-playwright-mcp.sh) becomes unresolvable when the menu bar "
            "app runs stack.sh from the installed LaunchAgent")

    def test_required_commands_resolve_under_launchd_env(self):
        line = _path_export_line()
        self.assertIsNotNone(line)  # see previous test for the message
        script = (
            f"{line}\n"
            "missing=\"\"\n"
            f"for c in {' '.join(REQUIRED_COMMANDS)}; do\n"
            '  command -v "$c" >/dev/null 2>&1 || missing="$missing $c"\n'
            "done\n"
            'echo "$missing"\n'
        )
        proc = subprocess.run(
            ["bash", "-c", script],
            env={"PATH": MINIMAL_PATH}, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        missing = proc.stdout.split()
        self.assertEqual(
            missing, [],
            f"not resolvable under launchd's minimal PATH even after "
            f"stack.sh's normalization: {missing} "
            f"(stderr={proc.stderr!r})")

    def test_npx_is_the_command_launchds_bare_path_actually_lacks(self):
        """Documents the premise: WITHOUT stack.sh's fix, npx is not on
        launchd's PATH at all. This is the failure C1 describes and the one
        test_required_commands_resolve_under_launchd_env guards against."""
        proc = subprocess.run(["bash", "-c", "command -v npx"],
                              env={"PATH": MINIMAL_PATH},
                              capture_output=True, text=True)
        self.assertNotEqual(
            proc.returncode, 0,
            "npx resolves under launchd's bare PATH on this machine, which "
            "means this test's premise for C1 no longer holds here")
