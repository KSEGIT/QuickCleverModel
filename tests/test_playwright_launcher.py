"""Check the arguments passed by the production browser launcher."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def old_system_bash():
    """macOS ships /bin/bash 3.2, where `set -u` rejects "${empty[@]}" (fixed in 4.4)."""
    try:
        version = subprocess.run(["/bin/bash", "-c", "echo ${BASH_VERSINFO[0]}${BASH_VERSINFO[1]}"],
                                 capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return "/bin/bash" if int(version) < 44 else None


class PlaywrightLauncherTests(unittest.TestCase):
    def launch(self, config="", bash="bash"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copy(ROOT / "start-playwright-mcp.sh", root)
            (root / ".env").write_text(config)
            npx = root / "npx"
            npx.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            npx.chmod(0o755)
            env = dict(os.environ, PATH=f"{root}:{os.environ['PATH']}")
            result = subprocess.run([bash, str(root / "start-playwright-mcp.sh"),
                                     "--headless"], env=env, text=True,
                                    capture_output=True, check=True)
            return result.stdout.splitlines()

    def test_pinned_package_and_small_responses(self):
        args = self.launch()
        self.assertIn("@playwright/mcp@0.0.82", args)
        self.assertEqual(args[args.index("--snapshot-mode") + 1], "none")
        self.assertEqual(args[args.index("--image-responses") + 1], "omit")
        self.assertIn("--isolated", args)
        self.assertIn("--headless", args)

    def test_env_overrides_snapshot_and_disables_eviction(self):
        args = self.launch("PW_MCP_SNAPSHOT=full\nPW_MCP_OUTPUT_MAX_SIZE=0\n")
        self.assertEqual(args[args.index("--snapshot-mode") + 1], "full")
        self.assertNotIn("--output-max-size", args)

    @unittest.skipUnless(old_system_bash(), "needs a bash older than 4.4 (macOS /bin/bash)")
    def test_eviction_opt_out_works_on_macos_system_bash(self):
        # The macOS CI runner runs `bash` as 3.2; an empty MAX_SIZE_ARGS aborted
        # the launcher with "unbound variable" there while Homebrew bash passed.
        args = self.launch("PW_MCP_OUTPUT_MAX_SIZE=0\n", bash=old_system_bash())
        self.assertNotIn("--output-max-size", args)
        self.assertIn("--headless", args)


if __name__ == "__main__":
    unittest.main()
