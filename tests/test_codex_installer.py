"""The client-only entrypoint must never start server installation."""

from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CodexClientDispatchTest(unittest.TestCase):
    def test_client_dispatch_preserves_arguments_and_skips_server_setup(self):
        with tempfile.TemporaryDirectory(prefix="codex-install-") as tmp:
            root = Path(tmp)
            shutil.copy2(ROOT / "install.sh", root / "install.sh")
            client = root / "setup-codex.sh"
            client.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$ARGS_FILE"\nexit 29\n')
            client.chmod(0o755)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            uname = fake_bin / "uname"
            uname.write_text('#!/bin/sh\ntouch "$PLATFORM_MARKER"\nprintf "Darwin\\n"\n')
            uname.chmod(0o755)
            env = dict(os.environ, PATH=str(fake_bin) + os.pathsep + os.environ["PATH"],
                       ARGS_FILE=str(root / "args"), PLATFORM_MARKER=str(root / "platform"))
            args = ["--base-url", "http://example.invalid:8080/v1", "--key-file", "a path/key", "--check"]
            result = subprocess.run([str(root / "install.sh"), "--codex-client", *args],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 29, result.stderr)
            self.assertEqual((root / "args").read_text().splitlines(), args)
            self.assertFalse((root / "platform").exists(), "Client setup ran a server platform check")


if __name__ == "__main__":
    unittest.main()
