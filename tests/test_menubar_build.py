"""Build smoke test for the menu bar app.

stdlib unittest, not pytest — see tests/test_cache_viz.py.
Skipped entirely when swiftc is absent, so the suite still passes on a box
without Xcode (`make up` does not need it; `make menubar` does).

Builds into a tempdir, NOT build/. The Makefile's APP variable is a `:=` that
callers can override, so `make menubar APP=<tmp>` builds a throwaway bundle
instead of overwriting build/BonsaiMenuBar.app — the exact bundle path the
installed LaunchAgent (com.bonsai.menubar) points at and, on a dev box, the
one a live login-started process is currently running from. Running this
suite must never clobber that binary out from under the running app.
"""
import os
import plistlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@unittest.skipIf(shutil.which("swiftc") is None, "swiftc not installed")
class MenuBarBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="bonsai-menubar-test-")
        cls.APP = os.path.join(cls._tmpdir, "BonsaiMenuBar.app")
        proc = subprocess.run(
            ["make", "menubar", f"APP={cls.APP}"], cwd=ROOT,
            capture_output=True, text=True)
        if proc.returncode != 0:
            shutil.rmtree(cls._tmpdir, ignore_errors=True)
            raise AssertionError("make menubar failed:\n" + proc.stdout + proc.stderr)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_bundle_has_an_executable(self):
        binary = os.path.join(self.APP, "Contents", "MacOS", "BonsaiMenuBar")
        self.assertTrue(os.path.isfile(binary), binary)
        self.assertTrue(os.access(binary, os.X_OK))

    def test_info_plist_marks_it_menu_bar_only(self):
        with open(os.path.join(self.APP, "Contents", "Info.plist"), "rb") as fh:
            info = plistlib.load(fh)
        self.assertTrue(info["LSUIElement"], "must not show a Dock icon")
        self.assertEqual(info["CFBundleIdentifier"], "com.bonsai.menubar")

    def test_info_plist_points_at_this_repo(self):
        with open(os.path.join(self.APP, "Contents", "Info.plist"), "rb") as fh:
            info = plistlib.load(fh)
        self.assertEqual(info["BonsaiRoot"], ROOT)
        self.assertTrue(os.path.isfile(os.path.join(info["BonsaiRoot"], "stack.sh")))
