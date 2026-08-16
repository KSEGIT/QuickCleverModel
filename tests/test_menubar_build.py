"""Build smoke test for the menu bar app.

stdlib unittest, not pytest — see tests/test_cache_viz.py.
Skipped entirely when swiftc is absent, so the suite still passes on a box
without Xcode (`make up` does not need it; `make menubar` does).
"""
import os
import plistlib
import shutil
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "build", "BonsaiMenuBar.app")


@unittest.skipIf(shutil.which("swiftc") is None, "swiftc not installed")
class MenuBarBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        proc = subprocess.run(["make", "menubar"], cwd=ROOT,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise AssertionError("make menubar failed:\n" + proc.stdout + proc.stderr)

    def test_bundle_has_an_executable(self):
        binary = os.path.join(APP, "Contents", "MacOS", "BonsaiMenuBar")
        self.assertTrue(os.path.isfile(binary), binary)
        self.assertTrue(os.access(binary, os.X_OK))

    def test_info_plist_marks_it_menu_bar_only(self):
        with open(os.path.join(APP, "Contents", "Info.plist"), "rb") as fh:
            info = plistlib.load(fh)
        self.assertTrue(info["LSUIElement"], "must not show a Dock icon")
        self.assertEqual(info["CFBundleIdentifier"], "com.bonsai.menubar")

    def test_info_plist_points_at_this_repo(self):
        with open(os.path.join(APP, "Contents", "Info.plist"), "rb") as fh:
            info = plistlib.load(fh)
        self.assertEqual(info["BonsaiRoot"], ROOT)
        self.assertTrue(os.path.isfile(os.path.join(info["BonsaiRoot"], "stack.sh")))
