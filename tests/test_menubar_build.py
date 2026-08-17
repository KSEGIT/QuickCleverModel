"""Build smoke test for the menu bar app.

stdlib unittest, not pytest — see tests/test_cache_viz.py.
Skipped unless this is macOS AND swiftc is present. swiftc alone is not
enough: the app imports SwiftUI and builds an .app bundle, so on a Linux box
with a Swift toolchain installed the build would fail rather than skip. The
repo targets Linux too (install.sh, docker/compose.linux.yaml), so the suite
must stay green there — `make up` needs no Xcode; `make menubar` does.

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
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("swiftc"),
                     "needs macOS and swiftc (the app imports SwiftUI)")
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


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("swiftc"),
                     "needs macOS and swiftc (the app imports SwiftUI)")
class AwkwardRepoPathTest(unittest.TestCase):
    """@ROOT@ crosses two escaping layers: XML, and a sed replacement.

    A checkout under e.g. ~/R&D breaks both. In a sed replacement an
    unescaped `&` means "the whole match", so the naive form wrote the
    literal `@ROOT@` into BonsaiRoot (measured: `/tmp/R@ROOT@D lab`), and
    `&` is also an XML metacharacter that must be encoded as `&amp;` or the
    plist will not parse. The build copies only the Makefile and menubar/
    into a temp path containing both an ampersand and a space, so CURDIR is
    the awkward path without relocating the real repo.
    """

    def test_ampersand_in_repo_path_survives_into_the_plist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "R&D lab")
            os.makedirs(root)
            shutil.copy(os.path.join(ROOT, "Makefile"), root)
            shutil.copytree(os.path.join(ROOT, "menubar"),
                            os.path.join(root, "menubar"))
            proc = subprocess.run(["make", "menubar"], cwd=root,
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0,
                             f"build failed under {root!r}:\n{proc.stdout}{proc.stderr}")
            plist = os.path.join(root, "build", "BonsaiMenuBar.app",
                                 "Contents", "Info.plist")
            with open(plist, "rb") as fh:
                info = plistlib.load(fh)   # raises if the XML is malformed
            self.assertNotIn("@ROOT@", info["BonsaiRoot"],
                             "sed treated & as the whole match")
            self.assertTrue(info["BonsaiRoot"].endswith("R&D lab"),
                            f"BonsaiRoot mangled: {info['BonsaiRoot']!r}")
