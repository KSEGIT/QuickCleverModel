"""Tests for the LaunchAgent template.

stdlib unittest, not pytest — see tests/test_cache_viz.py.

These tests render the template only. They never write to
~/Library/LaunchAgents or call launchctl: running the suite must not install
a login item as a side effect.
"""
import os
import plistlib
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "menubar", "com.bonsai.menubar.plist.in")


class LaunchAgentTemplateTest(unittest.TestCase):
    def setUp(self):
        with open(TEMPLATE) as fh:
            self.rendered = fh.read().replace("@ROOT@", ROOT)
        self.plist = plistlib.loads(self.rendered.encode())

    def test_every_placeholder_is_substituted(self):
        self.assertNotIn("@ROOT@", self.rendered)

    def test_label_matches_bundle_identifier(self):
        self.assertEqual(self.plist["Label"], "com.bonsai.menubar")

    def test_runs_at_load(self):
        self.assertTrue(self.plist["RunAtLoad"])

    def test_does_not_set_keepalive(self):
        """KeepAlive would respawn the app after the user chooses Quit."""
        self.assertNotIn("KeepAlive", self.plist)

    def test_points_at_the_built_binary(self):
        target = self.plist["ProgramArguments"][0]
        self.assertEqual(
            target,
            os.path.join(ROOT, "build", "BonsaiMenuBar.app",
                         "Contents", "MacOS", "BonsaiMenuBar"))

    def test_environment_path_includes_homebrew(self):
        """Regression test for C1: launchd starts this LaunchAgent with a bare
        PATH (/usr/bin:/bin:/usr/sbin:/sbin — no Homebrew), so stack.sh's own
        `export PATH=...` normalization (see tests/test_launchd_path.py) is
        the only thing standing between the app and an unresolvable `npx`
        when it shells out. That normalization only helps once the process is
        launched with an environment it can prepend to — if this plist's own
        EnvironmentVariables/PATH entry regressed (e.g. the key were deleted
        from the template), the rest of the suite would not catch it, since
        no other test here inspects EnvironmentVariables at all."""
        env = self.plist["EnvironmentVariables"]
        path_entries = env["PATH"].split(":")
        self.assertIn("/opt/homebrew/bin", path_entries)
        self.assertIn("/usr/bin", path_entries)
