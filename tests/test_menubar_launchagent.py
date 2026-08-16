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
