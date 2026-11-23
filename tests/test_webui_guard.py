"""Tests for webui-guard.sh, the hand-copied Open WebUI tags and the native
launcher's warnings.

stdlib unittest, not pytest — see tests/test_cache_viz.py for the convention.

webui-guard.sh is run for real with fake `curl` and `docker` first on PATH,
so no server and no Docker daemon are needed.
"""
import os
import re
import shutil
import stat
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUARD = os.path.join(ROOT, "webui-guard.sh")
UPDATE_SH = os.path.join(ROOT, "update.sh")
START_WEBUI = os.path.join(ROOT, "start-webui.sh")

# Fake curl: /health answers when FAKE_UI_UP=1; signin prints FAKE_SIGNIN as
# the HTTP code. Every call is logged, so a test can see what was asked.
FAKE_CURL = r"""#!/usr/bin/env bash
printf 'curl %s\n' "$*" >> "$FAKE_LOG"
for a in "$@"; do last="$a"; done
case "$last" in
  */health)
    [ "${FAKE_UI_UP:-0}" = 1 ] && exit 0
    if [ -n "${FAKE_UI_UP_AFTER:-}" ]; then
      n=$(grep -c '/health' "$FAKE_LOG")
      [ "$n" -ge "$FAKE_UI_UP_AFTER" ] && exit 0
    fi
    exit 7;;
  */api/v1/auths/signin) printf '%s' "${FAKE_SIGNIN:-400}"; exit 0;;
esac
exit 1
"""

FAKE_DOCKER = r"""#!/usr/bin/env bash
printf 'docker %s\n' "$*" >> "$FAKE_LOG"
exit "${FAKE_DOCKER_RC:-0}"
"""


class GuardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.bin = os.path.join(self.tmp, "bin")
        os.mkdir(self.bin)
        for name, body in (("curl", FAKE_CURL), ("docker", FAKE_DOCKER)):
            path = os.path.join(self.bin, name)
            with open(path, "w") as fh:
                fh.write(body)
            os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
        self.log = os.path.join(self.tmp, "log")
        open(self.log, "w").close()
        self.env_file = os.path.join(self.tmp, "env")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_guard(self, env_lines, args=(), **fake):
        with open(self.env_file, "w") as fh:
            fh.write("".join(line + "\n" for line in env_lines))
        env = {
            "PATH": self.bin + os.pathsep + os.environ.get("PATH", ""),
            "HOME": self.tmp,
            "QCM_ENV_FILE": self.env_file,
            "QCM_WEBUI_URL": "http://127.0.0.1:9",
            "FAKE_LOG": self.log,
        }
        env.update({k: str(v) for k, v in fake.items()})
        r = subprocess.run(["bash", GUARD, *args], capture_output=True,
                           text=True, env=env)
        with open(self.log) as fh:
            calls = fh.read()
        return r, calls

    def test_login_off_checks_nothing(self):
        r, calls = self.run_guard(["WEBUI_AUTH=false"], FAKE_UI_UP=1,
                                  FAKE_SIGNIN=200)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("signin", calls)
        self.assertNotIn("docker", calls)

    def test_a_ui_that_is_down_is_not_checked(self):
        r, calls = self.run_guard(["WEBUI_AUTH=true"], FAKE_UI_UP=0)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("signin", calls)

    def test_a_rejected_default_login_is_fine(self):
        r, calls = self.run_guard(["WEBUI_AUTH=true"], FAKE_UI_UP=1,
                                  FAKE_SIGNIN=400)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("signin", calls)
        self.assertNotIn("docker", calls)

    def test_a_working_default_login_stops_the_ui(self):
        """The real risk: a no-login volume with login turned on."""
        r, calls = self.run_guard(["WEBUI_AUTH=True"], FAKE_UI_UP=1,
                                  FAKE_SIGNIN=200)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("docker stop bonsai-webui", calls)
        self.assertIn("admin@localhost", r.stderr)
        self.assertIn("fresh", r.stderr.lower(),
                      "the message must say how to fix it")

    def test_it_says_so_when_it_cannot_stop_the_ui(self):
        r, _ = self.run_guard(["WEBUI_AUTH=true"], FAKE_UI_UP=1,
                              FAKE_SIGNIN=200, FAKE_DOCKER_RC=1)
        self.assertEqual(r.returncode, 3)
        self.assertIn("COULD NOT STOP", r.stderr)
        self.assertIn("STILL RUNNING", r.stderr)
        self.assertNotIn("stopped the", r.stderr)

    def test_it_waits_for_a_ui_that_is_still_starting(self):
        """FAKE_UI_UP_AFTER: /health answers from the Nth call on."""
        r, calls = self.run_guard(["WEBUI_AUTH=true"], FAKE_UI_UP_AFTER=2,
                                  FAKE_SIGNIN=400, args=["--wait", "10"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("signin", calls, "it gave up on a UI that was starting")

    def test_without_wait_it_checks_once(self):
        r, calls = self.run_guard(["WEBUI_AUTH=true"], FAKE_UI_UP_AFTER=2)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("signin", calls)

    def test_a_bad_model_id_is_refused(self):
        r, calls = self.run_guard(['WEBUI_MODEL=\'a"]},{"x\''])
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("WEBUI_MODEL", r.stderr)
        self.assertEqual(calls, "", "nothing may run after a bad model id")

    def test_a_plain_model_id_passes(self):
        r, _ = self.run_guard(["WEBUI_MODEL=bonsai-27b-ternary"])
        self.assertEqual(r.returncode, 0, r.stderr)


class OneModelIdRule(unittest.TestCase):
    """update.sh and webui-guard.sh both read the rule from lib/webui.sh."""

    def test_neither_script_defines_its_own(self):
        for path in (UPDATE_SH, GUARD):
            with open(path) as fh:
                body = fh.read()
            self.assertNotRegex(body, r"(?m)^\s*(QCM_)?MODEL_ID_RE=", path)
            self.assertIn("lib/webui.sh", body, path)


class OpenWebuiVersionIsNotHandCopied(unittest.TestCase):
    """docker/compose.linux.yaml is the one place the Open WebUI version is
    chosen. Every other copy must match it, and update.sh must hold none."""

    IMAGE_RE = re.compile(r"ghcr\.io/open-webui/open-webui:([A-Za-z0-9._-]+)")
    PIP_RE = re.compile(r"open-webui==([0-9][0-9A-Za-z.]*)")
    # Dated records of what was pinned before. They name old versions on
    # purpose.
    HISTORY = {os.path.join("docs", "dependency-audit.md")}
    SKIP_DIRS = {".git", ".worktrees", "node_modules", ".venv", "bench-site"}

    @classmethod
    def _compose_tag(cls):
        with open(os.path.join(ROOT, "docker", "compose.linux.yaml")) as fh:
            tags = cls.IMAGE_RE.findall(fh.read())
        assert len(tags) == 1, tags
        return tags[0]

    def _files(self):
        for base, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in self.SKIP_DIRS]
            for name in files:
                if not name.endswith((".yaml", ".yml", ".sh", ".md", ".py",
                                      ".example", ".txt", ".env")):
                    continue
                path = os.path.join(base, name)
                rel = os.path.relpath(path, ROOT)
                if rel in self.HISTORY:
                    continue
                with open(path, encoding="utf-8", errors="replace") as fh:
                    yield rel, fh.read()

    def test_every_copy_matches_the_linux_compose_file(self):
        want = self._compose_tag()
        bad = []
        for rel, text in self._files():
            for tag in self.IMAGE_RE.findall(text):
                if tag != want:
                    bad.append(f"{rel}: {tag}")
            for ver in self.PIP_RE.findall(text):
                if "v" + ver != want:
                    bad.append(f"{rel}: open-webui=={ver}")
        self.assertEqual(bad, [], f"copies differ from {want}")

    def test_update_sh_names_no_open_webui_tag(self):
        with open(UPDATE_SH) as fh:
            self.assertEqual(self.IMAGE_RE.findall(fh.read()), [],
                             "update.sh must read the tag from compose")

    def test_update_sh_reads_the_tag_from_the_compose_file(self):
        script = (f'export BONSAI_UPDATE_LIB=1; source "{UPDATE_SH}"; '
                  'printf "%s" "$WEBUI_IMAGE"')
        r = subprocess.run(["bash", "-c", script], capture_output=True,
                           text=True)
        self.assertEqual(r.stdout,
                         "ghcr.io/open-webui/open-webui:" + self._compose_tag(),
                         r.stderr)


def env_defines(var):
    path = os.path.join(ROOT, ".env")
    if not os.path.isfile(path):
        return False
    with open(path) as fh:
        return any(line.strip().startswith(var + "=") for line in fh)


@unittest.skipIf(env_defines("WEBUI_AUTH") or env_defines("WEBUI_MODEL"),
                 ".env sets WEBUI_AUTH or WEBUI_MODEL, which overrides the test")
class NativeLauncherSaysWhatItIgnores(unittest.TestCase):
    """start-webui.sh forces WEBUI_AUTH=False and lists every model. A
    setting in .env that it ignores must not be ignored in silence."""

    def _source(self, **extra):
        env = dict(os.environ)
        env.setdefault("BONSAI_API_KEY", "test-bonsai-key")
        env.pop("WEBUI_AUTH", None)
        env.pop("WEBUI_MODEL", None)
        env.update(extra)
        r = subprocess.run(["bash", "-c", 'source "$1"', "bash", START_WEBUI],
                           capture_output=True, text=True, env=env, cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stderr

    def test_login_on_is_reported_as_ignored(self):
        err = self._source(WEBUI_AUTH="true")
        self.assertIn("WEBUI_AUTH=true is ignored", err)
        self.assertIn("WITHOUT a login", err)

    def test_a_fixed_model_is_reported_as_ignored(self):
        err = self._source(WEBUI_MODEL="bonsai-27b-ternary")
        self.assertIn("WEBUI_MODEL=bonsai-27b-ternary is ignored", err)

    def test_the_default_says_nothing(self):
        self.assertEqual(self._source(), "")
        self.assertEqual(self._source(WEBUI_AUTH="false"), "")


if __name__ == "__main__":
    unittest.main()
