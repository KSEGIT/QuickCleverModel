"""Contract checks on .github/workflows/benchmark.yml.

stdlib unittest, not pytest — see tests/test_cache_viz.py for the convention.
No PyYAML: the checks read the file line by line, which is enough for the
few properties that matter here and keeps the suite dependency-free.

Why these properties: the repo is public, and this workflow reaches a real
machine. A push or pull_request trigger would run fork code on it; an input
named like a secret would print that secret on the run page; a `down` step
without always() would leave production stopped after any failure.
"""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW = os.path.join(ROOT, ".github", "workflows", "benchmark.yml")


def load():
    with open(WORKFLOW) as f:
        return f.read()


def top_level_block(text, key):
    """Lines of a top-level mapping `key:` up to the next top-level key."""
    lines = text.splitlines()
    out, inside = [], False
    for line in lines:
        if re.match(rf"^{re.escape(key)}:", line):
            inside = True
            out.append(line)
            continue
        if inside:
            if line and not line[0].isspace() and not line.startswith("#"):
                break
            out.append(line)
    return out


def input_names(text):
    """Names under on.workflow_dispatch.inputs (six-space indent)."""
    block = top_level_block(text, "on")
    names, inside = [], False
    for line in block:
        if re.match(r"^    inputs:\s*$", line):
            inside = True
            continue
        if inside:
            m = re.match(r"^      ([A-Za-z0-9_-]+):\s*$", line)
            if m:
                names.append(m.group(1))
            elif line.strip() and not line.startswith("       "):
                break
    return names


def steps(text):
    """Each step as its list of lines (split on `      - `)."""
    out, cur = [], None
    for line in text.splitlines():
        if re.match(r"^      - ", line):
            if cur:
                out.append(cur)
            cur = [line]
        elif cur is not None:
            if line and not line.startswith("        ") and line.strip():
                out.append(cur)
                cur = None
            else:
                cur.append(line)
    if cur:
        out.append(cur)
    return out


class BenchmarkWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.text = load()

    def test_only_manual_trigger(self):
        block = top_level_block(self.text, "on")
        self.assertTrue(block, "no top-level on:")
        triggers = [m.group(1) for line in block
                    if (m := re.match(r"^  ([A-Za-z_]+):", line))]
        self.assertEqual(triggers, ["workflow_dispatch"])

    def test_top_level_permissions_read_only(self):
        block = top_level_block(self.text, "permissions")
        body = [line.strip() for line in block[1:] if line.strip()]
        self.assertEqual(body, ["contents: read"])

    def test_one_run_at_a_time(self):
        block = "\n".join(top_level_block(self.text, "concurrency"))
        self.assertIn("group: rtx-benchmark", block)
        self.assertIn("cancel-in-progress: false", block)

    def test_benchmark_job_uses_environment(self):
        self.assertRegex(self.text, r"\n    environment: rtx-benchmark\n")

    def test_inputs_are_the_documented_set(self):
        self.assertEqual(input_names(self.text), [
            "worker_host", "ssh_user", "model_preset", "ctx", "suites",
            "repetitions", "restore_production"])

    def test_no_input_looks_like_a_secret(self):
        for name in input_names(self.text):
            for word in ("key", "secret", "token", "password"):
                self.assertNotIn(word, name.lower(), name)

    def test_down_step_always_runs(self):
        down = [s for s in steps(self.text)
                if any("worker.sh down" in line for line in s)]
        self.assertEqual(len(down), 1, "expected exactly one worker.sh down step")
        ifs = [line for line in down[0] if line.strip().startswith(("if:", "- if:"))]
        self.assertEqual(len(ifs), 1)
        self.assertIn("always()", ifs[0])
        self.assertIn("inputs.restore_production", ifs[0])
        # Skip only when settings never resolved (nothing was stopped).
        self.assertIn("env.WORKER_SSH != ''", ifs[0])

    def test_down_is_last_benchmark_step(self):
        all_steps = steps(self.text)
        down = next(i for i, s in enumerate(all_steps)
                    if any("worker.sh down" in line for line in s))
        publish_start = self.text.index("\n  publish:")
        # Every step before `publish:` must come at or before `down`.
        offset = 0
        for i, s in enumerate(all_steps):
            offset = self.text.index(s[0], offset)
            if offset < publish_start:
                self.assertLessEqual(i, down)

    def test_every_action_pinned_to_major(self):
        uses = re.findall(r"uses:\s*(\S+)", self.text)
        self.assertTrue(uses)
        for ref in uses:
            self.assertRegex(ref, r"^[\w.-]+/[\w.-]+@v\d+$", ref)

    def test_secrets_only_via_secrets_context(self):
        # Every mention of a secret name is inside ${{ ... secrets.NAME ... }}.
        for name in ("TS_OAUTH_CLIENT_ID", "TS_OAUTH_SECRET", "TS_AUTHKEY",
                     "WORKER_SSH_KEY", "WORKER_KNOWN_HOSTS"):
            with self.subTest(secret=name):
                for m in re.finditer(name, self.text):
                    before = self.text[max(0, m.start() - 8):m.start()]
                    self.assertEqual(before, "secrets.", f"{name} used outside secrets.")

    def test_secrets_never_inlined_into_scripts(self):
        # ${{ secrets.* }} belongs in with:/env:, never inside run: text,
        # where it would be pasted into the script body.
        for s in steps(self.text):
            in_run = False
            for line in s:
                stripped = line.strip()
                if re.match(r"^(- )?run:", stripped):
                    in_run = True
                elif re.match(r"^(- )?[a-z-]+:", stripped) and not line.startswith("          "):
                    in_run = False
                if in_run:
                    self.assertNotIn("secrets.", line)

    def test_inputs_never_inlined_into_scripts(self):
        for s in steps(self.text):
            in_run = False
            for line in s:
                stripped = line.strip()
                if re.match(r"^(- )?run:", stripped):
                    in_run = True
                elif re.match(r"^(- )?[a-z-]+:", stripped) and not line.startswith("          "):
                    in_run = False
                if in_run:
                    self.assertNotIn("${{", line)

    def test_model_presets(self):
        self.assertIn("/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf", self.text)
        self.assertIn("qwen3.5-9b-q4_k_m", self.text)
        self.assertIn("/models/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf", self.text)
        self.assertIn("qwen3.5-4b-q4_k_m", self.text)

    def test_worker_identity_masked_before_any_step_can_print_it(self):
        """Public logs: the host and user are masked first thing in Resolve
        settings, which runs before the tailnet ping, and never echoed."""
        benchmark = self.text[:self.text.index("\n  publish:")]
        resolve = self.text.index("- name: Resolve settings")
        mask_host = self.text.index('echo "::add-mask::$IN_WORKER_HOST"')
        mask_user = self.text.index('echo "::add-mask::$IN_SSH_USER"')
        first_check = self.text.index('[[ -n "$IN_WORKER_HOST" ]]')
        self.assertLess(resolve, mask_host)
        self.assertLess(max(mask_host, mask_user), first_check)
        self.assertLess(mask_host, self.text.index("ping: ${{ env.WORKER_HOST }}"))
        # Only Resolve settings holds them; no job-level env entry.
        self.assertEqual(benchmark.count("IN_WORKER_HOST:"), 1)
        self.assertEqual(benchmark.count("IN_SSH_USER:"), 1)
        # No echo prints them, except the mask itself and the lines the
        # grouped block writes to $GITHUB_ENV (NAME=value, not the log).
        for m in re.finditer(r'echo "([^"]*)"', benchmark):
            said = m.group(1)
            if said.startswith("::add-mask::") or re.match(r"^[A-Z0-9_]+=", said):
                continue
            for name in ("$IN_WORKER_HOST", "$IN_SSH_USER", "$WORKER_SSH", "$WORKER_HOST"):
                self.assertNotIn(name, said)

    def test_suite_steps_have_time_caps(self):
        for name, cap in (("Run single-slot suites", "fromJSON(env.PHASE1_TIMEOUT_MIN)"),
                          ("Run concurrency suite", "fromJSON(env.PHASE2_TIMEOUT_MIN)")):
            step = next(s for s in steps(self.text) if name in s[0])
            self.assertTrue(any("timeout-minutes:" in l and cap in l for l in step), name)
        for name in ("Start test server (1 slot)", "Restart test server (2 slots)",
                     "Restore production (worker.sh down)"):
            step = next(s for s in steps(self.text) if name in s[0])
            self.assertTrue(any("timeout-minutes:" in l for l in step), name)

    def test_phase_2_failure_does_not_fail_the_job(self):
        for name in ("Restart test server (2 slots)", "Run concurrency suite"):
            step = next(s for s in steps(self.text) if name in s[0])
            self.assertTrue(any("continue-on-error: true" in l for l in step), name)
        merge = next(s for s in steps(self.text) if "Merge phase reports" in s[0])
        self.assertIn('errors["concurrency"]', "\n".join(merge))

    def test_benchmark_checkout_drops_credentials(self):
        benchmark = self.text[:self.text.index("\n  publish:")]
        self.assertIn("persist-credentials: false", benchmark)

    def test_images_uploaded_once(self):
        self.assertNotIn("bench-images-", self.text)

    def test_gpu_name_from_variable(self):
        self.assertIn("BENCH_GPU_NAME: ${{ vars.BENCH_GPU_NAME }}", self.text)

    def test_publish_job_permissions(self):
        publish = self.text[self.text.index("\n  publish:"):]
        self.assertIn("needs: benchmark", publish)
        for perm in ("contents: write", "pages: write", "id-token: write"):
            self.assertIn(perm, publish)
        self.assertIn("name: github-pages", publish)


if __name__ == "__main__":
    unittest.main()
