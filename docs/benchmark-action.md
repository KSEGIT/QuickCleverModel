# RTX benchmark action

This page explains the GitHub Actions job that runs the browser-agent
benchmark on the RTX worker, and how to set it up and use it. The workflow
file is `.github/workflows/benchmark.yml`.

## What it does

- You start it by hand, from the Actions tab. It does not run on its own.
- It joins the worker's tailnet, connects over SSH, and swaps the worker's
  live model container for a test server for the length of the run.
- It runs the browser-agent suites you choose against that test server.
- It puts the live container back at the end — even if a suite failed — and
  checks that it answers before the job finishes.
- It saves the results, and shows them on a small dashboard site built by
  GitHub Pages.

### Why a manual trigger only

This repository is public. A workflow that runs on `push` or on a pull
request would also run for a pull request from a fork, with no review. That
must never reach the worker over SSH. So the workflow only has a
`workflow_dispatch` (manual) trigger.

GitHub itself lets anyone with write access to the repository start a
`workflow_dispatch` run. In practice, only the repository owner should start
one, because a run stops the live model for everyone (see "Stopping
production, on purpose" below).

## One-time setup

A repository owner does these steps once, before the first run.

### 1. Tailscale

1. In the Tailscale admin console, create an OAuth client and give it the
   tag `tag:ci`.
2. Add a Tailscale ACL rule that lets `tag:ci` reach the worker on port 22
   only. Do not give it any wider access.
3. Store the OAuth client ID and secret as repository secrets (see the table
   below). An auth key works too, as a fallback.

### 2. A dedicated SSH key

1. Make a new SSH key pair. Use it only for this job. Do not reuse a key you
   use anywhere else.
2. Add the public key to the worker's `authorized_keys` for the account this
   job will use.
3. Run `ssh-keyscan <worker-host>` by hand, once, from a machine you trust.
   Look at the output and check it names the worker's real host key before
   you store it. Do not skip this check.
4. Store the private key and the `ssh-keyscan` output as repository secrets.

Never write the worker's real Tailscale IP address or its SSH username in a
doc, an issue, or a commit message. Use a placeholder such as `100.x.y.z` or
`<worker-host>` instead. The setup below keeps both out of the workflow's
visible inputs for the same reason.

### 3. Repository secrets and variables

Create a GitHub Environment named `rtx-benchmark`, and add these secrets and
variables to it.

**Secrets** (never shown on the run page, even in this public repository):

| Secret | Holds |
| --- | --- |
| `TS_OAUTH_CLIENT_ID` | The Tailscale OAuth client ID from step 1 |
| `TS_OAUTH_SECRET` | The Tailscale OAuth client secret (or use `TS_AUTHKEY` instead, with a Tailscale auth key) |
| `WORKER_SSH_KEY` | The private half of the dedicated SSH key from step 2 |
| `WORKER_KNOWN_HOSTS` | The checked `ssh-keyscan` output for the worker |

**Variables** (defaults for a run; not secret, but still worth keeping out of
public docs where they'd be specific to your worker):

| Variable | Meaning | Default |
| --- | --- | --- |
| `WORKER_HOST` | The worker's tailnet name or address | none — must be set |
| `WORKER_SSH_USER` | The SSH account the job connects as | none — must be set |
| `LIVE_CONTAINER` | The production container the job pauses and restores | `bonsai-llama-1` |
| `LIVE_HEALTH_URL` | Health-check URL for the live container, checked on the worker | `http://127.0.0.1:8080/health` |
| `LIVE_ENV_FILE` | Worker path to the live container's env file, if it needs one | none — no key is sent if unset |
| `TEST_IMAGE` | The pinned test image the benchmark runs | `qcm-rtx-validation:922be44` |
| `MODELS_DIR` | Worker path to the model files | none — must be set |
| `TEMPLATE_FILE` | Worker path to the chat template for the test server | none — must be set |

A `workflow_dispatch` run also takes `worker_host` and `ssh_user` inputs that
can override these variables. Leave them blank when you start a run — the
job then uses the variables above. If you type a value instead, it shows on
the run's page, and this repository is public.

**`LIVE_ENV_FILE` must be readable by the SSH user, without `sudo`.** The
restore step reads the live server's API key from this file, on the worker,
to check that the live container answers after it restarts. If the SSH user
cannot read the file, the check sends no key. The live server can then
answer 401, and the job reports the live container as unhealthy — even
though the container itself is running. Set file permissions so the
dedicated SSH account (from step 2) can read this file directly.

### 4. Turn on GitHub Pages

1. Go to **Settings → Pages**.
2. Under **Build and deployment**, set **Source** to **GitHub Actions**.
3. Save. This step needs repository admin rights, so only the owner can do
   it, and it is a one-time setup step, not something each run repeats.
4. Go to **Settings → Environments → github-pages** (GitHub creates this
   environment the first time a Pages deployment runs, or you can create it
   yourself). Check its **Deployment branches and tags** rule. If it only
   allows a specific branch (often the default branch), and you dispatch a
   benchmark run from a different branch, the publish job stalls or is
   blocked — the environment rule never matches. Add every branch you might
   dispatch a run from, or set the rule to allow all branches.

## Starting a run

1. Go to the **Actions** tab, open the `benchmark.yml` workflow, and choose
   **Run workflow**.
2. Fill in the inputs you want to change. Leave the rest blank to use the
   defaults.

| Input | Meaning | Default |
| --- | --- | --- |
| `worker_host` | Overrides `WORKER_HOST` for this run only | (environment variable) |
| `ssh_user` | Overrides `WORKER_SSH_USER` for this run only | (environment variable) |
| `model_preset` | Which model to benchmark: `qwen3.5-9b` or `qwen3.5-4b` | `qwen3.5-9b` |
| `ctx` | Context size, in tokens | `32768` |
| `suites` | Space-separated list of suites to run: `fixture`, `long_context`, `live_web`, `concurrency` | `fixture long_context live_web` |
| `repetitions` | How many times to repeat each suite's tasks | `1` |
| `restore_production` | Restart the live container when the run ends | `true` |

3. Anyone with write access to the repository can start a run, but only the
   repository owner should — see "Why a manual trigger only" above.
4. Click **Run workflow** and wait. A run can take a while, because it
   downloads nothing new but does run real browser tasks against a live
   model, one suite at a time.

### About `restore_production`

Leave this at its default, `true`, for almost every run. When it is `true`,
the job always puts the live container back at the end, even if a suite
failed. When it is `false`, the job leaves the test server running and does
**not** restart the live container automatically — you must restart it
yourself. Only set it to `false` if you plan to inspect the test server by
hand right after the run.

## What happens during a run

1. The job joins the tailnet, then connects to the worker over SSH.
2. It stops the live container (`bonsai-llama-1` by default) and starts the
   test server in its place, bound to the worker's loopback address only.
3. It runs the suites you chose:

   | Suite | Server needs | What it runs | Pass means |
   | --- | --- | --- | --- |
   | `fixture` | 1 slot | Core browser tasks, repeated | Each task's own check passes |
   | `long_context` | 1 slot | A prompt near the context limit, asking the model to quote one exact line | The line quoted is exactly right |
   | `live_web` | 1 slot | A real DuckDuckGo image search that must save one real image file | An image file is actually saved |
   | `concurrency` | 2 slots | Four job-application runs: one at a time, then two at a time in pairs | Each run's own check passes |

   For `concurrency`, the job restarts the test server first, with 2 slots
   and a 49,152-token context, because that suite needs two agents running
   at once.
4. It saves raw results and any saved images as workflow artifacts, so you
   can look at them even without the dashboard.
5. At the end, whether or not a suite failed, it stops the test server and
   starts the live container again (unless you set `restore_production` to
   `false`). It checks the live container's health endpoint. If that check
   does not return HTTP 200, the job fails — that tells you production is
   not back up cleanly, instead of hiding the problem. This step retries on
   its own: starting the live container is tried up to 5 times, and if the
   whole step still fails, the workflow waits 20 seconds and runs it a
   second time. This covers a short network drop between the runner and the
   worker.
6. On success, a separate publish job downloads the run summary that this
   job uploaded as an artifact, then writes it to the `gh-pages` branch
   (`data/runs/<run_id>.json`), rebuilds the run index (`data/index.json`),
   and updates the Pages site from `bench-site/`. If the benchmark job never
   reaches its "Upload summary" step, there is nothing for the publish job
   to download, and publishing does not happen.

Only one run can happen at a time. If you start a second run while one is
already going, it waits its turn instead of running alongside the first.

## Reading the dashboard

Open the repository's Pages URL — **Settings → Pages** shows it once a run
has published. The dashboard shows:

- The latest run's pass rate, median time, and peak VRAM, for each suite.
- A pass-rate history chart, per suite, across all runs.
- A median-time history chart.
- A peak-VRAM-versus-context chart, with a line at 8,192 MiB (the worker's
  total VRAM).
- A table of every run, with its date, commit, model, context, and suites.

## Stopping production, on purpose

A run stops the real chat model for as long as the run takes. Anyone using
the live chat during that window sees it go down, then come back once the
run ends. Pick a quiet time to start a run, and tell anyone else who might
be using the live model first.
