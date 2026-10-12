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
3. Store the OAuth client ID and secret as `rtx-benchmark` environment
   secrets (see the table below). An auth key works too, as a fallback.

### 2. A dedicated SSH key

1. Make a new SSH key pair. Use it only for this job. Do not reuse a key you
   use anywhere else.
2. Add the public key to the worker's `authorized_keys` for the account this
   job will use.
3. Run `ssh-keyscan <worker-host>` by hand, once, from a machine you trust.
   Look at the output and check it names the worker's real host key before
   you store it. Do not skip this check.
4. Store the private key and the `ssh-keyscan` output as `rtx-benchmark`
   environment secrets.

Never write the worker's real Tailscale IP address or its SSH username in a
doc, an issue, or a commit message. Use a placeholder such as `100.x.y.z` or
`<worker-host>` instead. The setup below keeps both out of the workflow's
visible inputs for the same reason.

### 3. Environment secrets and variables

Create a GitHub Environment named `rtx-benchmark`, and add these secrets and
variables to it. Put them on the environment, not on the repository.

**Limit the environment to `main`.** In **Settings → Environments →
rtx-benchmark**, set **Deployment branches and tags** to **Selected
branches** and add only `main`. Without this rule, a user with write access
can push a changed workflow to another branch and run it. That run gets the
SSH key.

**Secrets** (never shown on the run page, even in this public repository):

| Secret | Holds |
| --- | --- |
| `TS_OAUTH_CLIENT_ID` | The Tailscale OAuth client ID from step 1 |
| `TS_OAUTH_SECRET` | The Tailscale OAuth client secret (or use `TS_AUTHKEY` instead, with a Tailscale auth key) |
| `WORKER_SSH_KEY` | The private half of the dedicated SSH key from step 2 |
| `WORKER_KNOWN_HOSTS` | The checked `ssh-keyscan` output for the worker |
| `WORKER_HOST` | Optional, but best: the worker's tailnet name or address (see below) |
| `WORKER_SSH_USER` | Optional, but best: the SSH account the job connects as |

The logs of this public repository are public. GitHub hides a secret's
value in every log line. So store `WORKER_HOST` and `WORKER_SSH_USER` as
secrets. You can use variables of the same names instead. The first step
("Resolve settings") then masks them, but a variable is plain text before
that step, so it can show in that step's header.

**Variables** (defaults for a run; not secret, but still worth keeping out of
public docs where they'd be specific to your worker):

| Variable | Meaning | Default |
| --- | --- | --- |
| `WORKER_HOST` | The worker's tailnet name or address, if not a secret | none — set it here or as a secret |
| `WORKER_SSH_USER` | The SSH account the job connects as, if not a secret | none — set it here or as a secret |
| `BENCH_GPU_NAME` | The GPU name the dashboard shows for each run, for example `RTX 3070 Ti` | none — the run records no GPU name |
| `LIVE_CONTAINER` | The production container the job pauses and restores | `bonsai-llama-1` |
| `LIVE_HEALTH_URL` | Health-check URL for the live container, checked on the worker | `http://127.0.0.1:8080/health` |
| `LIVE_ENV_FILE` | Worker path to the live container's env file, if it needs one | none — no key is sent if unset |
| `TEST_IMAGE` | The pinned test image the benchmark runs | `qcm-rtx-validation:922be44` |
| `MODELS_DIR` | Worker path to the model files | none — must be set |
| `TEMPLATE_FILE` | Worker path to the chat template for the test server | none — must be set |

A `workflow_dispatch` run also takes `worker_host` and `ssh_user` inputs that
can override these values. Leave them blank when you start a run — the
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
   dispatch a run from. Since `rtx-benchmark` allows only `main`, allowing
   `main` here is enough.

## Starting a run

1. Go to the **Actions** tab, open the `benchmark.yml` workflow, and choose
   **Run workflow**.
2. Fill in the inputs you want to change. Leave the rest blank to use the
   defaults.

| Input | Meaning | Default |
| --- | --- | --- |
| `worker_host` | Overrides `WORKER_HOST` for this run only | (environment variable) |
| `ssh_user` | Overrides `WORKER_SSH_USER` for this run only | (environment variable) |
| `model` | The LLM preset to benchmark. The choice list is every section name in `models.ini.in` (except `[*]`), in file order: `bonsai-27b-ternary`, `bonsai-27b-1bit`, `qwen3.5-9b-q4_k_m`, `granite-4.1-8b-q4_k_m`, `qwen3.5-4b-q4_k_m`, `qwen3.6-35b-a3b`, `gemma4-e4b`, `bonsai-27b-ternary-text` | `qwen3.5-9b-q4_k_m` |
| `decision_model` | The decision model for the `decide` suite's GLiNER runner. Choices: `fastino/GLiNER2.5-Decide`, `fastino/GLiNER2.5-multi-Decide`, `fastino/GLiNER2.5-Decide-1B`, `fastino/gliner2.5-base-v1`, `fastino/gliner2.5-small-v1`, or `none` to skip GLiNER and run only the two Qwen runners | `fastino/GLiNER2.5-Decide` |
| `ctx` | Context size, in tokens. Leave it empty to keep the chosen preset's own default — the job reads the real size back from the test server after start-up | (preset default) |
| `suites` | Space-separated list of suites to run: `fixture`, `long_context`, `live_web`, `decide`, `concurrency` | `fixture long_context live_web` |
| `repetitions` | How many times to repeat the `fixture` tasks and the `live_web` run. `long_context`, `decide`, and `concurrency` ignore it. | `1` |
| `restore_production` | Restart the live container when the run ends | `true` |

The test image (`TEST_IMAGE`) must already contain the preset you pick for
`model`. If its `models.ini.in` does not have that preset, the test server
still starts, but loading the preset fails — see "What happens during a run"
below.

`concurrency` only works with `model` set to `qwen3.5-9b-q4_k_m` or
`qwen3.5-4b-q4_k_m`. Every other preset fixes `parallel = 1`, so picking
`concurrency` with another model fails the run at once, before it touches
the worker.

3. Anyone with write access to the repository can start a run, but only the
   repository owner should — see "Why a manual trigger only" above.

### The `repetitions` limit

The job has 240 minutes. It must always keep time to put the live container
back. So the first step works out the longest time each suite step can
take, from the suites you chose and `repetitions`. Each suite step gets
that time as its own limit. If the total does not fit, the run stops at
once, with an error that gives the highest `repetitions` that fits.

- With the default suites, the most is 2, with or without `concurrency`.
  With `fixture` alone, the most is 3.
- Without `fixture`, the limit is much higher (26 for `live_web` alone).

`tests/bench/run_suites.py` holds the numbers (`step_minutes`,
`max_repetitions`), so the workflow and the suites always agree.
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

1. The job makes a new random bearer key for the test server. It uses this
   key only for the run, and masks it in the logs right away.
2. If `decide` is one of the suites, and `decision_model` is not `none`, the
   job installs GLiNER on the runner itself (`gliner2[local]`, with
   CPU-only PyTorch). It also restores a cached Hugging Face download of
   the chosen decision model. GLiNER then runs on this runner's own CPU,
   not on the worker's GPU. The dashboard labels its results
   "cpu (CI runner)".
3. The job joins the tailnet, then connects to the worker over SSH.
4. It stops the live container (`bonsai-llama-1` by default), and starts
   the test server in its place. The test server listens on the worker's
   loopback address only.

   The test server is the image's own `start-server.sh`, in **router**
   mode. It serves every preset in the image's `models.ini.in`, so your
   chosen `model` runs with its normal production settings. The router
   also reads the live container's own settings file (the `LIVE_ENV_FILE`
   variable). So it starts with the same knobs as production, plus this
   run's own bearer key, host, port, and — when you set `ctx` — context
   size.

   If the image's `models.ini.in` lacks your chosen preset, the test server
   still starts, but loading the preset fails. The job logs a warning.
   Every suite that needs the preset then records an error, instead of
   crashing.

   When you leave `ctx` empty, the job reads the preset's real context size
   back from the running server (its `/props` endpoint). It uses that value
   in the reports, and as `long_context`'s target.
5. It runs the suites you chose:

   | Suite | Server needs | What it runs | Pass means |
   | --- | --- | --- | --- |
   | `fixture` | 1 slot | Core browser tasks, repeated | Each task's own check passes |
   | `long_context` | 1 slot | A prompt near the context limit, asking the model to quote one exact line | The line quoted is exactly right |
   | `live_web` | 1 slot | A real DuckDuckGo image search that must save one real image file | An image file is actually saved |
   | `decide` | 1 slot, plus the runner's own CPU for GLiNER | The same 60 fixed-choice items answered three ways: free-text Qwen, schema-constrained Qwen, and GLiNER | Each runner's own accuracy, not a single pass/fail — see [docs/benchmarks.md](benchmarks.md#decide-suite) |
   | `concurrency` | 2 slots | Four job-application runs: one at a time, then two at a time in pairs | Each run's own check passes |

   `concurrency` always runs on its own direct, two-slot server, because
   every preset in `models.ini.in` fixes `parallel = 1`. This is why it only
   works with `model` set to `qwen3.5-9b-q4_k_m` or `qwen3.5-4b-q4_k_m`; see
   the note above. The job restarts the test server with 2 slots and a
   49,152-token context before this suite runs. If that restart or the
   `concurrency` suite itself fails, the job goes on: it records the error
   under `concurrency`, and still publishes the other suites' results.
6. It saves raw results, with any saved images, as one workflow artifact,
   so you can look at them even without the dashboard.
7. At the end, whether or not a suite failed, it stops the test server and
   starts the live container again (unless you set `restore_production` to
   `false`). It checks the live container's health endpoint. If that check
   does not return HTTP 200, the job fails — that tells you production is
   not back up cleanly, instead of hiding the problem. This step retries on
   its own: starting the live container is tried up to 5 times, and if the
   whole step still fails, the workflow waits 20 seconds and runs it a
   second time. This covers a short network drop between the runner and the
   worker.
8. On success, a separate publish job downloads the run summary that this
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
  The three `decide` runners each get their own card.
- A pass-rate history chart, per suite, across all runs.
- A median-time history chart.
- A time-per-decision chart, in milliseconds, with one line per `decide`
  runner (free-text Qwen, schema-constrained Qwen, GLiNER). It only appears
  once a published run has a `decide` suite.
- A peak-VRAM-versus-context chart, with a line at 8,192 MiB (the worker's
  total VRAM). It has one point for each suite in each run. A point uses
  the context size of the server that suite ran on. So the `concurrency`
  points show 49,152 tokens, even when the other suites in that run used a
  different `ctx`.
- A table of every run, with its date, commit, model, decision model,
  context, and suites.

## Stopping production, on purpose

A run stops the real chat model for as long as the run takes. Anyone using
the live chat during that window sees it go down, then come back once the
run ends. Pick a quiet time to start a run, and tell anyone else who might
be using the live model first.
