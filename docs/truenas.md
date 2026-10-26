# TrueNAS custom app

This opt-in configuration records the installed TrueNAS 25.10.7 / RTX 3070
deployment checked on 2026-10-25. It keeps the standard Linux and macOS presets
unchanged. It is part of the proposed v1.2.0 release; no v1.2.0 tag or published
image is implied. Read the [test limits](runtime-validation.md#truenas-rtx-3070-2026-10-25)
before choosing it for agent work.

## Preset and limits

| Stable API ID | Context | GPU K/V | Requested GPU layers |
| --- | ---: | --- | ---: |
| `qwen3.5-9b-q4_k_m` | 65,536, startup default | f16 | 99 |
| `qwen3.5-4b-q4_k_m` | 32,768 | f16 | 99 |
| `bonsai-27b-1bit` | 32,768 | q4_0 | 99 |
| `granite-4.1-8b-q4_k_m` | 32,768 | q4_0 | 32 |

The router keeps one model resident. Each has one slot, flash attention on,
ubatch 256, four context checkpoints, and a 256 MiB host prompt cache cap.
The inference container is limited to 8 GiB RAM with no extra swap; Open WebUI
is limited to 2 GiB with no extra swap. System RAM and GPU VRAM are separate
limits. The tested host had about 15.55 GiB system RAM and 8,192 MiB VRAM.

Qwen9 passed a 55,518-token whole-line recall probe at 64K. Qwen4 passed at
32K. Bonsai failed recall at 32K despite passing 20 API checks. Granite passed
19/20 API checks: its Messages two-tool case failed to use the second result.
No tested 128K configuration passed all trial criteria. These settings are
not a complete model/client certification or a memory guarantee under other
GPU workloads.

## Prepare TrueNAS and storage

Choose an Apps pool and enable the NVIDIA driver in the TrueNAS Apps settings.
The [TrueNAS 25.10 Apps reference](https://www.truenas.com/docs/scale/25.10/scaleuireference/apps/)
describes those controls. Use the TrueNAS driver controls rather than installing
host packages. Confirm the GPU is available with `nvidia-smi` in the NAS shell.

Create private datasets or directories for weights, configuration and WebUI
data before deployment. The YAML requires these existing absolute host paths:

| Variable | Host content | Container mount |
| --- | --- | --- |
| `QCM_MODELS_PATH` | The downloaded `models/` tree | `/app/models`, read only |
| `QCM_PRESET_PATH` | A copy of `docker/models.truenas.ini.in`, a file | `/app/models.ini.in`, read only |
| `QCM_WEBUI_DATA_PATH` | Persistent WebUI users, chats and model labels | `/app/backend/data`, writable |

Grant the containers access to those paths through the dataset permissions.
The model and preset mounts need read access; WebUI needs write access. A
missing preset file must not become a directory. The Compose binds use
`create_host_path: false` to fail on a missing source. See TrueNAS
[app storage guidance](https://apps.truenas.com/getting-started/app-storage/)
for host-path datasets and ACLs.

## Download and build

Use an x86_64 Linux Docker build host with Git, Bash and the existing `hf` CLI
available. The [dependency record](dependency-audit.md) records the downloader's
CLI requirements. Use the reviewed release-branch commit; this proposal does
not create a `v1.2.0` tag. Do not run `install.sh` or the systemd updater on
the TrueNAS appliance.

From the repository root, fetch the four model weights and Bonsai projector:

```sh
./fetch-models.sh --model bonsai-27b-1bit
./fetch-models.sh agents
```

The script verifies pinned byte counts and SHA256 values in `models.lock.tsv`,
including existing files. Transfer the resulting `models/` tree to
`QCM_MODELS_PATH` without flattening the family directories. Copy the tracked
`docker/models.truenas.ini.in` to `QCM_PRESET_PATH`. Model weights are mounted;
they are not part of the image. The Bonsai selector downloads both its Q1_0
weight and `Bonsai-27B-mmproj-Q8_0.gguf`, which the tested preset requires.

Build from the tracked Dockerfile, which defaults to Ampere architecture 86:

```sh
qcm_revision=$(git rev-parse HEAD)
qcm_image="quickclevermodel:$(git rev-parse --short=12 HEAD)-truenas"
docker build --platform linux/amd64 -f docker/Dockerfile \
  --build-arg QCM_REVISION="$qcm_revision" -t "$qcm_image" .
docker image inspect "$qcm_image" --format '{{.Id}}'
```

Record that image tag and ID. If the build host is separate, export it with
`docker save --output qcm-image.tar "$qcm_image"`, transfer the archive to
private NAS storage, then run `docker load --input /absolute/path/to/qcm-image.tar`
on the NAS. Alternatively, push it to your private registry and use that
immutable reference. For a registry image, run `docker pull "$QCM_IMAGE"` on
the NAS after setting `QCM_IMAGE` to that reference. The YAML uses
`pull_policy: never`, so the image must be cached in the NAS Docker engine
before installing the app.

This build procedure is newly documented. The historical tested image's tag
and revision do not prove it came from this Dockerfile. Test your new image
on the target before replacing a working installation.

## Render the private YAML

Keep the environment file and rendered YAML outside the checkout. Both contain
secrets. In the commands below, choose a private configuration directory:

```sh
qcm_private="$HOME/.config/quickclevermodel/truenas"
mkdir -p "$qcm_private"
chmod 700 "$qcm_private"
cp docker/truenas.env.example "$qcm_private/truenas.env"
chmod 600 "$qcm_private/truenas.env"
```

Edit that file. Fill every blank value: `QCM_IMAGE`, the three storage paths,
`BONSAI_API_KEY`, and `WEBUI_SECRET_KEY`. Use the image reference you built or
published. Generate each secret separately with `openssl rand -hex 32`; keep
the WebUI secret stable across restarts. The API key is shared only between
the inference service, WebUI and your authorised clients. WebUI login is
required.

Bindings default to loopback. For remote access, set `LLAMA_BIND2` and
`WEBUI_BIND` to a NAS LAN IPv4 address, and optionally `LLAMA_BIND3` and
`WEBUI_BIND2` to its Tailscale IPv4 address. Each address must exist before app
start; keep unused bindings at their distinct loopback defaults. API port is
8080 and UI port is 9090. Prefer an SSH tunnel when no remote listener is
needed. A direct HTTP listener sends credentials without TLS, so use a trusted
private network or an HTTPS proxy.

`WEBUI_URL` defaults to the first UI listener, and `CORS_ALLOW_ORIGIN` defaults
to both exact UI origins. Set explicit values for DNS or an HTTPS proxy. List
multiple origins with semicolons; do not use `*`. Restrict the selected
interfaces to the users who need them.

Render before pasting into TrueNAS. Docker's
[`config` command](https://docs.docker.com/reference/cli/docker/compose/config/)
resolves variables and validates the model. Its
[interpolation rules](https://docs.docker.com/compose/how-tos/environment-variables/variable-interpolation/)
give exported shell variables priority over `--env-file`, so clear any stale
exports with these names first.

```sh
umask 077
docker compose --env-file "$qcm_private/truenas.env" \
  -f docker/compose.truenas.yaml config --quiet
docker compose --env-file "$qcm_private/truenas.env" \
  -f docker/compose.truenas.yaml config > "$qcm_private/compose.rendered.yaml"
```

Inspect the private result for the intended image, absolute NAS paths,
listeners, GPU reservation, memory limits and required authentication. Do not
paste the unrendered template or assume the TrueNAS editor reads your local
environment file. Do not post the rendered YAML in issues or commits.

## Install and check

In **Apps > Discover**, open the menu and select **Install via YAML**. Enter a
lowercase alphanumeric app name, paste the rendered YAML into **Custom Config**,
then save. For an existing custom app, edit its YAML instead. See the official
[custom-app guide](https://apps.truenas.com/managing-apps/installing-custom-apps/).
The GPU reservation is in the YAML; this app does not need privileged mode.

Wait for both services to run. Check their logs for the expected Prism revision,
model context and successful model load. Open the UI on your selected address,
create or use the administrator account, and confirm that login is required.
Before sharing access, test that unauthenticated `/v1/models` returns 401 and
authenticated discovery returns exactly the four IDs above.

Send a short chat to each model, then run the existing
[runtime API probes](runtime-testing.md#api-contracts-and-tool-use) with the
listed context sizes and `--tools`. Keep reports private. A healthy container
or one short reply does not prove long-context recall or tool correctness.
Check GPU memory and OOM counters during your workload. Granite's known failure
must remain visible in the results.

## Add context labels in Open WebUI

Sign in as an administrator, open **Settings**, then **Models** in its
administrator section (direct path: `/?settings=admin%3Amodels`). Open each
base model and edit **Model Name**. Append `(64K)` to Qwen9 and `(32K)` to the
other three names. Select **Save & Update** while keeping the disabled
**Model ID** unchanged. These are
display labels; they do not change the server's context setting. The installed
v0.11.3 app retained all four IDs and passed four simple chat smokes after this
label change on 2026-10-26. The packaged YAML sets the default model, but does
not create these saved display labels automatically.

The pinned [v0.11.3 admin model settings](https://github.com/open-webui/open-webui/blob/v0.11.3/src/lib/components/admin/Settings/Models.svelte)
list provider models and save their display overrides. Workspace model presets
are a separate feature; creating a new preset can introduce a new ID.

Refresh the model selector and check both the names and authenticated API
discovery. Keep using the stable IDs in scripts and client settings. WebUI
stores labels in its data directory, so include that directory in backups.

## Backup, update and rollback

Before changing an existing app, save its current YAML, environment values,
image tag and ID, preset, and a restorable backup or snapshot of WebUI data.
Stop the app while taking a consistent WebUI data backup. Keep those files
private. Preserve model weights and keep the old image available; a mutable
tag alone is not a rollback copy.

Install updates through the TrueNAS custom-app editor after rendering the new
YAML. Use a maintenance window for GPU tests. Do not let two app instances
write the same WebUI data directory, and do not assume that two resident models
will fit. The Linux `update.sh` rollback process does not manage this custom app.

To roll back, stop the app, restore the saved YAML and preset, restore WebUI
data from the matching backup if it changed, then start the old image and test
all four IDs again. TrueNAS app rollback does not restore bind-mounted host
paths; restore those datasets separately as described in the
[25.10 rollback reference](https://www.truenas.com/docs/scale/25.10/scaleuireference/apps/).

For a migration from the old ternary runtime, keep both the old image and
`Ternary-Bonsai-27B-Q2_0.gguf`. Current Prism requires the separate official
`PQ2_0` file; do not overwrite or relabel the old artifact. The four-model
TrueNAS preset uses 1-bit Bonsai, but this warning still applies to an existing
installation that uses ternary aliases. See the
[ternary migration record](runtime-versions.md#ternary-format-migration).
