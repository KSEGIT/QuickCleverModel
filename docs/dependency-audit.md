# Dependency audit

Versions were checked against npm, PyPI and official GitHub releases on
2026-09-21. Version Sentinel was not available. Exact package pins replace
moving versions; this does not lock every transitive npm or Python dependency.

## Changed components

| Component | Previous | New | Reason |
| --- | --- | --- | --- |
| Playwright MCP | `@latest` (launcher comments described 0.0.79) | `0.0.82` | Reproducible direct package version; verified current tools and small responses. |
| OpenCode installer | Unversioned Homebrew/npm or remote install script | `opencode-ai@1.18.31` through npm | Same explicit release on supported platforms; keeps the existing provider format. |
| Hugging Face CLI installer | Unversioned `huggingface_hub[cli]` | `huggingface_hub==1.32.0` | Pin the CLI; remove the obsolete `cli` extra. Requires Python 3.10 or newer. |
| Open WebUI Docker | `ghcr.io/open-webui/open-webui:main` | `ghcr.io/open-webui/open-webui:v0.11.3` | Stop silently pulling development builds; match both Compose files and updater image checks. |
| Open WebUI macOS install instructions | Unversioned `open-webui` | `open-webui==0.11.3` | Match the container release. |

The installers keep already installed `hf` and `opencode` executables. Their
effective versions can differ from fresh-install pins. Existing Open WebUI
data must be backed up before deployment: its release notes describe database
migrations, and image rollback alone does not reverse a database migration.
The WebUI image uses a release tag, not a digest; the existing updater restores
image tags during rollback.

## Kept components

- `jinja2==3.1.6` remains the current PyPI release. The test suite needs no
  further Python packages.
- GitHub Actions already use `actions/checkout@v7` and
  `actions/setup-python@v7`. The current releases were v7.0.1 and v7.0.0.
  Keep the repository's major-version pin policy. There were no Docker,
  setup-node or cache actions to update.
- `@ai-sdk/openai-compatible` is a provider identifier in the generated
  OpenCode config. OpenCode 1.18.31 bundles version **2.0.41** and selects that
  built-in provider by this identifier. Do not append the npm registry's
  current **3.0.53** to it: that bypasses the bundled-provider lookup and moves
  to a different major version without a compatibility test.
- Claude integration uses the installed Claude CLI directly. There is no
  separate npm proxy/adapter package or Python proxy dependency to update.

## Browser compatibility and tests

The launcher keeps `--isolated`, `--snapshot-mode none`, and
`--image-responses omit`. Clients must call `browser_snapshot` or
`browser_find` when they need element information. Current tool schemas use
`target` for an element reference or selector; old hard-coded `ref` arguments
are rejected. Clients must refresh tool discovery after upgrading.

Playwright MCP 0.0.82 depends on the exact Playwright engine build
`1.64.0-alpha-1789764292000`; this is the engine version that its MCP initialize
response reports. The package requires Node 18 or newer. The smoke run used
Node 26.7.0, npm 11.19.0, and installed Google Chrome on macOS.

| Check | Result |
| --- | --- |
| `python3 -m unittest discover -s tests -p test_playwright_launcher.py -v` | PASS — 2 tests; exact pin, snapshot/image flags, `.env` override and eviction opt-out. The new pin test failed before the launcher change. |
| `python3 tests/smoke_playwright.py` | PASS — actual pinned MCP over HTTP: navigation, explicit discovery, form fill, dropdown, click, result extraction, invalid-reference error and recovery. Screenshot tool returned no inline image. Ordinary operations returned no automatic snapshot. |
| `python3 -m unittest discover -s tests -p test_opencode_config.py -v` | PASS — 18 tests at dependency-change validation. Closed two fixture file handles that produced ResourceWarnings. |
| `npx -y opencode-ai@1.18.31 --version` | PASS — prints `1.18.31`; this is not a model/client conversation test. |
| `python3 -m unittest discover -s tests -p test_update.py -q` | PASS — 217 tests, 2 macOS skips (`/dev/full` and `flock`). Run with localhost socket access for the HTTP fixture. |
| Temporary venv: install `huggingface_hub==1.32.0`, then `hf version` and `hf download --help` | PASS — reports 1.32.0; download accepts `--revision` and `--local-dir`. No model weights downloaded by this dependency check. |
| `bash -n start-playwright-mcp.sh setup-opencode.sh install.sh update.sh` | PASS. |
| WebUI container startup/database migration | SKIPPED — no deployment or container migration was performed. |

The browser smoke is opt-in because it starts Chrome and can download npm
packages on its first run. It binds temporary local ports, serves
`tests/fixtures/browser.html`, uses a temporary browser profile/output path,
and stops its server on exit. It does not use model inference. Claude plus
Playwright and other real agent-client results are reported separately in the
runtime validation record.

## Sources

- [Playwright MCP package metadata](https://registry.npmjs.org/@playwright%2fmcp/0.0.82),
  [release notes](https://github.com/microsoft/playwright-mcp/releases/tag/v0.0.82),
  [versioned options and tool schemas](https://github.com/microsoft/playwright-mcp/blob/v0.0.82/README.md).
- [OpenCode package metadata](https://registry.npmjs.org/opencode-ai/1.18.31),
  [release notes](https://github.com/anomalyco/opencode/releases/tag/v1.18.31),
  [bundled dependency pin](https://github.com/anomalyco/opencode/blob/v1.18.31/packages/opencode/package.json),
  [bundled-provider lookup](https://github.com/anomalyco/opencode/blob/v1.18.31/packages/opencode/src/provider/provider.ts).
- [Hugging Face package metadata](https://pypi.org/pypi/huggingface_hub/1.32.0/json),
  [v1 migration guide](https://huggingface.co/docs/huggingface_hub/en/concepts/migration),
  [CLI documentation](https://huggingface.co/docs/huggingface_hub/en/guides/cli).
- [Open WebUI release notes](https://github.com/open-webui/open-webui/releases/tag/v0.11.3),
  [Python package metadata](https://pypi.org/pypi/open-webui/0.11.3/json).
- [Jinja2 package metadata](https://pypi.org/pypi/jinja2/3.1.6/json),
  [checkout releases](https://github.com/actions/checkout/releases),
  [setup-python releases](https://github.com/actions/setup-python/releases).

## Open WebUI v0.13.4 update (2026-11-23)

| Component | Previous | New |
| --- | --- | --- |
| Open WebUI Docker (all three Compose files, `update.sh`) | `v0.11.3` | `v0.13.4` |
| Open WebUI macOS install instructions | `open-webui==0.11.3` | `open-webui==0.13.4` |

v0.13.4 was published on 2026-11-17, six days before this pin. The tag stays
a release tag, not a digest, for the same updater reason as above.

Notes from the release notes between v0.11.3 and v0.13.4 that touch this repo:

- v0.12.1, v0.13.0 and v0.13.1 change the database schema. Back up the
  `open-webui-data` volume first. An image rollback to v0.11.3 does not undo
  these migrations.
- v0.13.4 removes direct connections and the `ENABLE_DIRECT_CONNECTIONS`
  setting, so the TrueNAS Compose file no longer sets it.
- v0.13.1 changes retrieval defaults (`RAG_TOP_K` 3 to 10, `CHUNK_SIZE` 1000
  to 1024) and stops preloading attached files with native tool calls.
  Prompts with knowledge get longer, which matters at `BONSAI_CTX=8192`.
- v0.12.4 moves the Docker image to Python 3.12. The macOS venv stays on
  Python 3.11; PyPI lists `>=3.11,<3.13` for 0.13.4.
- No rename of `WEBUI_AUTH`, `ENABLE_SIGNUP`, `OPENAI_API_BASE_URL(S)`,
  `OPENAI_API_KEY(S)`, `ENABLE_PERSISTENT_CONFIG` or `WEBUI_SECRET_KEY`.

The Docker Compose files keep `WEBUI_AUTH=false` as the default (login is
opt-in, on a fresh data volume) and now default to `ENABLE_SIGNUP=false`. In
v0.13.4 the first account can sign up even with `ENABLE_SIGNUP=false`, and
becomes the admin. A new `WEBUI_MODEL` value in
`.env` fixes the chat model through `OPENAI_API_CONFIGS` and `DEFAULT_MODELS`;
see `.env.example`.

Sources: [release notes](https://github.com/open-webui/open-webui/releases/tag/v0.13.4)
and the releases before it,
[sign-up code](https://github.com/open-webui/open-webui/blob/v0.13.4/backend/open_webui/routers/auths.py),
[Python package metadata](https://pypi.org/pypi/open-webui/0.13.4/json),
[environment variables](https://docs.openwebui.com/reference/env-configuration).
