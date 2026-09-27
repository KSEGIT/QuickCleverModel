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
