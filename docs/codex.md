# Codex with your inference server

Keep GPT as your main Codex model and ask it to use local inference for a selected
task, or run a whole Codex session with a local model. Client setup installs both
options. Run it on the computer where you use Codex; installing the TrueNAS server
does not configure another computer automatically.

## Set up the client

You need Python 3.11 or later, an installed Codex CLI, a reachable inference API,
and that server's API key. Setup uses the installed CLI's catalog format. It
does not install or upgrade Codex.

On the inference machine, with the key already in the repository's `.env`:

```bash
./setup-codex.sh
```

On a separate client, keep the server key in a private text file containing only
the key, then run:

```bash
chmod 600 "$HOME/.config/qcm-api-key"
./setup-codex.sh --base-url http://SERVER_ADDRESS:8080/v1 --key-file "$HOME/.config/qcm-api-key" --name qcm
```

The installer provides the same client-only entrypoint:

```bash
./install.sh --codex-client --base-url http://SERVER_ADDRESS:8080/v1 --key-file "$HOME/.config/qcm-api-key" --name qcm
```

It runs on macOS or Linux and skips server installation. Normal `./install.sh`
still installs the Linux inference stack and prints client setup instructions.
Use an address reachable through the LAN, Tailscale, or an SSH tunnel. Port 8080
is the inference API; port 9090 is the chat UI.

Setup authenticates model discovery and makes a bounded Responses API request
before writing settings. It creates a profile for every advertised model and a
main `qcm` profile. Use `--model EXACT_MODEL_ID` to choose the main profile's model.
Context limits come from server presets or metadata. If the server does not
report its configured context, supply `--context-window TOKENS`. A model's
training limit does not prove the context available on your GPU.

Setup reads `.env` as data, without executing shell commands. `BONSAI_SERVER_URL`
selects a remote connect URL. `BONSAI_SERVER_KEY` takes precedence over
`BONSAI_API_KEY`. `BONSAI_ENV_FILE` selects another dotenv file. Bind settings such
as `BONSAI_HOST=0.0.0.0` do not select the client's address.

## Ask GPT to use local inference

Keep your usual GPT session and ask:

> Use my TrueNAS inference to review this function, then check its findings yourself.

For explicit skill selection:

> Use $quickclevermodel-inference to draft tests for this function, then review the result.

The skill calls the server for the requested task. GPT keeps control and checks
the returned text. It sends the selected prompt and context, rather than the
whole conversation. The helper does not execute model-proposed commands. Setup
does not enable automatic routing.

The global skill lives in `~/.agents/skills/quickclevermodel-inference`. Start a
new Codex session if it does not appear. With many installed skills, Codex can
omit descriptions from the initial context; explicitly select the skill when
needed. See the [official skills guide](https://learn.chatgpt.com/docs/build-skills).

## Run a fully local session

From the project you want to edit:

```bash
codex --profile qcm
```

Use `/model` to select another discovered model, or a per-model profile command
printed by setup. The catalog supplies each model's context limit when you switch.
Setup preserves your GPT default, unrelated providers, and other settings. See
the [official configuration guide](https://learn.chatgpt.com/docs/config-file/config-advanced).

## Saved files and refresh

Codex config uses `$CODEX_HOME`, or `~/.codex` when unset. For `--name qcm`, setup
saves the private key, `client.json`, and `models.json` under
`quickclevermodel/qcm/`, and creates `qcm.config.toml` plus per-model profiles.
It adds a provider to `config.toml`, with a backup before changes. The key file
has mode 600. Profiles and the catalog contain its path, not the key value.
Keep key files outside Git.

The installed skill and helper work after the source checkout is removed. Run
the same setup command again after changing the key, models, or context settings.
Setup updates its managed files and refuses unrelated collisions. A different
`--name` creates another connection. The most recent setup is the skill's default;
the helper's `--config PATH` selects another saved connection.

Add `--check` to validate the connection and prospective setup without writing
client files. Include the same URL and key-file options for a remote server:

```bash
./setup-codex.sh --check
```

Discovery alone does not prove Responses compatibility. See the
[official catalog guide](https://learn.chatgpt.com/docs/model-catalogs).

## Compatibility and limits

Codex requires the Responses API; this stack exposes `/v1/responses`.
Bonsai needs this repository's `bonsai-chat-template.jinja` for Codex's system
and developer messages. For `System message must be at the beginning` or a
template parser error, check the deployed template and rebuild the server image.
Other model families use their own templates.

Only one model is resident by default. Switching unloads the previous model and
can delay the first answer. Earlier repository tests found that Codex's
namespace-wrapped MCP tools were incompatible with llama.cpp. The on-request
helper sends text without tool definitions and does not depend on local MCP
support. A text response does not prove that Playwright, Context7, images, or all
coding tools work. See [runtime tests](runtime-testing.md) for separate checks.
