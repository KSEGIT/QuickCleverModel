#!/usr/bin/env python3
"""Run installed agent CLIs against an explicit test server in temporary fixtures."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import pathlib
import platform
import secrets
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import urllib.request


def child_environment(root):
    # Do not pass cloud credentials, personal provider routing, or injected hooks.
    keep = ("PATH", "HOME", "SHELL", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT")
    env = {key: value for key, value in os.environ.items() if key in keep}
    env.update({"XDG_CONFIG_HOME": str(root / "config"),
                "XDG_DATA_HOME": str(root / "data"),
                "XDG_CACHE_HOME": str(root / "cache"),
                "XDG_STATE_HOME": str(root / "state"),
                "CLAUDE_CONFIG_DIR": str(root / "claude"),
                "npm_config_cache": str(root / "npm-cache"),
                "TMPDIR": str(root / "tmp"),
                "DISABLE_AUTOUPDATER": "1", "DISABLE_TELEMETRY": "1",
                "DISABLE_ERROR_REPORTING": "1"})
    return env


def confined_command(argv, root):
    """Restrict client/tool file access; workspace-write alone permits broad reads."""
    root = root.resolve()
    if platform.system() == "Darwin" and shutil.which("sandbox-exec"):
        readable = ("/System", "/usr", "/bin", "/sbin", "/opt/homebrew", "/Library",
                    "/Applications", "/private/etc", "/dev", "/private/var/db")
        profile = "(version 1)(allow default)(deny file-read* file-write*)"
        profile += '(allow file-read-metadata)(allow file-read* (literal "/"))'
        profile += '(allow file-read* ' + " ".join(
            "(subpath " + json.dumps(path) + ")" for path in readable) + ")"
        profile += "(allow file-read* file-write* (subpath " + json.dumps(str(root)) + "))"
        profile += '(allow file-write* (literal "/dev/null") (literal "/dev/tty"))'
        return [shutil.which("sandbox-exec"), "-p", profile, *argv]
    raise RuntimeError("OS filesystem confinement unavailable; use macOS sandbox-exec for this harness")


def command(client, args, root, fixture, env, prompt, browser=False):
    base = args.base_url.rstrip("/").removesuffix("/v1")
    key = os.environ.get(args.key_env, "local-test")
    if client == "codex":
        env["QCM_TEST_API_KEY"] = key
        values = {
            "model_provider": '"qcm_test"',
            "model_providers.qcm_test.name": '"QCM isolated smoke"',
            "model_providers.qcm_test.base_url": json.dumps(base + "/v1"),
            "model_providers.qcm_test.env_key": '"QCM_TEST_API_KEY"',
            "model_providers.qcm_test.wire_api": '"responses"',
            "model_providers.qcm_test.request_max_retries": "0",
            "model_providers.qcm_test.stream_max_retries": "0",
            "model_context_window": str(args.context),
            "approval_policy": '"never"', "web_search": '"disabled"',
        }
        result = [client, "exec", "--ignore-user-config", "--ignore-rules", "--ephemeral",
                  "--sandbox", "workspace-write", "--json", "--model", args.model,
                  "--cd", str(fixture)]
        for key, value in values.items():
            result += ["-c", key + "=" + value]
        return result + [prompt]
    if client == "claude":
        env.update({"ANTHROPIC_BASE_URL": base, "ANTHROPIC_API_KEY": key,
                    "ANTHROPIC_MODEL": args.model,
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": args.model,
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": args.model,
                    "ANTHROPIC_DEFAULT_OPUS_MODEL": args.model,
                    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(args.context)})
        settings = root / "claude-settings.json"
        settings.write_text(json.dumps({"permissions": {
            "allow": ["Read(./**)", "Write(./**)", "Edit(./**)"] +
                     (["mcp__playwright__*"] if browser else []),
            "deny": ["Bash", "WebFetch", "WebSearch"]}}))
        mcp = {"mcpServers": {}}
        if browser:
            mcp["mcpServers"]["playwright"] = {
                "command": "npx", "args": ["-y", "@playwright/mcp@0.0.82", "--isolated", "--headless",
                    "--snapshot-mode", "none", "--image-responses", "omit", "--output-dir", str(root / "browser")]}
        return [client, "--bare", "--setting-sources", "", "--settings", str(settings),
                "--strict-mcp-config", "--mcp-config", json.dumps(mcp),
                "--no-session-persistence", "--permission-mode", "dontAsk",
                "--tools", "Read,Write,Edit", "--model", args.model,
                "--output-format", "stream-json", "--verbose", "--include-partial-messages",
                "--append-system-prompt", "Work only in the supplied fixture directory.",
                "-p", prompt]
    if client == "opencode":
        config = root / "opencode.json"
        config.write_text(json.dumps({
            "$schema": "https://opencode.ai/config.json",
            "provider": {"qcm_test": {"npm": "@ai-sdk/openai-compatible",
                "options": {"baseURL": base + "/v1", "apiKey": "{env:QCM_TEST_API_KEY}"},
                "models": {args.model: {"name": args.model,
                    "limit": {"context": args.context, "output": 1024}}}}},
            "model": "qcm_test/" + args.model,
            "permission": {"*": "deny", "read": "allow", "edit": "allow",
                           "external_directory": "deny"}}))
        env.update({"OPENCODE_CONFIG": str(config), "QCM_TEST_API_KEY": key,
                    "OPENCODE_DISABLE_AUTOUPDATE": "true",
                    "OPENCODE_DISABLE_DEFAULT_PLUGINS": "true"})
        return [client, "run", "--pure", "--format", "json", "--model", "qcm_test/" + args.model, prompt]
    raise ValueError(client)


def run_process(argv, env, cwd, timeout):
    """Kill the whole owned process group on timeout, including any MCP browser."""
    with subprocess.Popen(argv, env=env, cwd=cwd, stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                          start_new_session=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            error.stdout, error.stderr = stdout, stderr
            raise
        finally:
            # The parent CLI can exit while leaving an MCP subprocess alive.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def run_client(client, args, browser=False):
    executable = shutil.which(client)
    if not executable:
        return {"client": "claude-playwright" if browser else client,
                "status": "SKIPPED", "reason": "CLI not installed"}
    with tempfile.TemporaryDirectory(prefix="qcm-client-smoke-") as temporary:
        root = pathlib.Path(temporary)
        (root / "tmp").mkdir()
        fixture = root / "fixture"
        fixture.mkdir()
        token = secrets.token_hex(12)
        (fixture / "input.txt").write_text(token + "\n")
        (fixture / "AGENTS.md").write_text("Read input.txt before writing output.txt. Work only in this fixture.\n")
        env = child_environment(root)
        subprocess.run(["git", "init", "--quiet", str(fixture)], check=True, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        prompt = ("Read input.txt using a file tool. Create output.txt containing exactly the same "
                  "token followed by one newline. Do not change input.txt. Then say which file you wrote. "
                  "Work only in this temporary directory; do not inspect other directories.")
        http_server = None
        expected_output = token + "\n"
        if browser:
            if not shutil.which("npx"):
                return {"client": "claude-playwright", "status": "SKIPPED", "reason": "npx is not installed"}
            # Serve a copied fixture only, never the repository or a user directory.
            webroot = root / "web"
            webroot.mkdir()
            shutil.copyfile(pathlib.Path(__file__).with_name("fixtures") / "browser.html", webroot / "browser.html")
            http_server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(webroot)))
            threading.Thread(target=http_server.serve_forever, daemon=True).start()
            prompt = (f"Use Playwright MCP to navigate to http://127.0.0.1:{http_server.server_port}/browser.html. "
                      f"Inspect the form with an explicit snapshot. Fill Name with {token}, select Repair, click Book, "
                      "then read the result from the page with a snapshot. Write exactly that result and one newline "
                      "to output.txt. Use the browser tools to complete the form; do not calculate the result yourself.")
            expected_output = f"{token}: Repair booked\n"
        argv = command(client, args, root, fixture, env, prompt, browser=browser)
        started = time.monotonic()
        result = {"client": "claude-playwright" if browser else client, "model": args.model, "status": "FAIL"}
        try:
            try:
                argv = confined_command(argv, root)
            except RuntimeError as error:
                result.update(status="SKIPPED", reason=str(error))
                return result
            result["filesystem_confinement"] = "macOS sandbox-exec: fixture writes; system runtime reads"
            version = subprocess.run([executable, "--version"], env=env, cwd=fixture,
                                     capture_output=True, text=True, timeout=20)
            result["version"] = version.stdout.strip()
            process = run_process(argv, env, fixture, args.timeout)
            result.update(returncode=process.returncode, stdout=process.stdout, stderr=process.stderr)
            target = fixture / "output.txt"
            result["fixture_matches"] = (not target.is_symlink() and target.is_file()
                                         and target.read_text() == expected_output)
            result["input_preserved"] = (fixture / "input.txt").read_text() == token + "\n"
            events = []
            for line in process.stdout.splitlines():
                try:
                    event = json.loads(line)
                    if isinstance(event, dict):
                        events.append(event)
                except json.JSONDecodeError:
                    pass
            # The random token is absent from the prompt: a correct file proves a read/write round trip.
            result["json_events"] = len(events)
            if browser:
                names = {block.get("name") for event in events if isinstance(event, dict)
                         for block in event.get("message", {}).get("content", [])
                         if isinstance(block, dict) and block.get("type") == "tool_use"}
                required = {"mcp__playwright__browser_navigate", "mcp__playwright__browser_snapshot",
                            "mcp__playwright__browser_select_option", "mcp__playwright__browser_click"}
                result["browser_tool_names"] = sorted(name for name in names if name)
                result["browser_tools_observed"] = (required <= names and bool(names & {
                    "mcp__playwright__browser_fill_form", "mcp__playwright__browser_type"}))
            result["status"] = "PASS" if (process.returncode == 0 and result["fixture_matches"]
                                           and result["input_preserved"] and events
                                           and result.get("browser_tools_observed", True)) else "FAIL"
            if result["status"] == "FAIL":
                result["reason"] = "CLI error, missing structured events, or fixture read/edit failed; inspect transcript"
        except subprocess.TimeoutExpired as error:
            result.update(reason=f"timeout after {args.timeout}s",
                          stdout=(error.stdout or b"").decode(errors="replace") if isinstance(error.stdout, bytes)
                          else error.stdout,
                          stderr=(error.stderr or b"").decode(errors="replace") if isinstance(error.stderr, bytes)
                          else error.stderr)
        except (OSError, subprocess.SubprocessError) as error:
            result["reason"] = str(error)
        finally:
            if http_server:
                http_server.shutdown()
                http_server.server_close()
        result["seconds"] = time.monotonic() - started
        # TemporaryDirectory removes only the harness-owned fixture/config files.
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--client", choices=("codex", "claude", "opencode", "all"), default="all")
    parser.add_argument("--key-env", default="BONSAI_API_KEY")
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", required=True)
    parser.add_argument("--claude-playwright", action="store_true", help="Also run Claude against the local browser fixture")
    args = parser.parse_args()
    clients = ("codex", "claude", "opencode") if args.client == "all" else (args.client,)
    report = {"base_url": args.base_url, "model": args.model, "results": []}
    try:
        base = args.base_url.rstrip("/").removesuffix("/v1")
        headers = {"Authorization": "Bearer " + os.environ.get(args.key_env, "local-test")}
        with urllib.request.urlopen(urllib.request.Request(base + "/health", headers=headers), timeout=5) as response:
            if json.load(response).get("status") != "ok":
                raise ValueError("server is not healthy")
    except Exception as error:
        report["results"] = [{"client": client, "status": "SKIPPED", "reason": f"test server unavailable: {error}"}
                             for client in (*clients, *(("claude-playwright",) if args.claude_playwright else ()))]
    else:
        for client in clients:
            result = run_client(client, args)
            report["results"].append(result)
            pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
            print(f"{result['status']} {client}: {result.get('reason', 'fixture read and edit verified')}", flush=True)
        if args.claude_playwright:
            result = run_client("claude", args, browser=True)
            report["results"].append(result)
            print(f"{result['status']} claude-playwright: {result.get('reason', 'browser fixture verified')}", flush=True)
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    return 0 if all(result["status"] == "PASS" for result in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
