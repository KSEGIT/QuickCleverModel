#!/usr/bin/env python3
"""Opt-in integration test: python3 tests/smoke_playwright.py.

Requires Node/npm, installed Chrome, and network on the first run to fetch the
pinned MCP package. Runs the production launcher against a local fixture.
No model or inference server is involved.
"""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


class MCP:
    def __init__(self, port):
        self.url = f"http://localhost:{port}/mcp"
        self.session = None
        self.sequence = 0

    def request(self, method, params=None, notification=False):
        self.sequence += 1
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if not notification:
            payload["id"] = self.sequence
        headers = {"Content-Type": "application/json",
                   "Accept": "application/json, text/event-stream"}
        if self.session:
            headers["Mcp-Session-Id"] = self.session
            headers["MCP-Protocol-Version"] = "2025-03-26"
        request = urllib.request.Request(self.url, json.dumps(payload).encode(), headers)
        with urllib.request.urlopen(request, timeout=90) as response:
            self.session = response.headers.get("Mcp-Session-Id", self.session)
            if notification:
                return None
            if "text/event-stream" in response.headers.get("Content-Type", ""):
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    body = json.loads(line[6:])
                    if body.get("id") == self.sequence:
                        break
                else:
                    raise AssertionError("MCP stream ended without a response")
            else:
                body = json.load(response)
        if "error" in body:
            raise AssertionError(body["error"])
        return body["result"]

    def tool(self, name, arguments, expect_error=False, snapshot=False):
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        assert bool(result.get("isError")) == expect_error, result
        assert all(item["type"] != "image" for item in result.get("content", [])), result
        text = "\n".join(item.get("text", "") for item in result.get("content", []))
        if not snapshot:
            assert "### Snapshot" not in text, f"Unexpected automatic snapshot: {text}"
        return text


def main():
    with tempfile.TemporaryDirectory(prefix="qcm-browser-") as temporary:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        handler = partial(SimpleHTTPRequestHandler, directory=str(ROOT / "tests/fixtures"))
        fixture = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=fixture.serve_forever, daemon=True)
        thread.start()
        env = dict(os.environ, PW_MCP_PORT=str(port), PW_MCP_BROWSER="chrome",
                   PW_MCP_OUTPUT_DIR=temporary, PW_MCP_SNAPSHOT="none")
        # Explicit trailing flags override any developer .env and isolate output.
        with open(Path(temporary) / "server.log", "w+") as log:
            process = subprocess.Popen(
                ["bash", str(ROOT / "start-playwright-mcp.sh"), "--headless",
                 "--port", str(port), "--browser", "chrome", "--isolated", "--snapshot-mode", "none",
                 "--image-responses", "omit", "--output-dir", temporary],
                cwd=temporary, env=env, stdout=log, stderr=log, start_new_session=True)
            try:
                client = MCP(port)
                deadline = time.monotonic() + 60
                while True:
                    try:
                        init = client.request("initialize", {
                            "protocolVersion": "2025-03-26", "capabilities": {},
                            "clientInfo": {"name": "qcm-browser-smoke", "version": "1"}})
                        break
                    except (urllib.error.URLError, ConnectionError):
                        if process.poll() is not None or time.monotonic() >= deadline:
                            log.seek(0)
                            raise AssertionError(f"MCP failed to start: {log.read()}")
                        time.sleep(0.25)
                # MCP reports its pinned Playwright engine version in initialize.
                assert init["serverInfo"]["version"] == "1.64.0-alpha-1789764292000", init
                client.request("notifications/initialized", notification=True)
                tools = client.request("tools/list")["tools"]
                assert "browser_snapshot" in {tool["name"] for tool in tools}
                url = f"http://127.0.0.1:{fixture.server_port}/browser.html"
                client.tool("browser_navigate", {"url": url})
                snapshot = client.tool("browser_snapshot", {}, snapshot=True)

                def ref(label):
                    match = re.search(rf'{re.escape(label)}[^\n]*\[ref=([^\]]+)\]', snapshot)
                    assert match, snapshot
                    return match.group(1)

                name_ref = ref('textbox "Name"')
                service_ref = ref('combobox "Service"')
                button_ref = ref('button "Book"')
                client.tool("browser_fill_form", {"fields": [
                    {"name": "Name", "type": "textbox", "target": name_ref, "value": "Daniel"}]})
                client.tool("browser_select_option", {
                    "element": "Service", "target": service_ref, "values": ["Repair"]})
                client.tool("browser_click", {"element": "Book", "target": button_ref})
                result = client.tool("browser_snapshot", {}, snapshot=True)
                assert "Daniel: Repair booked" in result, result
                # Exercise a tool that would return an image without the flag.
                client.tool("browser_take_screenshot", {"scale": "css"})
                client.tool("browser_click", {"element": "missing", "target": "e999999"},
                            expect_error=True)
                client.tool("browser_navigate", {"url": url})
                recovered = client.tool("browser_snapshot", {}, snapshot=True)
                assert 'textbox "Name"' in recovered, recovered
                client.tool("browser_close", {})
                print("PASS: MCP 0.0.82 HTTP navigation, discovery, fill, select, click, "
                      "extraction, error recovery; explicit snapshots; no inline images")
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                fixture.shutdown()
                fixture.server_close()


if __name__ == "__main__":
    main()
