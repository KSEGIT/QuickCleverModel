#!/usr/bin/env python3
"""Opt-in real Playwright MCP smoke for the new local agent fixtures; no model."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
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

from playwright_agent_bench import FIXTURE, TASKS, browser_state
from smoke_playwright import MCP


def main():
    root = FIXTURE.parents[2]
    with tempfile.TemporaryDirectory(prefix="qcm-agent-fixture-") as temporary:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        fixture = ThreadingHTTPServer(("127.0.0.1", 0),
            partial(SimpleHTTPRequestHandler, directory=str(FIXTURE.parent)))
        threading.Thread(target=fixture.serve_forever, daemon=True).start()
        env = dict(os.environ, PW_MCP_PORT=str(port), PW_MCP_SNAPSHOT="none")
        with open(Path(temporary) / "mcp.log", "w+") as log:
            process = subprocess.Popen(["bash", str(root / "start-playwright-mcp.sh"),
                "--headless", "--port", str(port), "--browser", "chrome", "--isolated",
                "--snapshot-mode", "none", "--image-responses", "omit", "--output-dir", temporary],
                cwd=temporary, env=env, stdout=log, stderr=log, start_new_session=True)
            try:
                client = MCP(port)
                until = time.monotonic() + 60
                while True:
                    try:
                        client.request("initialize", {"protocolVersion": "2025-03-26",
                            "capabilities": {}, "clientInfo": {"name": "qcm-agent-fixture", "version": "1"}})
                        break
                    except urllib.error.URLError:
                        if process.poll() is not None or time.monotonic() > until:
                            log.seek(0)
                            raise AssertionError("MCP launch failed: " + log.read())
                        time.sleep(.25)
                client.request("notifications/initialized", notification=True)
                names = {tool["name"] for tool in client.request("tools/list")["tools"]}
                assert {"browser_evaluate", "browser_snapshot", "browser_file_upload"} <= names
                for task in TASKS:
                    url = f"http://127.0.0.1:{fixture.server_port}/{FIXTURE.name}?task={task}"
                    client.tool("browser_navigate", {"url": url})
                    assert browser_state(client)["task"] == task, task
                client.tool("browser_navigate", {"url": "about:blank"})
                url = f"http://127.0.0.1:{fixture.server_port}/{FIXTURE.name}?task=basic_form"
                client.tool("browser_navigate", {"url": url})
                snapshot = client.tool("browser_snapshot", {}, snapshot=True)
                def ref(label):
                    match = re.search(re.escape(label) + r'[^\n]*\[ref=([^\]]+)\]', snapshot)
                    assert match, (label, snapshot)
                    return match.group(1)
                client.tool("browser_fill_form", {"fields": [
                    {"name": "Name", "type": "textbox", "target": ref('textbox "Name"'), "value": "Ada"},
                    {"name": "Surname", "type": "textbox", "target": ref('textbox "Surname"'), "value": "Lovelace"}]})
                client.tool("browser_select_option", {"element": "Country",
                    "target": ref('combobox "Country"'), "values": ["UK"]})
                client.tool("browser_click", {"element": "Continue", "target": ref('button "Continue"')})
                assert browser_state(client)["complete"] is True
                large = f"http://127.0.0.1:{fixture.server_port}/{FIXTURE.name}?task=large_page"
                client.tool("browser_navigate", {"url": large})
                snapshot = client.tool("browser_snapshot", {}, snapshot=True)
                assert "Position 120" in snapshot and "Browser Engineer" in snapshot
                print("PASS: nine agent fixture modes, basic form, large-page snapshot, pinned MCP transport")
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
