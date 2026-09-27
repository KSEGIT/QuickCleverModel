#!/usr/bin/env python3
"""Live-web agent test: DuckDuckGo image search for oranges, save one image.

Run from the repository root: python3 tests/live_image_agent.py ...
Reuses the benchmark's chat client and MCP launcher settings, but talks to the
real web. Navigation is limited to duckduckgo.com; the image itself is
fetched by a harness tool (save_image) that checks it really is an image
before saving it.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import re
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import playwright_agent_bench as bench  # noqa: E402
from smoke_playwright import MCP  # noqa: E402

START_URL = "https://duckduckgo.com/"
TASK = ("Use DuckDuckGo to find pictures of oranges (the fruit) and download one. "
        f"Start at {START_URL}. Search for oranges, switch to the Images results, pick one "
        "image that clearly shows orange fruit, find the direct URL of the full image file, "
        "and call save_image with that URL. When the image is saved, reply with one short "
        "sentence saying what you saved.")
SYSTEM = (
    "You are a browser agent using Playwright tools on the real web. "
    "Operations do not include automatic snapshots: call browser_snapshot or browser_find to "
    "inspect the page before choosing element refs. Refs are bare IDs: use target \"e6\", "
    "not \"ref=e6\". Handle tool errors by inspecting state and changing the target; do not "
    "repeat a failed call with the same arguments. Only navigate to duckduckgo.com pages. "
    "To download, pass a direct http(s) image URL to save_image."
)
SAVE_TOOL = {"type": "function", "function": {
    "name": "save_image",
    "description": "Download an image from a direct http(s) URL and save it to disk. "
                   "Fails if the URL does not return an image.",
    "parameters": {"type": "object", "properties": {
        "url": {"type": "string", "description": "Direct URL of the image file"}},
        "required": ["url"], "additionalProperties": False}}}
TOOLS = bench.ALLOWED_TOOLS - {"browser_file_upload"}
MAX_TOOL_TEXT = 24000
LONG_URL = 300
STALE_LIMIT = 1500
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36")
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAGIC = ((b"\xff\xd8\xff", "jpg"), (b"\x89PNG\r\n\x1a\n", "png"), (b"GIF8", "gif"))


def image_kind(data):
    for prefix, kind in MAGIC:
        if data.startswith(prefix):
            return kind
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def save_image(url, out_dir):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None, "Error: url must be an absolute http(s) URL"
    request = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/140 Safari/537.36",
        "Accept": "image/avif,image/webp,image/*,*/*;q=0.8"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = response.read(MAX_IMAGE_BYTES + 1)
            ctype = response.headers.get("Content-Type", "")
    except (urllib.error.URLError, OSError, ValueError) as error:
        return None, f"Error: download failed: {error}"
    if len(data) > MAX_IMAGE_BYTES:
        return None, "Error: file larger than 15 MB"
    kind = image_kind(data)
    if not kind:
        return None, f"Error: URL did not return an image (Content-Type {ctype!r}). Find the direct image file URL."
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"orange-{int(time.time())}.{kind}"
    path.write_bytes(data)
    return {"path": str(path), "bytes": len(data), "kind": kind, "url": url}, \
        f"Saved {len(data)} bytes ({kind}) to {path.name}."


def shorten_urls(text):
    return re.sub(r"https?://\S{%d,}" % LONG_URL,
                  lambda m: m.group(0)[:120] + "...[long link shortened]", text)


def drop_stale_pages(messages):
    """Keep only the newest large page view; older ones become a short note."""
    for message in messages[:-1]:
        if message.get("role") == "tool" and len(message["content"]) > STALE_LIMIT:
            message["content"] = message["content"][:400] + "\n[older page view removed to save context]"


def allowed_navigation(url):
    host = urllib.parse.urlparse(url).hostname or ""
    return host == "duckduckgo.com" or host.endswith(".duckduckgo.com")


def run(args):
    out_dir = (Path(args.image_dir).resolve() if args.image_dir
               else Path(args.output).resolve().parent / "live-orange-images")
    with tempfile.TemporaryDirectory(prefix="qcm-live-agent-") as temporary:
        root = Path(temporary)
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        env = dict(os.environ, PW_MCP_PORT=str(port), PW_MCP_BROWSER="chrome",
                   PW_MCP_OUTPUT_DIR=str(root / "browser"), PW_MCP_SNAPSHOT="none")
        log = open(root / "mcp.log", "w+")
        process = subprocess.Popen(["bash", str(ROOT / "start-playwright-mcp.sh"), "--headless",
            "--port", str(port), "--browser", "chrome", "--isolated", "--snapshot-mode", "none",
            "--image-responses", "omit", "--output-dir", str(root / "browser"),
            "--user-agent", USER_AGENT],
            cwd=root, env=env, stdout=log, stderr=log, start_new_session=True)
        report = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "model": args.model, "task": TASK, "turns": [], "status": "FAIL",
                  "saved": None, "tool_errors": 0, "blocked_navigations": 0}
        started = time.monotonic()
        try:
            mcp = MCP(port)
            deadline = time.monotonic() + 60
            while True:
                try:
                    mcp.request("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                "clientInfo": {"name": "qcm-live-agent", "version": "1"}})
                    break
                except urllib.error.URLError:
                    if process.poll() is not None or time.monotonic() > deadline:
                        log.seek(0)
                        raise RuntimeError("Playwright MCP failed to start: " + log.read())
                    time.sleep(.25)
            mcp.request("notifications/initialized", notification=True)
            inventory = mcp.request("tools/list")["tools"]
            declarations = [{"type": "function", "function": {"name": t["name"],
                "description": t.get("description", ""), "parameters": t["inputSchema"]}}
                for t in inventory if t["name"] in TOOLS] + [SAVE_TOOL]
            report["tools"] = sorted(d["function"]["name"] for d in declarations)
            agent = bench.Agent(args, mcp, declarations)
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
            deadline = started + args.task_timeout
            for number in range(1, args.max_turns + 1):
                if time.monotonic() >= deadline:
                    report["reason"] = "task time limit"
                    break
                message, timing = agent.chat(messages, deadline)
                calls = message.get("tool_calls", [])
                turn = {"number": number, "llm_seconds": round(timing["seconds"], 2),
                        "prompt_tokens": timing["usage"].get("prompt_tokens"),
                        "content": message.get("content"), "calls": []}
                report["turns"].append(turn)
                messages.append(message)
                print(f"turn {number}: {timing['usage'].get('prompt_tokens')} prompt tokens, "
                      f"{[c['function']['name'] for c in calls] or 'answer'}", flush=True)
                if not calls:
                    report["answer"] = message.get("content") or ""
                    break
                for call in calls:
                    name = call["function"]["name"]
                    kind, arguments = bench.validate_call(name, call["function"].get("arguments"),
                                                          agent.names, agent.schemas)
                    entry = {"name": name, "arguments": arguments, "kind": kind}
                    if kind != "valid":
                        text = f"Error: {kind} tool call"
                        report["tool_errors"] += 1
                    elif name == "save_image":
                        saved, text = save_image(arguments["url"], out_dir)
                        if saved:
                            report["saved"] = saved
                        else:
                            report["tool_errors"] += 1
                    elif name == "browser_navigate" and not allowed_navigation(arguments.get("url", "")):
                        text = "Error: navigation is limited to duckduckgo.com. Use save_image for the image URL."
                        report["blocked_navigations"] += 1
                        report["tool_errors"] += 1
                    else:
                        raw = mcp.request("tools/call", {"name": name, "arguments": arguments})
                        text = bench.mcp_text(raw)
                        if raw.get("isError"):
                            report["tool_errors"] += 1
                    text = shorten_urls(text)
                    if len(text) > MAX_TOOL_TEXT:
                        entry["truncated_from"] = len(text)
                        text = text[:MAX_TOOL_TEXT] + "\n[output truncated by harness]"
                    entry["result_head"] = text[:600]
                    turn["calls"].append(entry)
                    if len(text) > STALE_LIMIT:
                        drop_stale_pages(messages)
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})
            else:
                report["reason"] = "turn cap reached"
        except (urllib.error.URLError, OSError, ValueError, AssertionError, RuntimeError) as error:
            report["reason"] = f"{type(error).__name__}: {error}"
        finally:
            report["seconds"] = round(time.monotonic() - started, 2)
            if report["saved"]:
                report["status"] = "PASS (image saved; subject needs visual check)"
            Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            log.close()
    print(json.dumps({k: report.get(k) for k in ("status", "reason", "seconds", "saved",
                                                 "tool_errors", "blocked_navigations", "answer")},
                     indent=2))
    return 0 if report["saved"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--key-env", default="BONSAI_API_KEY")
    parser.add_argument("--image-dir", default=None,
                        help="Directory to save the downloaded image (default: "
                             "live-orange-images next to --output)")
    parser.add_argument("--task-timeout", type=float, default=300)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--reasoning", choices=("default", "off", "on"), default="off")
    args = parser.parse_args()
    args.cache_prompt = True
    args.cv_path = ""
    args.output = str(Path(args.output).resolve())
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
