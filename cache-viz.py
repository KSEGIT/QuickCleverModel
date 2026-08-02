#!/usr/bin/env python3
"""Live visualiser for llama-server's prompt cache.

Serves a dashboard on :8090 showing, per request, how many prompt tokens were
REUSED from the KV cache versus how many had to be processed from scratch —
which is the whole reason prefill is fast on turn 2 of a conversation and slow
on turn 1.

Data comes from parsing run/logs/llama.log, because the /metrics endpoint
exposes throughput counters but nothing about cache reuse. Two line shapes
carry it:

    restored context checkpoint (... n_tokens = 485 ...)   -> tokens reused
    prompt eval time = 1324.43 ms /    79 tokens           -> tokens processed

Stdlib only; no install step.
"""
import http.server
import json
import os
import re
import socketserver
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(ROOT, "run", "logs", "llama.log")
PORT = int(os.environ.get("CACHE_VIZ_PORT", "8090"))

RE_TASK = re.compile(r"task\s+(\d+)")
RE_RESTORED = re.compile(r"restored context checkpoint .*?n_tokens = (\d+)")
RE_PROMPT = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
RE_EVAL = re.compile(r"\beval time =\s*([\d.]+) ms /\s*(\d+) tokens .*?([\d.]+) tokens per second")
RE_FULL = re.compile(r"forcing full prompt re-processing")

_lock = threading.Lock()
_requests = []          # completed request records, oldest first
_pending = {}           # task id -> partial record


def _parse_line(line: str) -> None:
    m = RE_TASK.search(line)
    if not m:
        return
    task = m.group(1)
    rec = _pending.setdefault(task, {"task": int(task), "reused": 0, "new": 0,
                                     "prefill_ms": 0.0, "gen_tok_s": 0.0,
                                     "gen_tokens": 0, "full_reprocess": False})
    if RE_FULL.search(line):
        rec["full_reprocess"] = True
    r = RE_RESTORED.search(line)
    if r:
        rec["reused"] = int(r.group(1))
    p = RE_PROMPT.search(line)
    if p:
        rec["prefill_ms"] = float(p.group(1))
        rec["new"] = int(p.group(2))
    e = RE_EVAL.search(line)
    if e and "prompt eval" not in line:
        rec["gen_tokens"] = int(e.group(2))
        rec["gen_tok_s"] = float(e.group(3))
        # generation timing is the last line of a request -> it is complete
        with _lock:
            _requests.append(_pending.pop(task))
            del _requests[:-200]        # keep the last 200


def tail_log() -> None:
    """Follow the log, tolerating the file being rotated or recreated."""
    pos = 0
    while True:
        try:
            if not os.path.exists(LOG):
                time.sleep(1)
                continue
            size = os.path.getsize(LOG)
            if size < pos:              # truncated by a restart
                pos = 0
                with _lock:
                    _requests.clear()
                _pending.clear()
            with open(LOG, "r", errors="replace") as fh:
                fh.seek(pos)
                for line in fh:
                    _parse_line(line)
                pos = fh.tell()
        except Exception:
            pass                        # a transient read error must not kill the tailer
        time.sleep(0.5)


def snapshot() -> dict:
    with _lock:
        reqs = list(_requests)
    reused = sum(r["reused"] for r in reqs)
    new = sum(r["new"] for r in reqs)
    total = reused + new
    # Cost per prompt token measured from the requests that actually did work,
    # used to estimate the wall-clock the cache avoided.
    rates = [r["prefill_ms"] / r["new"] for r in reqs if r["new"] > 0 and r["prefill_ms"] > 0]
    ms_per_tok = sum(rates) / len(rates) if rates else 0.0
    return {
        "requests": reqs[-40:],
        "hit_rate": (reused / total * 100) if total else 0.0,
        "reused_total": reused,
        "new_total": new,
        "saved_ms": reused * ms_per_tok,
        "ms_per_tok": ms_per_tok,
        "full_reprocess": sum(1 for r in reqs if r["full_reprocess"]),
        "n": len(reqs),
    }


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):        # keep the console quiet
        pass

    def do_GET(self):
        if self.path.startswith("/api/data"):
            body = json.dumps(snapshot()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prompt cache — llama-server</title>
<style>
  :root{
    color-scheme: light;
    --surface-1:#fcfcfb; --surface-2:#f3f3f1; --border:#e0e0dc;
    --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#77766f;
    --reused:#2a78d6;   /* categorical slot 1 */
    --fresh:#eb6834;    /* categorical slot 2 */
    --grid:#e6e6e2;
  }
  @media (prefers-color-scheme: dark){
    :root:where(:not([data-theme="light"])){
      color-scheme: dark;
      --surface-1:#1a1a19; --surface-2:#242422; --border:#38382f;
      --text-primary:#fff; --text-secondary:#c3c2b7; --text-muted:#8d8c81;
      --reused:#3987e5; --fresh:#d95926; --grid:#2e2e2a;
    }
  }
  :root[data-theme="dark"]{
    color-scheme: dark;
    --surface-1:#1a1a19; --surface-2:#242422; --border:#38382f;
    --text-primary:#fff; --text-secondary:#c3c2b7; --text-muted:#8d8c81;
    --reused:#3987e5; --fresh:#d95926; --grid:#2e2e2a;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--surface-1);color:var(--text-primary);
       font:14px/1.5 ui-sans-serif,-apple-system,"SF Pro Text",system-ui,sans-serif;padding:28px}
  h1{font-size:19px;margin:0 0 2px;letter-spacing:-.01em}
  .sub{color:var(--text-secondary);font-size:13px;margin-bottom:22px}
  .tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:12px;margin-bottom:22px}
  .tile{background:var(--surface-2);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
  .tile .label{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--text-muted)}
  .tile .value{font-size:27px;font-weight:600;margin-top:5px;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
  .tile .note{font-size:12px;color:var(--text-secondary);margin-top:2px}
  .card{background:var(--surface-2);border:1px solid var(--border);border-radius:10px;padding:16px 18px;margin-bottom:16px}
  .card h2{font-size:13px;margin:0 0 3px;font-weight:600}
  .card .desc{font-size:12px;color:var(--text-secondary);margin-bottom:14px}
  .legend{display:flex;gap:16px;font-size:12px;color:var(--text-secondary);margin-bottom:10px}
  .legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
  .scroll{overflow-x:auto}
  svg{display:block}
  .tt{position:fixed;pointer-events:none;background:var(--surface-1);border:1px solid var(--border);
      border-radius:7px;padding:8px 10px;font-size:12px;opacity:0;transition:opacity .1s;
      box-shadow:0 4px 14px rgba(0,0,0,.16);z-index:9;white-space:nowrap}
  .tt b{font-weight:600}
  table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}
  th,td{text-align:right;padding:5px 9px;border-bottom:1px solid var(--border)}
  th:first-child,td:first-child{text-align:left}
  th{color:var(--text-muted);font-weight:500;font-size:11px;text-transform:uppercase;letter-spacing:.05em}
  details{margin-top:6px} summary{cursor:pointer;color:var(--text-secondary);font-size:12px}
  .empty{color:var(--text-muted);font-size:13px;padding:22px 0;text-align:center}
</style></head><body>
<h1>Prompt cache</h1>
<div class="sub">Prompt tokens reused from the KV cache vs processed from scratch · llama-server :8080 · refreshes every 2s</div>

<div class="tiles">
  <div class="tile"><div class="label">Cache hit rate</div><div class="value" id="hit">—</div>
    <div class="note" id="hitnote">of all prompt tokens</div></div>
  <div class="tile"><div class="label">Tokens reused</div><div class="value" id="reused">—</div>
    <div class="note">never re-processed</div></div>
  <div class="tile"><div class="label">Prefill time saved</div><div class="value" id="saved">—</div>
    <div class="note" id="savednote">estimated</div></div>
  <div class="tile"><div class="label">Full re-processes</div><div class="value" id="full">—</div>
    <div class="note">cache missed entirely</div></div>
</div>

<div class="card">
  <h2>Prompt composition per request</h2>
  <div class="desc">Each bar is one request. The blue part cost nothing — it was already in the cache.</div>
  <div class="legend">
    <span><i style="background:var(--reused)"></i>Reused from cache</span>
    <span><i style="background:var(--fresh)"></i>Processed fresh</span>
  </div>
  <div class="scroll"><svg id="bars" height="230"></svg></div>
  <div class="empty" id="barsEmpty">Waiting for requests… send a message in the chat UI.</div>
</div>

<div class="card">
  <h2>Prefill time per request</h2>
  <div class="desc">The payoff. Turn 1 pays full price; later turns ride the cache.</div>
  <div class="scroll"><svg id="line" height="180"></svg></div>
  <details><summary>Table view</summary>
    <table id="tbl"><thead><tr><th>Req</th><th>Reused</th><th>Fresh</th><th>Prefill ms</th><th>Gen t/s</th></tr></thead>
    <tbody></tbody></table></details>
</div>

<div class="tt" id="tt"></div>
<script>
const $=s=>document.querySelector(s), tt=$("#tt");
const fmt=n=>n>=1000?(n/1000).toFixed(n>=10000?0:1)+"k":String(Math.round(n));
function showTip(e,html){tt.innerHTML=html;tt.style.opacity=1;
  const r=tt.getBoundingClientRect();
  tt.style.left=Math.min(e.clientX+14,innerWidth-r.width-8)+"px";
  tt.style.top=Math.max(8,e.clientY-r.height-12)+"px";}
function hideTip(){tt.style.opacity=0}

function drawBars(reqs){
  const svg=$("#bars"), host=svg.parentElement.clientWidth-2,
        W=Math.max(reqs.length*34+56,host), H=230, pad={t:12,r:12,b:26,l:52};
  svg.setAttribute("width",W); svg.setAttribute("viewBox",`0 0 ${W} ${H}`);
  const max=Math.max(...reqs.map(r=>r.reused+r.new),1);
  const ih=H-pad.t-pad.b, bw=Math.min(34,Math.max(8,(W-pad.l-pad.r)/reqs.length-10));
  let s="";
  // recessive gridlines + axis labels
  for(let i=0;i<=4;i++){const v=max*i/4, y=pad.t+ih-ih*i/4;
    s+=`<line x1="${pad.l}" y1="${y}" x2="${W-pad.r}" y2="${y}" stroke="var(--grid)" stroke-width="1"/>`;
    s+=`<text x="${pad.l-8}" y="${y+4}" text-anchor="end" font-size="10" fill="var(--text-muted)">${fmt(v)}</text>`;}
  reqs.forEach((r,i)=>{
    const x=pad.l+i*((W-pad.l-pad.r)/reqs.length)+4;
    const tot=r.reused+r.new;
    const hR=ih*r.reused/max, hN=ih*r.new/max;
    const yN=pad.t+ih-hN, yR=yN-hR;
    // fresh sits on the baseline with a 4px rounded top; 2px surface gap above it
    if(r.new>0) s+=`<rect x="${x}" y="${yN}" width="${bw}" height="${Math.max(hN,2)}" rx="3" fill="var(--fresh)"/>`;
    if(r.reused>0) s+=`<rect x="${x}" y="${yR}" width="${bw}" height="${Math.max(hR-2,2)}" rx="3" fill="var(--reused)"/>`;
    s+=`<rect class="hit" data-i="${i}" x="${x-3}" y="${pad.t}" width="${bw+6}" height="${ih}" fill="transparent"/>`;
    if(reqs.length<=24||i%3===0)
      s+=`<text x="${x+bw/2}" y="${H-9}" text-anchor="middle" font-size="10" fill="var(--text-muted)">${i+1}</text>`;
  });
  svg.innerHTML=s;
  svg.querySelectorAll(".hit").forEach(el=>{
    el.onmousemove=e=>{const r=reqs[+el.dataset.i];
      showTip(e,`<b>Request ${+el.dataset.i+1}</b><br>Reused <b>${r.reused}</b> · Fresh <b>${r.new}</b><br>
                 Prefill ${r.prefill_ms.toFixed(0)} ms · Gen ${r.gen_tok_s.toFixed(1)} t/s`);};
    el.onmouseleave=hideTip;});
}

function drawLine(reqs){
  const svg=$("#line"), host=svg.parentElement.clientWidth-2,
        W=Math.max(reqs.length*34+56,host), H=180, pad={t:12,r:12,b:26,l:52};
  svg.setAttribute("width",W); svg.setAttribute("viewBox",`0 0 ${W} ${H}`);
  const max=Math.max(...reqs.map(r=>r.prefill_ms),1), ih=H-pad.t-pad.b;
  const X=i=>pad.l+i*((W-pad.l-pad.r)/Math.max(reqs.length-1,1));
  const Y=v=>pad.t+ih-ih*v/max;
  let s="";
  for(let i=0;i<=3;i++){const v=max*i/3, y=pad.t+ih-ih*i/3;
    s+=`<line x1="${pad.l}" y1="${y}" x2="${W-pad.r}" y2="${y}" stroke="var(--grid)" stroke-width="1"/>`;
    s+=`<text x="${pad.l-8}" y="${y+4}" text-anchor="end" font-size="10" fill="var(--text-muted)">${fmt(v)}</text>`;}
  s+=`<path d="${reqs.map((r,i)=>(i?"L":"M")+X(i)+" "+Y(r.prefill_ms)).join(" ")}"
       fill="none" stroke="var(--fresh)" stroke-width="2" stroke-linejoin="round"/>`;
  reqs.forEach((r,i)=>{ // 2px surface ring so overlapping markers stay separable
    s+=`<circle cx="${X(i)}" cy="${Y(r.prefill_ms)}" r="4" fill="var(--fresh)" stroke="var(--surface-2)" stroke-width="2"/>`;
    s+=`<circle class="hitp" data-i="${i}" cx="${X(i)}" cy="${Y(r.prefill_ms)}" r="13" fill="transparent"/>`;});
  svg.innerHTML=s;
  svg.querySelectorAll(".hitp").forEach(el=>{
    el.onmousemove=e=>{const r=reqs[+el.dataset.i];
      showTip(e,`<b>Request ${+el.dataset.i+1}</b><br>Prefill <b>${r.prefill_ms.toFixed(0)} ms</b> for ${r.new} fresh tokens`);};
    el.onmouseleave=hideTip;});
}

async function tick(){
  let d; try{ d=await (await fetch("/api/data")).json(); }catch(e){ return; }
  const reqs=d.requests||[];
  $("#hit").textContent=reqs.length?d.hit_rate.toFixed(1)+"%":"—";
  $("#hitnote").textContent=`${fmt(d.reused_total)} of ${fmt(d.reused_total+d.new_total)} prompt tokens`;
  $("#reused").textContent=fmt(d.reused_total);
  $("#saved").textContent=d.saved_ms>=1000?(d.saved_ms/1000).toFixed(1)+"s":Math.round(d.saved_ms)+"ms";
  $("#savednote").textContent=d.ms_per_tok?`at ${d.ms_per_tok.toFixed(1)} ms/token measured`:"estimated";
  $("#full").textContent=d.full_reprocess;
  $("#barsEmpty").style.display=reqs.length?"none":"block";
  if(reqs.length){ drawBars(reqs); drawLine(reqs); }
  $("#tbl tbody").innerHTML=reqs.map((r,i)=>
    `<tr><td>${i+1}</td><td>${r.reused}</td><td>${r.new}</td><td>${r.prefill_ms.toFixed(0)}</td><td>${r.gen_tok_s.toFixed(1)}</td></tr>`).join("");
}
tick(); setInterval(tick,2000);
addEventListener("resize",()=>tick());
</script></body></html>"""


if __name__ == "__main__":
    threading.Thread(target=tail_log, daemon=True).start()
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"cache visualiser -> http://127.0.0.1:{PORT}")
        print(f"reading {LOG}")
        httpd.serve_forever()
