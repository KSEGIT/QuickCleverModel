/* RTX benchmark dashboard — plain JS, no build step, no external libraries.
 * Reads data/index.json (written by tests/bench/collect.py) and renders:
 *   - latest-run suite cards
 *   - pass-rate-over-runs line chart (one line per suite)
 *   - median-time-over-runs line chart (one line per suite)
 *   - peak-VRAM-vs-context scatter, with an 8,192 MiB reference line
 *   - a runs table (date, commit link, model, ctx, suites)
 * Charts are drawn as inline SVG built with the DOM API — no canvas
 * libraries, no chart frameworks. Colour is never the only signal: every
 * series has a legend entry with its name, every point has a native
 * tooltip (<title>), and every card/table cell repeats the number as text.
 */
(function () {
  "use strict";

  var REPO_URL = "https://github.com/KSEGIT/QuickCleverModel";
  var VRAM_CAP_MIB = 8192;

  // Fixed order, fixed colour per suite — never reassigned based on what a
  // given run happens to contain (dataviz: "colour follows the entity").
  var SUITES = [
    { key: "fixture", label: "Fixture", color: "--series-fixture" },
    { key: "long_context", label: "Long context", color: "--series-long-context" },
    { key: "live_web", label: "Live web", color: "--series-live-web" },
    { key: "concurrency_serial", label: "Concurrency (serial)", color: "--series-concurrency-serial" },
    { key: "concurrency_parallel", label: "Concurrency (parallel)", color: "--series-concurrency-parallel" }
  ];

  function suiteMeta(key) {
    for (var i = 0; i < SUITES.length; i++) {
      if (SUITES[i].key === key) return SUITES[i];
    }
    return { key: key, label: key, color: "--text-secondary" };
  }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function el(tag, attrs, children) {
    var isSvg = tag === "svg" || tag === "polyline" || tag === "line" ||
      tag === "circle" || tag === "text" || tag === "g" || tag === "rect" || tag === "title";
    var node = isSvg
      ? document.createElementNS("http://www.w3.org/2000/svg", tag)
      : document.createElement(tag);
    for (var key in (attrs || {})) {
      if (key === "class") node.setAttribute("class", attrs[key]);
      else if (key === "text") node.textContent = attrs[key];
      else node.setAttribute(key, attrs[key]);
    }
    (children || []).forEach(function (c) { if (c) node.appendChild(c); });
    return node;
  }

  function fmtNum(n, digits) {
    if (n === null || n === undefined) return "–"; // en dash for "no data"
    return n.toLocaleString(undefined, { maximumFractionDigits: digits === undefined ? 0 : digits });
  }

  function fmtSeconds(n) {
    if (n === null || n === undefined) return "–";
    return fmtNum(n, 1) + " s";
  }

  function fmtDate(iso) {
    if (!iso) return "–";
    var d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    return d.toISOString().slice(0, 16).replace("T", " ") + " UTC";
  }

  function shortCommit(sha) {
    if (!sha) return null;
    return sha.length > 7 ? sha.slice(0, 7) : sha;
  }

  function runPeakVram(run) {
    var peak = null;
    Object.keys(run.suites || {}).forEach(function (key) {
      var v = run.suites[key].peak_vram_mib;
      if (v !== null && v !== undefined && (peak === null || v > peak)) peak = v;
    });
    return peak;
  }

  // ---------------------------------------------------------------------
  // Latest-run cards
  // ---------------------------------------------------------------------

  function renderCards(run) {
    var grid = el("div", { class: "card-grid" });
    var keys = Object.keys(run.suites || {});
    if (keys.length === 0) {
      grid.appendChild(el("p", { text: "The latest run requested no suites." }));
      return grid;
    }
    SUITES.concat(keys.filter(function (k) { return !SUITES.some(function (s) { return s.key === k; }); })
      .map(function (k) { return { key: k, label: k, color: "--text-secondary" }; }))
      .forEach(function (meta) {
        var suite = run.suites[meta.key];
        if (!suite) return;
        var card = el("div", { class: "suite-card" });
        var swatch = el("span", { class: "swatch", style: "background:" + cssVar(meta.color) + ";" });
        card.appendChild(el("p", { class: "suite-name" }, [swatch, document.createTextNode(meta.label)]));

        var passOk = suite.total > 0 && suite.passed === suite.total;
        var pill = el("span", {
          class: "status-pill " + (passOk ? "status-good" : "status-critical"),
          text: (passOk ? "PASS " : "") + suite.passed + "/" + suite.total
        });
        var dl = el("dl", {}, [
          el("dt", { text: "Pass rate" }), el("dd", {}, [pill, document.createTextNode(" " + fmtNum(suite.pass_rate, 1) + "%")]),
          el("dt", { text: "Median time" }), el("dd", { text: fmtSeconds(suite.median_seconds) }),
          el("dt", { text: "Peak VRAM" }), el("dd", { text: suite.peak_vram_mib != null ? fmtNum(suite.peak_vram_mib) + " MiB" : "–" })
        ]);
        card.appendChild(dl);
        if (suite.error) {
          card.appendChild(el("p", { class: "error-text", text: "Error: " + suite.error }));
        }
        grid.appendChild(card);
      });
    return grid;
  }

  // ---------------------------------------------------------------------
  // Generic multi-series line chart (SVG, DOM API, no libraries)
  // ---------------------------------------------------------------------

  function drawLineChart(runs, opts) {
    // opts: { valueFn(suite) -> number|null, yFormat(n) -> string, yMax? }
    var W = 640, H = 260;
    var marginL = 46, marginR = 16, marginT = 16, marginB = 34;
    var plotW = W - marginL - marginR, plotH = H - marginT - marginB;

    var seriesData = SUITES.map(function (meta) {
      var points = runs.map(function (run, i) {
        var suite = run.suites && run.suites[meta.key];
        var v = suite ? opts.valueFn(suite) : null;
        return { i: i, v: (v === undefined ? null : v), run: run };
      });
      return { meta: meta, points: points };
    }).filter(function (s) { return s.points.some(function (p) { return p.v !== null; }); });

    var allVals = [];
    seriesData.forEach(function (s) { s.points.forEach(function (p) { if (p.v !== null) allVals.push(p.v); }); });
    var yMax = opts.yMax !== undefined ? opts.yMax : Math.max.apply(null, allVals.concat([0])) * 1.15;
    if (yMax <= 0) yMax = 1;
    var yMin = 0;

    var n = runs.length;
    function xAt(i) { return n <= 1 ? marginL + plotW / 2 : marginL + (plotW * i) / (n - 1); }
    function yAt(v) { return marginT + plotH - ((v - yMin) / (yMax - yMin)) * plotH; }

    var svg = el("svg", { class: "chart", viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": opts.ariaLabel || "" });

    // gridlines + y labels
    var ticks = 4;
    for (var t = 0; t <= ticks; t++) {
      var v = yMax * t / ticks;
      var y = yAt(v);
      svg.appendChild(el("line", { class: "gridline", x1: marginL, x2: W - marginR, y1: y, y2: y }));
      svg.appendChild(el("text", { x: marginL - 8, y: y + 3, "text-anchor": "end", "font-size": "10", text: opts.yFormat ? opts.yFormat(v) : fmtNum(v) }));
    }
    // x axis baseline + run date labels
    svg.appendChild(el("line", { class: "axis-line", x1: marginL, x2: W - marginR, y1: marginT + plotH, y2: marginT + plotH }));
    runs.forEach(function (run, i) {
      var x = xAt(i);
      svg.appendChild(el("line", { class: "axis-line", x1: x, x2: x, y1: marginT + plotH, y2: marginT + plotH + 4 }));
      var label = (run.created_utc || "").slice(5, 10) || String(i + 1);
      svg.appendChild(el("text", { x: x, y: H - 8, "text-anchor": "middle", "font-size": "10", text: label }));
    });

    // series lines + points
    seriesData.forEach(function (s) {
      var color = cssVar(s.meta.color);
      var seg = [];
      function flush() {
        if (seg.length > 1) {
          var pts = seg.map(function (p) { return xAt(p.i) + "," + yAt(p.v); }).join(" ");
          svg.appendChild(el("polyline", { class: "series-line", points: pts, stroke: color }));
        }
        seg = [];
      }
      s.points.forEach(function (p) {
        if (p.v === null) { flush(); return; }
        seg.push(p);
      });
      flush();
      s.points.forEach(function (p) {
        if (p.v === null) return;
        var cx = xAt(p.i), cy = yAt(p.v);
        var title = el("title", { text: s.meta.label + " — " + fmtDate(p.run.created_utc) + ": " + (opts.yFormat ? opts.yFormat(p.v) : fmtNum(p.v)) });
        svg.appendChild(el("circle", { class: "series-point", cx: cx, cy: cy, r: 4, fill: color }, [title]));
      });
    });

    var wrap = el("div", { class: "chart-wrap" }, [svg]);
    var legend = el("div", { class: "legend" }, seriesData.map(function (s) {
      return el("span", { class: "legend-item" }, [
        el("span", { class: "swatch", style: "background:" + cssVar(s.meta.color) + ";" }),
        document.createTextNode(s.meta.label)
      ]);
    }));
    if (seriesData.length === 0) {
      return el("div", {}, [el("p", { class: "panel-note", text: "No suites with this data yet." })]);
    }
    var container = el("div", {}, [wrap, legend]);
    return container;
  }

  // ---------------------------------------------------------------------
  // Peak VRAM vs context scatter
  // ---------------------------------------------------------------------

  function drawVramScatter(runs) {
    var W = 640, H = 260;
    var marginL = 54, marginR = 16, marginT = 16, marginB = 34;
    var plotW = W - marginL - marginR, plotH = H - marginT - marginB;

    var points = runs.map(function (run) {
      return { ctx: run.ctx, vram: runPeakVram(run), run: run };
    }).filter(function (p) { return p.ctx != null && p.vram != null; });

    if (points.length === 0) {
      return el("p", { class: "panel-note", text: "No run yet has both a context size and a peak VRAM reading." });
    }

    var xs = points.map(function (p) { return p.ctx; });
    var ys = points.map(function (p) { return p.vram; }).concat([VRAM_CAP_MIB]);
    var xMax = Math.max.apply(null, xs) * 1.1;
    var xMin = 0;
    var yMax = Math.max.apply(null, ys) * 1.1;
    var yMin = 0;

    function xAt(v) { return marginL + ((v - xMin) / (xMax - xMin)) * plotW; }
    function yAt(v) { return marginT + plotH - ((v - yMin) / (yMax - yMin)) * plotH; }

    var svg = el("svg", { class: "chart", viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "Peak VRAM against context size, in mebibytes" });

    var ticks = 4;
    for (var t = 0; t <= ticks; t++) {
      var yv = yMax * t / ticks;
      var y = yAt(yv);
      svg.appendChild(el("line", { class: "gridline", x1: marginL, x2: W - marginR, y1: y, y2: y }));
      svg.appendChild(el("text", { x: marginL - 8, y: y + 3, "text-anchor": "end", "font-size": "10", text: fmtNum(yv) }));
    }
    for (var xt = 0; xt <= ticks; xt++) {
      var xv = xMax * xt / ticks;
      var x = xAt(xv);
      svg.appendChild(el("text", { x: x, y: H - 8, "text-anchor": "middle", "font-size": "10", text: fmtNum(xv) }));
    }
    svg.appendChild(el("line", { class: "axis-line", x1: marginL, x2: W - marginR, y1: marginT + plotH, y2: marginT + plotH }));
    svg.appendChild(el("line", { class: "axis-line", x1: marginL, x2: marginL, y1: marginT, y2: marginT + plotH }));

    // 8,192 MiB reference line
    var refY = yAt(VRAM_CAP_MIB);
    if (refY >= marginT && refY <= marginT + plotH) {
      svg.appendChild(el("line", { class: "ref-line", x1: marginL, x2: W - marginR, y1: refY, y2: refY }));
      svg.appendChild(el("text", { x: W - marginR, y: refY - 4, "text-anchor": "end", "font-size": "10", text: "8,192 MiB" }));
    }

    points.forEach(function (p) {
      var cx = xAt(p.ctx), cy = yAt(p.vram);
      var color = cssVar("--series-fixture");
      var title = el("title", { text: (p.run.model || "run") + " — ctx " + fmtNum(p.ctx) + ", peak " + fmtNum(p.vram) + " MiB" });
      svg.appendChild(el("circle", { class: "series-point", cx: cx, cy: cy, r: 5, fill: color }, [title]));
      svg.appendChild(el("text", { x: cx, y: cy - 10, "text-anchor": "middle", "font-size": "10", text: fmtNum(p.vram) + " MiB" }));
    });

    return el("div", {}, [
      el("div", { class: "chart-wrap" }, [svg]),
      el("p", { class: "panel-note", text: "Dashed line: 8,192 MiB, the RTX 3070 Ti's VRAM budget." })
    ]);
  }

  // ---------------------------------------------------------------------
  // Runs table
  // ---------------------------------------------------------------------

  function renderTable(runs) {
    var table = el("table", { class: "runs-table" });
    var thead = el("thead", {}, [el("tr", {}, [
      el("th", { text: "Date" }), el("th", { text: "Commit" }), el("th", { text: "Model" }),
      el("th", { text: "Ctx" }), el("th", { text: "Suites" })
    ])]);
    var tbody = el("tbody");
    // Newest first for the table, even though the chart x-axis reads oldest-first.
    runs.slice().reverse().forEach(function (run) {
      var commit = shortCommit(run.commit);
      var commitCell = commit
        ? el("td", {}, [el("a", { href: REPO_URL + "/commit/" + run.commit, text: commit })])
        : el("td", { text: "–" });
      var tags = el("span", { class: "suite-tags" }, Object.keys(run.suites || {}).map(function (key) {
        var suite = run.suites[key];
        var meta = suiteMeta(key);
        return el("span", { class: "suite-tag", text: meta.label + " · " + fmtNum(suite.pass_rate, 0) + "%" });
      }));
      var dateText = fmtDate(run.created_utc);
      if (run.sample) dateText += " (sample)";
      var row = el("tr", {}, [
        el("td", { text: dateText }),
        commitCell,
        el("td", { text: run.model || "–" }),
        el("td", { text: run.ctx != null ? fmtNum(run.ctx) : "–" }),
        el("td", {}, [tags])
      ]);
      tbody.appendChild(row);
    });
    table.appendChild(thead);
    table.appendChild(tbody);
    return el("div", { class: "table-scroll" }, [table]);
  }

  // ---------------------------------------------------------------------
  // Page assembly
  // ---------------------------------------------------------------------

  function panel(title, note, body) {
    var children = [el("h2", { text: title })];
    if (note) children.push(el("p", { class: "panel-note", text: note }));
    children.push(body);
    return el("section", { class: "panel" }, children);
  }

  function render(index) {
    var app = document.getElementById("app");
    var headerNote = document.getElementById("header-note");
    app.textContent = "";

    var runs = (index && index.runs) || [];
    var hasSample = runs.some(function (r) { return r.sample; });

    if (hasSample) {
      headerNote.textContent = "";
      var headerRow = el("p", {}, [
        document.createTextNode(runs.length + " run" + (runs.length === 1 ? "" : "s") + " recorded. "),
        el("span", { class: "badge badge-sample", text: "Sample data — no real run yet" })
      ]);
      headerNote.appendChild(headerRow);
    } else if (runs.length > 0) {
      headerNote.textContent = runs.length + " run" + (runs.length === 1 ? "" : "s") + " recorded. Latest: " + fmtDate(runs[runs.length - 1].created_utc) + ".";
    } else {
      headerNote.textContent = "No runs recorded yet.";
    }

    if (runs.length === 0) {
      app.appendChild(el("div", { class: "empty-note" }, [
        el("p", { text: "No runs yet." }),
        el("p", { text: "Trigger the “RTX benchmark” workflow from the Actions tab to populate this dashboard." })
      ]));
      return;
    }

    var latest = runs[runs.length - 1];
    app.appendChild(panel("Latest run", "Suite results for the most recent run (" + fmtDate(latest.created_utc) + ").", renderCards(latest)));

    app.appendChild(panel("Pass rate over runs", "One line per suite; a suite with no data for a run is skipped, not zero.",
      drawLineChart(runs, { valueFn: function (s) { return s.total > 0 ? s.pass_rate : null; }, yFormat: function (v) { return fmtNum(v) + "%"; }, yMax: 100, ariaLabel: "Pass rate per suite across runs, 0 to 100 percent" })));

    app.appendChild(panel("Median time over runs", "Median wall-clock seconds per suite.",
      drawLineChart(runs, { valueFn: function (s) { return s.median_seconds; }, yFormat: function (v) { return fmtNum(v, 0) + "s"; }, ariaLabel: "Median seconds per suite across runs" })));

    app.appendChild(panel("Peak VRAM vs. context", "Each point is one run: its context size and the highest VRAM use seen across its suites.",
      drawVramScatter(runs)));

    app.appendChild(panel("All runs", null, renderTable(runs)));
  }

  function renderError(message) {
    var app = document.getElementById("app");
    document.getElementById("header-note").textContent = "Could not load run data.";
    app.textContent = "";
    app.appendChild(el("div", { class: "empty-note" }, [
      el("p", { text: "Could not load data/index.json." }),
      el("p", { text: String(message) })
    ]));
  }

  fetch("data/index.json")
    .then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    })
    .then(render)
    .catch(function (err) { renderError(err && err.message ? err.message : err); });
})();
