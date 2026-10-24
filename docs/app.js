/* Static results browser -- the client half of slmbench/viewer.py.
 *
 * Adapted from the lokalni.ai BenchMAX-cs viewer: every model here runs with
 * thinking disabled, so the thinking/non-thinking pairing is gone and a tab
 * opens on META.default_models instead. Per-item scores can be partial
 * (summaries, JSON fields), which the build folds into the ok flag and the
 * extra line under each output.
 *
 * Same interface as the live viewer in analysis/viewer.py, with the one part
 * that needed a server replaced. There, a filter and a page number went to
 * /api/samples and Python answered with the ten cards to draw. Here the same
 * answer is assembled in the browser out of files the build wrote:
 *
 *   data/index.js                  models, benchmarks, headline scores
 *   data/<bench>/<view>/meta.js    prompts, references, per-item verdicts
 *   data/<bench>/<view>/c<k>.js    model outputs, 25 items to a file
 *
 * They are loaded by appending <script> tags, not with fetch(): a file:// page
 * is its own opaque origin, so fetch() and XMLHttpRequest are refused, while a
 * script element is not. Each data file calls BM.put(key, payload), which is
 * how it reports back. That is the entire reason this file is not simply the
 * live viewer's JS with a different URL in it.
 *
 * Filtering, searching and counting run off meta.js, which carries one small
 * integer per item per model rather than the answers themselves. That keeps
 * "show me every item some model got wrong" instant over 1,012 items and ten
 * models, and means the only thing ever loaded on demand is the text of the
 * ten cards actually on screen -- with the next page's chunk prefetched, so
 * paging does not blink.
 */

var BM = (function () {
  "use strict";

  var STORE = {};     // key -> payload (null marks a file that failed to load)
  var WAIT = {};      // key -> [callback]
  var META = null;
  var BENCH = {};
  var STATE = {};     // per benchmark, so switching tabs keeps your selection

  // ---- data loading -------------------------------------------------------

  function put(key, payload) {
    STORE[key] = payload;
    var waiting = WAIT[key] || [];
    delete WAIT[key];
    waiting.forEach(function (cb) { cb(payload); });
  }

  function need(key, url, cb) {
    if (key in STORE) { cb(STORE[key]); return; }
    if (WAIT[key]) { WAIT[key].push(cb); return; }
    WAIT[key] = [cb];
    var s = document.createElement("script");
    s.src = url;
    s.async = true;
    s.onerror = function () { put(key, null); };
    document.head.appendChild(s);
  }

  function needAll(specs, cb) {
    var left = specs.length, out = new Array(specs.length);
    if (!left) { cb(out); return; }
    specs.forEach(function (sp, i) {
      need(sp[0], sp[1], function (v) {
        out[i] = v;
        if (--left === 0) cb(out);
      });
    });
  }

  function viewKey(st) { return st.ds ? st.ds + "|" + st.lang : st.lang; }
  function viewPath(bench, key) {
    return "data/" + bench + "/" + key.replace("|", "__") + "/";
  }
  function metaSpec(bench, key) {
    return [bench + "|" + key + "|meta", viewPath(bench, key) + "meta.js"];
  }
  function chunkSpec(bench, key, k) {
    return [bench + "|" + key + "|" + k, viewPath(bench, key) + "c" + k + ".js"];
  }

  // ---- helpers ------------------------------------------------------------

  function esc(s) {
    return (s == null ? "" : String(s)).replace(/[&<>"']/g, function (c) {
      return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c];
    });
  }
  function num(n) { return (n == null ? "-" : n.toLocaleString("en-US")); }
  function pctv(x) { return (x == null ? "-" : (100 * x).toFixed(1)); }
  function isThink() { return true; }
  function defaults(avail) {
    return (META.default_models || []).filter(function (t) { return avail.indexOf(t) >= 0; });
  }

  // ---- state lives in the location hash, so a view is a link --------------

  function parseHash() {
    var h = location.hash.replace(/^#/, "");
    var i = h.indexOf("?");
    var tab = i < 0 ? h : h.slice(0, i);
    return {tab: tab || "overall", p: new URLSearchParams(i < 0 ? "" : h.slice(i + 1))};
  }

  function pushHash(tab, st) {
    var p = new URLSearchParams();
    if (st) {
      p.set("lang", st.lang);
      if (st.ds) p.set("ds", st.ds);
      if (st.only !== "all") p.set("only", st.only);
      if (st.q) p.set("q", st.q);
      if (st.page) p.set("p", st.page);
      p.set("models", st.models.join(","));
    }
    var s = p.toString();
    setHash("#" + tab + (s ? "?" + s : ""));
  }

  // history.replaceState() takes a URL, and a file:// document is an opaque
  // origin: some browsers refuse the call outright there. Assigning the hash
  // does the same job (at the cost of a history entry), and the hashchange
  // handler ignores a hash that names the tab already on screen.
  function setHash(h) {
    try {
      history.replaceState(null, "", h);
    } catch (e) {
      if (location.hash !== h) location.hash = h;
    }
  }

  function defaultState(b, p) {
    var lang = p && p.get("lang");
    if (!lang || b.langs.indexOf(lang) < 0) lang = b.langs[0];
    var ds = null;
    if (b.datasets && b.datasets.length) {
      ds = (p && p.get("ds")) || b.datasets[0];
      if (b.datasets.indexOf(ds) < 0) ds = b.datasets[0];
    }
    var st = {lang: lang, ds: ds, only: (p && p.get("only")) || "all",
              q: (p && p.get("q")) || "", page: parseInt((p && p.get("p")) || "0", 10) || 0,
              models: null, per: 5, token: 0};
    var avail = b.views[viewKey(st)] || [];
    var wanted = ((p && p.get("models")) || "").split(",").filter(Boolean);
    st.models = wanted.length ? wanted.filter(function (t) { return avail.indexOf(t) >= 0; })
                              : defaults(avail);
    if (!st.models.length) st.models = avail.slice(0, 2);
    return st;
  }

  // ---- controls -----------------------------------------------------------

  function seg(role, options, current) {
    return '<div class="seg" data-role="' + role + '">' + options.map(function (o) {
      return '<button data-v="' + esc(o.v) + '"' + (o.v === current ? ' class="on"' : "") +
             ">" + esc(o.t) + "</button>";
    }).join("") + "</div>";
  }

  function controlsHTML(b, st) {
    var avail = b.views[viewKey(st)] || [];
    var html = '<div class="controls">';
    html += '<div class="ctl"><span class="lbl">' +
            (b.datasets.length ? "direction" : "language") + "</span>" +
            seg("lang", b.langs.map(function (l) {
              return {v: l, t: (b.lang_labels && b.lang_labels[l]) || l};
            }), st.lang) + "</div>";
    if (b.datasets.length) {
      // WMT24 ships only en->cs in BenchMAX, so the dataset switch offers what
      // this direction actually has rather than a button that leads nowhere.
      var dss = b.datasets.filter(function (d) { return b.views[d + "|" + st.lang]; });
      html += '<div class="ctl"><span class="lbl">dataset</span>' +
              seg("ds", dss.map(function (d) { return {v: d, t: d}; }), st.ds) + "</div>";
    }
    html += '<div class="ctl"><span class="lbl">show</span>' +
            seg("only", [{v: "all", t: "all"}, {v: "wrong", t: "any wrong"},
                         {v: "disagree", t: "disagreement"}, {v: "empty", t: "empty"},
                         {v: "right", t: "all correct"}], st.only) + "</div>";
    html += '<div class="ctl"><input class="q" type="search" placeholder="search inputs" value="' +
            esc(st.q) + '"></div>';
    html += '<div class="ctl"><span class="lbl">per page</span>' +
            seg("per", [{v: 5, t: "5"}, {v: 10, t: "10"}, {v: 25, t: "25"}], st.per) + "</div>";
    html += "</div>";

    html += '<div class="controls" data-role="models"><span class="lbl">models</span>';
    META.models.forEach(function (m) {
      if (avail.indexOf(m.tag) < 0) return;
      var on = st.models.indexOf(m.tag) >= 0;
      html += '<button class="chip' + (on ? " on" : "") +
              '" data-tag="' + esc(m.tag) + '" title="' + esc(m.params || "") + '">' + esc(m.tag) + "</button>";
    });
    html += '<span class="ctl"><button class="chip lite" data-pick="default">default</button>' +
            '<button class="chip lite" data-pick="all">all</button>' +
            '<button class="chip lite" data-pick="none">none</button></span>';
    html += "</div>";
    return html;
  }

  // ---- one tile per model: the run's own score, then what is on disk ------
  // The headline is the harness's number for this view (first key of scores),
  // never something recomputed here; the second line is what the build counted,
  // including the locale-aware re-extraction where it applies.

  function statsHTML(b, st, models, meta) {
    var scores = (META.scores[b.id] || {})[viewKey(st)] || {};
    if (!models.length) return "";
    var html = '<div class="strip">';
    models.forEach(function (t) {
      var vals = scores[t] || {}, keys = Object.keys(vals);
      var s = meta.summary[t] || {};
      var head = keys.length ? pctv(vals[keys[0]])
               : (s.graded ? pctv(s.ok / s.graded) : "—");
      var label = keys.length ? keys[0] : "correct";
      var rest = keys.slice(1).map(function (k) { return esc(k) + " " + pctv(vals[k]); });
      if (s.graded) rest.push(s.ok + "/" + s.graded + " correct");
      if (s.empty) rest.push(s.empty + " empty");
      if (s.corrected != null) rest.push("locale-aware " + pctv(s.corrected));
      html += '<div class="stat"><div class="t">' + esc(t) + "</div>" +
              '<div class="v">' + head + ' <span class="m">' + esc(label) + "</span></div>" +
              '<div class="m">' + rest.join(" · ") + "</div></div>";
    });
    return html + "</div>";
  }

  function pagerHTML(matched, total, st) {
    var pages = Math.max(1, Math.ceil(matched / st.per));
    var page = Math.min(st.page, pages - 1);
    return '<div class="pager">' +
      '<button data-nav="prev"' + (page <= 0 ? " disabled" : "") + ">← prev</button>" +
      '<button data-nav="next"' + (page >= pages - 1 ? " disabled" : "") + ">next →</button>" +
      "<span>page " + (page + 1) + " of " + pages + "</span>" +
      "<span>" + num(matched) + " of " + num(total) + " items" +
      (matched !== total ? " match" : "") + "</span></div>";
  }

  function outHTML(tag, o, flag) {
    var mark = flag === 1 ? '<span class="ok">✓</span>'
             : flag === 0 ? '<span class="bad">✗</span>'
             : '<span class="dim">–</span>';
    var body = (o.o || "").trim()
      ? '<pre class="clamp">' + esc(o.o) + "</pre>"
      : '<div class="empty">(no output recorded)</div>';
    return '<div class="out">' +
      '<div class="ohead">' + mark + '<span class="tag">' + esc(tag) + "</span></div>" +
      body + (o.x ? '<div class="ox">' + esc(o.x) + "</div>" : "") + "</div>";
  }

  function cardHTML(meta, idx, row, models) {
    var it = meta.items[idx];
    var cols = models.map(function (t) {
      return row && row[t] ? outHTML(t, row[t], meta.ok[t][idx]) : "";
    }).join("");
    var html = '<article class="card"><div class="chead"><span class="idx">#' + (idx + 1) +
      '</span><span class="pill">' + esc(it.id) + "</span>" +
      (it.m ? '<span class="dim">' + esc(it.m) + "</span>" : "") + "</div>" +
      '<div class="qbox"><div class="lbl">input</div><pre class="clamp">' + esc(it.q) + "</pre></div>";
    if (it.t) {
      html += '<div class="tbox"><div class="lbl">reference</div><pre class="clamp">' +
              esc(it.t) + "</pre></div>";
    }
    return html + '<div class="outs">' + cols + "</div></article>";
  }

  // ---- filtering ----------------------------------------------------------
  // Runs off meta.ok / meta.empty, which are one integer per item per model:
  //   1 correct, 0 wrong, -1 not graded, -2 this model has no output.

  function matching(meta, st) {
    var q = st.q.trim().toLowerCase();
    var models = st.models;
    var keep = [];
    for (var i = 0; i < meta.items.length; i++) {
      if (q && meta.items[i].q.toLowerCase().indexOf(q) < 0) continue;
      var anyWrong = false, allRight = models.length > 0, seen = false;
      var anyEmpty = false, sawOk = false, sawBad = false;
      for (var j = 0; j < models.length; j++) {
        var f = meta.ok[models[j]][i];
        if (f === -2) { allRight = false; continue; }
        seen = true;
        if (f === 0) { anyWrong = true; sawBad = true; }
        if (f === 1) sawOk = true;
        if (f !== 1) allRight = false;
        if (meta.empty[models[j]][i]) anyEmpty = true;
      }
      if (st.only === "wrong" && !anyWrong) continue;
      if (st.only === "right" && !(seen && allRight)) continue;
      if (st.only === "empty" && !anyEmpty) continue;
      if (st.only === "disagree" && !(sawOk && sawBad)) continue;
      keep.push(i);
    }
    return keep;
  }

  // ---- render -------------------------------------------------------------

  function pane(id) {
    return document.querySelector('.tabpane[data-tab="' + id + '"]');
  }

  function load(id) {
    var b = BENCH[id], st = STATE[id], el = pane(id);
    if (b.datasets.length && !b.views[viewKey(st)]) {
      for (var i = 0; i < b.datasets.length; i++) {
        if (b.views[b.datasets[i] + "|" + st.lang]) { st.ds = b.datasets[i]; break; }
      }
    }
    var key = viewKey(st);
    var avail = b.views[key] || [];
    st.models = st.models.filter(function (t) { return avail.indexOf(t) >= 0; });
    pushHash(id, st);

    el.querySelector(".ctlbox").innerHTML = controlsHTML(b, st);
    var body = el.querySelector(".body");
    if (!avail.length) {
      body.innerHTML = '<p class="empty">Nothing on disk for this view.</p>';
      return;
    }
    body.innerHTML = '<div class="loading">loading…</div>';

    var token = (st.token += 1);
    var spec = metaSpec(id, key);
    need(spec[0], spec[1], function (meta) {
      if (token !== st.token) return;   // a newer request already won
      if (!meta) {
        body.innerHTML = '<p class="bad">missing data file: ' + esc(spec[1]) +
          ' — run <code>python -m slmbench.viewer</code>.</p>';
        return;
      }
      var keep = matching(meta, st);
      var pages = Math.max(1, Math.ceil(keep.length / st.per));
      if (st.page > pages - 1) { st.page = pages - 1; load(id); return; }
      if (st.page < 0) st.page = 0;

      var shown = keep.slice(st.page * st.per, st.page * st.per + st.per);
      var head = statsHTML(b, st, st.models, meta) + pagerHTML(keep.length, meta.total, st);
      body.innerHTML = head + shown.map(function () { return '<div class="skel"></div>'; }).join("");

      var chunks = {};
      shown.forEach(function (idx) { chunks[Math.floor(idx / meta.chunk)] = 1; });
      var specs = Object.keys(chunks).map(function (k) { return chunkSpec(id, key, k); });
      needAll(specs, function () {
        if (token !== st.token) return;
        var html = head + (shown.length
          ? shown.map(function (idx) {
              var k = Math.floor(idx / meta.chunk);
              var rows = STORE[chunkSpec(id, key, k)[0]] || [];
              return cardHTML(meta, idx, rows[idx % meta.chunk], st.models);
            }).join("")
          : '<p class="empty">No items match this filter.</p>');
        html += pagerHTML(keep.length, meta.total, st);
        body.innerHTML = html;
        clampAll(body);
        prefetch(id, key, meta, keep, st);
      });
    });
  }

  // The next page's chunk, fetched while you read this one. Paging is the main
  // gesture in these tabs and a chunk is a few hundred kilobytes.
  function prefetch(id, key, meta, keep, st) {
    var nxt = keep.slice((st.page + 1) * st.per, (st.page + 1) * st.per + st.per);
    var chunks = {};
    nxt.forEach(function (idx) { chunks[Math.floor(idx / meta.chunk)] = 1; });
    Object.keys(chunks).forEach(function (k) {
      var sp = chunkSpec(id, key, k);
      need(sp[0], sp[1], function () {});
    });
  }

  // A <pre> only gets the "show more" affordance when it is actually cut off.
  function clampAll(root) {
    root.querySelectorAll("pre.clamp").forEach(function (el) {
      if (el.scrollHeight <= el.clientHeight + 4) {
        el.classList.remove("clamp");
        return;
      }
      var btn = document.createElement("button");
      btn.className = "more";
      btn.textContent = "show all";
      btn.addEventListener("click", function () {
        var open = el.classList.toggle("open");
        btn.textContent = open ? "show less" : "show all";
      });
      el.parentNode.insertBefore(btn, el.nextSibling);
    });
  }

  // ---- SVG export ---------------------------------------------------------
  // The vs-published charts are inline SVG painted by the page's stylesheet, so
  // an exported file has to carry its own paint: clone it, resolve every class
  // to literal presentation attributes in the light palette (a figure lands on
  // white paper and on white slides, whatever theme the page was read in), lay
  // an opaque white plate underneath, then serialise. Attributes rather than an
  // embedded <style>, because several things one feeds an .svg to ignore
  // stylesheets and would paint every bar black.

  var SVGNS = "http://www.w3.org/2000/svg";

  function inlineStyles(root) {
    var styles = (META.svg_export || {}).styles || {};
    var all = [root].concat(Array.prototype.slice.call(root.querySelectorAll("*")));
    all.forEach(function (el) {
      var cls = el.getAttribute && el.getAttribute("class");
      if (!cls) return;
      cls.split(/\s+/).forEach(function (tok) {
        var decls = styles[tok];
        if (!decls) return;
        Object.keys(decls).forEach(function (k) { el.setAttribute(k, decls[k]); });
      });
      el.removeAttribute("class");
    });
  }

  function downloadSVG(fig) {
    var src = fig && fig.querySelector("svg");
    if (!src) return;
    var box = (src.getAttribute("viewBox") || "").trim().split(/[\s,]+/);
    var svg = src.cloneNode(true);
    svg.setAttribute("xmlns", SVGNS);
    if (box.length === 4) {
      svg.setAttribute("width", box[2]);
      svg.setAttribute("height", box[3]);
    }
    inlineStyles(svg);
    svg.setAttribute("font-family", (META.svg_export || {}).font || "sans-serif");
    var bg = document.createElementNS(SVGNS, "rect");
    bg.setAttribute("x", box[0] || 0);
    bg.setAttribute("y", box[1] || 0);
    bg.setAttribute("width", box[2] || "100%");
    bg.setAttribute("height", box[3] || "100%");
    bg.setAttribute("fill", "#ffffff");
    svg.insertBefore(bg, svg.firstChild);
    var blob = new Blob(['<?xml version="1.0" encoding="UTF-8"?>\n',
                         new XMLSerializer().serializeToString(svg)],
                        {type: "image/svg+xml;charset=utf-8"});
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = (fig.dataset.name || "chart") + ".svg";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
  }

  // ---- events -------------------------------------------------------------

  function bindPane(id) {
    var el = pane(id);
    el.addEventListener("click", function (ev) {
      var st = STATE[id];
      var segbtn = ev.target.closest(".seg button");
      if (segbtn) {
        var role = segbtn.parentNode.dataset.role;
        st[role] = role === "per" ? parseInt(segbtn.dataset.v, 10) : segbtn.dataset.v;
        st.page = 0;
        load(id); return;
      }
      var pick = ev.target.closest("[data-pick]");
      if (pick) {
        var avail = BENCH[id].views[viewKey(st)] || [];
        st.models = pick.dataset.pick === "all" ? avail.slice()
                  : pick.dataset.pick === "none" ? [] : defaults(avail);
        st.page = 0;
        load(id); return;
      }
      var chip = ev.target.closest(".chip");
      if (chip) {
        var tag = chip.dataset.tag, i = st.models.indexOf(tag);
        if (i >= 0) st.models.splice(i, 1); else st.models.push(tag);
        // Keep the on-disk order, not click order, so columns do not jump.
        st.models.sort(function (a, b2) {
          return META.order.indexOf(a) - META.order.indexOf(b2);
        });
        load(id); return;
      }
      var nav = ev.target.closest("[data-nav]");
      if (nav) {
        st.page += (nav.dataset.nav === "next" ? 1 : -1);
        if (st.page < 0) st.page = 0;
        load(id); return;
      }
    });
    var timer = null;
    el.addEventListener("input", function (ev) {
      if (!ev.target.classList.contains("q")) return;
      clearTimeout(timer);
      var v = ev.target.value;
      timer = setTimeout(function () {
        STATE[id].q = v; STATE[id].page = 0; load(id);
      }, 250);
    });
  }

  function show(tab, params) {
    if (tab !== "overall" && !BENCH[tab]) tab = "overall";
    document.querySelectorAll(".tabbtn").forEach(function (b) {
      b.classList.toggle("on", b.dataset.tab === tab);
    });
    document.querySelectorAll(".tabpane").forEach(function (p) {
      p.classList.toggle("on", p.dataset.tab === tab);
    });
    if (tab === "overall") { pushHash(tab, null); return; }
    if (!STATE[tab]) { STATE[tab] = defaultState(BENCH[tab], params); bindPane(tab); }
    load(tab);
  }

  function start() {
    META = STORE.index;
    if (!META) {
      document.body.insertAdjacentHTML("afterbegin",
        '<p class="bad">data/index.js did not load — run <code>python -m slmbench.viewer</code>.</p>');
      return;
    }
    META.benches.forEach(function (b) { BENCH[b.id] = b; });

    document.querySelectorAll(".tabbtn").forEach(function (btn) {
      btn.addEventListener("click", function () { show(btn.dataset.tab, null); });
    });
    document.addEventListener("click", function (ev) {
      var btn = ev.target.closest && ev.target.closest(".svgdl");
      if (btn) downloadSVG(btn.closest("figure"));
    });
    // Paging with the arrow keys, since browsing is the whole point of these
    // tabs -- but never while the search box has focus.
    document.addEventListener("keydown", function (ev) {
      if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
      var el = document.activeElement;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" ||
                 el.isContentEditable)) return;
      if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") return;
      var tab = parseHash().tab;
      if (!STATE[tab]) return;
      var st = STATE[tab];
      if (ev.key === "ArrowLeft" && st.page > 0) { st.page--; load(tab); }
      if (ev.key === "ArrowRight") { st.page++; load(tab); }
    });
    window.addEventListener("hashchange", function () {
      var h = parseHash();
      if (!document.querySelector('.tabpane[data-tab="' + h.tab + '"].on')) show(h.tab, h.p);
    });
    var h0 = parseHash();
    show(h0.tab, h0.p);
  }

  return {put: put, start: start};
})();
