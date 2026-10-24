"""Static results site in the lokalni.ai house style, after the BenchMAX-cs viewer
(https://lokalniai.github.io/benchmax-cs-experiments/viewer/).

    python -m slmbench.viewer [--out outputs] [--results results] [--site viewer/site]

Charts and tables are rendered here as static SVG/HTML. The sample browser is viewer/static/app.js, which
loads data/<task>/<lang>/{meta,c<k>}.js through <script> tags, so the site also works from file://.
"""

import argparse
import csv
import html
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

from .common import LANGS, OUTPUTS, RESULTS, ROOT, TASKS, data_path, read_jsonl
from .generate import model_configs

POST_URL = "https://lokalni.ai/blog/male-modely/"
POST_TITLE = "Malé modely pro koncová zařízení"
LEDE = ("Twelve small instruction-tuned models (0.27–4 B) on six practical skills, each tested in English "
        "and in Czech. Every model runs in BF16 with greedy decoding and thinking switched off.")
DEFAULT_MODELS = ["lfm2.5-1.2b", "qwen3.5-2b", "gemma-4-e2b", "ministral-3-3b"]
LANG_NAME = {"en": "English", "cs": "Czech"}
CHUNK = 25

# task -> (tab label, radar axis label, headline name, description)
BENCH = {
    "nexus": ("Tool calling", "Tool calling", "call accuracy",
              "BenchMAX Nexus (318 queries per language, 2–12 tools in context). A call counts when the function "
              "and every argument match exactly, with defaults filled in, as in BenchMAX. Tool descriptions stay in "
              "English. Each model uses its native tool format; tiny-aya, whose template has no tools, gets "
              "BenchMAX's JSON tool prompt."),
    "jsonmode": ("Structured output", "Structured output", "structured score",
                 "NousResearch json-mode-eval (100 requests); the Czech set is a Kimi K3 translation that keeps the "
                 "schema, the keys and the values that must be copied verbatim. Score = mean of valid JSON, valid "
                 "against the schema, and the share of gold fields reproduced. No constrained decoding."),
    "eurlex": ("Summarization", "Summarization", "judge score",
               "EUR-Lex-Sum, the same 50 EU legal acts in both languages (≤ 16k tokens). Prompt and listwise "
               "judge follow “Summarization Is Not Dead Yet” (arXiv 2606.08000): Kimi K3 scores informativeness, "
               "faithfulness, coherence and conciseness 1–5 against the source, with the official EUR-Lex summary "
               "in every group as a human reference. Score = harmonic mean of informativeness and faithfulness "
               "(coverage and precision), rescaled to 0–100, so a short but correct one-liner does not score well; "
               "a summary in the wrong language scores 0. Up to 2,048 output tokens. tiny-aya (8k context) sees "
               "truncated documents."),
    "minds": ("Classification", "Classification", "accuracy",
              "MInDS-14 banking intents, natively collected in en-US (563) and cs-CZ (574): pick one of 14 intents "
              "from the label list. Chance is 7 %."),
    "gmmlu": ("Knowledge", "Knowledge", "accuracy",
              "Global-MMLU-Lite, 400 parallel questions per language (human-translated Czech), 200 culturally "
              "sensitive and 200 culturally agnostic. Zero-shot, letter answer. Chance is 25 %."),
    "ifeval": ("Instruction following", "Instructions", "IFEval strict",
               "BenchMAX IFEval (429 prompts per language) with BenchMAX's language-aware checkers. Score = mean "
               "of prompt-level and instruction-level strict accuracy."),
}
# per-bench table columns: (metric key in scores.csv, column header, scale) - scale 1 = already 0-100
TABLE_METRICS = {
    "nexus": [("radar", "accuracy"), ("text_fallback", "% via JSON text"), ("errors", "errors")],
    "jsonmode": [("radar", "score"), ("valid_json", "valid JSON"), ("schema_valid", "schema"),
                 ("field_acc", "fields")],
    "eurlex": [("radar", "score"), ("informativeness", "inform."), ("faithfulness", "faithf."),
               ("coherence", "coher."), ("conciseness", "concise"), ("lang_ok", "right lang.")],
    "minds": [("radar", "accuracy"), ("truncated", "hit limit")],
    "gmmlu": [("radar", "accuracy"), ("cultural:CA", "agnostic"), ("cultural:CS", "sensitive")],
    "ifeval": [("prompt_strict", "prompt strict"), ("inst_strict", "inst. strict"), ("prompt_loose", "prompt loose"),
               ("inst_loose", "inst. loose")],
}
LIKERT = {"informativeness", "faithfulness", "coherence", "conciseness"}

# Light-palette paint for exported SVGs (a figure lands on white paper whatever the page theme was).
SVG_STYLES = {
    "ct-title": {"font-size": "13px", "font-weight": "600", "fill": "#1f211c"},
    "ct-lgd": {"font-size": "9.5px", "fill": "#585a54"},
    "ct-name": {"font-size": "11.5px", "font-weight": "600", "fill": "#1f211c"},
    "ct-cond": {"font-size": "9px", "fill": "#585a54"},
    "ct-cap": {"font-size": "9.5px", "fill": "#585a54"},
    "ct-axis": {"font-size": "10px", "font-weight": "600", "fill": "#585a54"},
    "ct-lang": {"font-size": "8.5px", "fill": "#585a54"},
    "ct-tick": {"font-size": "8.5px", "fill": "#585a54"},
    "ct-val": {"font-size": "9px", "fill": "#1f211c"},
    "ct-grid": {"stroke": "#d5d7d2", "stroke-width": "1", "fill": "none"},
    "ct-base": {"stroke": "#585a54", "stroke-width": "1", "opacity": "0.45"},
    "ct-band": {"fill": "#d5d7d2", "opacity": "0.35"},
    "ct-track": {"fill": "#d5d7d2", "opacity": "0.5"},
    "cs": {"fill": "#8b3fa8"}, "en": {"fill": "#0e6e7a"}, "ref": {"fill": "#585a54"},
    "rd-en": {"fill": "#0e6e7a", "fill-opacity": "0.10", "stroke": "#0e6e7a", "stroke-width": "2"},
    "rd-cs": {"fill": "#8b3fa8", "fill-opacity": "0.10", "stroke": "#8b3fa8", "stroke-width": "2"},
    "rd-dot-en": {"fill": "#0e6e7a", "stroke": "#eceee9", "stroke-width": "2"},
    "rd-dot-cs": {"fill": "#8b3fa8", "stroke": "#eceee9", "stroke-width": "2"},
    "rd-hit": {"fill": "transparent"},
}

esc = html.escape


def f1(x):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1f}"


# ------------------------------------------------------------------------------------------------ data

def load_scores(results: Path) -> dict:
    """{(model, task, lang): {metric: value}}"""
    out = defaultdict(dict)
    p = results / "scores.csv"
    if p.exists():
        for r in csv.DictReader(open(p)):
            out[(r["model"], r["task"], r["lang"])][r["metric"]] = float(r["value"])
    return out


def radar_values(scores, models):
    """{model: {task: {lang: value or None}}}"""
    return {m: {t: {l: scores.get((m, t, l), {}).get("radar") for l in LANGS} for t in TASKS} for m in models}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


# ---------------------------------------------------------------------------------------------- charts

def marker(lang: str, x: float, y: float, r: float = 4) -> str:
    """Shape is the second channel next to colour: the house teal/purple pair is only ΔE 7.2 apart under
    deuteranopia (validate_palette.js), which is legal only with a secondary encoding."""
    if lang == "en":
        return f"<circle class='rd-dot-en' cx='{x:.1f}' cy='{y:.1f}' r='{r}'/>"
    d = r * 1.3
    return f"<path class='rd-dot-cs' d='M{x:.1f} {y - d:.1f}L{x + d:.1f} {y:.1f}L{x:.1f} {y + d:.1f}L{x - d:.1f} {y:.1f}Z'/>"


def radar_svg(model: str, params: str, vals: dict) -> str:
    W, H, cx, cy, R = 320, 268, 160, 148, 88
    tasks = list(TASKS)
    n = len(tasks)

    def pt(i, v):
        a = -math.pi / 2 + 2 * math.pi * i / n
        return cx + R * v / 100 * math.cos(a), cy + R * v / 100 * math.sin(a)

    s = [f"<svg class='ctchart' viewBox='0 0 {W} {H}' role='img' "
         f"aria-label='{esc(model)}: score per skill in English and Czech'>"]
    s.append(f"<text class='ct-name' x='6' y='14'>{esc(model)}</text>")
    s.append(f"<text class='ct-cond' x='6' y='26'>{esc(params)}</text>")
    for k, lang in enumerate(LANGS):  # mean per language, the one direct label each polygon gets
        m = mean(vals[t][lang] for t in tasks)
        x = W - 78 + 40 * k
        s.append(marker(lang, x + 4, 9.5, 3.5) +
                 f"<text class='ct-lang' x='{x + 12}' y='13'>{lang}</text>"
                 f"<text class='ct-val' x='{x}' y='26'>{f1(m)}</text>")
    for ring in (25, 50, 75, 100):
        d = "M" + "L".join(f"{x:.1f} {y:.1f}" for x, y in (pt(i, ring) for i in range(n))) + "Z"
        s.append(f"<path class='ct-grid' d='{d}' fill='none'/>")
    for ring in (50, 100):
        x, y = pt(0, ring)
        s.append(f"<text class='ct-tick' x='{x + 3:.1f}' y='{y + 3:.1f}'>{ring}</text>")
    for i, t in enumerate(tasks):
        x, y = pt(i, 100)
        s.append(f"<line class='ct-grid' x1='{cx}' y1='{cy}' x2='{x:.1f}' y2='{y:.1f}'/>")
        lx, ly = pt(i, 114)
        anchor = "middle" if abs(lx - cx) < 8 else ("start" if lx > cx else "end")
        dy = -2 if ly < cy - R * 0.9 else (9 if ly > cy + R * 0.9 else 3.5)
        s.append(f"<text class='ct-axis' x='{lx:.1f}' y='{ly + dy:.1f}' text-anchor='{anchor}'>"
                 f"{esc(BENCH[t][1])}</text>")
    dots = []
    for lang in LANGS:
        pts = [pt(i, vals[t][lang] or 0) for i, t in enumerate(tasks)]
        if all(vals[t][lang] is None for t in tasks):
            continue
        d = "M" + "L".join(f"{x:.1f} {y:.1f}" for x, y in pts) + "Z"
        s.append(f"<path class='rd-{lang}' d='{d}'/>")
        for (x, y), t in zip(pts, tasks):
            v = vals[t][lang]
            tip = f"{model} · {LANG_NAME[lang]} · {BENCH[t][0]}: {f1(v) if v is not None else 'no data'}"
            dots.append(f"<g><title>{esc(tip)}</title><circle class='rd-hit' cx='{x:.1f}' cy='{y:.1f}' r='9'/>"
                        f"{marker(lang, x, y)}</g>")
    s += dots
    s.append("</svg>")
    return "".join(s)


def _bar(cls, x0, x1, y, h=8, r=2.5):
    if x1 - x0 < r:
        return f"<rect class='{cls}' x='{x0:.1f}' y='{y:.1f}' width='{max(x1 - x0, 0):.1f}' height='{h}'/>"
    return (f"<path class='{cls}' d='M{x0:.1f} {y:.1f}H{x1 - r:.1f}Q{x1:.1f} {y:.1f} {x1:.1f} {y + r:.1f}"
            f"V{y + h - r:.1f}Q{x1:.1f} {y + h:.1f} {x1 - r:.1f} {y + h:.1f}H{x0:.1f}Z'/>")


def bars_svg(title: str, rows: list, ref_rows: list = ()) -> str:
    """Horizontal bars per model, Czech above English, on a 0-100 axis (the BenchMAX viewer's layout).
    rows / ref_rows: [(name, sub, {"cs": v, "en": v})]."""
    W, x0, x1 = 720, 150, 640
    sc = lambda v: x0 + (x1 - x0) * max(0.0, min(v, 100)) / 100  # noqa: E731
    block = 34
    top = 40
    n = len(rows) + len(ref_rows)
    plot_h = n * block + (14 if ref_rows else 0)
    H = top + plot_h + 26
    s = [f"<svg class='ctchart' viewBox='0 0 {W} {H}' role='img' aria-label='{esc(title)}: score by model and language'>"]
    s.append(f"<text class='ct-title' x='6' y='13'>{esc(title)}</text>")
    for k, lang in enumerate(("cs", "en")):
        x = W - 130 + 60 * k
        s.append(f"<rect class='{lang}' x='{x}' y='5' width='11' height='9' rx='2'/>"
                 f"<text class='ct-lgd' x='{x + 15}' y='13'>{LANG_NAME[lang]}</text>")
    for v in (0, 25, 50, 75, 100):
        cls = "ct-base" if v == 0 else "ct-grid"
        s.append(f"<line class='{cls}' x1='{sc(v):.1f}' y1='{top - 6}' x2='{sc(v):.1f}' y2='{top + plot_h}'/>"
                 f"<text class='ct-tick' x='{sc(v):.1f}' y='{top + plot_h + 13}' text-anchor='middle'>{v}</text>")
    s.append(f"<text class='ct-tick' x='6' y='{top + plot_h + 13}'>score (0–100)</text>")
    y = top
    for i, (name, sub, vals) in enumerate(list(rows) + list(ref_rows)):
        if i == len(rows) and ref_rows:
            y += 14
            s.append(f"<rect class='ct-band' x='0' y='{y - 8}' width='{W}' height='{len(ref_rows) * block + 4}' rx='5'/>")
        s.append(f"<text class='ct-name' x='6' y='{y + 11}'>{esc(name)}</text>")
        if sub:
            s.append(f"<text class='ct-cond' x='6' y='{y + 23}'>{esc(sub)}</text>")
        for k, lang in enumerate(("cs", "en")):
            by = y + 2 + 11 * k
            v = vals.get(lang)
            s.append(f"<text class='ct-lang' x='{x0 - 18}' y='{by + 6.5}'>{lang}</text>"
                     f"<rect class='ct-track' x='{x0}' y='{by}' width='{x1 - x0}' height='8' rx='2'/>")
            if v is None:
                s.append(f"<text class='ct-val na' x='{x0 + 4}' y='{by + 6.5}'>n/a</text>")
                continue
            s.append(f"<g><title>{esc(name)} · {LANG_NAME[lang]}: {f1(v)}</title>{_bar(lang, x0, sc(v), by)}</g>"
                     f"<text class='ct-val' x='{sc(v) + 4:.1f}' y='{by + 6.5}'>{f1(v)}</text>")
        y += block
    s.append("</svg>")
    return "".join(s)


def figure(svg: str, name: str, cls: str = "chartblock ctfig") -> str:
    return (f"<figure class='{cls}' data-name='{esc(name)}'>{svg}<figcaption class='figfoot'>"
            f"<button class='svgdl' type='button'>download SVG</button></figcaption></figure>")


KEYROW = ("<div class='keyrow'><span><svg class='ctchart' width='12' height='12' viewBox='0 0 12 12' aria-hidden='true'>"
          f"{marker('en', 6, 6, 4)}</svg> English</span><span><svg class='ctchart' width='12' height='12' "
          f"viewBox='0 0 12 12' aria-hidden='true'>{marker('cs', 6, 6, 4)}</svg> Czech</span>"
          "<span class='dim'>hover a point for its value; every number is also in the table below</span></div>")


# ---------------------------------------------------------------------------------------------- tables

def overall_table(models, cfgs, rv) -> str:
    tasks = list(TASKS)
    cols = [(t, l) for t in tasks for l in LANGS]
    best = {c: max((rv[m][c[0]][c[1]] for m in models if rv[m][c[0]][c[1]] is not None), default=None) for c in cols}
    h = ["<div class='wrap'><table><tr><th rowspan='2'>model</th><th rowspan='2'>size</th>"]
    h += [f"<th class='grp{' pairstart' if i else ''}' colspan='2'>{esc(BENCH[t][0])}</th>" for i, t in enumerate(tasks)]
    h.append("<th class='grp pairstart' colspan='3'>mean</th></tr><tr>")
    h += [f"<th class='num{' pairstart' if l == 'en' else ''}'>{l}</th>" for t in tasks for l in LANGS]
    h.append("<th class='num pairstart'>en</th><th class='num'>cs</th><th class='num'>cs − en</th></tr>")
    for m in models:
        h.append(f"<tr><td><code>{esc(m)}</code></td><td class='dim'>{esc(cfgs[m].get('params', ''))}</td>")
        for t, l in cols:
            v = rv[m][t][l]
            cls = "num" + (" pairstart" if l == "en" else "") + (" best" if v is not None and v == best[(t, l)] else "")
            h.append(f"<td class='{cls}'>{f1(v)}</td>")
        me, mc = (mean(rv[m][t][l] for t in tasks) for l in LANGS)
        gap = mc - me if me is not None and mc is not None else None
        h.append(f"<td class='num pairstart'>{f1(me)}</td><td class='num'>{f1(mc)}</td>"
                 f"<td class='num gap'>{'' if gap is None else ('+' if gap > 0 else '')}{f1(gap)}</td></tr>")
    h.append("</table></div>")
    return "".join(h)


def bench_table(task, models, scores, extra_rows=()) -> str:
    mets = TABLE_METRICS[task]
    h = ["<div class='wrap'><table><tr><th rowspan='2'>model</th>"]
    h += [f"<th class='grp{' pairstart' if i else ''}' colspan='2'>{esc(lbl)}</th>" for i, (_, lbl) in enumerate(mets)]
    h.append("</tr><tr>" + "".join(f"<th class='num{' pairstart' if l == 'en' and i else ''}'>{l}</th>"
                                   for i, _ in enumerate(mets) for l in LANGS) + "</tr>")
    for m in list(models) + list(extra_rows):
        h.append(f"<tr><td><code>{esc(m)}</code></td>")
        for i, (k, _) in enumerate(mets):
            for l in LANGS:
                v = scores.get((m, task, l), {}).get(k)
                txt = "–" if v is None else (f"{v:.2f}" if k in LIKERT else (f"{v:.0f}" if k in ("errors", "truncated") else f1(v)))
                h.append(f"<td class='num{' pairstart' if l == 'en' and i else ''}'>{txt}</td>")
        h.append("</tr>")
    h.append("</table></div>")
    return "".join(h)


# -------------------------------------------------------------------------------------- sample browser

def item_view(task, it) -> dict:
    """What a card shows for an item: q (input), t (reference), m (meta line)."""
    g = it["gold"]
    if task == "nexus":
        tools = ", ".join(t["function"]["name"] for t in it["tools"])
        args = {k: v for k, v in g["arguments"].items() if v is not None}
        return {"q": it["messages"][-1]["content"] + f"\n\n[tools: {tools}]",
                "t": f"{g['name']}({json.dumps(args, ensure_ascii=False)})", "m": g["source"]}
    if task == "jsonmode":
        return {"q": it["messages"][-1]["content"] + "\n\n[schema] " + json.dumps(g["schema"], ensure_ascii=False),
                "t": json.dumps(g["json"], ensure_ascii=False, indent=1)}
    if task == "eurlex":
        doc = it["document"]
        q = doc[:6000] + (f"\n\n[… {len(doc) - 6000:,} more characters]" if len(doc) > 6000 else "")
        return {"q": q, "t": g["reference_summary"], "m": f"CELEX {it['id']} · {len(doc):,} characters"}
    if task == "minds":
        text = it["messages"][-1]["content"].split("\n\n")[-2].split(": ", 1)[-1].strip("\"„“")
        return {"q": text, "t": g["label"]}
    if task == "gmmlu":
        body = it["messages"][-1]["content"].split("\n\n", 1)[1].rsplit("\n\n", 1)[0]
        return {"q": body, "t": g["answer"], "m": f"{g['category']} · {'culturally sensitive' if g['cultural'] == 'CS' else 'culturally agnostic'}"}
    return {"q": it["messages"][-1]["content"], "t": "\n".join(g["instruction_id_list"])}


def output_view(task, out, sc) -> tuple[dict, int, int]:
    """(card payload {o, x}, ok flag, empty flag) for one model's answer."""
    if out is None:
        return None, -2, 0
    if "error" in out:
        return {"o": "", "x": "generation error: " + out["error"][:300]}, 0, 1
    parts = [f"→ {tc['name']}({tc['arguments']})" for tc in out.get("tool_calls") or []]
    if out.get("content"):
        parts.append(out["content"])
    o = "\n".join(parts)
    x = []
    s = sc.get("score") if sc else None
    if task == "nexus" and sc:
        x.append("parsed from text" if sc.get("via") == "text" else ("native tool call" if sc.get("via") else "no call found"))
    elif task == "jsonmode" and sc:
        x.append(f"JSON {'✓' if sc.get('valid_json') else '✗'} · schema {'✓' if sc.get('schema_valid') else '✗'} · "
                 f"fields {100 * sc.get('field_acc', 0):.0f}%")
    elif task == "eurlex" and sc:
        dims = " ".join(f"{d[:3].lower()} {sc[d]:.1f}" for d in ("Informativeness", "Faithfulness", "Coherence", "Conciseness") if d in sc)
        x.append(f"score {100 * (s or 0):.0f}" + (f" · {dims}" if dims else " · not judged yet") +
                 ("" if sc.get("lang_ok", 1) else " · wrong language"))
        if out.get("doc_kept") is not None and out["doc_kept"] < 1:
            x.append(f"document truncated to {100 * out['doc_kept']:.0f}%")
    elif task in ("minds", "gmmlu") and sc:
        x.append(f"parsed: {sc.get('pred') or '—'}")
    elif task == "ifeval" and sc:
        inst = sc.get("inst_strict") or []
        x.append(f"{sum(inst)}/{len(inst)} instructions followed")
    if out.get("finish_reason") == "length":
        x.append("hit the token limit")
    if out.get("completion_tokens") is not None:
        x.append(f"{out['completion_tokens']} tokens")
    if task == "eurlex":
        ok = -1
    elif s is None:
        ok = -1
    else:
        ok = int(s >= 0.999)
    return {"o": o, "x": " · ".join(x)}, ok, int(not o.strip())


def write_js(path: Path, key: str, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"BM.put({json.dumps(key)},{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))});\n",
                    encoding="utf-8")


def build_samples(site: Path, models, outputs: Path, results: Path) -> dict:
    """Write data/<task>/<lang>/... and return the views {task: {lang: [models with outputs]}}."""
    views = {}
    for task in TASKS:
        views[task] = {}
        for lang in LANGS:
            dp = data_path(task, lang)
            if not dp.exists():
                continue
            items = read_jsonl(dp)
            outs, scs = {}, {}
            for m in models:
                p = outputs / m / f"{task}.{lang}.jsonl"
                if p.exists():
                    outs[m] = {r["id"]: r for r in read_jsonl(p)}
                    sp = results / "items" / m / f"{task}.{lang}.jsonl"
                    scs[m] = {r["id"]: r for r in read_jsonl(sp)} if sp.exists() else {}
            have = [m for m in models if m in outs]
            views[task][lang] = have
            keep = [it for it in items if any(it["id"] in outs[m] for m in have)] or items
            meta = {"items": [], "ok": {m: [] for m in have}, "empty": {m: [] for m in have},
                    "summary": {}, "total": len(keep), "chunk": CHUNK}
            chunks = []
            for i, it in enumerate(keep):
                meta["items"].append({"id": it["id"], **item_view(task, it)})
                row = {}
                for m in have:
                    payload, ok, empty = output_view(task, outs[m].get(it["id"]), scs[m].get(it["id"]))
                    meta["ok"][m].append(ok)
                    meta["empty"][m].append(empty)
                    if payload is not None:
                        row[m] = payload
                if i % CHUNK == 0:
                    chunks.append([])
                chunks[-1].append(row)
            for m in have:
                flags = meta["ok"][m]
                meta["summary"][m] = {"graded": sum(f in (0, 1) for f in flags), "ok": sum(f == 1 for f in flags),
                                      "empty": sum(meta["empty"][m])}
            base = site / "data" / task / lang
            if base.exists():
                shutil.rmtree(base)
            write_js(base / "meta.js", f"{task}|{lang}|meta", meta)
            for k, rows in enumerate(chunks):
                write_js(base / f"c{k}.js", f"{task}|{lang}|{k}", rows)
    return views


# ------------------------------------------------------------------------------------------------ page

MAST = ("<header class=mast><a class=brand href='https://lokalni.ai' aria-label='lokalni.ai'>"
        "<img class=logo-light src='assets/logo_dark.png' alt='lokalni.ai' width=102 height=32>"
        "<img class=logo-dark src='assets/logo_light.png' alt='lokalni.ai' width=102 height=32></a>"
        "<span class=mastsep>&frasl;</span><a class=proj href='{url}'>{title}</a><span class=spacer></span>"
        "<button class=tgl type=button data-theme-toggle aria-label='Switch theme' title='Switch theme'>"
        "<svg class=sun viewBox='0 0 24 24' aria-hidden=true><circle cx=12 cy=12 r=4 /><path d='M12 2v2M12 20v2M4.9 "
        "4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M6.3 17.7l-1.4 1.4M19.1 4.9l-1.4 1.4' /></svg>"
        "<svg class=moon viewBox='0 0 24 24' aria-hidden=true><path d='M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z' />"
        "</svg></button></header>")

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Malé modely pro koncová zařízení — results</title>
<link rel="icon" href="assets/favicon.ico" sizes="any">
<script>try{var t=localStorage.getItem('theme');if(t==='dark')document.documentElement.setAttribute('data-theme','dark');}catch(e){}</script><link rel=preconnect href='https://fonts.googleapis.com'><link rel=preconnect href='https://fonts.gstatic.com' crossorigin><link rel=stylesheet href='https://fonts.googleapis.com/css2?family=Barlow:ital,wght@0,300;0,400;0,500;0,600;1,400&family=Barlow+Condensed:wght@600;700&family=JetBrains+Mono:ital,wght@0,400;0,500;1,400&display=swap'>
<link rel="stylesheet" href="style.css?v={v}">
</head>
<body>
"""

FOOT = """</main>
<script src="app.js?v={v}"></script>
<script src="data/index.js?v={v}"></script>
<script>BM.start();</script>
<script>document.addEventListener('click',function(e){{var b=e.target.closest&&e.target.closest('[data-theme-toggle]');if(!b)return;var d=document.documentElement,n=d.getAttribute('data-theme')==='dark'?'light':'dark';if(n==='dark')d.setAttribute('data-theme','dark');else d.removeAttribute('data-theme');try{{localStorage.setItem('theme',n);}}catch(err){{}}}});</script>
</body>
</html>
"""

METHOD = """<div class='caveat'><ul>
<li><b>Models.</b> Instruction-tuned BF16 checkpoints, served with vLLM 0.31, greedy decoding, the checkpoint's own chat template with <code>enable_thinking=False</code>, and its native tool-call parser.</li>
<li><b>Prompts.</b> Every input is fully in the test language; tool descriptions and JSON schemas stay in English. The Czech json-mode set and the Czech summarization prompt are Kimi K3 translations; everything else is human-made or human-checked Czech.</li>
<li><b>Judge.</b> Summaries only: Kimi K3 (e-INFRA CZ), reasoning effort high, one judge, listwise with hidden and shuffled system names.</li>
</ul></div>"""


def build(outputs: Path, results: Path, site: Path):
    cfgs = model_configs()
    models = list(cfgs)
    scores = load_scores(results)
    rv = radar_values(scores, models)
    site.mkdir(parents=True, exist_ok=True)
    static = ROOT / "viewer" / "static"
    for f in ("style.css", "app.js"):
        shutil.copy(static / f, site / f)
    shutil.copytree(static / "assets", site / "assets", dirs_exist_ok=True)
    views = build_samples(site, models, outputs, results)
    v = format(abs(hash((tuple(sorted(map(str, scores.items()))), (static / "app.js").stat().st_mtime))) % 16**8, "08x")

    page = [HEAD.replace("{v}", v), MAST.format(url=esc(POST_URL), title=esc(POST_TITLE)),
            f"<div class='sub lede'>{esc(LEDE)}</div>", "<nav class='tabs'><button class='tabbtn' data-tab='overall'>overall</button>"]
    page += [f"<button class='tabbtn' data-tab='{t}'>{esc(BENCH[t][0])}</button>" for t in TASKS]
    page.append("</nav><main>")

    # overall
    page.append("<section class='tabpane' data-tab='overall'>" + KEYROW + "<div class='radargrid'>")
    page += [figure(radar_svg(m, cfgs[m].get("params", ""), rv[m]), f"radar-{m}") for m in models]
    page.append("</div><h2>models side by side</h2>" + overall_table(models, cfgs, rv))
    gap_rows = [(m, cfgs[m].get("params", ""), {l: mean(rv[m][t][l] for t in TASKS) for l in LANGS}) for m in models]
    page.append("<h2>mean over the six skills</h2><div class='viz'>" +
                figure(bars_svg("Mean score over the six skills", gap_rows), "mean-by-language") + "</div>")
    page.append("<h2>how it was run</h2>" + METHOD + "</section>")

    # one tab per skill
    for t in TASKS:
        label, _, _, desc = BENCH[t]
        rows = [(m, cfgs[m].get("params", ""), {l: rv[m][t][l] for l in LANGS}) for m in models]
        ref = []
        if t == "eurlex" and any(("human", t, l) in scores for l in LANGS):
            ref = [("official EUR-Lex summary", "human-written", {l: scores.get(("human", t, l), {}).get("radar") for l in LANGS})]
        page.append(f"<section class='tabpane' data-tab='{t}'><div class='sub'>{esc(desc)}</div>")
        page.append("<div class='viz'>" + figure(bars_svg(label, rows, ref), f"{t}-by-model") + "</div>")
        page.append(bench_table(t, models, scores, ["human"] if ref else []))
        page.append("<h2>samples</h2><div class='ctlbox'></div><div class='body'></div></section>")
    page.append(FOOT.replace("{v}", v).replace("{{", "{").replace("}}", "}"))
    (site / "index.html").write_text("".join(page), encoding="utf-8")

    # index.js for the browser
    bench_meta = [{"id": t, "label": BENCH[t][0], "langs": list(LANGS), "datasets": [], "lang_labels": LANG_NAME,
                   "note": "", "views": views[t]} for t in TASKS]
    sc_idx = {t: {l: {} for l in LANGS} for t in TASKS}
    for t in TASKS:
        for l in LANGS:
            for m in models:
                s = scores.get((m, t, l))
                if s and "radar" in s:
                    d = {BENCH[t][2]: s["radar"] / 100}
                    for k, lbl in TABLE_METRICS[t][1:]:
                        if k in s and k not in LIKERT and k not in ("errors", "truncated"):
                            d[lbl] = s[k] / 100
                    sc_idx[t][l][m] = d
    index = {"models": [{"tag": m, "params": cfgs[m].get("params", "")} for m in models], "order": models,
             "default_models": DEFAULT_MODELS, "benches": bench_meta, "scores": sc_idx,
             "svg_export": {"styles": SVG_STYLES, "font": "Barlow, Helvetica, Arial, sans-serif"}}
    write_js(site / "data" / "index.js", "index", index)
    n_files = sum(1 for _ in (site / "data").rglob("*.js"))
    size = sum(f.stat().st_size for f in site.rglob("*") if f.is_file())
    print(f"wrote {site}/index.html, {n_files} data files, {size / 1e6:.1f} MB total")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUTPUTS)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--site", type=Path, default=ROOT / "viewer" / "site")
    args = ap.parse_args()
    build(args.out, args.results, args.site)


if __name__ == "__main__":
    main()
