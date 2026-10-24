"""Score generations and write results/scores.csv (long format) plus per-item details.

    python -m slmbench.score [--out outputs] [--results results]

Every axis is on a 0-100 scale. Items with generation errors or missing outputs score 0.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from rapidfuzz import fuzz

from .common import (DATA, LANGS, OUTPUTS, RESULTS, ROOT, TASKS, data_path, load_yaml, read_jsonl, strip_think,
                     write_jsonl)
from .generate import model_configs
from .judge import DIMS, HUMAN
from .prepare import fill_args

AXES = {"nexus": "Tool calling", "jsonmode": "Structured output", "eurlex": "Summarization",
        "minds": "Classification", "gmmlu": "Knowledge", "ifeval": "Instruction following"}

# ------------------------------------------------------------------------------------------------ nexus

_NEXUS_FUNCS = None


def _funcs():
    global _NEXUS_FUNCS
    if _NEXUS_FUNCS is None:
        _NEXUS_FUNCS = json.loads((DATA / "nexus_functions.json").read_text(encoding="utf-8"))
    return _NEXUS_FUNCS


def parse_call(out: dict):
    """First native tool call; otherwise a JSON {"name", "parameters"|"arguments"} in the text
    (BenchMAX GenericOutputParser), which is how tiny-aya is prompted and also rescues near-misses."""
    if out.get("tool_calls"):
        tc = out["tool_calls"][0]
        return tc["name"], json.loads(tc["arguments"] or "{}"), "native"
    text = strip_think(out.get("content"))
    for pat in (r"```json([\s\S]*?)```", r"```([\s\S]*?)```", r"({[\s\S]*})"):
        m = re.search(pat, text)
        if not m:
            continue
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            args = obj.get("parameters", obj.get("arguments", {}))
            if isinstance(args, dict):
                return obj["name"], args, "text"
    return None, None, None


def score_nexus(it, out):
    try:
        name, args, via = parse_call(out)
    except (json.JSONDecodeError, TypeError):
        return {"score": 0.0, "via": "native", "error": "bad arguments json"}
    gold = it["gold"]
    ok = False
    if name == gold["name"]:
        try:
            ok = fill_args(_funcs()[name], args) == gold["arguments"]
        except TypeError:
            ok = False
    return {"score": float(ok), "via": via, "pred_name": name}


# --------------------------------------------------------------------------------------------- jsonmode

def extract_json(text: str):
    text = strip_think(text)
    cands = [m.group(1) for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", text)] + [text]
    for c in cands:
        c = c.strip()
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            pass
        i = c.find("{")
        if i >= 0:
            try:
                return json.JSONDecoder().raw_decode(c[i:])[0]
            except json.JSONDecodeError:
                pass
    raise ValueError("no JSON found")


def leaves(x, path=()):
    if isinstance(x, dict) and x:
        for k, v in x.items():
            yield from leaves(v, path + (k,))
    elif isinstance(x, list) and x:
        for i, v in enumerate(x):
            yield from leaves(v, path + (i,))
    else:
        yield path, x


def _norm(s: str) -> str:
    return " ".join(s.casefold().split())


def leaf_match(g, p) -> bool:
    if isinstance(g, bool) or isinstance(p, bool) or g is None or p is None:
        return g == p and type(g) is type(p)
    if isinstance(g, (int, float)) and isinstance(p, (int, float)):
        return float(g) == float(p)
    if isinstance(g, str) and isinstance(p, str):
        return _norm(g) == _norm(p) or (len(g) > 20 and fuzz.ratio(_norm(g), _norm(p)) >= 90)
    return g == p


def _unwrap(pred, gold):
    """Models often wrap the object in one extra key named after the schema title ({"WirelessAccessPoint": {...}}).
    That stays a schema failure, but field accuracy is computed on the inner object, so a correct extraction is
    not scored like nonsense."""
    if isinstance(pred, dict) and len(pred) == 1 and isinstance(gold, dict):
        (k, v), = pred.items()
        if k not in gold:
            if isinstance(v, list) and len(v) == 1:
                v = v[0]
            if isinstance(v, dict):
                return v
    return pred


def score_jsonmode(it, out):
    import jsonschema

    try:
        pred = extract_json(out.get("content"))
    except ValueError:
        return {"score": 0.0, "valid_json": 0, "schema_valid": 0, "field_acc": 0.0}
    try:
        jsonschema.validate(pred, it["gold"]["schema"])
        schema_ok = 1
    except jsonschema.ValidationError:
        schema_ok = 0
    except jsonschema.SchemaError:
        schema_ok = int(isinstance(pred, dict))
    gold_json = it["gold"]["json"]
    gold = dict(leaves(gold_json))
    pl = dict(leaves(_unwrap(pred, gold_json)))
    acc = sum(leaf_match(v, pl[k]) for k, v in gold.items() if k in pl) / max(len(gold), 1)
    return {"score": (1 + schema_ok + acc) / 3, "valid_json": 1, "schema_valid": schema_ok, "field_acc": acc}


# ----------------------------------------------------------------------------------------------- eurlex

_JUDGE = {}


def _judged(lang):
    if lang not in _JUDGE:
        p = RESULTS / "judge" / f"eurlex.{lang}.jsonl"
        _JUDGE[lang] = {}
        for r in read_jsonl(p) if p.exists() else []:
            for m, s in r["scores"].items():
                _JUDGE[lang].setdefault((r["id"], m), []).append(s)  # human appears in every group
    return _JUDGE[lang]


def detect_lang(text: str) -> str:
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0
    try:
        return detect(text)
    except Exception:  # noqa: BLE001 - empty or symbol-only text
        return "unk"


def summary_score(lang, doc_id, model, text) -> dict:
    """Harmonic mean of informativeness (coverage) and faithfulness (precision), rescaled from 1-5 to 0-1;
    0 when the summary is empty, unjudged or not in the target language.

    Not the mean of all four dimensions: in the pilot, a one-sentence "summary" got informativeness 1 but 5 on
    faithfulness, coherence and conciseness, and so outranked the official summary. The harmonic mean needs both
    coverage and correctness; coherence and conciseness are reported alongside."""
    text = strip_think(text)
    if not text:
        return {"score": 0.0, "lang_ok": 0, "judged": 0}
    lang_ok = int(detect_lang(text) == lang)
    s = _judged(lang).get((doc_id, model))
    if not s:
        return {"score": 0.0, "lang_ok": lang_ok, "judged": 0}
    dims = {d: sum(x[d] for x in s) / len(s) for d in DIMS}
    i, f = dims["Informativeness"], dims["Faithfulness"]
    content = 2 * i * f / (i + f)
    return {"score": lang_ok * (content - 1) / 4, "lang_ok": lang_ok, "judged": 1, **dims}


def score_eurlex(it, out, model):
    return summary_score(it["lang"], it["id"], model, out.get("content"))


# ------------------------------------------------------------------------------------------------ minds

def parse_label(text: str, labels: list[str]) -> str | None:
    t = re.sub(r"[\s\-]+", "_", strip_think(text).casefold()).strip("_`*\"'„“”.,:;()[]")
    if t in labels:
        return t
    hits = [(m.start(), lab) for lab in labels for m in re.finditer(rf"(?<![a-z]){lab}(?![a-z])", t)]
    return min(hits)[1] if hits else None


def score_minds(it, out, labels):
    pred = parse_label(out.get("content") or "", labels)
    return {"score": float(pred == it["gold"]["label"]), "pred": pred}


# ------------------------------------------------------------------------------------------------ gmmlu

_LEAD = re.compile(r"^\W*\(?([ABCD])(?:$|[.):\]*]|\s*\n|\s+[-–])")
_SAID = re.compile(r"(?i:answer|odpověď|odpověd\w*|správn\w*)\W{0,15}(?i:is|je|zní)?\W{0,10}\(?([ABCD])(?![\w])")
_BOLD = re.compile(r"\*\*\(?([ABCD])[.)]?\*\*")


def parse_letter(text: str) -> str | None:
    """Leading letter ("C", "C.", "**C**", "(C)"), else the last explicit "answer is X" / "odpověď je X", else
    the last bold option. Never a bare standalone letter: in Czech "A" is also the word "and"."""
    t = strip_think(text)
    m = _LEAD.match(t)
    if m:
        return m.group(1)
    for pat in (_SAID, _BOLD):
        hits = pat.findall(t)
        if hits:
            return hits[-1]
    return None


def score_gmmlu(it, out):
    pred = parse_letter(out.get("content") or "")
    return {"score": float(pred == it["gold"]["answer"]), "pred": pred,
            "cultural": it["gold"]["cultural"], "category": it["gold"]["category"]}


# ----------------------------------------------------------------------------------------------- ifeval

def _ifeval():
    sys.path.insert(0, str(ROOT / "third_party" / "benchmax_ifeval"))
    os.environ.setdefault("NLTK_DATA", str(Path(os.environ.get("WD_CACHE_DIR", Path.home() / ".cache")) / "nltk_data"))
    import instructions_registry  # noqa: E402

    return instructions_registry.INSTRUCTION_DICT


def _follows(registry, gold, response) -> list[bool]:
    """BenchMAX tasks/ifeval/utils.py:test_instruction_following_strict, condensed."""
    res = []
    for i, iid in enumerate(gold["instruction_id_list"]):
        inst = registry[iid](iid, gold["lang"])
        inst.build_description(**{k: v for k, v in gold["kwargs"][i].items() if v})
        args = inst.get_instruction_args()
        if args and "prompt" in args:
            inst.build_description(prompt=gold["prompt"])
        res.append(bool(response.strip() and inst.check_following(response)))
    return res


def _loose_variants(response: str) -> list[str]:
    r = response.split("\n")
    base = [response, "\n".join(r[1:]).strip(), "\n".join(r[:-1]).strip(), "\n".join(r[1:-1]).strip()]
    return base + [x.replace("*", "") for x in base]


def score_ifeval(it, out, registry):
    resp = strip_think(out.get("content"))
    strict = _follows(registry, it["gold"], resp)
    variants = [_follows(registry, it["gold"], v) for v in _loose_variants(resp)]
    loose = [any(v[i] for v in variants) for i in range(len(strict))]
    return {"score": float(all(strict)), "inst_strict": strict, "prompt_loose": float(all(loose)), "inst_loose": loose}


# ------------------------------------------------------------------------------------------------- main

def aggregate(task, rows) -> dict:
    n = len(rows)
    mean = lambda k: 100 * sum(r[k] for r in rows) / n  # noqa: E731
    agg = {"radar": mean("score")}
    if task == "ifeval":
        flat = lambda k: [x for r in rows for x in r[k]]  # noqa: E731
        agg["prompt_strict"] = agg["radar"]
        agg["inst_strict"] = 100 * sum(flat("inst_strict")) / len(flat("inst_strict"))
        agg["prompt_loose"] = mean("prompt_loose")
        agg["inst_loose"] = 100 * sum(flat("inst_loose")) / len(flat("inst_loose"))
        agg["radar"] = (agg["prompt_strict"] + agg["inst_strict"]) / 2
    elif task == "jsonmode":
        agg |= {k: mean(k) for k in ("valid_json", "schema_valid", "field_acc")}
    elif task == "nexus":
        agg["text_fallback"] = 100 * sum(r.get("via") == "text" for r in rows) / n
    elif task == "eurlex":
        agg["lang_ok"] = mean("lang_ok")
        judged = [r for r in rows if r.get("judged")]
        for d in DIMS:
            agg[d.lower()] = sum(r[d] for r in judged) / len(judged) if judged else float("nan")
    elif task == "gmmlu":
        for key in ("cultural", "category"):
            for v in sorted({r[key] for r in rows}):
                sub = [r for r in rows if r[key] == v]
                agg[f"{key}:{v}"] = 100 * sum(r["score"] for r in sub) / len(sub)
    agg["errors"] = sum("gen_error" in r for r in rows)
    agg["truncated"] = sum(r.get("finish_reason") == "length" for r in rows)
    return agg


def main():
    global RESULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUTPUTS)
    ap.add_argument("--results", type=Path, default=RESULTS)
    args = ap.parse_args()
    RESULTS = args.results  # also where the judge results are read from
    labels = list(load_yaml("prompts.yaml")["minds"]["labels"])
    registry = _ifeval()
    table = []
    for model in model_configs():
        for task in TASKS:
            for lang in LANGS:
                p = args.out / model / f"{task}.{lang}.jsonl"
                if not p.exists():
                    continue
                outs = {r["id"]: r for r in read_jsonl(p)}
                items = [it for it in read_jsonl(data_path(task, lang)) if it["id"] in outs]
                rows = []
                for it in items:
                    out = outs[it["id"]]
                    extra = {}
                    if "error" in out:  # scored as an empty answer
                        extra = {"gen_error": out["error"]}
                        out = {"content": ""}
                    if task == "nexus":
                        r = score_nexus(it, out)
                    elif task == "jsonmode":
                        r = score_jsonmode(it, out)
                    elif task == "eurlex":
                        r = score_eurlex(it, out, model)
                    elif task == "minds":
                        r = score_minds(it, out, labels)
                    elif task == "gmmlu":
                        r = score_gmmlu(it, out)
                    else:
                        r = score_ifeval(it, out, registry)
                    rows.append({"id": it["id"], "finish_reason": out.get("finish_reason"), **r, **extra})
                write_jsonl(RESULTS / "items" / model / f"{task}.{lang}.jsonl", rows)
                for metric, value in aggregate(task, rows).items():
                    table.append({"model": model, "task": task, "axis": AXES[task], "lang": lang,
                                  "metric": metric, "value": round(value, 3), "n": len(rows)})
    # official EUR-Lex summaries as a human reference point
    for lang in LANGS:
        items = read_jsonl(data_path("eurlex", lang))
        rows = [summary_score(lang, it["id"], HUMAN, it["gold"]["reference_summary"]) for it in items]
        rows = [r for r in rows if r["judged"]]  # only documents the judge has seen
        if rows:
            for metric, value in aggregate("eurlex", rows).items():
                table.append({"model": HUMAN, "task": "eurlex", "axis": AXES["eurlex"], "lang": lang,
                              "metric": metric, "value": round(value, 3), "n": len(rows)})
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "task", "axis", "lang", "metric", "value", "n"])
        w.writeheader()
        w.writerows(table)
    radar = defaultdict(dict)
    for r in table:
        if r["metric"] == "radar":
            radar[r["model"]][f"{r['task']}.{r['lang']}"] = r["value"]
    for m, v in radar.items():
        print(f"{m:22s} " + "  ".join(f"{k}={x:5.1f}" for k, x in v.items()))


if __name__ == "__main__":
    main()
