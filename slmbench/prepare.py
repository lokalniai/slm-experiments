"""Build the frozen evaluation sets in data/<task>.<lang>.jsonl.

    python -m slmbench.prepare all            # everything (Kimi translations are cached)
    python -m slmbench.prepare nexus minds    # selected tasks

Each item: {"id", "task", "lang", "messages", "max_tokens", "gold", optional "tools", optional "document"}.
If "document" is present, it replaces "{document}" in the last message at generation time, after truncation
to the model's context if needed.
"""

import asyncio
import json
import random
import re
import sys

import pandas as pd
from huggingface_hub import hf_hub_download

from .common import CONFIGS, DATA, LANGS, TASKS, data_path, fill, load_yaml, read_jsonl, write_jsonl

SEED = 0
PROMPTS = load_yaml("prompts.yaml")


def item(task, lang, id_, messages, gold, **extra):
    return {"id": str(id_), "task": task, "lang": lang, "messages": messages,
            "max_tokens": TASKS[task], "gold": gold, **extra}


def hf_file(repo, filename, revision="main"):
    return hf_hub_download(repo, filename, repo_type="dataset", revision=revision)


# ---------------------------------------------------------------------------------------------- nexus

_JSON_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean",
               "list": "array", "dict": "object"}


def _json_type(type_str: str) -> str | None:
    t = type_str.strip()
    if "datetime" in t:
        return "string"
    return _JSON_TYPES.get(re.split(r"[\[|]", t)[0].strip().lower())  # List[str] -> array, unknown -> any


def nexus_functions() -> dict:
    """NexusRaven API list as OpenAI tools plus the argument specs used for scoring.

    Mirrors BenchMAX tasks/nexus/utils.py:build_functions (description = the docstring, per-parameter
    descriptions empty, optional parameters default to None) without exec-ing generated code.
    """
    df = pd.read_parquet(hf_file("Nexusflow/NexusRaven_API_evaluation",
                                 "standardized_api_list/train-00000-of-00001-16cfa01b84238b38.parquet"))
    funcs = {}
    for _, r in df.iterrows():
        desc = r["description"].strip("\n")
        if desc.startswith("    "):
            desc = desc.strip().replace("\n    ", "\n")
        props, required, defaults = {}, [], {}
        for a in sorted(r["args_dicts"], key=lambda a: a["required"], reverse=True):
            type_str = a["type"]
            if type_str == "None":  # type only given in the docstring, e.g. "cveId (str)"
                m = re.search(re.escape(a["name"]) + r" \((.+?)\)", r["description"])
                type_str = m.group(1) if m else ""
            jt = _json_type(type_str)
            props[a["name"]] = ({"type": jt} if jt else {}) | {"description": ""}
            if a["required"]:
                required.append(a["name"])
            else:
                d = a["default"]
                if isinstance(d, str) and d.startswith("JSON:"):
                    d = json.loads(d.removeprefix("JSON:"))
                elif d == "None":
                    d = None
                defaults[a["name"]] = d
        funcs[r["name"]] = {
            "tool": {"type": "function", "function": {
                "name": r["name"], "description": desc,
                "parameters": {"type": "object", "properties": props, "required": required}}},
            "required": required, "defaults": defaults,
        }
    return funcs


def fill_args(spec: dict, args: dict) -> dict:
    """Bind call arguments like Python would (BenchMAX compares `locals()` of the called stub)."""
    allowed = set(spec["required"]) | set(spec["defaults"])
    unknown = set(args) - allowed
    missing = set(spec["required"]) - set(args)
    if unknown or missing:
        raise TypeError(f"unknown={sorted(unknown)} missing={sorted(missing)}")
    return {**spec["defaults"], **args}


def prep_nexus():
    funcs = nexus_functions()
    with open(DATA / "nexus_functions.json", "w", encoding="utf-8") as f:
        json.dump(funcs, f, ensure_ascii=False, indent=1)
    for lang in LANGS:
        rows = read_jsonl(hf_file("LLaMAX/BenchMAX_Multiple_Functions", f"nexus_{lang}.jsonl"))
        out = []
        for i, r in enumerate(rows):
            name = r["python_function_name"]
            gold_args = fill_args(funcs[name], json.loads(r["python_args_dict"]))
            out.append(item("nexus", lang, i, [{"role": "user", "content": r["prompt"]}],
                            {"name": name, "arguments": gold_args, "source": r["dataset"]},
                            tools=[funcs[k]["tool"] for k in r["context_functions"]]))
        write_jsonl(data_path("nexus", lang), out)


# ------------------------------------------------------------------------------------------- jsonmode

JSONMODE_TRANSLATE = """You are a professional English-to-Czech translator preparing the Czech version of a benchmark in which a model must turn a user's request into a JSON object that follows a given JSON schema.

You get the JSON schema, the user's message in English, and the gold JSON answer. Return a JSON object {"user": ..., "gold": ...}:
- "user": the user's message translated into natural, fluent Czech, as a Czech user would write it. Keep all facts, numbers and the level of detail.
- "gold": the gold answer adapted to your Czech message. Keep exactly the same keys, nesting, types and non-string values. Translate a string value only if it is natural-language text that you translated in the user message, and then use exactly the same wording as in your Czech message, so that the value can still be extracted from it. Copy all other strings unchanged.

Never translate: JSON keys and field names, enum values allowed by the schema, identifiers, codes, IDs, URLs, e-mail addresses, file names, units, product and brand names, or names of people and organisations.

Output only the JSON object, with no code fences or commentary.

JSON schema:
{schema}

User message:
{user}

Gold answer:
{gold}"""


def _same_shape(a, b) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same_shape(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(map(_same_shape, a, b))
    if isinstance(a, str):
        return isinstance(b, str)
    return a == b  # numbers, booleans and null must be copied unchanged


async def _translate_jsonmode(rows):
    import jsonschema

    from .kimi import Kimi

    kimi = Kimi("jsonmode_cs")

    async def one(i, r):
        schema = json.loads(r["schema"])
        gold = json.loads(r["completion"])

        def validate(content):
            out = json.loads(content)
            if not isinstance(out.get("user"), str) or not _same_shape(gold, out.get("gold")):
                raise ValueError("translated gold changed shape")
            jsonschema.validate(out["gold"], schema)

        prompt = fill(JSONMODE_TRANSLATE, schema=json.dumps(schema, ensure_ascii=False),
                      user=r["prompt"][1]["content"], gold=json.dumps(gold, ensure_ascii=False))
        rec = await kimi.chat([{"role": "user", "content": prompt}], effort="low", validate=validate)
        return i, json.loads(rec["content"])

    done = await asyncio.gather(*(one(i, r) for i, r in enumerate(rows)), return_exceptions=True)
    return done


def prep_jsonmode():
    df = pd.read_parquet(hf_file("NousResearch/json-mode-eval", "data/train-00000-of-00001.parquet"))
    rows = df.to_dict("records")
    for r in rows:
        r["prompt"] = list(r["prompt"])
    pre = PROMPTS["jsonmode"]["system_preamble"]

    def make(lang, i, r, user, gold):
        system = r["prompt"][0]["content"]
        assert system.startswith(pre["en"]), system[:100]
        system = pre[lang] + system[len(pre["en"]):]
        return item("jsonmode", lang, i, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    {"json": gold, "schema": json.loads(r["schema"])})

    write_jsonl(data_path("jsonmode", "en"),
                [make("en", i, r, r["prompt"][1]["content"], json.loads(r["completion"])) for i, r in enumerate(rows)])
    out, failed = [], []
    for res in asyncio.run(_translate_jsonmode(rows)):
        if isinstance(res, Exception):
            failed.append(str(res))
            continue
        i, tr = res
        out.append(make("cs", i, rows[i], tr["user"], tr["gold"]))
    if failed:
        print(f"jsonmode: {len(failed)} translations failed, re-run to retry: {failed[:3]}", file=sys.stderr)
    write_jsonl(data_path("jsonmode", "cs"), sorted(out, key=lambda x: int(x["id"])))


# --------------------------------------------------------------------------------------------- eurlex

EURLEX_N = 50
EURLEX_MAX_DOC_TOKENS = 16000  # fits the 32k context of every model except tiny-aya (8k, truncated)
EURLEX_PROMPT_TRANSLATE = """Translate the following prompt for a language model from English into Czech. It is a prompt that asks for a summary of an EU legislative document written in Czech. Use precise Czech legal and EU terminology as on the Czech EUR-Lex portal (e.g. nařízení, směrnice, rozhodnutí, doporučení, stanovisko, odůvodnění, normativní část / články, Úřední věstník Evropské unie, Rada, Komise, Evropský parlament, Soudní dvůr). Where the prompt names the English modal verbs that express obligation, permission and recommendation, replace them with their usual Czech equivalents in EU legislation. Wherever the prompt refers to the document's or the summary's language, it must say Czech. Keep the numbering and structure, and keep the placeholder {document} unchanged at the very end. Output only the translated prompt.

Prompt:
"""
LANG_NAMES = {"en": "english", "cs": "czech"}


def eurlex_prompt(lang: str) -> str:
    path = CONFIGS / "prompts" / f"eurlex.{lang}.txt"
    if not path.exists():
        assert lang == "cs"
        from .kimi import Kimi

        en = (CONFIGS / "prompts" / "eurlex.en.txt").read_text(encoding="utf-8")

        def validate(content):
            if not content.rstrip().endswith("{document}") or content.count("{document}") != 1:
                raise ValueError("placeholder lost")

        rec = asyncio.run(Kimi("eurlex_prompt").chat(
            [{"role": "user", "content": EURLEX_PROMPT_TRANSLATE + en}], effort="high", validate=validate))
        path.write_text(rec["content"].strip() + "\n", encoding="utf-8")
        print(f"wrote {path} - please proofread it", file=sys.stderr)
    return path.read_text(encoding="utf-8").strip()


def prep_eurlex():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")  # reference tokenizer for the length filter
    docs = {}
    for lang in LANGS:
        with open(hf_file("dennlinger/eur-lex-sum", f"data/{LANG_NAMES[lang]}/test.json"), encoding="utf-8") as f:
            text = f.read().strip()
        rows = json.loads(text) if text.startswith("[") else [json.loads(l) for l in text.splitlines()]
        docs[lang] = {r["celex_id"]: r for r in rows}
    ids = sorted(set(docs["en"]) & set(docs["cs"]))

    def n_tokens(s):
        return len(tok(s, add_special_tokens=False)["input_ids"])

    ok = [c for c in ids if all(n_tokens(docs[l][c]["reference"]) <= EURLEX_MAX_DOC_TOKENS for l in LANGS)]
    chosen = sorted(random.Random(SEED).sample(ok, EURLEX_N))
    print(f"eurlex: {len(ok)}/{len(ids)} parallel documents <= {EURLEX_MAX_DOC_TOKENS} tokens, sampled {len(chosen)}")
    for lang in LANGS:
        prompt = eurlex_prompt(lang)
        # In EUR-Lex-Sum, "reference" is the source document and "summary" the official summary.
        write_jsonl(data_path("eurlex", lang), [
            item("eurlex", lang, c, [{"role": "user", "content": prompt}],
                 {"reference_summary": docs[lang][c]["summary"]}, document=docs[lang][c]["reference"])
            for c in chosen])


# ---------------------------------------------------------------------------------------------- minds

MINDS_LOCALES = {"en": "en-US", "cs": "cs-CZ"}


def prep_minds():
    cfg = PROMPTS["minds"]
    names = list(cfg["labels"])
    for lang in LANGS:
        loc = MINDS_LOCALES[lang]
        df = pd.read_parquet(hf_file("PolyAI/minds14", f"{loc}/train-00000-of-00001.parquet"),
                             columns=["path", "transcription", "intent_class"])
        labels = "\n".join(f"- {n}: {cfg['labels'][n][lang]}" for n in names)
        write_jsonl(data_path("minds", lang), [
            item("minds", lang, r["path"],
                 [{"role": "user", "content": fill(cfg["template"][lang], labels=labels, text=r["transcription"].strip())}],
                 {"label": names[int(r["intent_class"])]})
            for r in df.to_dict("records")])


# ---------------------------------------------------------------------------------------------- gmmlu

def prep_gmmlu():
    tpl = PROMPTS["gmmlu"]["template"]
    for lang in LANGS:
        df = pd.read_parquet(hf_file("CohereLabs/Global-MMLU-Lite", f"{lang}/test-00000-of-00001.parquet"))
        write_jsonl(data_path("gmmlu", lang), [
            item("gmmlu", lang, r["sample_id"],
                 [{"role": "user", "content": fill(tpl[lang], question=r["question"].strip(), a=r["option_a"],
                                                   b=r["option_b"], c=r["option_c"], d=r["option_d"])}],
                 {"answer": r["answer"].strip(), "cultural": r["cultural_sensitivity_label"],
                  "category": r["subject_category"]})
            for r in df.to_dict("records")])


# --------------------------------------------------------------------------------------------- ifeval

def prep_ifeval():
    for lang in LANGS:
        rows = read_jsonl(hf_file("LLaMAX/BenchMAX_Rule-based", f"ifeval_{lang}.jsonl"))
        write_jsonl(data_path("ifeval", lang), [
            item("ifeval", lang, r["key"], [{"role": "user", "content": r["prompt"]}],
                 {"instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"],
                  "lang": r.get("lang", lang), "prompt": r["prompt"]})
            for r in rows])


STEPS = {"nexus": prep_nexus, "jsonmode": prep_jsonmode, "eurlex": prep_eurlex, "minds": prep_minds,
         "gmmlu": prep_gmmlu, "ifeval": prep_ifeval}

if __name__ == "__main__":
    wanted = sys.argv[1:] or ["all"]
    for name in STEPS if wanted == ["all"] else wanted:
        STEPS[name]()
        for lang in LANGS:
            p = data_path(name, lang)
            if p.exists():
                print(f"{p.name}: {len(read_jsonl(p))} items")
