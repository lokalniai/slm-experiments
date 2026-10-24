"""Kimi K3 judge for EUR-Lex summaries, adapted from "Summarization Is Not Dead Yet" (arXiv 2606.08000,
Appendix C.1 and Fig. 21).

Per document and language, the model summaries are split at random into two groups, and each group also
gets the official EUR-Lex summary ("human") as a shared anchor. Each group is judged in one listwise call,
with the order randomized and system names hidden. The judge gives 1-5 Likert scores on informativeness,
faithfulness, coherence and conciseness, plus an overall ranking. Single judge, reasoning effort "high".

    python -m slmbench.judge [--langs en cs] [--limit N] [--out outputs] [--results results]
"""

import argparse
import asyncio
from pathlib import Path
import random
import re
import string
import sys

from .common import LANGS, OUTPUTS, RESULTS, data_path, fill, read_jsonl, strip_think, write_jsonl
from .generate import model_configs
from .kimi import Kimi

DIMS = ("Informativeness", "Faithfulness", "Coherence", "Conciseness")
N_GROUPS = 2
HUMAN = "human"

JUDGE_PROMPT = """You are an expert summary evaluator. You will be given a source document and {n} candidate summaries (labeled Summary A through Summary {last}) in randomized order. Your task is to score all {n} summaries on each of four dimensions independently, then provide an overall ranking of summary quality.

Dimension definitions and scoring anchors.
1. Informativeness. Does the summary capture the key information from the source? Key information refers to the points that a knowledgeable reader of the source would consider essential.
  • Score 1: Misses most key points or covers only a small fraction of the main content.
  • Score 3: Covers the most important points but omits several secondary details relevant to a complete understanding.
  • Score 5: Captures all major points and key supporting details without omitting information that would change the reader's understanding.
2. Faithfulness. Is every claim in the summary factually accurate? Verify each claim first against the source document; for claims about widely accepted general knowledge that the source presupposes, supplement with credible external references.
  • Score 1: Contains multiple claims that are factually incorrect or that cannot be verified from the source or credible external references.
  • Score 3: Mostly accurate but contains one or two statements that are imprecise or weakly supported.
  • Score 5: Every claim is factually accurate and supported by the source or verifiable through credible external references.
3. Coherence. Is the summary well-organized, grammatically correct, and easy to read? Coherence concerns surface readability and linguistic quality, not length or comprehensiveness; a longer summary is not less coherent for being longer.
  • Score 1: Difficult to follow due to grammatical errors, disorganized structure, or unclear references.
  • Score 3: Generally readable but contains some awkward transitions or minor grammatical issues.
  • Score 5: Well-structured, grammatically correct, and reads naturally throughout.
4. Conciseness. Does the summary avoid unnecessary repetition and verbosity? Conciseness concerns brevity and absence of redundancy, not grammatical quality; a grammatically rough but tight summary should not lose conciseness points for grammar alone.
  • Score 1: Highly repetitive or padded with irrelevant content.
  • Score 3: Mostly concise but contains some redundant phrases or minor verbosity.
  • Score 5: Conveys all essential information without repetition, padding, or off-topic content.

Instructions.
1. Read the source. Read the source document carefully before evaluating any summary.
2. Per-dimension scoring. For each of the four dimensions, assign each of the {n} summaries a Likert score from 1 (very poor) to 5 (excellent). Multiple summaries may receive the same score on a given dimension. Use intermediate scores (2 and 4) for summaries that fall between the described anchors.
3. Overall ranking. After scoring all four dimensions, provide an overall ranking of all {n} summaries from worst (rank 1) to best (rank {n}). Each rank must be used exactly once; ties are not permitted.
4. Bias mitigation. Score each summary on its own content, not its position in the list (A through {last}) or its surface style. Do not let stylistic familiarity favor one summary over another. Do not let your impression of an earlier summary anchor your scores for later ones.
5. Dimension independence. Evaluate each dimension independently. Do not allow your assessment of one dimension to influence your scores on another.
6. Language. If the source is in a language other than English, evaluate in that language and apply the same criteria.

Output format.
Return your scores and ranking in the following exact format:
{format}

Source document: {document}

{summaries}"""


def judge_prompt(document: str, summaries: list[str]) -> str:
    labels = string.ascii_uppercase[: len(summaries)]
    slots = ", ".join(f"{x}=[1-5]" for x in labels)
    fmt = "\n".join(f"{d}: {slots}" for d in DIMS)
    fmt += "\nOverall: " + " < ".join(f"[{'/'.join(labels)}]" for _ in labels)
    body = "\n\n".join(f"Summary {x}: {s}" for x, s in zip(labels, summaries))
    return fill(JUDGE_PROMPT, n=len(summaries), last=labels[-1], format=fmt, summaries=body, document=document)


def parse_judgment(text: str, labels: str) -> tuple[dict, list]:
    scores = {}
    for d in DIMS:
        m = re.search(rf"^\W*{d}\W*:\s*(.+)$", text, re.M | re.I)
        if not m:
            raise ValueError(f"missing {d}")
        pairs = dict(re.findall(r"\b([A-Z])\s*=\s*\[?([1-5])\]?", m.group(1)))
        if set(pairs) != set(labels):
            raise ValueError(f"{d}: labels {sorted(pairs)} != {list(labels)}")
        scores[d] = {k: int(v) for k, v in pairs.items()}
    m = re.search(r"^\W*Overall\W*:\s*(.+)$", text, re.M | re.I)
    order = re.findall(r"\b([A-Z])\b", m.group(1)) if m else []
    if sorted(order) != sorted(labels):
        raise ValueError(f"bad overall ranking {order}")
    return scores, order  # order: worst ... best


def collect(lang: str, models: list[str], out_root: Path) -> tuple[list[dict], dict]:
    items = read_jsonl(data_path("eurlex", lang))
    outs = {}
    for m in models:
        p = out_root / m / f"eurlex.{lang}.jsonl"
        if p.exists():
            outs[m] = {r["id"]: strip_think(r.get("content")) for r in read_jsonl(p)}
    return items, outs


async def judge_lang(lang: str, models: list[str], limit: int | None, out_root: Path, results: Path):
    items, outs = collect(lang, models, out_root)
    items = items[:limit]
    missing = [m for m in models if m not in outs or any(it["id"] not in outs[m] for it in items)]
    if missing:
        print(f"[{lang}] incomplete eurlex outputs, judging without: {missing}", file=sys.stderr)
    present = [m for m in models if m not in missing]
    kimi = Kimi(f"judge_eurlex_{lang}")

    async def one(it, g, members):
        rng = random.Random(f"{it['id']}|{lang}|{g}|order")
        members = [m for m in members if m == HUMAN or outs[m].get(it["id"])]  # empty summaries score 0
        rng.shuffle(members)
        labels = string.ascii_uppercase[: len(members)]
        texts = [it["gold"]["reference_summary"] if m == HUMAN else outs[m][it["id"]] for m in members]
        prompt = judge_prompt(it["document"], texts)
        rec = await kimi.chat([{"role": "user", "content": prompt}], effort="high", max_tokens=32000,
                              validate=lambda c: parse_judgment(c, labels))
        scores, order = parse_judgment(rec["content"], labels)
        lab2m = dict(zip(labels, members))
        return {"id": it["id"], "lang": lang, "group": g, "labels": lab2m,
                "scores": {lab2m[x]: {d: scores[d][x] for d in DIMS} for x in labels},
                "rank": {lab2m[x]: r + 1 for r, x in enumerate(order)},  # 1 = worst
                "judge_prompt_tokens": rec["prompt_tokens"], "judge_reasoning_tokens": rec["reasoning_tokens"],
                "raw": rec["content"]}

    jobs = []
    for it in items:
        rng = random.Random(f"{it['id']}|{lang}|groups")
        shuffled = present[:]
        rng.shuffle(shuffled)
        for g in range(N_GROUPS):
            jobs.append(one(it, g, shuffled[g::N_GROUPS] + [HUMAN]))
    res = await asyncio.gather(*jobs, return_exceptions=True)
    errors = [r for r in res if isinstance(r, Exception)]
    for e in errors[:5]:
        print(f"[{lang}] judge error: {e}", file=sys.stderr)
    write_jsonl(results / "judge" / f"eurlex.{lang}.jsonl", [r for r in res if not isinstance(r, Exception)])
    print(f"[{lang}] judged {len(res) - len(errors)}/{len(res)} groups")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
    ap.add_argument("--limit", type=int, default=None, help="first N documents (pilot)")
    ap.add_argument("--out", type=Path, default=OUTPUTS)
    ap.add_argument("--results", type=Path, default=RESULTS)
    args = ap.parse_args()
    models = list(model_configs())
    for lang in args.langs:
        asyncio.run(judge_lang(lang, models, args.limit, args.out, args.results))


if __name__ == "__main__":
    main()
