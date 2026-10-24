"""Shared paths, task registry and small helpers."""

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "results"

LANGS = ("en", "cs")

# task -> generation budget (new tokens). Radar axes are defined in score.py.
TASKS = {
    "nexus": 512,  # tool calling (BenchMAX default)
    "jsonmode": 1024,  # structured output
    "eurlex": 2048,  # summarization; the paper uses 1024, which cut off most summaries of capable models, Czech more
    "minds": 64,  # intent classification
    "gmmlu": 256,  # knowledge; room for models that explain before the letter (scored on knowledge, not format)
    "ifeval": 2048,  # instruction following (BenchMAX default)
}


def data_path(task: str, lang: str) -> Path:
    return DATA / f"{task}.{lang}.jsonl"


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_yaml(name: str) -> dict:
    with open(CONFIGS / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def fill(template: str, **kw) -> str:
    """Placeholder substitution that leaves other braces alone (documents and schemas contain them)."""
    for k, v in kw.items():
        template = template.replace("{" + k + "}", str(v))
    return template


_THINK = re.compile(r"<think>.*?</think>|<\|channel\>.*?<channel\|>", re.S)  # Gemma 4: <|channel>thought…<channel|>


def strip_think(text: str | None) -> str:
    """Drop any reasoning block a model emits despite thinking being disabled."""
    if not text:
        return ""
    text = _THINK.sub("", text)
    if "</think>" in text:  # opening tag was part of the prompt
        text = text.split("</think>", 1)[1]
    return text.strip()
