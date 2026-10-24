"""Run models over all evaluation sets with a local vLLM OpenAI-compatible server.

    python -m slmbench.generate --models qwen3.5-0.8b smollm3-3b [--limit 3] [--out outputs]

For each model: download the snapshot, start `vllm serve` (BF16, greedy, thinking disabled through
chat_template_kwargs, native tool-call parser), send every item, and write
<out>/<model>/<task>.<lang>.jsonl. Finished items are skipped on re-runs.
"""

import argparse
import asyncio
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import socket

import httpx
from huggingface_hub import snapshot_download
from openai import AsyncOpenAI

from .common import LANGS, OUTPUTS, TASKS, data_path, load_yaml, read_jsonl, write_jsonl

CONCURRENCY = 64


def free_port() -> int:
    """Jobs can share a node (two GPU jobs on fobos10 once fought over a fixed port), so each server takes a free one."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
IGNORE = ["*.gguf", "*.onnx", "*.litertlm", "*.ipynb", "*.png", "*.jpg", "*.pth", "*.bin", "original/*"]
CHAT_KWARGS = {"enable_thinking": False}


def model_configs() -> dict:
    cfg = load_yaml("models.yaml")
    return {name: {**cfg["defaults"], **m} for name, m in cfg["models"].items()}


# Same wording as BenchMAX (tasks/nexus/utils.py:get_tool_prompt), for models whose template has no tools.
def tool_prompt(tools: list[dict]) -> str:
    s = "Respond to the human as helpfully and accurately as possible. You have access to the following tools:\n\n"
    s += "\n\n".join(json.dumps(t, indent=4, ensure_ascii=False) for t in tools)
    s += ('\n\nReminder to ALWAYS respond with a valid json. Use tools if necessary. Respond in the format '
          '{"name": function name, "parameters": dictionary of argument name and its value}. '
          'Do not use variables or output comments.')
    return s


class Server:
    def __init__(self, name: str, cfg: dict, log_path: Path):
        self.name, self.cfg, self.log_path = name, cfg, log_path
        self.proc = None
        self.port = free_port()

    def start(self, model_path: str):
        vllm = Path(sys.executable).parent / "vllm"
        cmd = [str(vllm), "serve", model_path, "--served-model-name", self.name, "--port", str(self.port),
               "--dtype", "bfloat16", "--max-model-len", str(self.cfg["max_model_len"]),
               "--generation-config", "vllm",  # ignore checkpoint sampling defaults: plain greedy for all
               "--gpu-memory-utilization", "0.85", "--seed", "0"]
        if self.cfg.get("tool_parser"):
            cmd += ["--enable-auto-tool-choice", "--tool-call-parser", self.cfg["tool_parser"]]
        if self.cfg.get("trust_remote_code"):
            cmd += ["--trust-remote-code"]
        cmd += [str(a) for a in self.cfg["extra_args"]]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(self.log_path, "w")
        self.log.write(" ".join(cmd) + "\n")
        self.log.flush()
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 1800:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vLLM exited with {self.proc.returncode}, see {self.log_path}")
            try:  # healthy AND serving this model, so a neighbour's server can never be mistaken for ours
                r = httpx.get(f"http://localhost:{self.port}/v1/models", timeout=5)
                if r.status_code == 200 and any(m["id"] == self.name for m in r.json()["data"]):
                    return time.time() - t0
            except (httpx.HTTPError, ValueError, KeyError):
                pass
            time.sleep(5)
        raise TimeoutError("vLLM did not become healthy in 30 min")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.proc:
            self.log.close()


class Runner:
    def __init__(self, name: str, cfg: dict, port: int):
        self.name, self.cfg, self.port = name, cfg, port
        self.native_tools = bool(cfg.get("tool_parser"))
        # vLLM's Mistral tokenizer mode rejects chat_template_kwargs (Ministral has no thinking mode anyway)
        self.chat_kwargs = CHAT_KWARGS if cfg.get("chat_template_kwargs", True) else None
        self.client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="none", timeout=3600, max_retries=2)
        self.http = httpx.AsyncClient(timeout=600)
        self.sem = asyncio.Semaphore(CONCURRENCY)

    def build(self, it: dict, doc: str | None):
        msgs = copy.deepcopy(it["messages"])
        if doc is not None:
            msgs[-1]["content"] = msgs[-1]["content"].replace("{document}", doc)
        tools = it.get("tools")
        if tools and not self.native_tools:
            msgs[0]["content"] = tool_prompt(tools) + "\n\n" + msgs[0]["content"]
            tools = None
        return msgs, tools

    async def count_tokens(self, msgs, tools) -> int:
        body = {"model": self.name, "messages": msgs, "add_generation_prompt": True}
        if self.chat_kwargs:
            body["chat_template_kwargs"] = self.chat_kwargs
        if tools:
            body["tools"] = tools
        r = await self.http.post(f"http://localhost:{self.port}/tokenize", json=body)
        r.raise_for_status()
        return r.json()["count"]

    async def fit_document(self, it: dict):
        """Cut the document from the end until prompt + generation budget fits the served context."""
        doc = it["document"]
        budget = self.cfg["max_model_len"] - it["max_tokens"] - 16
        keep = len(doc)
        for _ in range(8):
            msgs, tools = self.build(it, doc[:keep])
            n = await self.count_tokens(msgs, tools)
            if n <= budget:
                return msgs, tools, keep / len(doc)
            keep = int(keep * min(budget / n, 0.97) * 0.98)
        raise ValueError(f"could not fit document into {budget} tokens")

    async def one(self, it: dict) -> dict:
        async with self.sem:
            rec = {"id": it["id"]}
            try:
                if "document" in it:
                    msgs, tools, kept = await self.fit_document(it)
                    rec["doc_kept"] = round(kept, 4)
                else:
                    msgs, tools = self.build(it, None)
                kw = {"tools": tools, "tool_choice": "auto"} if tools else {}
                t0 = time.time()
                resp = await self.client.chat.completions.create(
                    model=self.name, messages=msgs, temperature=0, max_tokens=it["max_tokens"],
                    extra_body={"chat_template_kwargs": self.chat_kwargs} if self.chat_kwargs else None, **kw)
                ch = resp.choices[0]
                rec.update({
                    "content": ch.message.content,
                    "tool_calls": [{"name": tc.function.name, "arguments": tc.function.arguments}
                                   for tc in (ch.message.tool_calls or [])],
                    "finish_reason": ch.finish_reason,
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "seconds": round(time.time() - t0, 2),
                })
            except Exception as e:  # noqa: BLE001 - keep going, the error is recorded and retried on re-run
                rec["error"] = f"{type(e).__name__}: {e}"[:2000]
            return rec

    async def run_file(self, items: list[dict], out_path: Path):
        done = {}
        if out_path.exists():
            done = {r["id"]: r for r in read_jsonl(out_path) if "error" not in r}
        todo = [it for it in items if it["id"] not in done]
        new = await asyncio.gather(*(self.one(it) for it in todo))
        order = {it["id"]: i for i, it in enumerate(items)}
        rows = sorted([*done.values(), *new], key=lambda r: order.get(r["id"], 1e9))
        write_jsonl(out_path, rows)
        return len(todo), sum("error" in r for r in new)


def download(cfg: dict) -> tuple[str, str]:
    kw = {"allow_patterns": cfg["include"]} if cfg.get("include") else {"ignore_patterns": IGNORE}
    path = snapshot_download(cfg["repo"], **kw)
    return path, Path(path).name  # snapshot dir is named by commit hash


def run_model(name: str, cfg: dict, tasks: list[str], limit: int | None, out_root: Path):
    out_dir = out_root / name
    t0 = time.time()
    path, revision = download(cfg)
    print(f"[{name}] downloaded {cfg['repo']}@{revision[:10]} in {time.time() - t0:.0f}s", flush=True)
    server = Server(name, cfg, out_dir / "vllm.log")
    meta = {"model": name, "repo": cfg["repo"], "revision": revision, "config": cfg, "limit": limit}
    try:
        meta["startup_s"] = round(server.start(path))
        print(f"[{name}] server up in {meta['startup_s']}s", flush=True)

        async def run_all():  # one event loop for the shared async clients
            runner = Runner(name, cfg, server.port)
            for task in tasks:
                for lang in LANGS:
                    items = read_jsonl(data_path(task, lang))[:limit]
                    t1 = time.time()
                    n, errors = await runner.run_file(items, out_dir / f"{task}.{lang}.jsonl")
                    meta[f"{task}.{lang}_s"] = round(time.time() - t1)
                    print(f"[{name}] {task}.{lang}: {n} generated, {errors} errors, {time.time() - t1:.0f}s",
                          flush=True)

        asyncio.run(run_all())
    finally:
        server.stop()
        meta["vllm"] = subprocess.run([sys.executable, "-c", "import vllm;print(vllm.__version__)"],
                                      capture_output=True, text=True).stdout.strip()
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--limit", type=int, default=None, help="first N items per file (smoke test)")
    ap.add_argument("--out", type=Path, default=OUTPUTS)
    args = ap.parse_args()
    cfgs = model_configs()
    names = list(cfgs) if args.models == ["all"] else args.models
    failed = []
    for name in names:
        try:
            run_model(name, cfgs[name], args.tasks, args.limit, args.out)
        except Exception as e:  # noqa: BLE001 - one broken model must not stop the others
            print(f"[{name}] FAILED: {type(e).__name__}: {e}", flush=True)
            failed.append(name)
    if failed:
        print("failed models:", " ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
