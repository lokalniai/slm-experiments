"""Kimi K3 client for the e-INFRA CZ LLM API (translations and the summarization judge).

The service allows at most 4 parallel requests per user, so every call goes through one semaphore.
Responses are cached on disk by request hash, so re-running a step never re-queries finished items.
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path

from openai import AsyncOpenAI

from .common import ROOT

BASE_URL = "https://llm.ai.e-infra.cz/v1"
MODEL = "kimi-k3"
# The service allows 4. We use 3: a request the client gave up on can still hold a server slot, and the
# first full judge run lost 6 calls to 429 max_parallel_requests that way.
MAX_PARALLEL = 3
# absolute path: on compute nodes $HOME is the node's site-local storage, not plzen1
TOKEN_FILE = Path(os.environ.get("EINFRA_TOKEN_FILE", "/storage/plzen1/home/kasnezde/.config/einfra_token"))
CACHE_DIR = ROOT / "data" / "kimi_cache"


def _api_key() -> str:
    return os.environ.get("E_INFRA_API_TOKEN") or TOKEN_FILE.read_text().strip()


class Kimi:
    def __init__(self, cache_name: str):
        self.client = AsyncOpenAI(base_url=BASE_URL, api_key=_api_key(), timeout=1800, max_retries=0)
        self.sem = asyncio.Semaphore(MAX_PARALLEL)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_path = CACHE_DIR / f"{cache_name}.jsonl"
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                self.cache[r["key"]] = r
        self.lock = asyncio.Lock()

    async def chat(self, messages: list[dict], effort: str = "low", max_tokens: int = 16000,
                   validate=None, attempts: int = 4) -> dict:
        """Return {"content", "reasoning_tokens", ...}. `validate(content)` may raise to trigger a retry."""
        req = {"model": MODEL, "messages": messages, "reasoning_effort": effort,
               "max_tokens": max_tokens, "temperature": 0}
        key = hashlib.sha256(json.dumps(req, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        last_err = None
        for attempt in range(attempts):
            try:
                async with self.sem:
                    resp = await self.client.chat.completions.create(**req)
                content = resp.choices[0].message.content
                if not content:
                    raise ValueError(f"empty content (finish_reason={resp.choices[0].finish_reason})")
                if validate is not None:
                    validate(content)
                usage = resp.usage
                rec = {"key": key, "content": content, "finish_reason": resp.choices[0].finish_reason,
                       "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                       "reasoning_tokens": getattr(usage.completion_tokens_details, "reasoning_tokens", None)}
                async with self.lock:
                    self.cache[key] = rec
                    with open(self.cache_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                return rec
            except Exception as e:  # noqa: BLE001 - API errors, timeouts and validation failures alike
                last_err = e
                throttled = getattr(e, "status_code", None) == 429
                await asyncio.sleep(min((60 if throttled else 10) * 2**attempt, 600))
        raise RuntimeError(f"Kimi request failed after {attempts} attempts: {last_err}")
