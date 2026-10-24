# Malé modely pro koncová zařízení

Report, 2026-10-24. Small instruction-tuned LLMs for on-device use, on six practical skills, each in English
and in Czech. Per-item results: [results/](results/) (regenerate with `python -m slmbench.score`). Viewer:
https://lokalniai.github.io/slm-experiments/ · Blogpost (in Czech): https://lokalni.ai/blog/male-modely/

## TL;DR

1. **Gemma 4 E2B is the best all-rounder, and the one whose Czech holds up best among the strong models.** It
   averages 79.1 in English and 74.2 in Czech over the six skills. It leads summarization in both languages
   (80.3 / 79.3), where the official EUR-Lex summaries score 71.4 / 81.0.
2. **tiny-aya-global has the smallest Czech gap (−2.4).** Its English is mid-table (70.3), but its Czech (67.9)
   is second only to Gemma, and it leads Czech instruction following (74.5). It does this even though its chat
   template has no tool support at all.
3. **Below about 1.5 B parameters, Czech collapses.** LFM2.5-350M, LFM2.5-1.2B and Granite-4.0-H-350M reach
   52–61 in English but only 20–29 in Czech. On Czech intent classification and summarization they score close
   to 0: they echo the customer's greeting, repeat a word of the instruction, return nothing, or answer in
   English. Qwen3.5-0.8B is the exception at this size: its classification is the same in both languages
   (86.0 / 88.7).
4. **Strong English does not guarantee strong Czech.** Nemotron-3-Nano-4B (76.4 EN, second-best English) and
   SmolLM3-3B (73.9) lose 14 and 21 points in Czech. Most of the loss is summarization: 58.1 → 32.3 and
   64.5 → 24.5. Their Czech summaries run on article by article until they hit the token limit, and SmolLM3
   answers 11 of the 50 Czech documents in English.
5. **No model is best everywhere.**
   - Structured output is nearly solved from 2 B up: Nemotron, Granite 4.2, Qwen3.5-2B and Gemma all score
     95–97 in both languages.
   - Tool calling is the hardest skill: the best scores are 67.3 (EN, Qwen3.5-2B) and 57.2 (CS, Ministral).
     Granite 4.2 is strong elsewhere but often asks a clarifying question instead of calling the tool
     (33.0 / 19.8).
   - Knowledge favours the largest models: Ministral leads with 70.8 / 57.8.
   - FunctionGemma-270M, a base for fine-tuning, is not usable as-is, not even for tool calling (9.4 / 5.0).

## Setup

| | |
|---|---|
| Engine | vLLM 0.31.0 (torch 2.13, CUDA 13.0), OpenAI-compatible server, one model at a time, FlashInfer sampler off |
| Precision | BF16 checkpoints only, no quantization. Ministral uses the `-BF16` repo, because the default one is FP8. |
| Decoding | Greedy (`temperature=0`), the checkpoints' own sampling defaults ignored (`--generation-config vllm`), one run per item |
| Thinking | Off for all models: `enable_thinking=False` in the chat template, a no-op for models without a thinking mode. A stray reasoning block is stripped before scoring (Gemma 4 emitted one in rare cases). |
| Prompting | Zero-shot, each model's own chat template. Every input is fully in the test language; tool descriptions and JSON schemas stay in English. |
| Tool calls | Each model's native format, parsed by its vLLM tool parser. tiny-aya's template has no tool support, so it gets BenchMAX's JSON tool prompt in the user turn instead. |
| GPUs | 1× L40S (fobos, Plzeň) for 9 models; 1× A100-40GB (zia) for Ministral, FunctionGemma and Granite-350M |
| Judge | Kimi K3 via the e-INFRA CZ LLM API, `reasoning_effort=high`, one judge, summarization only |

### Models

| Model | HF id | Size | Tool parser | Context used |
|---|---|---|---|---|
| `functiongemma-270m` | google/functiongemma-270m-it | 0.27 B | functiongemma | 32k |
| `granite-4.0-h-350m` | ibm-granite/granite-4.0-h-350m | 0.34 B | granite4 | 32k |
| `lfm2.5-350m` | LiquidAI/LFM2.5-350M | 0.35 B | lfm2 | 32k |
| `qwen3.5-0.8b` | Qwen/Qwen3.5-0.8B | 0.87 B | qwen3_coder | 32k |
| `lfm2.5-1.2b` | LiquidAI/LFM2.5-1.2B-Instruct | 1.2 B | lfm2 | 32k |
| `qwen3.5-2b` | Qwen/Qwen3.5-2B | 2.3 B | qwen3_coder | 32k |
| `gemma-4-e2b` | google/gemma-4-E2B-it | 2 B effective (5.1 B with per-layer embeddings) | gemma4 | 32k |
| `smollm3-3b` | HuggingFaceTB/SmolLM3-3B | 3.1 B | hermes | 32k |
| `tiny-aya-global` | CohereLabs/tiny-aya-global | 3.3 B | none (JSON tool prompt) | **8k** |
| `granite-4.2-3b` | ibm-granite/granite-4.2-3b | 3.7 B | qwen3_coder | 32k |
| `ministral-3-3b` | mistralai/Ministral-3-3B-Instruct-2512-BF16 | 3.8 B | mistral | 32k |
| `nemotron-3-nano-4b` | nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | 4.0 B | qwen3_coder | 32k |

The exact checkpoint revisions are in `outputs/<model>/meta.json`. Qwen3.5, Gemma 4 and Ministral are
vision-language models and run text-only (`--language-model-only`).

### Skills

Each skill has an English set and a Czech one. Most pairs are parallel: Czech-native or human-translated
data where it exists, otherwise a Kimi K3 translation.

| Skill | Data (EN / CS) | Items per lang | Score (0–100) | Max new tokens |
|---|---|---|---|---|
| Tool calling | BenchMAX Nexus `nexus_en` / `nexus_cs` (human-checked translation); 2–12 NexusRaven tools in context | 318 | Exact function name and arguments, defaults filled in (BenchMAX protocol) | 512 |
| Structured output | NousResearch/json-mode-eval / its Kimi K3 translation (keys, enums, IDs and numbers kept; validated against the schema) | 100 | Mean of valid JSON, valid against the schema, and the share of gold fields reproduced. No constrained decoding. | 1024 |
| Summarization | EUR-Lex-Sum `english` / `czech`: the same 50 EU legal acts, ≤ 16k tokens in both languages | 50 | Kimi K3 listwise judge (below) | 2048 |
| Classification | MInDS-14 `en-US` / `cs-CZ`: banking intents, natively collected in each language, 14 labels | 563 / 574 | Accuracy (chance 7 %) | 64 |
| Knowledge | Global-MMLU-Lite `en` / `cs`: the same 400 questions, human-translated Czech, 200 culturally sensitive + 200 agnostic | 400 | Accuracy (chance 25 %) | 256 |
| Instruction following | BenchMAX IFEval `ifeval_en` / `ifeval_cs` with BenchMAX's language-aware checkers | 429 | Mean of prompt-level and instruction-level strict accuracy | 2048 |

### Summarization protocol

Adapted from *Summarization Is Not Dead Yet* (Liu et al., arXiv 2606.08000):

- **Prompt.** Their EurLexSum prompt (Fig. 19): instrument type and legal basis, scope, obligations with
  article references, defined terms, entry into force, no extrapolation. The Czech version is a Kimi K3
  translation, fully in Czech.
- **Judging.** Their judge prompt (Fig. 21). Per document and language, the 12 summaries are split at random
  into two groups of 6, and each group also gets the official EUR-Lex summary as a shared human reference.
  Each group of 7 is judged in one listwise call, with the order shuffled and names hidden. The judge gives
  1–5 Likert scores for informativeness, faithfulness, coherence and conciseness, plus an overall ranking.
  That's 200 calls in total.
- **Score.** The harmonic mean of informativeness (coverage) and faithfulness (precision), rescaled from 1–5
  to 0–100. A summary not in the target language (by `langdetect`) scores 0. Coherence and conciseness are
  reported, but not part of the score.
- **Truncation.** tiny-aya's documents are cut from the end to fit its 8k context. It keeps 40–100 % of each
  (median 80 % EN, 59 % CS), and the kept fraction is recorded per item.

## Results

### All skills

The radar value per skill and language, 0–100; bold marks the best per column.

| Model | Size | Tools EN | Tools CS | JSON EN | JSON CS | Summ. EN | Summ. CS | Class. EN | Class. CS | Know. EN | Know. CS | IFEval EN | IFEval CS | **Mean EN** | **Mean CS** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `functiongemma-270m` | 0.27B | 9.4 | 5.0 | 40.7 | 20.3 | 3.7 | 4.0 | 6.9 | 0.9 | 20.8 | 10.2 | 21.4 | 17.1 | 17.1 | 9.6 |
| `granite-4.0-h-350m` | 0.34B | 50.6 | 32.7 | 94.1 | 78.8 | 21.2 | 1.0 | 53.8 | 4.9 | 30.5 | 22.0 | 64.1 | 36.5 | 52.4 | 29.3 |
| `lfm2.5-350m` | 0.35B | 36.8 | 16.4 | 71.3 | 47.8 | 15.6 | 1.0 | 70.0 | 0.9 | 42.2 | 29.2 | 77.8 | 26.5 | 52.3 | 20.3 |
| `qwen3.5-0.8b` | 0.87B | 45.6 | 27.4 | 94.7 | 55.9 | 53.0 | 38.5 | 86.0 | 88.7 | 47.0 | 35.0 | 60.5 | 38.9 | 64.5 | 47.4 |
| `lfm2.5-1.2b` | 1.2B | 34.0 | 9.7 | 74.3 | 63.1 | 45.7 | 1.5 | 74.1 | 17.6 | 53.2 | 26.8 | 86.0 | 38.7 | 61.2 | 26.2 |
| `qwen3.5-2b` | 2.3B | **67.3** | 51.3 | 95.7 | 95.5 | 72.0 | 62.8 | 85.6 | 87.6 | 63.8 | 47.8 | 71.3 | 51.7 | 75.9 | 66.1 |
| `gemma-4-e2b` | 2B eff. | 62.0 | 54.7 | 95.2 | 95.0 | **80.3** | **79.3** | **94.0** | 92.3 | 58.2 | 54.2 | 85.0 | 69.8 | **79.1** | **74.2** |
| `smollm3-3b` | 3.1B | 56.6 | 51.3 | 90.8 | 85.9 | 64.5 | 24.5 | 90.9 | 70.6 | 65.5 | 40.0 | 74.8 | 48.0 | 73.9 | 53.4 |
| `tiny-aya-global` | 3.3B | 49.1 | 43.7 | 90.0 | 92.3 | 48.6 | 49.8 | 89.5 | 92.0 | 59.8 | 55.0 | 84.9 | **74.5** | 70.3 | 67.9 |
| `granite-4.2-3b` | 3.7B | 33.0 | 19.8 | 96.3 | 96.2 | 60.7 | 50.8 | 91.3 | 89.4 | 65.2 | 46.8 | 82.0 | 63.3 | 71.4 | 61.0 |
| `ministral-3-3b` | 3.8B | 64.5 | **57.2** | 78.5 | 60.2 | 60.6 | 69.5 | 92.4 | 91.1 | **70.8** | **57.8** | 64.0 | 55.7 | 71.8 | 65.3 |
| `nemotron-3-nano-4b` | 4.0B | 54.7 | 37.1 | **97.3** | **96.6** | 58.1 | 32.3 | 92.4 | **92.9** | 67.2 | 49.2 | **88.6** | 67.9 | 76.4 | 62.7 |

### Czech vs English

The mean over the six skills, sorted by Czech score.

| Model | EN | CS | CS − EN |
|---|---|---|---|
| `gemma-4-e2b` | 79.1 | 74.2 | −4.9 |
| `tiny-aya-global` | 70.3 | 67.9 | −2.4 |
| `qwen3.5-2b` | 75.9 | 66.1 | −9.8 |
| `ministral-3-3b` | 71.8 | 65.3 | −6.5 |
| `nemotron-3-nano-4b` | 76.4 | 62.7 | −13.7 |
| `granite-4.2-3b` | 71.4 | 61.0 | −10.4 |
| `smollm3-3b` | 73.9 | 53.4 | −20.5 |
| `qwen3.5-0.8b` | 64.5 | 47.4 | −17.1 |
| `granite-4.0-h-350m` | 52.4 | 29.3 | −23.1 |
| `lfm2.5-1.2b` | 61.2 | 26.2 | −35.0 |
| `lfm2.5-350m` | 52.3 | 20.3 | −32.0 |
| `functiongemma-270m` | 17.1 | 9.6 | −7.5 |

Czech beats English by a clear margin only once: Ministral's summarization (69.5 vs 60.6). The other places
where Czech is ahead are within 3 points: classification for both Qwens, tiny-aya and Nemotron, and JSON and
summarization for tiny-aya.

### Summarization in detail

The judge's mean Likert scores (1–5), the share of summaries in the right language, and how many of the 50
hit the 2,048-token limit.

| Model | Inform. EN / CS | Faithf. EN / CS | Coher. EN / CS | Concise EN / CS | Right lang. CS | Hit limit EN / CS |
|---|---|---|---|---|---|---|
| official EUR-Lex summary | 3.78 / 4.26 | 4.05 / 4.29 | 4.16 / 4.26 | 3.43 / 3.65 | 100 % | – |
| `gemma-4-e2b` | 4.08 / 4.26 | 4.44 / 4.18 | 4.96 / 4.28 | 4.16 / 3.74 | 100 % | 1 / 16 |
| `ministral-3-3b` | 4.56 / 4.46 | 2.82 / 3.32 | 4.82 / 4.54 | 3.74 / 3.88 | 100 % | 0 / 7 |
| `qwen3.5-2b` | 4.36 / 3.54 | 3.60 / 3.72 | 4.60 / 3.50 | 3.64 / 2.78 | 100 % | 5 / 37 |
| `granite-4.2-3b` | 3.88 / 3.44 | 3.18 / 2.76 | 3.52 / 3.18 | 2.70 / 3.10 | 100 % | 12 / 36 |
| `tiny-aya-global` | 3.16 / 3.14 | 2.86 / 2.94 | 4.44 / 4.58 | 3.72 / 4.38 | 100 % | 0 / 0 |
| `nemotron-3-nano-4b` | 3.70 / 2.38 | 3.12 / 2.50 | 4.28 / 2.08 | 4.12 / 2.00 | 100 % | 2 / 36 |
| `smollm3-3b` | 3.32 / 2.00 | 4.04 / 2.54 | 3.94 / 2.20 | 3.08 / 1.58 | 78 % | 12 / 41 |
| `qwen3.5-0.8b` | 3.34 / 2.24 | 3.28 / 3.28 | 3.28 / 2.40 | 2.38 / 1.62 | 100 % | 25 / 46 |
| `lfm2.5-1.2b` | 2.68 / 1.04 | 3.20 / 1.10 | 4.48 / 1.22 | 4.54 / 1.12 | 100 % | 0 / 12 |
| `granite-4.0-h-350m` | 1.18 / 1.08 | 4.76 / 1.67 | 4.52 / 1.83 | 4.56 / 1.50 | 14 % | 1 / 1 |
| `lfm2.5-350m` | 1.94 / 1.02 | 1.46 / 1.10 | 2.56 / 1.06 | 2.22 / 1.22 | 88 % | 1 / 1 |
| `functiongemma-270m` | 1.06 / 1.02 | 1.38 / 1.62 | 2.56 / 1.80 | 1.48 / 1.58 | 86 % | 1 / 9 |

- **Ministral covers the most, but makes things up.** It has the highest informativeness (4.56 / 4.46) but
  the lowest faithfulness among the strong models (2.82 / 3.32). The score combines both, so it isn't rewarded
  for adding claims the act doesn't make.
- **Why not the mean of all four dimensions.** Granite-350M writes one-sentence summaries in English: 1.18
  informativeness, but 4.5+ on the other three. With that average it would outscore the official summaries;
  with the harmonic mean it gets 21.2.
- **Czech summaries are longer.** Most long Czech summaries go through the act article by article; only a few
  (1–7 per model) are repetition loops. The token limit was raised from the paper's 1,024 to 2,048 for this
  reason. At 1,024 the smoke test cut off most summaries of the capable models, and Czech ones more often,
  because Czech needs more tokens per word.
- **The judge likes LLM summaries.** It rates Gemma above the official summary in English (80.3 vs 71.4).
  This matches the paper's finding that LLM judges favour LLM output on surface quality. The official Czech
  summaries score higher than the English ones (81.0 vs 71.4) on the same acts.

### Other sub-metrics

- **Structured output**, as valid JSON / schema-valid / field accuracy:
  - Ministral: 98 / 53 / 85 in EN and 95 / 25 / 61 in CS. It often wraps the object in a key named after the
    schema title (`{"WirelessAccessPoint": {…}}`). That fails the schema check, but field accuracy is
    computed on the inner object.
  - Qwen3.5-0.8B: schema validity drops from 97 % (EN) to 38 % (CS).
- **Tool calling.**
  - Granite 4.2 makes no call at all on 45 of 318 English and 117 of 318 Czech requests; it asks for details
    instead.
  - tiny-aya's JSON answers (its only option) are parsed from text. SmolLM3 and Granite 4.2 also have 3–15 %
    of their calls parsed from text instead of the native format.
- **Knowledge.** Culturally agnostic questions are easier than culturally sensitive ones for every model except
  FunctionGemma, which is near chance. Ministral, for example, scores 73.5 / 68.0 in EN and 59.5 / 56.0 in CS.
- **IFEval.** Prompt-level strict accuracy: Nemotron 86.9 (EN), tiny-aya 70.9 (CS).

## Caveats

- **One greedy run per item.** On the smaller sets (100 JSON requests, 50 documents), a few points are within
  noise.
- **One judge.** The paper averages five judges with self-exclusion. Kimi is not a family under test, but a
  single judge's preferences are not averaged out.
- **Translated inputs.**
  - The Czech json-mode set and the Czech summarization prompt are machine translations (Kimi K3, validated
    for structure, spot-checked).
  - The Czech Nexus and IFEval prompts are BenchMAX's translations, checked by native speakers.
  - Global-MMLU-Lite Czech is human-translated.
  - MInDS-14 and EUR-Lex are Czech-native.
- **Format counts.** Answers are parsed leniently (a letter in "The answer is C", a label inside markdown), but
  a model that doesn't follow the requested format loses the item. This is deliberate for on-device use, and it
  is most visible for the sub-1.5 B models in Czech.
- **Truncated context.** tiny-aya sees truncated EUR-Lex documents; its 8k context would not fit them.
- **Changes after the smoke test.** The token budgets for knowledge (32 → 256), classification (32 → 64) and
  summarization (1,024 → 2,048), and the summarization score, were changed after the smoke test and before the
  full run, for the reasons given above.

## Reproduction

```bash
python -m slmbench.prepare all                                  # data/*.jsonl (Kimi translations cached)
qsub -v MODELS="qwen3.5-2b gemma-4-e2b" jobs/generate.pbs       # outputs/<model>/<task>.<lang>.jsonl
python -m slmbench.judge                                        # results/judge/ (Kimi K3)
python -m slmbench.score                                        # results/scores.csv, results/items/
python -m slmbench.viewer --site docs                           # the static site
```

Compute:
- 9 models in 71 min on two L40S jobs in Plzeň (32 + 39 min wall time, 1.5–4 min vLLM start-up per model).
- 3 models in 84 min on an A100 in Brno, where every vLLM start-up took 15–19 min because the venv is on Plzeň
  storage. Hence `plzen=True` in `jobs/generate.pbs`.
- The judge: 200 Kimi K3 calls, about 6 h at 3–4 parallel requests.
