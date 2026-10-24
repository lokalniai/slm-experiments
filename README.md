# Malé modely pro koncová zařízení: small LLMs on six skills, in English and Czech

Outputs, scores and an interactive viewer for 12 small instruction-tuned LLMs (0.27–4 B parameters) aimed at
on-device use. Each model is tested on six practical skills, each in English and in Czech. The models ran on
MetaCentrum GPUs.

- [💻 **Interactive viewer**](https://lokalniai.github.io/slm-experiments/):
  - one radar of the six skills where you pick up to four models and switch between English and Czech
  - all results: per-model radars, the full table and the means
  - per-skill charts and tables, and every model answer next to its input and reference
- [🖋 **Blogpost (in Czech)**](https://lokalni.ai/blog/male-modely/)
- [📄 **Report**](REPORT.md): setup, results and caveats

---

## What was run

| Skill | Data (EN / CS) | Items per lang | Score |
|---|---|---|---|
| Tool calling | BenchMAX Nexus | 318 | exact function name and arguments |
| Structured output | json-mode-eval / its Kimi K3 translation | 100 | valid JSON, schema-valid, field accuracy |
| Summarization | EUR-Lex-Sum (the same 50 EU acts in both languages) | 50 | Kimi K3 listwise judge (protocol of arXiv 2606.08000) |
| Classification | MInDS-14 `en-US` / `cs-CZ` banking intents | 563 / 574 | accuracy |
| Knowledge | Global-MMLU-Lite | 400 | accuracy |
| Instruction following | BenchMAX IFEval | 429 | prompt- and instruction-level strict accuracy |

The models:
- FunctionGemma-270M, Granite-4.0-H-350M, LFM2.5-350M, Qwen3.5-0.8B, LFM2.5-1.2B, Qwen3.5-2B, Gemma-4-E2B,
  SmolLM3-3B, tiny-aya-global, Granite-4.2-3B, Ministral-3-3B and Nemotron-3-Nano-4B.
- All use BF16 checkpoints, served by vLLM 0.31.0.
- Decoding is greedy with thinking disabled, using each model's native chat template and tool-call format.

## Layout

| Path | Contents |
|---|---|
| `slmbench/` | `prepare.py` (data), `generate.py` (vLLM runner), `judge.py` (Kimi K3), `score.py`, `viewer.py` |
| `configs/` | models and their vLLM settings, prompt templates (EN and CS) |
| `data/` | the frozen evaluation sets, one JSONL per skill and language |
| `outputs/<model>/` | raw generations per skill and language; `meta.json` with the checkpoint revision |
| `results/` | `scores.csv` (every metric), `items/` (per-item scores), `judge/` (raw judge verdicts) |
| `docs/` | the built viewer, served by GitHub Pages |
| `jobs/generate.pbs` | the PBS job used on MetaCentrum |
| `third_party/benchmax_ifeval/` | IFEval checkers from BenchMAX (MIT) |

## Reproduce

```bash
python -m slmbench.prepare all
qsub -v MODELS="qwen3.5-2b gemma-4-e2b" jobs/generate.pbs
python -m slmbench.judge
python -m slmbench.score
python -m slmbench.viewer --site docs
```

## Data sources

| Source | License |
|---|---|
| [BenchMAX](https://github.com/CONE-MT/BenchMAX) (Nexus and IFEval sets) | CC BY 4.0; checker code MIT |
| [NexusRaven API evaluation](https://huggingface.co/datasets/Nexusflow/NexusRaven_API_evaluation) (tool definitions) | no license stated |
| [NousResearch/json-mode-eval](https://huggingface.co/datasets/NousResearch/json-mode-eval) | no license stated |
| [EUR-Lex-Sum](https://huggingface.co/datasets/dennlinger/eur-lex-sum) | CC BY 4.0 |
| [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) | CC BY 4.0 |
| [Global-MMLU-Lite](https://huggingface.co/datasets/CohereLabs/Global-MMLU-Lite) | Apache 2.0 |

The summarization prompt and judge rubric follow Liu et al., *Summarization Is Not Dead Yet*
([arXiv 2606.08000](https://arxiv.org/abs/2606.08000)).

Computational resources were provided by the e-INFRA CZ project (ID:90254), supported by the Ministry of
Education, Youth and Sports of the Czech Republic.
