---
base_model: thinkingmachines/Inkling-2-Mini
tags:
- pi
- tinker
- harbor
- rl
- coding
---

# PINGI — pilot report

**Pilot complete; project paused. This repository contains documentation and evaluation records, not adapter weights.** PINGI adapts [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) for the [`pi`](https://pi.dev/) coding harness (`read`, `bash`, `edit`, `write`) using agentic RL on Tinker and Harbor.

## Pilot result (2026-11-22)

| Model | Held-out Pass@1 | Median turns | Median tool calls | Edit error rate |
|---|---:|---:|---:|---:|
| Base Inkling-2-Mini | 10/20 (50%) | 15.5 | 13 | 0% |
| `pingi-pilot` final checkpoint | 11/20 (55%) | 14.0 | 13 | 0% |

**One newly solved task, none lost.** Both models ran on the same 20 frozen Harbor tasks excluded from training and dev: 10 from `scale800-harbor` and 10 from `termgrade-train-harbor`. Both used thinking `none`, temperature `0.0`, and a 25-turn cap. No task-level eval errors were recorded. One additional solve is a preliminary signal, not a general performance claim or a direct comparison with `harness-hub/Qwen3.8-27B-pi-adapted`, Terminal-Bench, or SWE-bench.

[Held-out manifest](pilot-results/heldout_tasks.json) · [Base per-task scores](pilot-results/base.jsonl) · [Pilot per-task scores](pilot-results/pingi.jsonl) · [Eval settings and file hashes](pilot-results/run.json)

## Build record

The [pilot build log](https://github.com/lyffseba/pingi/blob/main/docs/PILOT.md) covers the Docker smoke test, the switch to the dedicated Modal `pingi` environment, the run settings, and what broke along the way. The pilot started from base Inkling-2-Mini, trained for 20 batches (640 nominal rollouts) with LoRA rank 32 and thinking `none`, and recorded **$18.73** in Tinker training-run cost, excluding Modal and the standalone paired eval.

Tinker project `pingi`, session `pingi-pilot`: the final training and sampler checkpoints are saved privately. The larger run (`pingi`) has not started. No pilot weights have been exported or made public.

[Source code](https://github.com/lyffseba/pingi) · [Smoke record](https://huggingface.co/lyffseba/pingi-smo) · [XAI research notes](https://github.com/lyffseba/xai)
