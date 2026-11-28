---
base_model: thinkingmachines/Inkling-2-Mini
tags:
- pi
- tinker
- harbor
- rl
- coding
---

# PINGI — findings report

**Pilot complete; not released; no next run started.** This repository contains documentation and evaluation records, not adapter weights. PINGI adapts [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) for the [`pi`](https://pi.dev/) coding harness (`read`, `bash`, `edit`, `write`) using agentic RL on Tinker and Harbor.

## Latest held-out result (2026-11-27)

| Model | Held-out Pass@1 | Median turns | Median tool calls |
|---|---:|---:|---:|
| Base Inkling-2-Mini | 27/60 (45.0%) | 15.0 | 11.5 |
| `pingi-pilot` final checkpoint | 23/60 (38.3%) | 16.0 | 13.0 |

**3 newly solved, 7 lost.** The same private final sampler was behind base Inkling on 60 fresh tasks from the training mixture. `scale800` was tied at 14/30. `termgrade` was 9/30 against 13/30. Thinking was `none`, temperature `0.0`, and the turn cap was 25. No task errors were recorded. This is not a Terminal-Bench, SWE-bench, or Qwen result.

[60-task manifest](followup-eval/heldout_tasks.json) · [Base scores](followup-eval/results/base.jsonl) · [Pilot scores](followup-eval/results/pingi.jsonl) · [Hashes](followup-eval/results/run.json)

## Earlier 20-task check (2026-11-22)

| Model | Held-out Pass@1 | Median turns | Median tool calls |
|---|---:|---:|---:|
| Base Inkling-2-Mini | 10/20 (50%) | 15.5 | 13 |
| `pingi-pilot` final checkpoint | 11/20 (55%) | 14.0 | 13 |

That check gained one task and lost none. It is a separate frozen set. Do not average it with the 60-task result.

[20-task manifest](pilot-results/heldout_tasks.json) · [Base scores](pilot-results/base.jsonl) · [Pilot scores](pilot-results/pingi.jsonl)

## Separate high-thinking check (2026-11-28)

This is not a third held-out score. It reran the 10 tasks where the models already disagreed, at thinking `high` and an 8,192-token generation cap.

| Model | These 10 tasks at `high` | Same 10 tasks at `none` |
|---|---:|---:|
| Base Inkling-2-Mini | 0/10 | 7/10 |
| `pingi-pilot` final checkpoint | 1/10 | 3/10 |

The pilot's only pass was `task_12182_baae2f`, a task it had lost at `none`. The generated score file says +1 / −0. That is only the comparison at `high`. It is not an improvement over the `none` result on the same tasks. No task errors were recorded. Tinker reported **$0.35**, excluding Modal. Do not average this with either held-out check.

[Note](high-check/README.md) · [Manifest](high-check/heldout_tasks.json) · [Base scores](high-check/results/base.jsonl) · [Pilot scores](high-check/results/pingi.jsonl) · [Hashes](high-check/results/run.json)

## Build record

The pilot trained for 20 batches from base Inkling-2-Mini, with LoRA rank 32 and thinking `none`. It recorded **$18.73** in Tinker training cost, excluding Modal and the standalone evals. Final checkpoints remain private on Tinker and had no scheduled expiration when rechecked on 2026-11-28 at 09:35 UTC. Periodic checkpoints were still present then. The earliest expires at 16:31 UTC the same day. No `high` training run and no full `pingi` run have been started. The saved results cannot authorize a weight upload.

[Findings](https://github.com/lyffseba/pingi/blob/main/docs/FINDINGS.md) · [Next-run checklist](https://github.com/lyffseba/pingi/blob/main/docs/NEXT_RUN.md) · [Source](https://github.com/lyffseba/pingi) · [Smoke record](https://huggingface.co/lyffseba/pingi-smo) · [XAI notes](https://github.com/lyffseba/xai)
