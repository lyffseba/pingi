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

**Do not scale this pilot, and do not treat this repository as a weight release.** It contains the score files only. The checkpoints remain private.

PINGI is a LoRA adapter for [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) inside the [`pi`](https://pi.dev/) tool loop. The result that matters is the 60-task check below. The other checks are separate and must not be averaged with it.

## The result that matters (2026-11-27)

| 60 fresh tasks, thinking off | Base | Pilot |
|---|---:|---:|
| Repair an existing repository (`scale800`) | 14/30 | 14/30 |
| Write a new tool from a contract (`termgrade`) | 13/30 | 9/30 |
| All 60 | 27/60 | 23/60 |

The pilot newly solved 3 tasks and lost 7. The whole deficit is the contract-writing row. No task errors were recorded. Median turns were 15.0 for the base model and 16.0 for the pilot. This is not Terminal-Bench, SWE-bench, or a Qwen comparison.

[Manifest](followup-eval/heldout_tasks.json) · [Base scores](followup-eval/results/base.jsonl) · [Pilot scores](followup-eval/results/pingi.jsonl) · [Hashes](followup-eval/results/run.json)

## Earlier 20-task check (2026-11-22)

The same checkpoint scored 11/20 against the base model's 10/20, with one new solve and no losses. It is a different frozen set. Do not average it with 23/60.

[Manifest](pilot-results/heldout_tasks.json) · [Base scores](pilot-results/base.jsonl) · [Pilot scores](pilot-results/pingi.jsonl)

## Thinking high made both models worse (2026-11-28)

This reran only the 10 tasks where the models already disagreed. Read the columns downward.

| Same 10 tasks | Base | Pilot |
|---|---:|---:|
| Thinking off | 7/10 | 3/10 |
| Thinking high | 0/10 | 1/10 |

The bottom row is not a win. The base model had solved 7 of these 10 with thinking off. The pilot's only pass was `task_12182_baae2f`, and it lost the three tasks it had won before. The generation cap was 8,192 rather than 4,096. Tinker reported **$0.35**, excluding Modal. No training run was started.

[Note](high-check/README.md) · [Manifest](high-check/heldout_tasks.json) · [Base scores](high-check/results/base.jsonl) · [Pilot scores](high-check/results/pingi.jsonl) · [Hashes](high-check/results/run.json)

## What is saved

Training cost **$18.73** on Tinker, excluding Modal and the later evaluations. Final checkpoints were rechecked on 2026-11-28 at 09:41 UTC: private, with no scheduled expiration. Periodic checkpoints were still present then. The earliest expires at 16:31 UTC the same day. No `high` training run and no full `pingi` run have been started.

[Findings](https://github.com/lyffseba/pingi/blob/main/docs/FINDINGS.md) · [Next-run checklist](https://github.com/lyffseba/pingi/blob/main/docs/NEXT_RUN.md) · [Source](https://github.com/lyffseba/pingi) · [XAI note](https://huggingface.co/spaces/lyffseba/xai/blob/main/pingi.html)
