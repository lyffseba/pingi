# PINGI findings through 2026-11-28

**Status: paused. Nothing is training or evaluating.** This page is the complete record of what was measured. It is not a model release and not permission to start the next paid run.

## Bottom line

The `none` pilot is real, saved, and not good enough to scale. A 20-task check showed one extra solve. A later 60-task check on fresh tasks from the same mixture reversed that: the pilot solved fewer tasks than base Inkling and lost more than it gained. A later `high` inference check on the 10 disagreement tasks made both models worse. No `high` training run has been started. No adapter weights have been published.

Do not pool the 20-task, 60-task, and 10-task numbers into one score. They are separate checks.

## What ran

| Stage | When | What it was | Result |
|---|---|---|---|
| Smoke | 2026-11-21 | 3 Docker steps on Terminal-Bench 2.0, thinking `none` | Infrastructure check only. Not a benchmark. [Record](SMOKE.md). |
| Pilot training | 2026-11-21, finished 22:03 CET | 20 synchronous batches, 640 nominal rollouts, LoRA rank 32, learning rate `1e-5`, temperature `0.8`, thinking `none` | Completed. Tinker session `pingi-pilot`. Recorded Tinker cost **$18.73**, excluding Modal and later evals. |
| 20-task paired eval | 2026-11-22 | Same final sampler vs base, thinking `none`, temperature `0.0`, 25 turns | **11/20 vs 10/20**. +1 newly solved, −0 lost. |
| 60-task paired eval | 2026-11-27 | Fresh manifest, same checkpoint and settings | **23/60 vs 27/60**. +3 newly solved, −7 lost. |
| 10-task `high` inference check | 2026-11-28 | The 7 losses and 3 wins from the 60-task check, not a new held-out set. Thinking `high`, temperature `0.0`, 25 turns, 8,192-token generation cap | **1/10 vs 0/10**. On the same 10 tasks at `none`, the scores were 3/10 vs 7/10. Tinker reported **$0.35**, excluding Modal. |
| `high` training pilot | Not started | Would be a new training run from base Inkling with thinking `high` (`0.9`) | Not approved. The inference check does not justify it. |

The first attempt to start the 60-task eval was stopped after about 3 minutes, before any task result was saved. The completed eval is `20261127-192206-004981Z`. Its exact aborted Tinker cost was not separately invoiced here; do not treat the planning estimate as that bill.

## 20-task pilot result

| Model | Pass@1 | Median turns | Median tool calls |
|---|---:|---:|---:|
| Base Inkling-2-Mini | 10/20 (50%) | 15.5 | 13 |
| `pingi-pilot` final sampler | 11/20 (55%) | 14.0 | 13 |

The only change was `life4__textdistance.c3aca916.combine_file__bbyj98tv`, solved by the pilot and not by the base model. No task errors. The set was 10 `scale800-harbor` and 10 `termgrade-train-harbor` tasks excluded from training and in-loop dev.

Evidence: [manifest](pilot-results/heldout_tasks.json) · [base](pilot-results/base.jsonl) · [pilot](pilot-results/pingi.jsonl) · [hashes](pilot-results/run.json).

## 60-task follow-up result

| Model | Pass@1 | Median turns | Median tool calls |
|---|---:|---:|---:|
| Base Inkling-2-Mini | 27/60 (45.0%) | 15.0 | 11.5 |
| `pingi-pilot` final sampler | 23/60 (38.3%) | 16.0 | 13.0 |

Paired counts on the same 60 tasks: 20 solved by both, 3 by the pilot only, 7 by the base only, 30 by neither. No task errors.

| Source | Base | Pilot |
|---|---:|---:|
| `harness-hub/scale800-harbor` | 14/30 | 14/30 |
| `harness-hub/termgrade-train-harbor` | 13/30 | 9/30 |

Those source names are also the task types. Every `scale800` task was a repair to an existing repository, and the models tied at 14/30. Every `termgrade` task was a new tool written from a written contract, and the pilot scored 9/30 against the base model's 13/30. That is the whole 4-task deficit. The contract instructions are long and exact. On 4 of the 7 losses, the pilot used fewer turns than the base model, with a median of 14 against 19, so it often stopped earlier and was wrong. The saved scores do not include the full trajectories.

The recorded eval telemetry counted **zero `edit` tool calls** for both models, so the 0% edit-error rate is not evidence about edit quality. Both models used the same `read`, `bash`, `edit`, and `write` tool definitions; they simply did not record an `edit` call on these tasks.

This was not Terminal-Bench, SWE-bench, or a comparison with `harness-hub/Qwen3.8-27B-pi-adapted`. It was a same-mixture generalization check. It does not prove the adapter has no useful behavior, and it does not justify more `none` RL or a weight release.

Evidence: [manifest](followup-eval/heldout_tasks.json) · [base](followup-eval/results/base.jsonl) · [pilot](followup-eval/results/pingi.jsonl) · [hashes](followup-eval/results/run.json). Manifest SHA-256: `c0db6562d074188784f81bc1ab476886081dfa43818cd73ac450e9f67ab83142`.

## 10-task high-thinking check

This reran the disagreement slice. It is not a third held-out benchmark.

| Model | These 10 tasks at `high` | Same 10 tasks at `none` |
|---|---:|---:|
| Base Inkling-2-Mini | 0/10 | 7/10 |
| `pingi-pilot` final sampler | 1/10 | 3/10 |

The pilot's only pass was `task_12182_baae2f`, which it had lost at `none`. It lost the three tasks it had won before. No task errors. The generation cap was 8,192 rather than the 4,096 used in the `none` evals, so the drop cannot be attributed to thinking mode alone. It is still large enough to reject a `high` training pilot on this evidence. The recorded Tinker cost was **$0.35** through 09:00 UTC, excluding Modal, inside the approved $10 budget.

Evidence: [note](high-check/README.md) · [manifest](high-check/heldout_tasks.json) · [base](high-check/results/base.jsonl) · [pilot](high-check/results/pingi.jsonl) · [hashes](high-check/results/run.json).

## What the build taught us

- Tinker can train `thinkingmachines/Inkling-2-Mini`. River and Modular were not the training backend for this model.
- Modal environment `main` shut sandboxes down before commands ran. Environment `pingi` worked. The driver must keep the lid open on AC power; locking the screen is fine.
- W&B was disabled after a config-upload crash. Local metrics were enough to finish the pilot.
- One training batch stalled on a Modal exec-status retry. Terminating that one sandbox let the batch finish. The run still completed and saved `final`.
- The old ~$7 pilot estimate was not a cap. The recorded Tinker bill was **$18.73**, excluding Modal and standalone evals.
- A `high` training pilot is not a cheap flag flip. The `none` run sampled 6,801,485 tokens and trained 15,470,364. Thinking `high` emits more tokens inside the same 4,096-token generation cap. Applying the project's own 3–4× token claim to the $18.73 bill gives a planning range of roughly **$55–75 on Tinker, plus Modal**. That is not a quote. The later inference check is a reason not to spend it. Raising `max_tokens` above 4,096 could cost more.
- A weight upload now requires one paired eval whose recorded checkpoint and task manifest match the checkpoint being uploaded, plus a net gain over base. The 60-task result fails that test. `allow_unverified=True` can bypass the check, so do not use it for a public release.

## What is saved for later

Final checkpoints rechecked on 2026-11-28 at 09:35 UTC:

- Tinker project `pingi`, session `pingi-pilot` (`00751c13-86fd-503f-99f9-e4db8ba28e63`), finished successfully, not corrupted.
- `weights/final` and `sampler_weights/final` are private and have no scheduled expiration.
- Periodic checkpoints `000005` through `000020` were still present and private. The earliest expires at 16:31 UTC on 2026-11-28. Use `final` for later work.
- Modal app `pingi` had 0 tasks at the documentation check.
- No local `pingi-rl` or `pingi-eval` process was running.

The checkpoint URIs document provenance. They do not grant access to another account.

## What not to claim

- Do not say the pilot beats base Inkling. The 60-task check says it does not. The 1/10 on the disagreement slice does not either.
- Do not average the 20-task, 60-task, and 10-task checks.
- Do not say thinking `high` repaired the pilot. On that slice it made both models worse.
- Do not say it matches or trails `Qwen3.8-27B-pi-adapted`. That comparison has not been run.
- Do not publish the private checkpoints or call the Hugging Face repo a weight release.
- Do not start the reserved full session `pingi`, an MVP run, or a `high` training pilot until a budget is explicitly approved.
