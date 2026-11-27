# PINGI findings through 2026-11-27

**Status: paused and ready for a future run. Nothing is training or evaluating.** This page is the complete record of what was measured. It is not a model release and not permission to start the next paid run.

## Bottom line

The `none` pilot is real, saved, and not good enough to scale. A 20-task check showed one extra solve. A later 60-task check on fresh tasks from the same mixture reversed that: the pilot solved fewer tasks than base Inkling and lost more than it gained. No `high` training run has been started. No adapter weights have been published.

Do not pool the 20-task and 60-task numbers into one score. They are separate frozen sets.

## What ran

| Stage | When | What it was | Result |
|---|---|---|---|
| Smoke | 2026-11-21 | 3 Docker steps on Terminal-Bench 2.0, thinking `none` | Infrastructure check only. Not a benchmark. [Record](SMOKE.md). |
| Pilot training | 2026-11-21, finished 22:03 CET | 20 synchronous batches, 640 nominal rollouts, LoRA rank 32, learning rate `1e-5`, temperature `0.8`, thinking `none` | Completed. Tinker session `pingi-pilot`. Recorded Tinker cost **$18.73**, excluding Modal and later evals. |
| 20-task paired eval | 2026-11-22 | Same final sampler vs base, thinking `none`, temperature `0.0`, 25 turns | **11/20 vs 10/20**. +1 newly solved, −0 lost. |
| 60-task paired eval | 2026-11-27 | Fresh manifest, same checkpoint and settings | **23/60 vs 27/60**. +3 newly solved, −7 lost. |
| `high` pilot | Not started | Would be a new training run from base Inkling with thinking `high` (`0.9`) | No budget approved. No process launched. |

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

The loss is concentrated in `termgrade`. `scale800` was tied. The recorded eval telemetry counted **zero `edit` tool calls** for both models, so the 0% edit-error rate is not evidence about edit quality. Both models used the same `read`, `bash`, `edit`, and `write` tool definitions; they simply did not record an `edit` call on these tasks.

This was not Terminal-Bench, SWE-bench, or a comparison with `harness-hub/Qwen3.8-27B-pi-adapted`. It was a same-mixture generalization check. It does not prove the adapter has no useful behavior, and it does not justify more `none` RL or a weight release.

Evidence: [manifest](followup-eval/heldout_tasks.json) · [base](followup-eval/results/base.jsonl) · [pilot](followup-eval/results/pingi.jsonl) · [hashes](followup-eval/results/run.json). Manifest SHA-256: `c0db6562d074188784f81bc1ab476886081dfa43818cd73ac450e9f67ab83142`.

## What the build taught us

- Tinker can train `thinkingmachines/Inkling-2-Mini`. River and Modular were not the training backend for this model.
- Modal environment `main` shut sandboxes down before commands ran. Environment `pingi` worked. The driver must keep the lid open on AC power; locking the screen is fine.
- W&B was disabled after a config-upload crash. Local metrics were enough to finish the pilot.
- One training batch stalled on a Modal exec-status retry. Terminating that one sandbox let the batch finish. The run still completed and saved `final`.
- The old ~$7 pilot estimate was not a cap. The recorded Tinker bill was **$18.73**, excluding Modal and standalone evals.
- A `high` pilot is not a cheap flag flip. The `none` run sampled 6,801,485 tokens and trained 15,470,364. Thinking `high` emits more tokens inside the same 4,096-token generation cap. Applying the project's own 3–4× token claim to the $18.73 bill gives a planning range of roughly **$55–75 on Tinker, plus Modal**. That is not a quote and not a stop. Raising `max_tokens` above 4,096 could cost more.
- A weight upload now requires one paired eval whose recorded checkpoint and task manifest match the checkpoint being uploaded, plus a net gain over base. The 60-task result fails that test. `allow_unverified=True` can bypass the check, so do not use it for a public release.

## What is saved for later

Verified on 2026-11-27:

- Tinker project `pingi`, session `pingi-pilot` (`00751c13-86fd-503f-99f9-e4db8ba28e63`), finished successfully, not corrupted.
- `weights/final` and `sampler_weights/final` are private and have no scheduled expiration.
- Periodic checkpoints `000005` through `000020` expire on 2026-11-28. Use `final` for later work.
- Modal app `pingi` had 0 tasks at the documentation check.
- No local `pingi-rl` or `pingi-eval` process was running.

The checkpoint URIs document provenance. They do not grant access to another account.

## What not to claim

- Do not say the pilot beats base Inkling. The larger paired check says it does not.
- Do not say it matches or trails `Qwen3.8-27B-pi-adapted`. That comparison has not been run.
- Do not publish the private checkpoints or call the Hugging Face repo a weight release.
- Do not start the reserved full session `pingi`, an MVP run, or a `high` pilot until a budget is explicitly approved.
