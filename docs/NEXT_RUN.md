# Ready for a future run

**No next run is approved or running.** This checklist is here so the next launch does not depend on chat history.

## Current decision

Do not continue the `none` adapter just because the 20-task check was positive. The 60-task check put the same final sampler behind base Inkling: **23/60 vs 27/60**, +3 / −7. Read [FINDINGS.md](FINDINGS.md) before spending more.

The reserved full session name remains `pingi`. It has not been created. The finished session `pingi-pilot` cannot be renamed or reused as the live driver.

## Saved checkpoint

Use these private Tinker paths, verified on 2026-11-27 with no scheduled expiration:

- Resume training: `tinker://00751c13-86fd-503f-99f9-e4db8ba28e63:train:0/weights/final`
- Evaluate: `tinker://00751c13-86fd-503f-99f9-e4db8ba28e63:train:0/sampler_weights/final`

Periodic checkpoints expire on 2026-11-28. Do not plan a resume around `000020`.

## Before any paid command

1. Confirm the dollar budget in writing. The trainer and evaluator do not enforce a spending cap.
2. Check the Tinker balance at https://tinker.thinkingmachines.ai/billing/balance.
3. Keep the Mac on AC power with the lid open. Screen lock is fine. Closing the lid kills the local driver.
4. Use Modal environment `pingi`, not `main`.
5. Use `behavior_if_log_dir_exists=raise` for a new run. Do not delete the pilot log directory.
6. Leave W&B off unless its config-upload bug is fixed. The pilot used `wandb_project=None`.
7. Do not download Harbor datasets inside the training process. The local cache is already populated for both training mixtures and Terminal-Bench 2.0.
8. Do not upload weights. The release gate is not bound to a checkpoint and manifest.

## If the next experiment is a `high` pilot

This starts from base Inkling. It does not repair the `none` result. Approve a budget before launching. The planning range is roughly **$55–75 on Tinker, plus Modal**, assuming the same 20-batch pilot schedule and the current 4,096-token generation cap. It is not a quote.

Use a new session name, `pingi-pilot-high`. Do not use `pingi` or `pingi-pilot`.

```sh
PINGI_MODAL_ENVIRONMENT=pingi uv run pingi-rl \
  preset=pilot \
  session_name=pingi-pilot-high \
  wandb_project=None \
  thinking_mode=high \
  max_tokens=4096 \
  behavior_if_log_dir_exists=raise
```

Do not set `load_checkpoint_path` for that experiment. Loading `weights/final` would continue the `none` adapter rather than train a fresh `high` pilot. After it finishes, evaluate it on a manifest frozen before launch, against base Inkling at the same `high` setting. Do not compare a `high` checkpoint with the old `none` scores.

## If the next experiment is more `none` training

Do not start it on the current evidence. If it is approved later anyway, resume only from `weights/final`, use a new session name, and freeze a new held-out manifest that excludes the pilot training tasks, dev tasks, the 20-task set, and the 60-task set.

## After a future run

Save the paired scores, manifest hash, checkpoint path, and cost before updating the public report. Keep infrastructure failures separate from wrong answers. Update [FINDINGS.md](FINDINGS.md), the Hugging Face report, and the XAI links together so the public pages do not keep an older result as the latest one.
