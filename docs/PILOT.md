# PINGI pilot: build log and results

**Status: completed and paused.** Training finished on 2026-11-21 at 22:03 CET; paired evaluation finished on 2026-11-22. This is a public pilot report, **not a public adapter release**. The larger run, whose Tinker session is reserved as `pingi`, has not started.

## What the pilot measured

| Model | Held-out Pass@1 | Median turns | Median tool calls | Edit error rate |
|---|---:|---:|---:|---:|
| Base `thinkingmachines/Inkling-2-Mini` | 10/20 (50%) | 15.5 | 13 | 0% |
| `pingi-pilot` (`sampler_weights/final`) | 11/20 (55%) | 14.0 | 13 | 0% |

**Paired result: +1 newly solved, −0 lost (+5 percentage points).** Both models solved 10 tasks and neither solved 9. The single change was `life4__textdistance.c3aca916.combine_file__bbyj98tv`. No task-level eval errors were recorded. Both models used the same 20 tasks (10 `scale800-harbor`, 10 `termgrade-train-harbor`), the same `read`, `bash`, `edit`, `write` harness, thinking `none`, temperature `0.0`, and a 25-turn cap. These tasks were excluded from training and in-loop dev. This is one small paired result: it does not establish a general improvement and is not a Terminal-Bench, SWE-bench, or Qwen comparison.

**Evidence:** [task manifest](pilot-results/heldout_tasks.json) · [base per-task scores](pilot-results/base.jsonl) · [pilot per-task scores](pilot-results/pingi.jsonl) · [eval settings and file hashes](pilot-results/run.json). These records omit machine-local paths; prompts, generated traces, credentials, and weights are not part of the report.

## How we built it

| Stage | What happened |
|---|---|
| Smoke | Ran a [three-step Docker smoke test](SMOKE.md) to verify the `pi` tools, Harbor verifier, Tinker training, and checkpoint save. Not a benchmark. |
| Pilot setup | Pre-cached both Harbor datasets, then used Modal rather than building hundreds of task images on local Docker. Modal `main` had sandbox shutdowns before commands could run; environment `pingi` worked. The driver reuses one app and limits concurrent sandbox creation and setup. |
| Pilot training | Started from base Inkling-2-Mini on Tinker: LoRA rank 32, learning rate `1e-5`, temperature `0.8`, thinking `none`. Completed 20 batches with 640 nominal rollouts (`4 × 8 × 20`; excludes eval and extra sampling). The trainer filters constant-reward groups, so 640 is not the count of training examples used. |
| Operations | W&B was disabled after a config-upload failure; metrics remained local. One early batch stopped progressing on an exec status retry. Terminating only the stuck sandbox let the batch finish without restarting the driver. The run ultimately completed successfully. |
| Cost | **$18.73** in recorded Tinker training-run cost, including in-loop work, but excluding Modal and the separate paired evaluation. The initial ~$7 estimate was not a spending cap and underestimated this run. |
| Evaluation and pause | Evaluated the saved final sampler on the frozen 20-task manifest. Paused after recording the results; no larger run or public pilot weight release. |

## Tinker checkpoint record

- Project: `pingi`; session: `pingi-pilot` (`00751c13-86fd-503f-99f9-e4db8ba28e63`).
- Resume: `tinker://00751c13-86fd-503f-99f9-e4db8ba28e63:train:0/weights/final`
- Eval: `tinker://00751c13-86fd-503f-99f9-e4db8ba28e63:train:0/sampler_weights/final`
- Tinker confirmed both final checkpoints were private and had no scheduled expiration when verified on 2026-11-22. Periodic saves `000005`, `000010`, `000015`, and `000020` had the configured seven-day retention. Use `final` for later work. The URIs document provenance; they do not grant access.

## Reproduce the setup

Create or select a Tinker project, then set `TINKER_PROJECT_ID` to its ID in the gitignored `.env` alongside `TINKER_API_KEY` and your Modal credentials. The `TINKER_PROJECT_ID` example in the repo is for the original project; use your own if you are reproducing the run. Use Modal environment `pingi`, or set `PINGI_MODAL_ENVIRONMENT` to your chosen environment. For a Mac driver, keep it on AC power with the lid open while jobs run.

Download both Harbor datasets before training; the trainer reads the local cache:

```sh
uv run hf download harness-hub/scale800-harbor --repo-type dataset --max-workers 2 \
  --local-dir ~/.cache/harbor/tasks/harness-hub__scale800-harbor
uv run hf download harness-hub/termgrade-train-harbor --repo-type dataset --max-workers 2 \
  --local-dir ~/.cache/harbor/tasks/harness-hub__termgrade-train-harbor
```

For a new run, choose a new log directory rather than deleting an existing run:

```sh
PINGI_MODAL_ENVIRONMENT=pingi uv run pingi-rl \
  preset=pilot session_name=pingi-pilot wandb_project=None \
  behavior_if_log_dir_exists=raise
```

For an account with access to the evaluated checkpoint, the published manifest can be reused with the cached datasets:

```sh
PINGI_MODAL_ENVIRONMENT=pingi uv run pingi-eval \
  checkpoint_path=tinker://00751c13-86fd-503f-99f9-e4db8ba28e63:train:0/sampler_weights/final \
  compare_with_base=True heldout_file=docs/pilot-results/heldout_tasks.json \
  datasets=harness-hub/scale800-harbor,harness-hub/termgrade-train-harbor \
  sandbox=modal max_tasks=20 thinking_mode=none temperature=0.0 max_turns=25
```

## Before expanding or releasing weights

Freeze a larger fresh set that excludes training and dev tasks. Compare the same checkpoint with base Inkling again, then do a matched Qwen run if the gain holds. The current upload check scores the latest paired results but does not bind them to the requested checkpoint or task manifest; fix that before using it to authorize a release. Set a separate budget for the next eval before starting it.
