# Pilot setup

Names:

| Thing | Name |
|---|---|
| Project, GitHub, Hugging Face, Modal app | `pingi` |
| This ~$7 pilot | Tinker session `pingi-pilot` |
| Final full run | Tinker session `pingi` |

The pilot is 640 rollouts (`4 x 8 x 20`) of `harness-hub/scale800-harbor` and `harness-hub/termgrade-train-harbor` on Tinker. Thinking mode is `none`. Sandboxes run in the Modal environment `pingi`, not `main`.

```sh
PINGI_MODAL_ENVIRONMENT=pingi uv run pingi-rl \
  preset=pilot \
  session_name=pingi-pilot \
  wandb_name=pingi-pilot \
  wandb_project=None \
  behavior_if_log_dir_exists=delete
```

`wandb_project=None` is intentional for this launch. The cookbook's hyperparameter upload crashed the previous Forge run before training started. Metrics still land in the local run directory.

## Required before launch

| Piece | Status check | Where |
|---|---|---|
| Tinker | `TINKER_API_KEY` | https://tinker-console.thinkingmachines.ai |
| Modal | `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`, environment `pingi` | https://modal.com/settings/lyffseba/usage |
| Harbor tasks | already in `~/.cache/harbor/tasks/` | do not download inside the training process |
| W&B | optional; this launch is local-only | https://forge.coreweave.com |

Download both Harbor datasets before training. `pingi-rl` only reads the local cache:

```sh
uv run hf download harness-hub/scale800-harbor --repo-type dataset --max-workers 2 \
  --local-dir ~/.cache/harbor/tasks/harness-hub__scale800-harbor
uv run hf download harness-hub/termgrade-train-harbor --repo-type dataset --max-workers 2 \
  --local-dir ~/.cache/harbor/tasks/harness-hub__termgrade-train-harbor
```

## Modal

Use environment `pingi`. In the old `main` environment, Modal created a sandbox ID and then reported the container already shut down (`NotFoundError`, error code `C0MJ62F2`) before any command could run. A new environment in the same workspace executes `echo` and Harbor `test.sh`. The code defaults to `PINGI_MODAL_ENVIRONMENT=pingi`.

Sandbox creates are capped (`PINGI_MODAL_CONCURRENCY=2`) so AppGetOrCreate rate limits do not starve the Tinker heartbeat. One shared Modal app is looked up once.

Local Docker is only the fallback for `termgrade-train-harbor`, which shares one image. Do not run `scale800-harbor` locally: it is 750 separate images.

A clean start means stop the local `pingi-rl` process and terminate sandboxes in the `pingi` environment before launching again. Tinker keeps old sessions visible until they expire; the live one is the new `pingi-pilot` session.

## Reference Benchmarks

Our primary apples-to-apples reference is [`harness-hub/Qwen3.8-27B-pi-adapted`](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted) (same `pi` harness, same `scale800 + termgrade` mixture, thinking OFF, LoRA r=32: **55.8%** Terminal-Bench 2.1, **64.7%** SWE-bench Pro hard-51). See [BENCHMARKS.md](BENCHMARKS.md) for the full table (including base `Inkling-2-Mini` and `unsloth/Inkling-2-Mini-GGUF` quantization reference).

After the pilot checkpoint exists, compare it to base `Inkling-2-Mini` on the held-out file written next to the run log:

```sh
uv run pingi-eval \
  checkpoint_path=tinker://<pilot-run>/sampler_weights/final \
  compare_with_base=True \
  heldout_file=<run-log>/heldout_tasks.json \
  datasets=harness-hub/scale800-harbor,harness-hub/termgrade-train-harbor \
  sandbox=modal \
  max_tasks=20
```

Do not treat Unsloth's KL table as a PINGI pass-rate score.
