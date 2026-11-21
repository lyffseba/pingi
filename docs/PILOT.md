# Pilot setup

The ~$7 run is:

```sh
uv run pingy-rl preset=pilot session_name=pingy-pilot behavior_if_log_dir_exists=delete
```

That is 640 rollouts (`4 x 8 x 20`) of `harness-hub/scale800-harbor` and `harness-hub/termgrade-train-harbor` on Tinker, with Modal sandboxes. Thinking mode is `none`.

## Required before launch

| Piece | Status check | Where |
|---|---|---|
| Tinker | `TINKER_API_KEY` | https://tinker-console.thinkingmachines.ai |
| Modal | `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` | https://modal.com/settings |
| W&B | `WANDB_API_KEY` | https://wandb.ai/authorize |

`pingy-rl` refuses to start `preset=pilot` without Modal. If W&B is missing it still trains and writes local metrics, but there is no live chart.

## Reference Benchmarks

Our primary apples-to-apples reference is [`harness-hub/Qwen3.8-27B-pi-adapted`](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted) (same `pi` harness, same `scale800 + termgrade` mixture, thinking OFF, LoRA r=32: **55.8%** Terminal-Bench 2.1, **64.7%** SWE-bench Pro hard-51). See [BENCHMARKS.md](BENCHMARKS.md) for the full table (including base `Inkling-2-Mini` and `unsloth/Inkling-2-Mini-GGUF` quantization reference).

After the pilot checkpoint exists, compare it to base `Inkling-2-Mini` on the held-out file written next to the run log:

```sh
uv run pingy-eval \
  checkpoint_path=tinker://<pilot-run>/sampler_weights/final \
  compare_with_base=True \
  heldout_file=<run-log>/heldout_tasks.json \
  datasets=harness-hub/scale800-harbor,harness-hub/termgrade-train-harbor \
  sandbox=modal \
  max_tasks=20
```

Do not treat Unsloth's KL table as a PINGY pass-rate score.
