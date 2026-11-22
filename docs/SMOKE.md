# Smoke run: pingi-smo

Completed 2026-11-21. This was an infrastructure check, not a benchmark. See the [completed pilot report](PILOT.md) for the separate held-out comparison.

## What ran

```sh
uv run pingi-rl preset=smoke behavior_if_log_dir_exists=delete wandb_project=None session_name=pingi-smo
```

- Model: `thinkingmachines/Inkling-2-Mini`
- Thinking: `none`
- 3 steps, 2 groups x 2 rollouts, `eval_every=0`
- Dataset: local `terminal-bench-2.0`
- Sandbox: Docker
- Local log: `/Users/juan.gallego/tinker-runs/pingi/pingi-smoke-none-r32-lr1e-05-g2x2-20261121-0847`

## Tinker console

The run is session `e7bba24a-6366-5c8c-93b8-21e624a621a6`. The public record is `pingi-smo`; the original Tinker display name remains `pingy-smo`.

https://tinker-console.thinkingmachines.ai

Checkpoints:

- `tinker://e7bba24a-6366-5c8c-93b8-21e624a621a6:train:0/weights/final`
- `tinker://e7bba24a-6366-5c8c-93b8-21e624a621a6:train:0/sampler_weights/final`

## Hugging Face

https://huggingface.co/lyffseba/pingi-smo

The Hub repo is the run record. The actual checkpoint is published on Tinker, not uploaded as safetensors, because Tinker returns `403 Checkpoint downloads are disabled for base model thinkingmachines/Inkling-2-Mini`.

Published paths:

- `tinker://e7bba24a-6366-5c8c-93b8-21e624a621a6:train:0/weights/final`
- `tinker://e7bba24a-6366-5c8c-93b8-21e624a621a6:train:0/sampler_weights/final`

## Train metrics

| Step | test_passed | reward | turns / episode |
|---:|---:|---:|---:|
| 0 | 1.0 | 1.015 | 12 |
| 1 | 0.0 | 0.0 | 15 (max turns) |
| 2 | 1.0 | 1.054 | 11 |
