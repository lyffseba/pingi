---
base_model: thinkingmachines/Inkling-2-Mini
license: apache-2.0
tags:
- pi
- inkling-2-mini
- tinker
- smoke-test
---

# pingi-smo

Public smoke-test checkpoint for **PINGI** ([GitHub](https://github.com/lyffseba/pingi) · [Main HF Hub](https://huggingface.co/lyffseba/pingi) · [xai Space](https://huggingface.co/spaces/lyffseba/xai)) on `thinkingmachines/Inkling-2-Mini`.

The smoke weights are published on Tinker, not stored in this repo. Thinking Machines disables file downloads for this base model (`403 Checkpoint downloads are disabled`). For the separate completed pilot and paired held-out results, see the [pilot report](https://huggingface.co/lyffseba/pingi). The pilot weights remain private.

## Published Tinker checkpoints

- Weights: `tinker://e7bba24a-6366-5c8c-93b8-21e624a621a6:train:0/weights/final`
- Sampler: `tinker://e7bba24a-6366-5c8c-93b8-21e624a621a6:train:0/sampler_weights/final`

Both paths were published with `publish_checkpoint_from_tinker_path`, so another Tinker account can load them.

- Console session: `e7bba24a-6366-5c8c-93b8-21e624a621a6`
- Legacy Tinker display name: `pingy-smo`
- [Tinker Console](https://tinker-console.thinkingmachines.ai)

LoRA rank 32, attention + MLP + unembed, AdamW.

## What this run is

A 3-step Docker smoke test on `terminal-bench-2.0`, thinking mode `none`. It checks that the `pi` tool loop, Harbor grader, Tinker session, and checkpoint publish all work. It is not a finished coding model.

| Step | test_passed | reward | turns / episode |
|---:|---:|---:|---:|
| 0 | 1.0 | 1.015 | 12 |
| 1 | 0.0 | 0.0 | 15 (turn cap) |
| 2 | 1.0 | 1.054 | 11 |

## Sample it

```python
import tinker

service = tinker.ServiceClient()
sampler = service.create_sampling_client(
    model_path="tinker://e7bba24a-6366-5c8c-93b8-21e624a621a6:train:0/sampler_weights/final",
    base_model="thinkingmachines/Inkling-2-Mini",
).result()
```
