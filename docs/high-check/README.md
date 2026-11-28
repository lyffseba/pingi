# High-thinking check

**Completed on 2026-11-28. This was an inference comparison, not a training run, and it is not a new held-out score.** The approved budget was $10 total, including Tinker and Modal. Tinker reported **$0.35** through 09:00 UTC for session `6557f3ea-6edb-5642-9dbb-2e0670a6f06a`. That figure excludes Modal. No second run was started.

The manifest is the 10 tasks where `pingi-pilot` and base Inkling disagreed on the 60-task `none` check: 7 losses and 3 wins. Both models used thinking `high`, temperature `0.0`, 25 turns, 8,192 generated tokens per step, and the 49,152-token trajectory cap. The generation cap was higher than the `none` evals, so this is not a pure thinking comparison.

| Model | These 10 tasks at `high` | Same 10 tasks at `none` |
|---|---:|---:|
| Base Inkling-2-Mini | 0/10 | 7/10 |
| `pingi-pilot` final sampler | 1/10 | 3/10 |

The pilot's only pass was `task_12182_baae2f`, a task it had lost at `none`. It lost all three tasks it had won before. No task errors were recorded. Median turns were 10.0 for the base model and 17.0 for the pilot. Do not average these 10 tasks with the 20-task or 60-task results. It does not change the separate 23/60 versus 27/60 result, and it is not evidence that the pilot beats the base model. The checkpoint remains private.

Evidence: [manifest](heldout_tasks.json) · [base](results/base.jsonl) · [pilot](results/pingi.jsonl) · [hashes](results/run.json) · [annotated score table](results/summary.md). Manifest SHA-256: `841d2417b431c6c99784ceaf5e6db9284cc4fa38bf2dd58c70fe02d0ee6d48ae`. The score table includes the generator's +1 / −0 line, but that line is only the `high` comparison. It is not an improvement over the `none` scores on the same tasks.

The command below is the record of what ran. It is not approval to run it again.

```sh
PINGI_MODAL_ENVIRONMENT=pingi uv run pingi-eval \
  checkpoint_path=tinker://00751c13-86fd-503f-99f9-e4db8ba28e63:train:0/sampler_weights/final \
  compare_with_base=True heldout_file=docs/high-check/heldout_tasks.json \
  datasets=harness-hub/scale800-harbor,harness-hub/termgrade-train-harbor sandbox=modal \
  max_tasks=10 thinking_mode=high temperature=0.0 max_turns=25 max_tokens=8192 concurrency=2
```
