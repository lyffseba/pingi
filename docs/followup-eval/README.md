# PINGI 60-task follow-up

**Completed on 2026-11-27. The pilot gain did not hold.** This is a same-mixture generalization check of the saved private `pingi-pilot` final sampler against base Inkling-2-Mini. It is not a Qwen comparison and not authorization to release weights or start another run.

## Result

| Model | Pass@1 | Median turns | Median tool calls |
|---|---:|---:|---:|
| Base Inkling-2-Mini | 27/60 (45.0%) | 15.0 | 11.5 |
| `pingi-pilot` | 23/60 (38.3%) | 16.0 | 13.0 |

**+3 newly solved, −7 lost.** No task errors. `scale800-harbor` was tied at 14/30. `termgrade-train-harbor` was 9/30 for the pilot against 13/30 for the base model. The recorded telemetry had zero `edit` calls on both sides, so edit-error rate is not informative. Do not pool this set with the earlier 20-task result.

[Manifest](heldout_tasks.json) · [base scores](results/base.jsonl) · [pilot scores](results/pingi.jsonl) · [settings and hashes](results/run.json).

Manifest SHA-256: `c0db6562d074188784f81bc1ab476886081dfa43818cd73ac450e9f67ab83142`.

## How the 60 tasks were chosen

The pilot ran 20 synchronous batches of 8 groups. Its saved ordered dataset config and completed batch records identify the first 160 training tasks as scheduled. This manifest uses the unscheduled portion of that candidate pool. It excludes those 160 tasks, the 16 in-loop dev tasks, all 20 original pilot evaluation tasks, and normalized exact instruction duplicates. Selection was deterministic by SHA-256 rank with seed `pingi-followup-60-v1`, limited to prebuilt task images, with at most three selected `scale800-harbor` tasks per project. The evaluator checked each instruction hash against the local task cache before creating a Tinker client. Near-duplicate tasks, overlap in base-model pretraining, and changes to remote image tags are not ruled out.

## Protocol that ran

Both models used the frozen manifest, thinking `none`, temperature `0.0`, 25 turns, 4,096 generated tokens per step, 49,152 trajectory tokens, and the same `pi` tools. Task errors would have been kept out of the final score files. This run had none. The planning estimate was about **$1.32 on Tinker, plus Modal**, based on the earlier 40-rollout eval. That was not a hard cap. The exact invoice for this eval was not separated here.

An earlier launch of the same command was stopped after about 3 minutes and saved no task results. The completed artifact is `20261127-192206-004981Z`.

```sh
PINGI_MODAL_ENVIRONMENT=pingi uv run pingi-eval \
  checkpoint_path=tinker://00751c13-86fd-503f-99f9-e4db8ba28e63:train:0/sampler_weights/final \
  compare_with_base=True heldout_file=docs/followup-eval/heldout_tasks.json \
  datasets=harness-hub/scale800-harbor,harness-hub/termgrade-train-harbor sandbox=modal \
  max_tasks=60 thinking_mode=none temperature=0.0 max_turns=25 concurrency=4
```
