# PINGI follow-up evaluation plan

**Prepared, not run.** Compare the saved private `pingi-pilot` final checkpoint with base Inkling-2-Mini on 60 fresh tasks, 30 per Harbor source. This is a same-mixture generalization check, not a test on entirely unseen projects, a comparison with Qwen, or authorization to release weights.

## Selection

The pilot ran 20 synchronous batches of 8 groups. Its saved ordered dataset config and completed batch records identify the first 160 training tasks as scheduled. The [frozen manifest](heldout_tasks.json) uses the unscheduled portion of that candidate pool. It excludes the 160 scheduled tasks, the 16 in-loop dev tasks, all 20 original pilot evaluation tasks, and normalized exact instruction duplicates. Selection was deterministic by SHA-256 rank with seed `pingi-followup-60-v1`, limited to prebuilt task images, with at most three selected `scale800-harbor` tasks per project. The evaluator checks each instruction hash against the local task cache before creating a Tinker client. Near-duplicate tasks, overlap in base-model pretraining, and changes to remote image tags are not ruled out.

Manifest SHA-256: `c0db6562d074188784f81bc1ab476886081dfa43818cd73ac450e9f67ab83142`.

## Protocol

Use the same frozen manifest for both models, thinking `none`, temperature `0.0`, 25 turns, 4,096 generated tokens per step and 49,152 trajectory tokens, with the same `pi` tools. Report paired wins and losses, per-source pass counts, and median turns. Preserve task errors separately instead of counting them as model failures. Save individual task results as they finish, plus the manifest hash and checkpoint path. Do not combine this result with the 20-task pilot after looking at outcomes.

The prior 40-rollout comparison had about **$0.44 in estimated Tinker usage** from its billing rows. Scaling linearly to 120 rollouts gives **about $1.32 on Tinker**, before Modal and any storage charges. This is a planning estimate, not a hard cap. Confirm the available balance and spend limit before running. No retraining is needed.

```sh
PINGI_MODAL_ENVIRONMENT=pingi uv run pingi-eval \
  checkpoint_path=tinker://00751c13-86fd-503f-99f9-e4db8ba28e63:train:0/sampler_weights/final \
  compare_with_base=True heldout_file=docs/followup-eval/heldout_tasks.json \
  datasets=harness-hub/scale800-harbor,harness-hub/termgrade-train-harbor sandbox=modal \
  max_tasks=60 thinking_mode=none temperature=0.0 max_turns=25 concurrency=4
```
