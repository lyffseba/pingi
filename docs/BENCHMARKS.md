# Benchmarks & Reference Targets

## Measured Harbor checks

These are separate frozen sets. Do not average them.

| Check | Base Inkling | `pingi-pilot` | Paired change |
|---|---:|---:|---|
| 20 held-out tasks, 2026-11-22 | 10/20 | 11/20 | +1 / −0 |
| 60 fresh tasks, 2026-11-27 | 27/60 | 23/60 | +3 / −7 |

Both used thinking `none`, temperature `0.0`, and a 25-turn cap. Neither is Terminal-Bench or SWE-bench. The 60-task loss was 14/30 versus 14/30 on `scale800` and 9/30 versus 13/30 on `termgrade`. Full evidence is in [FINDINGS.md](FINDINGS.md).

## Diagnostic, not a third held-out set

On 2026-11-28 the same checkpoint was rerun at thinking `high` on only the 10 tasks where the models already disagreed. The generation cap was 8,192, not 4,096. Do not add these numbers to the table above.

| Slice | Base Inkling | `pingi-pilot` |
|---|---:|---:|
| Same 10 tasks at `none` | 7/10 | 3/10 |
| Same 10 tasks at `high` | 0/10 | 1/10 |

Thinking `high` made both models worse on that slice. It does not change 23/60 versus 27/60. Evidence: [high-check/README.md](high-check/README.md).

## Primary reference target: `harness-hub/Qwen3.8-27B-pi-adapted`

`harness-hub/Qwen3.8-27B-pi-adapted` is our primary target reference because it uses the **same `pi` harness**, the **same Harbor dataset mixture** (`scale800-harbor` + `termgrade-train-harbor`), **thinking OFF**, **LoRA rank 32**, and **verifier correctness + 0.1 tool-efficiency shaping**:

| Model | Active / Total Params | Thinking | Terminal-Bench 2.1 (`pi`) | SWE-bench Pro hard-51 (`pi`) | Median Turns (TB 2.1 / SWE-Pro) |
|---|---:|---|---:|---:|---:|
| `Qwen/Qwen3.8-27B` (base) | 27B / 27B | off | 39.5% (41.0% on 83-task overlap) | 39.2% | 18 / 59 |
| `Qwen/Qwen3.8-27B` (base) | 27B / 27B | `xhigh` | 54.2% | 74.5% | 18 / 74 |
| **`harness-hub/Qwen3.8-27B-pi-adapted`** | 27B / 27B | **off** | **55.8%** (+16.3 pts; 16 solved / 2 lost) | **64.7%** (+25.5 pts; 15 solved / 2 lost) | **8 / 23** |
| `thinkingmachines/Inkling-2-Mini` (official card, default thinking) | 3.4B (5.4B eff.) / 26.4B | default | 71.6% (`pi v0.83.0`) | 77.9% (`mini-swe-agent`) | — |
| `unsloth/Inkling-2-Mini-GGUF` (`UD-Q4_K_XL` / `UD-Q8_K_XL`) | 3.4B (5.4B eff.) / 26.4B | base quant | Quantization of base (98.0% top-1 vs Q8) | Quantization of base | — |
| `lyffseba/pingi-smo` | 3.4B (5.4B eff.) / 26.4B | `none` | *3-step smoke only — not a held-out eval* | *not evaluated* | — |
| **PINGI** | 3.4B (5.4B eff.) / 26.4B | **`none`** | *not evaluated* | *not evaluated* | — |

*Numbers for `Qwen3.8-27B-pi-adapted` are from [harness-hub/Qwen3.8-27B-pi-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted) (100-step budget, temp 0.8). Numbers for `Inkling-2-Mini` are from [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini). Do not mix different harnesses or thinking settings as equivalent.*

## Why PINGI vs `Qwen3.8-27B-pi-adapted` Is Interesting

- `Qwen3.8-27B` activates **27B dense parameters** per token.
- `Inkling-2-Mini` activates **3.4B parameters** (5.4B effective across the 3-pass looped core) with a bundled DSpark speculator and 1M context — fitting on a 32–48 GB Mac or single DGX Spark.
- The full run remains reserved as session `pingi` and has not started. A head-to-head comparison with `Qwen3.8-27B-pi-adapted` still needs the same tasks, harness, thinking setting, and inference budget. The current Harbor checks do not justify that spend yet.

## Running Paired Held-Out Evaluation

Every `pingi-rl` run writes `heldout_tasks.json` (tasks excluded from both training and in-loop dev). Score any checkpoint against base `Inkling-2-Mini` on the exact same tasks:

```sh
uv run pingi-eval \
  checkpoint_path=tinker://<run-id>/sampler_weights/final \
  compare_with_base=True \
  heldout_file=<run-log>/heldout_tasks.json \
  datasets=harness-hub/scale800-harbor,harness-hub/termgrade-train-harbor \
  sandbox=modal \
  max_tasks=20
```
