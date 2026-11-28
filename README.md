# PINGI — Pi-Native `Inkling-2-Mini`

**PINGI** adapts [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
(26.4B total / 3.4B active / 5.4B effective via 3× looped core, 1M context, multimodal, Apache 2.0)
specifically for the [`pi`](https://pi.dev/) coding agent harness using **Tinker** agentic RL on Harbor tasks.

| Link | URL |
|---|---|
| GitHub | https://github.com/lyffseba/pingi |
| Hugging Face (main hub) | https://huggingface.co/lyffseba/pingi |
| Hugging Face (smoke checkpoint) | https://huggingface.co/lyffseba/pingi-smo |
| Companion research (`xai`) | https://github.com/lyffseba/xai · https://huggingface.co/spaces/lyffseba/xai |
| Docs | [docs/FINDINGS.md](docs/FINDINGS.md) · [docs/NEXT_RUN.md](docs/NEXT_RUN.md) · [docs/PILOT.md](docs/PILOT.md) · [docs/BENCHMARKS.md](docs/BENCHMARKS.md) |

## Pilot complete — not ready to scale

The saved `none` pilot is behind base Inkling on the later 60-task check: **23/60 versus 27/60**, with 3 new solves and 7 losses. An earlier 20-task check had shown 11/20 versus 10/20. A separate `high` inference check on the 10 disagreements scored 1/10 versus 0/10. Do not combine those checks. Training cost **$18.73** on Tinker, excluding Modal and the separate evals. Checkpoints remain private. No next run is approved. See [findings](docs/FINDINGS.md) and the [next-run checklist](docs/NEXT_RUN.md).

## Design Decisions

1. **Best RL Backend = Tinker**: Thinking Machines hosts `thinkingmachines/Inkling-2-Mini` natively on Tinker
   (handling NVFP4/MXFP8 + 3× looped MoE LoRA internals for you):
   - **Prefill**: `$0.15 / 1M` (**`$0.015 / 1M` cached** — huge savings on multi-turn `pi` rollouts)
   - **Sample**: `$0.50 / 1M`
   - **Train**: `$0.75 / 1M` (64K train / 1M sample)
2. **Strict Binary Thinking Policy** (no adaptive effort drift):
   - **Default = `none` (`reasoning_effort = 0.0`)** — fast, crisp non-thinking `pi` tool calls (~3× cheaper to train and ~4× fewer tokens per task, matching the `Qwen3.8-27B-pi-adapted` insight).
   - **Toggle = `high` (`reasoning_effort = 0.9`)** — only thinking OFF (`none`) or ON (`high`).
3. **True `pi`-Native Tool Suite**:
   - Trains directly on `pi`’s 4 core tools (`read`, `bash`, `edit`, `write`) + `pi` system rules.
   - Verifier-gated reward shaping: only rewards tool efficiency & clean exact-match `edit`s when `tests/test.sh` passes.

---

## Smoke run

The completed smoke run is documented in [docs/SMOKE.md](docs/SMOKE.md) and published as [`lyffseba/pingi-smo`](https://huggingface.co/lyffseba/pingi-smo). The weights stay on Tinker; downloads are disabled for `Inkling-2-Mini`.

## 1. Get Your API Keys

| Service | Purpose | Link |
|---|---|---|
| **Tinker (Required)** | RL & SFT training on `Inkling-2-Mini` | Sign up: https://auth.thinkingmachines.ai/sign-up <br> API key: https://tinker-console.thinkingmachines.ai |
| **Modal (Recommended for full RL)** | Parallel cloud Docker sandboxes (or use `--sandbox docker` locally for $0!) | https://modal.com/settings (or run `uvx modal setup`) |
| **Hugging Face (Required for datasets & upload)** | Download `harness-hub` tasks & publish `lyffseba/pingi` | https://huggingface.co/settings/tokens |
| **Weights & Biases (Recommended)** | Live reward, pass@1, turn count & `edit` error curves | https://wandb.ai/authorize |
| **Modular Cloud (Optional — Stage 0)** | Free credits + cheap verified `pi` trajectory mining before RL | https://console.modular.com/ |

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

---

## 2. Install

```bash
uv sync --extra dev
```

---

## 3. Download Harbor Tasks

Fast starter set (Terminal-Bench 2.0):

```bash
uvx harbor datasets download terminal-bench@2.0 -o ~/.cache/harbor/tasks/terminal-bench-2.0
```

Training mixture (`scale800` + `termgrade`): [populate the local Harbor cache](docs/PILOT.md#reproduce-the-setup) before running `pingi-rl`.
- `harness-hub/scale800-harbor`
- `harness-hub/termgrade-train-harbor`

---

## 4. Run PINGI Training (Budget Ladder — Non-Thinking Default)

| Preset | Command | Nominal rollouts | Tinker cost | Goal |
|---|---|---:|---:|---|
| **`smoke`** | `uv run pingi-rl preset=smoke` | 12 (`2×2×3`, Docker) | **~$0.25** (estimate) | Verify Tinker + tools + Docker end-to-end |
| **`pilot`** | `PINGI_MODAL_ENVIRONMENT=pingi uv run pingi-rl preset=pilot session_name=pingi-pilot wandb_project=None` | 640 (`4×8×20`, Modal environment `pingi`) | **$18.73** (recorded) | Completed; see the [pilot report](docs/PILOT.md). |
| **`mvp`** | `uv run pingi-rl preset=mvp` | 1,600 (`4×8×50`, Modal) | **~$15 – $40** | Strong daily-driver PINGI on `scale800 + termgrade` |
| **`full`** | `uv run pingi-rl preset=full` | 5,120 (`4×8×160`, Modal) | **~$75 – $135** | Full `Qwen3.8-27B-pi-adapted` scale |

Costs above exclude Modal and standalone evaluation; estimates are not spending caps. The MVP and full presets have not been run. Do not warm-start another `none` run from the pilot just because its checkpoint exists. The reserved full session name is `pingi`, and it has not been created.

### Optional: Train a `High-Thinking` (`high`) Variant Later

```bash
uv run pingi-rl preset=mvp thinking_mode=high max_tokens=16384 max_trajectory_tokens=49152
```

### Optional Stage 0 + Stage 1: Cheap Verified `pi` Trace Mining (Modular Cloud) + Tinker SFT Warmup

```bash
# Stage 0: Mine verifier-passing pi traces using Modular Cloud free credits
uv run pingi-mine dataset=terminal-bench-2.0 model=deepseek/deepseek-v4.1-flash-1017 sandbox=docker max_tasks=30

# Stage 1: Warm-start Inkling-2-Mini on those verified traces (~$2-$6 on Tinker)
uv run pingi-sft jsonl_path=./artifacts/sft_verified_traces.jsonl thinking_mode=none num_epochs=3

# Then feed the SFT checkpoint into RL:
uv run pingi-rl preset=pilot load_checkpoint_path=tinker://<your-sft-run>/weights/final
```

---

## 5. Paired Benchmark Evaluation (Base vs PINGI)

See the [pilot protocol and command](docs/PILOT.md#reproduce-the-setup) for a comparison of base `Inkling-2-Mini` and a PINGI checkpoint on the same frozen held-out tasks. See [Benchmarks](docs/BENCHMARKS.md) for how this result relates to the reference targets.

---

## 6. Release status

The [Hugging Face page](https://huggingface.co/lyffseba/pingi) hosts reports and per-task scores, not model files. A weight upload now requires paired evidence whose checkpoint and task manifest match the checkpoint being uploaded, and that evidence must show a net gain. The saved 60-task result fails that test. Do not publish weights from the current results.
