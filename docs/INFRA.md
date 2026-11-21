# Infrastructure & Expansion Stack

PINGI is designed to scale cleanly from a **$0.25 local smoke test** → **$7 pilot** → **$15–$40 MVP** → **full-scale runs and production serving**.

## Active Now (Smoke & Pilot)

| Component | Role | Link | Env Vars |
|---|---|---|---|
| **Tinker** | RL & SFT training on `thinkingmachines/Inkling-2-Mini` (`$0.15/1M` prefill, `$0.015/1M` cached, `$0.50/1M` sample, `$0.75/1M` train) | https://tinker-console.thinkingmachines.ai <br> https://tinker.thinkingmachines.ai/billing/balance | `TINKER_API_KEY` |
| **Modal** | Parallel Harbor sandboxes in environment `pingi`, app `pingi`. Do not use environment `main` for this project. | https://modal.com/settings/lyffseba/usage | `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `PINGI_MODAL_ENVIRONMENT=pingi` |
| **Local Docker** | Zero-cost sandboxes for `preset=smoke` | Docker Desktop | *(local daemon)* |
| **Hugging Face** | Harbor datasets + model hub (`lyffseba/pingi`, `lyffseba/pingi-smo`) | https://huggingface.co/lyffseba | `HF_TOKEN` |
| **GitHub** | Source of truth (`lyffseba/pingi`) + research notes (`lyffseba/xai`) | https://github.com/lyffseba/pingi | `gh` CLI |
| **Weights & Biases** | Live pass@1, reward, turn count & `edit` error curves | https://wandb.ai/authorize | `WANDB_API_KEY` |

## Ready for Future Expansion (Stage 0 Mining, Full Runs & Serving)

| Component | Role | Link | Env Vars / How Used |
|---|---|---|---|
| **River AI** | Cheap Stage 0 verified `pi` trajectory mining (`DeepSeek-V4.1-Flash-1017` at `$0.12/$0.0024/$0.48`, `Qwen3.8-27B-FP8`); also alternative RL/OPSD backend for Qwen/GLM/DeepSeek | https://river.ai <br> https://docs.river.ai/ | `RIVER_API_KEY` via `uv run pingi-mine provider=river ...` |
| **Modular Cloud / MAX** | Stage 0 verified `pi` trajectory mining (`deepseek/deepseek-v4.1-flash-1017`, `qwen/qwen3.8-27b`) + future high-throughput serving across NVIDIA & AMD | https://console.modular.com/ <br> https://docs.modular.com/ | `MODULAR_API_KEY` via `uv run pingi-mine provider=modular ...` |
| **CoreWeave Forge** | Dedicated NVIDIA GPU clusters (H200/B200) for future full-scale self-hosted TRL `AsyncGRPO` / SkyRL runs or dedicated vLLM + DSpark endpoints | https://forge.coreweave.com/account-settings/lyff-org/billing | Provisioned when moving beyond Tinker or hosting dedicated vLLM endpoints |

## Ecosystem Links

- Code & scripts: https://github.com/lyffseba/pingi
- Model Hub (main): https://huggingface.co/lyffseba/pingi
- Smoke checkpoint record: https://huggingface.co/lyffseba/pingi-smo
- Research notes (`xai`): https://github.com/lyffseba/xai · https://huggingface.co/spaces/lyffseba/xai · https://huggingface.co/datasets/lyffseba/xai-studies
