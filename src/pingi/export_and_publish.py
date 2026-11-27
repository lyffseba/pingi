"""Download trained PINGI weights from Tinker and publish to Hugging Face."""

from __future__ import annotations

import logging
from pathlib import Path

import chz
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

MODEL_CARD_TEMPLATE = """---
base_model: thinkingmachines/Inkling-2-Mini
library_name: peft
license: apache-2.0
tags:
- pi
- inkling-2-mini
- tinker
- rl
- harbor
- agents
- coding
pipeline_tag: image-text-to-text
---

# PINGI (`Inkling-2-Mini` — `pi`-Native)

**PINGI** is a LoRA adapter for [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
trained with agentic reinforcement learning on **Tinker** directly inside a **`pi`-native Harbor environment**
(`read`, `bash`, `edit`, `write`).

## Thinking Mode Policy (Binary Only)

PINGI intentionally avoids adaptive/intermediate effort modes:

| Mode | `reasoning_effort` | Role |
|---|---:|---|
| **No-Thinking (Default)** | `"none"` (`0.0`) | Default mode used in training and serving (fast, ~3–4× fewer tokens) |
| **High Thinking** | `"high"` (`0.9`) | Optional deep-reasoning toggle |

## Serving with vLLM + DSpark

```bash
vllm serve thinkingmachines/Inkling-2-Mini \\
  --enable-lora --max-lora-rank 32 \\
  --lora-modules pingi={hf_repo} \\
  --enable-auto-tool-choice \\
  --tool-call-parser inkling \\
  --reasoning-parser inkling \\
  --default-chat-template-kwargs '{{"reasoning_effort": "none"}}' \\
  --speculative-config '{{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}}'
```

## Connect to `pi`

Add to `~/.pi/agent/models.json` (see `configs/pi_models.json`):

```json
{{
  "providers": {{
    "pingi-local": {{
      "baseUrl": "http://localhost:8000/v1",
      "api": "openai-completions",
      "apiKey": "dummy",
      "models": [
        {{ "id": "pingi", "name": "PINGI (No-Thinking Default)" }}
      ]
    }}
  }}
}}
```
"""


@chz.chz
class ExportConfig:
    """Export a Tinker checkpoint to disk and optionally upload to Hugging Face."""

    # Example: tinker://.../weights/000160
    tinker_path: str
    output_dir: str = "./artifacts/pingi-lora"
    hf_repo: str | None = "lyffseba/pingi"
    private: bool = False
    push_to_hub: bool = False
    # Public upload requires a paired held-out eval that beats the base model.
    allow_unverified: bool = False
    # Use when more than one local eval directory names the same checkpoint.
    eval_dir: str | None = None


def main(cfg: ExportConfig) -> None:
    from tinker_cookbook import weights

    if cfg.push_to_hub and not cfg.allow_unverified:
        from pingi.release_gate import check_release

        gate = check_release(
            cfg.tinker_path,
            eval_dir=Path(cfg.eval_dir) if cfg.eval_dir else None,
        )
        if not gate.ok:
            raise SystemExit(
                "Refusing Hugging Face upload until the held-out gate passes:\n"
                + "\n".join(f"- {r}" for r in gate.reasons)
            )
        logger.info(
            "Release gate passed: %.1f%% -> %.1f%% on %d tasks (+%d / -%d)",
            100 * gate.base_pass,
            100 * gate.pingi_pass,
            gate.n_tasks,
            gate.newly_solved,
            gate.lost,
        )

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading Tinker weights from %s -> %s", cfg.tinker_path, out)
    weights.download(tinker_path=cfg.tinker_path, output_dir=str(out))

    readme = out / "README.md"
    readme.write_text(
        MODEL_CARD_TEMPLATE.format(hf_repo=cfg.hf_repo or "lyffseba/pingi"),
        encoding="utf-8",
    )
    logger.info("Wrote model card to %s", readme)

    if cfg.push_to_hub:
        if not cfg.hf_repo:
            raise ValueError("Set hf_repo=... when push_to_hub=True")
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=cfg.hf_repo, repo_type="model", exist_ok=True, private=cfg.private)
        api.upload_folder(
            folder_path=str(out),
            repo_id=cfg.hf_repo,
            repo_type="model",
            commit_message=f"Upload PINGI checkpoint exported from {cfg.tinker_path}",
        )
        logger.info("Published PINGI to https://huggingface.co/%s", cfg.hf_repo)
    else:
        logger.info("Export complete locally. Pass push_to_hub=True to upload to Hugging Face.")


def entrypoint() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = chz.entrypoint(ExportConfig)
    main(cfg)


if __name__ == "__main__":
    entrypoint()
