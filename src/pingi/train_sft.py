"""Stage 1 Tinker SFT Warm-Start Trainer for PINGI (`uv run pingi-sft`).

Trains `thinkingmachines/Inkling-2-Mini` on verified `pi` trajectories JSONL
(mined by `pingi-mine` on Modular Cloud or exported from earlier runs), using
the exact same `PingiEffortRenderer` (`none=0.0` default / `high=0.9`) and `pi`
tool schemas as RL so the resulting checkpoint feeds seamlessly into `pingi-rl`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import chz
import tinker
from dotenv import load_dotenv
from tinker_cookbook import cli_utils, model_info, tokenizer_utils
from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.supervised import train as sl_train
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder

from pingi.pi_tools import PI_SYSTEM_PROMPT, PiSandboxTools
from pingi.thinking import (
    DEFAULT_THINKING_MODE,
    PingiEffortRenderer,
    ThinkingMode,
    resolve_thinking_effort,
)

logger = logging.getLogger(__name__)


def normalize_trace_for_inkling(
    raw_messages: list[dict[str, Any]],
    renderer: PingiEffortRenderer,
    thinking_mode: ThinkingMode,
) -> list[dict[str, Any]]:
    """Normalize mined OpenAI-format pi traces into Inkling renderer messages.

    - Re-applies PINGI's canonical 4-tool declaration prefix (`read`, `bash`, `edit`, `write`).
    - Drops any teacher `reasoning_content` when training the `none` (non-thinking) model
      so SFT strictly matches zero-thinking inference.
    - Keeps OpenAI-format `tool_calls` (`{function: {name, arguments: json-string}}`),
      which `tinker_cookbook.third_party.openai_compat` validates as `ToolCall`.
    """
    system_prompt = PI_SYSTEM_PROMPT
    rest: list[dict[str, Any]] = []
    for msg in raw_messages:
        role = msg.get("role")
        if role == "system" and not rest:
            if msg.get("content"):
                system_prompt = str(msg["content"])
            continue
        cleaned = dict(msg)
        if thinking_mode == "none":
            cleaned.pop("reasoning_content", None)
            cleaned.pop("thinking", None)
        rest.append(cleaned)

    dummy_tools = PiSandboxTools(sandbox=None)
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=dummy_tools.tool_specs(),
        system_prompt=system_prompt,
    )
    return list(prefix) + rest


def load_verified_traces(jsonl_path: Path, min_score: float = 1.0) -> list[list[dict[str, Any]]]:
    conversations: list[list[dict[str, Any]]] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if float(row.get("score", 1.0)) < min_score:
                continue
            msgs = row.get("messages")
            if isinstance(msgs, list) and msgs:
                conversations.append(msgs)
    return conversations


class PingiSFTDataset(SupervisedDataset):
    """In-memory batchable SupervisedDataset of verified pi trajectories."""

    def __init__(self, datums: list[tinker.Datum], batch_size: int) -> None:
        self._datums = list(datums)
        self._batch_size = max(1, batch_size)

    def get_batch(self, index: int) -> list[tinker.Datum]:
        start = index * self._batch_size
        return self._datums[start : start + self._batch_size]

    def set_epoch(self, seed: int = 0) -> None:
        rng = random.Random(seed)
        rng.shuffle(self._datums)

    def __len__(self) -> int:
        return (len(self._datums) + self._batch_size - 1) // self._batch_size


@chz.chz
class PingiSFTDatasetBuilder(SupervisedDatasetBuilder):
    jsonl_path: str
    model_name: str
    thinking_mode: ThinkingMode = DEFAULT_THINKING_MODE
    renderer_name: str | None = None
    max_length: int = 32 * 1024
    batch_size: int = 4
    min_score: float = 1.0
    eval_fraction: float = 0.1
    seed: int = 42

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        tok = tokenizer_utils.get_tokenizer(self.model_name)
        r_name = self.renderer_name or model_info.get_recommended_renderer_name(self.model_name)
        renderer = PingiEffortRenderer(
            get_renderer(r_name, tok), thinking_mode=self.thinking_mode
        )

        raw_convos = load_verified_traces(Path(self.jsonl_path), min_score=self.min_score)
        if not raw_convos:
            raise ValueError(f"No verified traces found in {self.jsonl_path} with score >= {self.min_score}")

        datums: list[tinker.Datum] = []
        for convo in raw_convos:
            normalized = normalize_trace_for_inkling(convo, renderer, self.thinking_mode)
            datums.append(
                conversation_to_datum(
                    normalized,  # type: ignore[arg-type]
                    renderer,  # type: ignore[arg-type]
                    max_length=self.max_length,
                    train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
                )
            )

        rng = random.Random(self.seed)
        rng.shuffle(datums)

        if len(datums) >= 10 and self.eval_fraction > 0:
            n_eval = max(1, int(len(datums) * self.eval_fraction))
            eval_ds: SupervisedDataset | None = PingiSFTDataset(datums[:n_eval], self.batch_size)
            train_ds = PingiSFTDataset(datums[n_eval:], self.batch_size)
        else:
            train_ds = PingiSFTDataset(datums, self.batch_size)
            eval_ds = None

        logger.info(
            "Built PINGI SFT dataset: %d train examples, %d eval examples (thinking=%s)",
            len(train_ds._datums),
            len(eval_ds._datums) if isinstance(eval_ds, PingiSFTDataset) else 0,
            self.thinking_mode,
        )
        return train_ds, eval_ds


@chz.chz
class PingiSFTConfig:
    """CLI configuration for PINGI Stage 1 SFT cold-start on Tinker."""

    jsonl_path: str = "./artifacts/sft_verified_traces.jsonl"
    model_name: str = "thinkingmachines/Inkling-2-Mini"
    thinking_mode: ThinkingMode = DEFAULT_THINKING_MODE
    lora_rank: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 4
    max_length: int = 32 * 1024
    min_score: float = 1.0
    load_checkpoint_path: str | None = None
    log_path: str | None = None
    wandb_project: str | None = "pingi"
    wandb_name: str | None = None
    save_every: int = 20
    eval_every: int = 20
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def run(cfg: PingiSFTConfig) -> None:
    effort = resolve_thinking_effort(cfg.thinking_mode)
    logger.info("PINGI SFT thinking_mode=%s (effort=%.1f)", cfg.thinking_mode, effort)

    run_name = f"pingi-sft-{cfg.thinking_mode}-r{cfg.lora_rank}-{datetime.now().strftime('%Y%m%d-%H%M')}"
    log_path = cfg.log_path or cli_utils.runs_path("pingi_sft", run_name)

    dataset_builder = PingiSFTDatasetBuilder(
        jsonl_path=cfg.jsonl_path,
        model_name=cfg.model_name,
        thinking_mode=cfg.thinking_mode,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        min_score=cfg.min_score,
    )
    train_cfg = sl_train.Config(
        log_path=log_path,
        model_name=cfg.model_name,
        recipe_name="pingi_sft_warmup",
        load_checkpoint_path=cfg.load_checkpoint_path,
        dataset_builder=dataset_builder,
        learning_rate=cfg.learning_rate,
        num_epochs=cfg.num_epochs,
        lora_rank=cfg.lora_rank,
        save_every=cfg.save_every,
        eval_every=cfg.eval_every,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name or run_name,
        base_url=cfg.base_url,
    )
    cli_utils.check_log_dir(log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)
    await sl_train.main(train_cfg)


def entrypoint() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = chz.entrypoint(PingiSFTConfig)
    asyncio.run(run(cfg))


if __name__ == "__main__":
    entrypoint()
