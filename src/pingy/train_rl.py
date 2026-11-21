"""Launch PINGY RL training on Tinker for `thinkingmachines/Inkling-2-Mini`.

Default thinking mode is strictly `none` (0.0) — fast, low-token `pi` execution.
Only binary `none` (0.0) or `high` (0.9) is permitted — no adaptive effort levels.
Explicit CLI overrides always take precedence over preset defaults.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import chz
from dotenv import load_dotenv
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl.train import AsyncConfig, Config, main

from pingy.datasets import load_datasets_spec, split_train_dev_heldout
from pingy.docker_sandbox import docker_sandbox_factory
from pingy.env import PingyDatasetBuilder, modal_sandbox_factory
from pingy.thinking import DEFAULT_THINKING_MODE, ThinkingMode, resolve_thinking_effort

logger = logging.getLogger(__name__)

PresetName = Literal["custom", "smoke", "pilot", "mvp", "full"]
LossFnName = Literal["importance_sampling", "ppo", "cispo", "gspo", "dro"]

PRESETS: dict[str, dict[str, object]] = {
    # ~$0.25 on Tinker + $0 local Docker (12 rollouts, eval_every=0 skips step-0 eval so it finishes in ~3-5m)
    "smoke": {
        "sandbox": "docker",
        "datasets": "terminal-bench-2.0",
        "group_size": 2,
        "groups_per_batch": 2,
        "max_turns": 15,
        "max_steps": 3,
        "eval_every": 0,
        "save_every": 3,
        "sampling_steps_ahead": None,
    },
    # ~$7 Half-MVP Pilot on Tinker + $0 Modal free tier (640 rollouts: 4x8x20, resumes cleanly into mvp)
    "pilot": {
        "sandbox": "modal",
        "datasets": "harness-hub/scale800-harbor,harness-hub/termgrade-train-harbor",
        "group_size": 4,
        "groups_per_batch": 8,
        "max_turns": 25,
        "max_steps": 20,
        "eval_every": 5,
        "save_every": 5,
        "sampling_steps_ahead": 2,
    },
    # ~$15-$40 Strong MVP (1,600 rollouts: 4x8x50)
    "mvp": {
        "sandbox": "modal",
        "datasets": "harness-hub/scale800-harbor,harness-hub/termgrade-train-harbor",
        "group_size": 4,
        "groups_per_batch": 8,
        "max_turns": 25,
        "max_steps": 50,
        "eval_every": 10,
        "save_every": 10,
        "sampling_steps_ahead": 2,
    },
    # ~$75-$135 Full Qwen3.8-pi scale (5,120 rollouts: 4x8x160)
    "full": {
        "sandbox": "modal",
        "datasets": "harness-hub/scale800-harbor,harness-hub/termgrade-train-harbor",
        "group_size": 4,
        "groups_per_batch": 8,
        "max_turns": 25,
        "max_steps": 160,
        "eval_every": 10,
        "save_every": 10,
        "sampling_steps_ahead": 2,
    },
}

_CUSTOM_DEFAULTS: dict[str, object] = {
    "sandbox": "modal",
    "datasets": "terminal-bench-2.0",
    "group_size": 4,
    "groups_per_batch": 8,
    "max_turns": 25,
    "max_steps": 160,
    "eval_every": 10,
    "save_every": 10,
    "sampling_steps_ahead": 2,
}


@chz.chz
class PingyRLConfig:
    """CLI configuration for PINGY agentic RL on Tinker."""

    # Budget preset: "smoke" (~$0.25), "pilot" (~$7), "mvp" (~$15-$40), "full" (~$75-$135), or "custom"
    preset: PresetName = "custom"

    # Base model (64K train / 1M sample = $0.15 prefill, $0.015 cached, $0.50 sample, $0.75 train)
    model_name: str = "thinkingmachines/Inkling-2-Mini"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Binary thinking policy: strictly "none" (DEFAULT, 0.0) or "high" (0.9)
    thinking_mode: ThinkingMode = DEFAULT_THINKING_MODE

    # Sampling & context budgets (48K trajectory safely inside Inkling-2-Mini's 64K train limit)
    max_tokens: int = 4096
    max_trajectory_tokens: int = 48 * 1024
    temperature: float = 0.8

    # Overridable preset fields (None = inherit from preset/defaults; explicit values always win!)
    datasets: str | None = None
    sandbox: Literal["modal", "docker"] | None = None
    max_turns: int | None = None
    group_size: int | None = None
    groups_per_batch: int | None = None
    max_steps: int | None = None
    eval_every: int | None = None
    save_every: int | None = None

    # Dataset splitting & timeouts
    eval_fraction: float = 0.05
    max_eval_tasks: int = 16
    seed: int = 42
    sandbox_timeout: int = 2400
    command_timeout: int = 120
    grader_timeout: int = 300

    # Reward shaping (strictly gated on verifier correctness >= 1.0)
    w_tool_efficiency: float = 0.10
    tool_budget: float = 20.0
    w_clean_edits: float = 0.05

    # RL optimization
    loss_fn: LossFnName = "importance_sampling"
    learning_rate: float = 1e-5
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1
    sampling_steps_ahead: int | None = None
    remove_constant_reward_groups: bool = True
    rollout_json_export: bool = True

    # Checkpointing & observability
    log_path: str | None = None
    wandb_project: str | None = "pingy"
    wandb_name: str | None = None
    session_name: str | None = None
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def resolve_preset_values(cfg: PingyRLConfig) -> dict[str, object]:
    """Merge preset defaults with explicit CLI overrides (explicit overrides always win)."""
    base = dict(_CUSTOM_DEFAULTS)
    if cfg.preset != "custom":
        base.update(PRESETS[cfg.preset])

    for key in (
        "sandbox",
        "datasets",
        "group_size",
        "groups_per_batch",
        "max_turns",
        "max_steps",
        "eval_every",
        "save_every",
        "sampling_steps_ahead",
    ):
        override = getattr(cfg, key)
        if override is not None:
            base[key] = override
    return base


async def run(cfg: PingyRLConfig) -> None:
    effort = resolve_thinking_effort(cfg.thinking_mode)
    params = resolve_preset_values(cfg)
    sandbox_kind = str(params["sandbox"])
    datasets_spec = str(params["datasets"])
    group_size = int(params["group_size"])  # type: ignore[arg-type]
    groups_per_batch = int(params["groups_per_batch"])  # type: ignore[arg-type]
    max_turns = int(params["max_turns"])  # type: ignore[arg-type]
    max_steps = int(params["max_steps"]) if params["max_steps"] is not None else None  # type: ignore[arg-type]
    eval_every = int(params["eval_every"])  # type: ignore[arg-type]
    save_every = int(params["save_every"])  # type: ignore[arg-type]
    raw_ahead = params["sampling_steps_ahead"]
    sampling_steps_ahead = int(raw_ahead) if raw_ahead is not None else None  # type: ignore[arg-type]

    logger.info(
        "PINGY preset=%s thinking_mode=%s (effort=%.1f) loss_fn=%s sandbox=%s rollouts=%s",
        cfg.preset,
        cfg.thinking_mode,
        effort,
        cfg.loss_fn,
        sandbox_kind,
        group_size * groups_per_batch * max_steps if max_steps else "unbounded",
    )

    if sandbox_kind == "modal" and not (
        os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET")
    ) and not (Path.home() / ".modal.toml").is_file():
        raise SystemExit(
            "Pilot/MVP needs Modal sandboxes. Add MODAL_TOKEN_ID and MODAL_TOKEN_SECRET "
            "to pingy/.env from https://modal.com/settings, then rerun "
            "`uv run pingy-rl preset=pilot session_name=pingy-pilot`."
        )

    wandb_project = cfg.wandb_project
    if wandb_project and not os.environ.get("WANDB_API_KEY"):
        logger.warning(
            "WANDB_API_KEY is missing, so this run will log locally only. "
            "Add a key from https://wandb.ai/authorize to get live curves."
        )
        wandb_project = None

    groups = load_datasets_spec(datasets_spec)
    train_tasks, eval_tasks, heldout_tasks = split_train_dev_heldout(
        groups,
        dev_fraction=cfg.eval_fraction,
        heldout_fraction=cfg.eval_fraction,
        max_dev_tasks=cfg.max_eval_tasks,
        max_heldout_tasks=max(cfg.max_eval_tasks, 20),
        seed=cfg.seed,
    )
    logger.info(
        "Loaded %d train, %d dev, %d held-out tasks",
        len(train_tasks),
        len(eval_tasks),
        len(heldout_tasks),
    )

    sandbox_factory = modal_sandbox_factory if sandbox_kind == "modal" else docker_sandbox_factory
    renderer_name = cfg.renderer_name or model_info.get_recommended_renderer_name(cfg.model_name)

    run_name = (
        f"pingy-{cfg.preset}-{cfg.thinking_mode}-r{cfg.lora_rank}-lr{cfg.learning_rate}-"
        f"g{group_size}x{groups_per_batch}-{datetime.now().strftime('%Y%m%d-%H%M')}"
    )
    log_path = cfg.log_path or cli_utils.runs_path("pingy", run_name)
    wandb_name = cfg.wandb_name or run_name

    dataset_builder = PingyDatasetBuilder(
        train_tasks=train_tasks,
        eval_tasks=eval_tasks,
        batch_size=groups_per_batch,
        group_size=group_size,
        model_name=cfg.model_name,
        thinking_mode=cfg.thinking_mode,
        renderer_name=renderer_name,
        max_turns=max_turns,
        sandbox_timeout=cfg.sandbox_timeout,
        command_timeout=cfg.command_timeout,
        grader_timeout=cfg.grader_timeout,
        max_trajectory_tokens=cfg.max_trajectory_tokens,
        max_generation_tokens=cfg.max_tokens,
        w_tool_efficiency=cfg.w_tool_efficiency,
        tool_budget=cfg.tool_budget,
        w_clean_edits=cfg.w_clean_edits,
        sandbox_factory=sandbox_factory,
    )

    train_config = Config(
        learning_rate=cfg.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cfg.model_name,
        recipe_name="pingy_harbor_rl",
        lora_rank=cfg.lora_rank,
        loss_fn=cfg.loss_fn,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        session_name=cfg.session_name or wandb_name,
        log_path=log_path,
        base_url=cfg.base_url,
        load_checkpoint_path=cfg.load_checkpoint_path,
        kl_penalty_coef=cfg.kl_penalty_coef,
        num_substeps=cfg.num_substeps,
        remove_constant_reward_groups=cfg.remove_constant_reward_groups,
        rollout_json_export=cfg.rollout_json_export,
        eval_every=eval_every,
        save_every=save_every,
        async_config=AsyncConfig(
            sampling_steps_ahead=sampling_steps_ahead,
            groups_per_batch=groups_per_batch,
        )
        if sampling_steps_ahead is not None
        else None,
        max_steps=max_steps,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)
    heldout_path = Path(log_path) / "heldout_tasks.json"
    heldout_path.parent.mkdir(parents=True, exist_ok=True)
    heldout_path.write_text(
        json.dumps(
            [
                {"task_name": t.task_name, "source": t.source, "task_dir": str(t.task_dir)}
                for t in heldout_tasks
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote held-out manifest to %s", heldout_path)
    await main(train_config)


def entrypoint() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = chz.entrypoint(PingyRLConfig)
    asyncio.run(run(cfg))


if __name__ == "__main__":
    entrypoint()
