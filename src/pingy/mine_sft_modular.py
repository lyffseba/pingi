"""Optional Stage 0: Mine cheap, verifier-passing `pi` trajectories via Modular Cloud.

Uses Modular Cloud's OpenAI-compatible endpoint (`https://api.modular.com/v1`)
with strong shared models (e.g. `deepseek/deepseek-v4.1-flash-1017` or `qwen/qwen3.8-27b`)
to solve Harbor tasks using PINGY's exact `pi` tools (`read`, `bash`, `edit`, `write`).
Only trajectories that PASS the Harbor verifier are written to JSONL for optional
SFT warm-starting before Tinker RL.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

import chz
from dotenv import load_dotenv
from openai import AsyncOpenAI

from pingy.datasets import PingyTask, load_datasets_spec
from pingy.docker_sandbox import docker_sandbox_factory
from pingy.env import initialize_task_sandbox, modal_sandbox_factory
from pingy.pi_tools import PI_SYSTEM_PROMPT, PiSandboxTools
from pingy.reward import PingyHarborReward

logger = logging.getLogger(__name__)

OPENAI_PI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read the contents of a file (1-indexed offset/limit).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": ["integer", "null"]},
                    "limit": {"type": ["integer", "null"]},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command in the current working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": ["integer", "null"]},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Make exact, unique, non-overlapping text replacements in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "oldText": {"type": "string"},
                                "newText": {"type": "string"},
                            },
                            "required": ["oldText", "newText"],
                        },
                    },
                },
                "required": ["path", "edits"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Create or overwrite a file with new content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
]


PROVIDER_DEFAULTS: dict[str, tuple[str, str, str]] = {
    # provider -> (base_url, default_model, env_var)
    "modular": (
        "https://api.modular.com/v1",
        "deepseek/deepseek-v4.1-flash-1017",
        "MODULAR_API_KEY",
    ),
    "river": (
        "https://api.river.ai/v1",
        "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "RIVER_API_KEY",
    ),
}


@chz.chz
class MineConfig:
    provider: Literal["modular", "river", "custom"] = "modular"
    dataset: str = "terminal-bench-2.0"
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    sandbox: Literal["docker", "modal"] = "docker"
    output_jsonl: str = "./artifacts/sft_verified_traces.jsonl"
    max_tasks: int = 20
    max_turns: int = 25
    concurrency: int = 2


def resolve_mining_provider(cfg: MineConfig) -> tuple[str, str, str]:
    """Return `(base_url, model, api_key_env)` for the chosen Stage 0 provider."""
    if cfg.provider in PROVIDER_DEFAULTS:
        default_url, default_model, default_env = PROVIDER_DEFAULTS[cfg.provider]
        return (
            cfg.base_url or default_url,
            cfg.model or default_model,
            cfg.api_key_env or default_env,
        )
    if not cfg.base_url or not cfg.model:
        raise ValueError("provider=custom requires explicit base_url=... and model=...")
    return cfg.base_url, cfg.model, cfg.api_key_env or "OPENAI_API_KEY"


async def _dispatch_tool(
    pi_tools: PiSandboxTools, name: str, raw_args: str, call_id: str | None = None
) -> str:
    args = json.loads(raw_args or "{}")
    return await pi_tools.call_tool(name=name, arguments=args, call_id=call_id)


async def _run_one(
    client: AsyncOpenAI,
    model: str,
    task: PingyTask,
    cfg: MineConfig,
    sem: asyncio.Semaphore,
) -> dict[str, Any] | None:
    factory = modal_sandbox_factory if cfg.sandbox == "modal" else docker_sandbox_factory
    async with sem:
        sandbox = await factory(task, 900)
        try:
            workdir = await initialize_task_sandbox(sandbox, task)
            pi_tools = PiSandboxTools(sandbox=sandbox, workdir=workdir)
            reward_fn = PingyHarborReward(
                tests_dir=task.task_dir / "tests",
                sandbox=sandbox,
                telemetry=pi_tools.telemetry,
                grader_timeout=max(300, task.verifier_timeout or 0),
            )

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": f"{PI_SYSTEM_PROMPT}\n\n<cwd>\n{workdir}\n</cwd>"},
                {"role": "user", "content": task.instruction},
            ]

            for _ in range(cfg.max_turns):
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    tools=OPENAI_PI_TOOLS,  # type: ignore[arg-type]
                    temperature=0.6,
                )
                msg = resp.choices[0].message
                assistant_dict: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
                if getattr(msg, "reasoning_content", None):
                    assistant_dict["reasoning_content"] = msg.reasoning_content
                if msg.tool_calls:
                    assistant_dict["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
                messages.append(assistant_dict)

                if not msg.tool_calls:
                    break

                for tc in msg.tool_calls:
                    out = await _dispatch_tool(
                        pi_tools, tc.function.name, tc.function.arguments, call_id=tc.id
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.function.name,
                            "content": out,
                        }
                    )

            score, metrics = await reward_fn(messages)
            logger.info(
                "Task %s: passed=%s score=%.3f calls=%d",
                task.task_name,
                metrics.get("test_passed"),
                score,
                pi_tools.telemetry.total_calls,
            )
            if metrics.get("test_passed", 0.0) >= 1.0:
                return {
                    "task_name": task.task_name,
                    "source": task.source,
                    "teacher_model": model,
                    "score": score,
                    "metrics": metrics,
                    "messages": messages,
                }
            return None
        finally:
            await sandbox.cleanup()


async def run(cfg: MineConfig) -> None:
    base_url, model, env_var = resolve_mining_provider(cfg)
    api_key = os.environ.get(env_var)
    if not api_key:
        raise RuntimeError(f"Set {env_var} in pingy/.env to mine traces with provider={cfg.provider}.")

    groups = load_datasets_spec(cfg.dataset)
    tasks = [t for g in groups for t in g][: cfg.max_tasks]

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    sem = asyncio.Semaphore(cfg.concurrency)
    results = await asyncio.gather(*(_run_one(client, model, t, cfg, sem) for t in tasks))

    out_path = Path(cfg.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = [r for r in results if r is not None]
    payload = "".join(json.dumps(row) + "\n" for row in kept)
    await asyncio.to_thread(out_path.write_text, payload, "utf-8")
    logger.info("Saved %d/%d verified pi trajectories to %s", len(kept), len(tasks), out_path)


def entrypoint() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = chz.entrypoint(MineConfig)
    asyncio.run(run(cfg))


if __name__ == "__main__":
    entrypoint()
