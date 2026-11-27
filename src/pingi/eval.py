"""Paired Benchmark Evaluator for PINGI (`uv run pingi-eval`).

Evaluates base `thinkingmachines/Inkling-2-Mini` and/or any trained PINGI Tinker
checkpoint on the exact same Harbor tasks under `pi` tools, saving full JSONL
results and printing a paired pass@1 + efficiency comparison table.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import chz
import tinker
from dotenv import load_dotenv
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.rollout_runner import run_rollout
from tinker_cookbook.tool_use import build_agent_tool_env

from pingi.datasets import PingiTask, load_datasets_spec
from pingi.docker_sandbox import docker_sandbox_factory
from pingi.env import (
    TaskSandboxFactory,
    build_pingi_rollout_config,
    initial_pi_messages,
    initialize_task_sandbox,
    modal_sandbox_factory,
)
from pingi.pi_tools import PiSandboxTools
from pingi.reward import PingiHarborReward
from pingi.thinking import DEFAULT_THINKING_MODE, PingiEffortRenderer, ThinkingMode

logger = logging.getLogger(__name__)


@dataclass
class EvalTaskResult:
    task_name: str
    passed: bool
    correctness: float
    shaped_reward: float
    turns: int
    total_tool_calls: int
    edit_calls: int
    edit_failures: int
    time_seconds: float
    error: str | None = None


@chz.chz
class PingiEvalConfig:
    """Evaluate base Inkling-2-Mini vs a PINGI Tinker checkpoint on Harbor tasks."""

    model_name: str = "thinkingmachines/Inkling-2-Mini"
    checkpoint_path: str | None = None  # e.g. tinker://.../weights/000020 (None = base only)
    compare_with_base: bool = False  # True = run BOTH base and checkpoint on identical tasks
    thinking_mode: ThinkingMode = DEFAULT_THINKING_MODE
    datasets: str = "terminal-bench-2.0"
    sandbox: Literal["modal", "docker"] = "docker"
    max_tasks: int = 16
    max_turns: int = 25
    max_tokens: int = 4096
    max_trajectory_tokens: int = 48 * 1024
    temperature: float = 0.0
    concurrency: int = 4
    sandbox_timeout: int = 2400
    command_timeout: int = 120
    grader_timeout: int = 300
    output_dir: str = "./artifacts/evals"
    heldout_file: str | None = None
    base_url: str | None = None


async def _eval_one_task(
    task: PingiTask,
    policy: TinkerTokenCompleter,
    renderer: PingiEffortRenderer,
    sandbox_factory: TaskSandboxFactory,
    cfg: PingiEvalConfig,
    sem: asyncio.Semaphore,
) -> EvalTaskResult:
    start = time.monotonic()
    async with sem:
        sandbox = None
        try:
            sandbox = await sandbox_factory(task, cfg.sandbox_timeout)
            workdir = await initialize_task_sandbox(sandbox, task)
            pi_tools = PiSandboxTools(
                sandbox=sandbox,
                workdir=workdir,
                default_command_timeout=cfg.command_timeout,
            )
            effective_grader_timeout = max(cfg.grader_timeout, task.verifier_timeout or 0)
            rollout_cfg = build_pingi_rollout_config(
                max_turns=cfg.max_turns,
                max_trajectory_tokens=cfg.max_trajectory_tokens,
                grader_timeout=effective_grader_timeout,
            )
            reward_fn = PingiHarborReward(
                tests_dir=task.task_dir / "tests",
                sandbox=sandbox,
                telemetry=pi_tools.telemetry,
                grader_timeout=effective_grader_timeout,
            )
            env = build_agent_tool_env(
                renderer=renderer,
                tools=pi_tools.all_tools(),
                initial_messages=initial_pi_messages(task, renderer, pi_tools, workdir),
                reward_fn=reward_fn,
                rollout_config=rollout_cfg,
                model_name=cfg.model_name,
                max_turns=cfg.max_turns,
                max_trajectory_tokens=cfg.max_trajectory_tokens,
                max_generation_tokens=cfg.max_tokens,
            )
            traj = await run_rollout(policy, env, config=rollout_cfg)
            final_metrics: dict[str, float] = {}
            for tr in reversed(traj.transitions):
                if "correctness" in tr.metrics or "reward" in tr.metrics:
                    final_metrics = dict(tr.metrics)
                    break
            correctness = float(final_metrics.get("correctness", 0.0))
            shaped = float(final_metrics.get("reward", 0.0))
            return EvalTaskResult(
                task_name=task.task_name,
                passed=correctness >= 1.0,
                correctness=correctness,
                shaped_reward=shaped,
                turns=len(traj.transitions),
                total_tool_calls=pi_tools.telemetry.total_calls,
                edit_calls=pi_tools.telemetry.edit_calls,
                edit_failures=pi_tools.telemetry.edit_failures,
                time_seconds=round(time.monotonic() - start, 2),
            )
        except Exception as exc:
            logger.exception("Eval failed on %s", task.task_name)
            return EvalTaskResult(
                task_name=task.task_name,
                passed=False,
                correctness=0.0,
                shaped_reward=0.0,
                turns=0,
                total_tool_calls=0,
                edit_calls=0,
                edit_failures=0,
                time_seconds=round(time.monotonic() - start, 2),
                error=str(exc),
            )
        finally:
            if sandbox is not None:
                await sandbox.cleanup()


def format_summary_table(
    label_a: str,
    results_a: list[EvalTaskResult],
    label_b: str | None = None,
    results_b: list[EvalTaskResult] | None = None,
) -> str:
    def _stats(rows: list[EvalTaskResult]) -> tuple[float, float, float, float]:
        n = max(1, len(rows))
        pass_rate = 100.0 * sum(1 for r in rows if r.passed) / n
        med_turns = statistics.median([r.turns for r in rows]) if rows else 0.0
        med_calls = statistics.median([r.total_tool_calls for r in rows]) if rows else 0.0
        edits = sum(r.edit_calls for r in rows)
        edit_err = (100.0 * sum(r.edit_failures for r in rows) / edits) if edits else 0.0
        return pass_rate, med_turns, med_calls, edit_err

    lines = [
        "| Model | Tasks | Pass@1 | Median Turns | Median Tool Calls | Edit Error Rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for lbl, rows in [(label_a, results_a)] + (
        [(label_b, results_b)] if label_b and results_b is not None else []
    ):
        pr, mt, mc, ee = _stats(rows)
        lines.append(f"| {lbl} | {len(rows)} | {pr:.1f}% | {mt:.1f} | {mc:.1f} | {ee:.1f}% |")

    if label_b and results_b is not None:
        by_a = {r.task_name: r.passed for r in results_a}
        by_b = {r.task_name: r.passed for r in results_b}
        common = sorted(set(by_a) & set(by_b))
        newly_solved = sum(1 for t in common if not by_a[t] and by_b[t])
        lost = sum(1 for t in common if by_a[t] and not by_b[t])
        lines.append(
            f"\nPaired diff ({label_b} vs {label_a} on {len(common)} tasks): "
            f"+{newly_solved} newly solved / -{lost} lost."
        )
    return "\n".join(lines)


async def _run_suite(
    label: str,
    sampling_client: tinker.SamplingClient,
    tasks: list[PingiTask],
    renderer: PingiEffortRenderer,
    sandbox_factory: TaskSandboxFactory,
    cfg: PingiEvalConfig,
    progress_path: Path,
) -> list[EvalTaskResult]:
    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )
    sem = asyncio.Semaphore(cfg.concurrency)
    logger.info("Running eval [%s] on %d tasks (thinking=%s)", label, len(tasks), cfg.thinking_mode)
    completed = 0

    async def record(task: PingiTask) -> EvalTaskResult:
        nonlocal completed
        result = await _eval_one_task(task, policy, renderer, sandbox_factory, cfg, sem)
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result)) + "\n")
        completed += 1
        logger.info("Eval [%s] %d/%d: %s", label, completed, len(tasks), task.task_name)
        return result

    return list(await asyncio.gather(*(record(task) for task in tasks)))


def _filter_heldout(
    tasks: list[PingiTask], manifest: bytes | None, max_tasks: int
) -> list[PingiTask]:
    if max_tasks < 1:
        raise ValueError("max_tasks must be positive")
    if manifest is None:
        return tasks[:max_tasks]

    rows = json.loads(manifest)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Held-out manifest must contain a nonempty list of tasks")
    by_name: dict[str, list[PingiTask]] = {}
    for task in tasks:
        by_name.setdefault(task.task_name, []).append(task)
    selected = []
    seen = set()
    for row in rows:
        name = row["task_name"]
        if name in seen:
            raise ValueError(f"Duplicate held-out task: {name}")
        seen.add(name)
        source = row.get("source")
        candidates = [t for t in by_name.get(name, []) if source is None or t.source == source]
        if len(candidates) != 1:
            raise ValueError(f"Held-out task {name!r} did not match exactly one cached task")
        task = candidates[0]
        fingerprint = row.get("instruction_sha256")
        if fingerprint is not None and fingerprint != hashlib.sha256(task.instruction.encode()).hexdigest():
            raise ValueError(f"Held-out task {name!r} instruction changed since the manifest was frozen")
        selected.append(task)
    return selected[:max_tasks]


async def run(cfg: PingiEvalConfig) -> None:
    if cfg.concurrency < 1:
        raise ValueError("concurrency must be positive")
    if cfg.compare_with_base and not cfg.checkpoint_path:
        raise ValueError("compare_with_base requires checkpoint_path")
    groups = load_datasets_spec(cfg.datasets)
    manifest = Path(cfg.heldout_file).read_bytes() if cfg.heldout_file else None
    tasks = _filter_heldout([t for g in groups for t in g], manifest, cfg.max_tasks)
    if not tasks:
        raise ValueError("No evaluation tasks selected")
    sandbox_factory = modal_sandbox_factory if cfg.sandbox == "modal" else docker_sandbox_factory

    tok = tokenizer_utils.get_tokenizer(cfg.model_name)
    r_name = model_info.get_recommended_renderer_name(cfg.model_name)
    renderer = PingiEffortRenderer(get_renderer(r_name, tok), thinking_mode=cfg.thinking_mode)

    service = tinker.ServiceClient(base_url=cfg.base_url)

    out_dir = Path(cfg.output_dir) / datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%fZ")
    out_dir.mkdir(parents=True)
    manifest_hash = hashlib.sha256(manifest).hexdigest() if manifest is not None else None
    evidence = {
        "model_name": cfg.model_name,
        "checkpoint_path": cfg.checkpoint_path,
        "compare_with_base": cfg.compare_with_base,
        "datasets": cfg.datasets,
        "renderer": r_name,
        "heldout_manifest_sha256": manifest_hash,
        "tasks": [{"task_name": t.task_name, "source": t.source} for t in tasks],
        "thinking_mode": cfg.thinking_mode,
        "temperature": cfg.temperature,
        "max_turns": cfg.max_turns,
        "max_tokens": cfg.max_tokens,
        "max_trajectory_tokens": cfg.max_trajectory_tokens,
        "sandbox": cfg.sandbox,
    }
    (out_dir / "run.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    logger.info("Eval artifacts: %s", out_dir)

    models = {}
    if not cfg.checkpoint_path or cfg.compare_with_base:
        models["base"] = await service.create_sampling_client(base_model=cfg.model_name)
    if cfg.checkpoint_path:
        models["pingi"] = await service.create_sampling_client(
            model_path=cfg.checkpoint_path, base_model=cfg.model_name
        )
    results = {}
    for label, client in models.items():
        results[label] = await _run_suite(
            label, client, tasks, renderer, sandbox_factory, cfg, out_dir / f"{label}.jsonl.partial"
        )
    errors = sum(row.error is not None for rows in results.values() for row in rows)
    if errors:
        raise RuntimeError(f"{errors} task errors; incomplete eval evidence retained in {out_dir}")

    for label, rows in results.items():
        (out_dir / f"{label}.jsonl").write_text(
            "".join(json.dumps(asdict(row)) + "\n" for row in rows), encoding="utf-8"
        )
        (out_dir / f"{label}.jsonl.partial").unlink()
    base_label = f"{cfg.model_name.removeprefix('thinkingmachines/')} (base)"
    if cfg.compare_with_base:
        table = format_summary_table(base_label, results["base"], "PINGI", results["pingi"])
    elif "pingi" in results:
        table = format_summary_table("PINGI", results["pingi"])
    else:
        table = format_summary_table(base_label, results["base"])
    (out_dir / "summary.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table + f"\n\nSaved detailed results to {out_dir}")


def entrypoint() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = chz.entrypoint(PingiEvalConfig)
    asyncio.run(run(cfg))


if __name__ == "__main__":
    entrypoint()
