"""Tinker RL Environment & DatasetBuilder for PINGI (pi-native Inkling-2-Mini)."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import chz
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.rl.rollout_limits import RolloutLimits
from tinker_cookbook.rl.rollout_presets import (
    ParseErrorPolicy,
    RolloutConfig,
    TerminationRewardPolicy,
)
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.sandbox import SandboxInterface
from tinker_cookbook.sandbox.modal_sandbox import ModalSandbox
from tinker_cookbook.tool_use import build_agent_tool_env

from pingi.datasets import PingiTask
from pingi.pi_tools import PI_SYSTEM_PROMPT, PiSandboxTools
from pingi.reward import PingiHarborReward
from pingi.thinking import DEFAULT_THINKING_MODE, PingiEffortRenderer, ThinkingMode

logger = logging.getLogger(__name__)

TaskSandboxFactory = Callable[[PingiTask, int], Awaitable[SandboxInterface]]


_MODAL_APP: Any = None
_MODAL_APP_LOCK = asyncio.Lock()
_MODAL_CREATE_LOCK = asyncio.Semaphore(int(os.environ.get("PINGI_MODAL_CONCURRENCY", "2")))
_SETUP_SEMAPHORE = asyncio.Semaphore(int(os.environ.get("PINGI_SETUP_CONCURRENCY", "8")))


async def _modal_app() -> Any:
    """Look up the shared Modal app once. Parallel lookups hit AppGetOrCreate rate limits."""
    global _MODAL_APP
    if _MODAL_APP is not None:
        return _MODAL_APP
    async with _MODAL_APP_LOCK:
        if _MODAL_APP is None:
            import modal

            app_name = os.environ.get("PINGI_RESOURCE_NAME", "pingi")
            environment_name = os.environ.get("PINGI_MODAL_ENVIRONMENT", "pingi")
            _MODAL_APP = await modal.App.lookup.aio(
                app_name,
                create_if_missing=True,
                environment_name=environment_name,
            )
    return _MODAL_APP


async def modal_sandbox_factory(task: PingiTask, timeout: int) -> SandboxInterface:
    """Create a Modal sandbox for a Harbor task.

    Prefers prebuilt `docker_image` from `task.toml` (instant start for scale800
    and termgrade-train-harbor), falling back to building `environment/Dockerfile`.
    Sandbox creates are capped so Modal rate limits do not starve the Tinker heartbeat.
    """
    import modal

    if task.docker_image:
        image = modal.Image.from_registry(task.docker_image)
    elif task.dockerfile_path is not None:
        image = modal.Image.from_dockerfile(
            path=str(task.dockerfile_path),
            context_dir=str(task.dockerfile_path.parent),
        )
    else:
        raise RuntimeError(
            f"Task {task.task_name} has neither docker_image nor environment/Dockerfile."
        )
    async with _MODAL_CREATE_LOCK:
        app = await _modal_app()
        sandbox = await modal.Sandbox.create.aio(app=app, image=image, timeout=timeout)
    return ModalSandbox(
        timeout=timeout,
        image=image,
        app=app,
        sandbox=sandbox,
    )


async def initialize_task_sandbox(sandbox: Any, task: PingiTask) -> str:
    """Resolve effective workdir and run `environment/setup.sh` if the task requires it.

    Workdir resolution order:
      1. `task.toml` `[environment] workdir` (e.g. `/app` in termgrade-train-harbor)
      2. Container's own `WORKDIR` (`pwd`, e.g. `/testbed` in scale800-harbor) if not `/`
      3. `/app` if it exists, otherwise `/root`
    """
    if task.configured_workdir:
        workdir = task.configured_workdir
        await sandbox.run_command(f"mkdir -p {shlex.quote(workdir)}", workdir="/", timeout=30)
    else:
        res = await sandbox.run_command(
            'p=$(pwd); if [ "$p" != "/" ] && [ -n "$p" ]; then echo "$p"; '
            'elif [ -d /testbed ]; then echo /testbed; '
            'elif [ -d /app ]; then echo /app; else echo /root; fi',
            workdir=None,
            timeout=15,
        )
        workdir = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else "/root"

    setup_script = task.setup_script_path
    if setup_script is not None:
        setup_target = f"{workdir.rstrip('/')}/setup.sh"
        await sandbox.write_file(
            setup_target, setup_script.read_bytes(), executable=True, timeout=60
        )
        setup_res = None
        async with _SETUP_SEMAPHORE:
            setup_res = await sandbox.run_command(
                f"bash {shlex.quote(setup_target)} > /tmp/pingi_setup.log 2>&1; "
                f"rc=$?; rm -f {shlex.quote(setup_target)}; exit $rc",
                workdir=workdir,
                timeout=900,
            )
        if setup_res.exit_code != 0:
            logger.warning(
                "setup.sh for task %s exited with %d (see /tmp/pingi_setup.log)",
                task.task_name,
                setup_res.exit_code,
            )

    return workdir


def build_pingi_rollout_config(
    *,
    max_turns: int,
    max_trajectory_tokens: int,
    grader_timeout: int,
    max_tool_calls: int | None = None,
) -> RolloutConfig:
    """Build an agentic RolloutConfig tailored for multi-turn `pi` tool-use on Inkling-2-Mini.

    - Retries up to 2 consecutive malformed tool calls with corrective feedback (`ParseErrorPolicy`)
    - Runs tool calls sequentially so filesystem/bash edits in one turn stay deterministic
    - Enforces grade-then-clamp (`zero_reward_on_limit=True`) so budget-blowing rollouts don't profit
    """
    return RolloutConfig(
        limits=RolloutLimits(
            max_turns=max_turns,
            max_trajectory_tokens=max_trajectory_tokens,
            max_tool_calls=max_tool_calls or max(40, max_turns * 2),
        ),
        parse_errors=ParseErrorPolicy(max_consecutive=2),
        termination=TerminationRewardPolicy(
            zero_reward_on_limit=True,
            skip_grading_on_timeout=True,
            grader_timeout_seconds=float(grader_timeout + 60),
        ),
        tool_execution="sequential",
    )


def initial_pi_messages(
    task: PingiTask,
    renderer: PingiEffortRenderer,
    pi_tools: PiSandboxTools,
    workdir: str,
) -> list[Message]:
    system_prompt = f"{PI_SYSTEM_PROMPT}\n\n<cwd>\n{workdir}\n</cwd>"
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=pi_tools.tool_specs(),
        system_prompt=system_prompt,
    )
    return prefix + [{"role": "user", "content": task.instruction}]


class PingiEnvGroupBuilder(EnvGroupBuilder):
    """Spawns a group of pi-native Harbor sandboxes for one task."""

    def __init__(
        self,
        task: PingiTask,
        model_name: str,
        renderer_name: str | None,
        thinking_mode: ThinkingMode,
        max_turns: int,
        group_size: int,
        sandbox_timeout: int = 2400,
        command_timeout: int = 120,
        grader_timeout: int = 300,
        max_trajectory_tokens: int = 32 * 1024,
        max_generation_tokens: int | None = None,
        context_overflow_reward: float = -0.1,
        w_tool_efficiency: float = 0.10,
        tool_budget: float = 20.0,
        w_clean_edits: float = 0.05,
        sandbox_factory: TaskSandboxFactory | None = None,
    ) -> None:
        self.task = task
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.thinking_mode = thinking_mode
        self.max_turns = max_turns
        self.group_size = group_size
        self.sandbox_timeout = sandbox_timeout
        self.command_timeout = command_timeout
        self.grader_timeout = grader_timeout
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_generation_tokens = max_generation_tokens
        self.context_overflow_reward = context_overflow_reward
        self.w_tool_efficiency = w_tool_efficiency
        self.tool_budget = tool_budget
        self.w_clean_edits = w_clean_edits
        self.sandbox_factory = sandbox_factory or modal_sandbox_factory
        self._sandboxes: list[SandboxInterface] = []

    async def make_envs(self) -> Sequence[Env]:
        self._sandboxes = []
        tests_dir = self.task.task_dir / "tests"

        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name
        )
        base_renderer = get_renderer(renderer_name, tokenizer)
        renderer = PingiEffortRenderer(base_renderer, thinking_mode=self.thinking_mode)

        effective_grader_timeout = max(self.grader_timeout, self.task.verifier_timeout or 0)
        rollout_cfg = build_pingi_rollout_config(
            max_turns=self.max_turns,
            max_trajectory_tokens=self.max_trajectory_tokens,
            grader_timeout=effective_grader_timeout,
        )

        # Spawn group_size sandboxes concurrently; clean up any created ones on partial failure.
        results = await asyncio.gather(
            *(self.sandbox_factory(self.task, self.sandbox_timeout) for _ in range(self.group_size)),
            return_exceptions=True,
        )
        sandboxes: list[SandboxInterface] = []
        first_err: BaseException | None = None
        for item in results:
            if isinstance(item, BaseException):
                if first_err is None:
                    first_err = item
            else:
                sandboxes.append(item)
        if first_err is not None:
            for sb in sandboxes:
                try:
                    await sb.cleanup()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Cleanup after partial sandbox failure failed: %s", exc)
            raise first_err

        self._sandboxes = sandboxes
        workdirs = await asyncio.gather(
            *(initialize_task_sandbox(sb, self.task) for sb in sandboxes)
        )

        envs: list[Env] = []
        for sandbox, effective_workdir in zip(sandboxes, workdirs, strict=True):
            pi_tools = PiSandboxTools(
                sandbox=sandbox,
                workdir=effective_workdir,
                default_command_timeout=self.command_timeout,
            )
            reward_fn = PingiHarborReward(
                tests_dir=tests_dir,
                sandbox=sandbox,
                telemetry=pi_tools.telemetry,
                grader_timeout=effective_grader_timeout,
                w_tool_efficiency=self.w_tool_efficiency,
                tool_budget=self.tool_budget,
                w_clean_edits=self.w_clean_edits,
            )
            envs.append(
                build_agent_tool_env(
                    renderer=renderer,
                    tools=pi_tools.all_tools(),
                    initial_messages=initial_pi_messages(
                        self.task, renderer, pi_tools, effective_workdir
                    ),
                    reward_fn=reward_fn,
                    rollout_config=rollout_cfg,
                    model_name=self.model_name,
                    max_turns=self.max_turns,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                    max_generation_tokens=self.max_generation_tokens,
                    context_overflow_reward=self.context_overflow_reward,
                )
            )
        return envs

    async def cleanup(self) -> None:
        for sandbox in self._sandboxes:
            try:
                await sandbox.cleanup()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Sandbox cleanup failed: %s", exc)
        self._sandboxes.clear()

    def logging_tags(self) -> list[str]:
        return ["pingi", f"thinking:{self.thinking_mode}", f"src:{self.task.source}"]


class PingiRLDataset(RLDataset):
    def __init__(self, env_group_builders: list[PingiEnvGroupBuilder], batch_size: int) -> None:
        self.env_group_builders = env_group_builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = start + self.batch_size
        return self.env_group_builders[start:end]

    def __len__(self) -> int:
        return (len(self.env_group_builders) + self.batch_size - 1) // self.batch_size


@chz.chz
class PingiDatasetBuilder(RLDatasetBuilder):
    train_tasks: list[PingiTask]
    eval_tasks: list[PingiTask]
    batch_size: int
    group_size: int
    model_name: str
    thinking_mode: ThinkingMode = DEFAULT_THINKING_MODE
    renderer_name: str | None = None
    max_turns: int = 25
    sandbox_timeout: int = 2400
    command_timeout: int = 120
    grader_timeout: int = 300
    max_trajectory_tokens: int = 32 * 1024
    max_generation_tokens: int | None = None
    context_overflow_reward: float = -0.1
    w_tool_efficiency: float = 0.10
    tool_budget: float = 20.0
    w_clean_edits: float = 0.05
    sandbox_factory: TaskSandboxFactory | None = None

    def _builders(self, tasks: list[PingiTask], group_size: int) -> list[PingiEnvGroupBuilder]:
        return [
            PingiEnvGroupBuilder(
                task=task,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
                thinking_mode=self.thinking_mode,
                max_turns=self.max_turns,
                group_size=group_size,
                sandbox_timeout=self.sandbox_timeout,
                command_timeout=self.command_timeout,
                grader_timeout=self.grader_timeout,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
                context_overflow_reward=self.context_overflow_reward,
                w_tool_efficiency=self.w_tool_efficiency,
                tool_budget=self.tool_budget,
                w_clean_edits=self.w_clean_edits,
                sandbox_factory=self.sandbox_factory,
            )
            for task in tasks
        ]

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        train_ds = PingiRLDataset(
            self._builders(self.train_tasks, self.group_size),
            batch_size=self.batch_size,
        )
        eval_ds = (
            PingiRLDataset(
                self._builders(self.eval_tasks, group_size=1),
                batch_size=self.batch_size,
            )
            if self.eval_tasks
            else None
        )
        return train_ds, eval_ds
