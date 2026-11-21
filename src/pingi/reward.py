"""PINGI reward function: Harbor verifier correctness + gated pi-native shaping.

Shaping bonuses (tool efficiency, clean exact-match edits, avoiding cat/sed in bash)
are strictly gated on `correctness >= 1.0`, preventing reward hacking where a model
exits early with fewer tool calls without solving the task.
"""

from __future__ import annotations

import json
import logging
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pingi.pi_tools import PiTelemetry

logger = logging.getLogger(__name__)


def compute_shaped_reward(
    correctness: float,
    telemetry: PiTelemetry,
    *,
    w_tool_efficiency: float = 0.10,
    tool_budget: float = 20.0,
    w_clean_edits: float = 0.05,
    w_read_discipline: float = 0.03,
    w_cat_penalty: float = 0.03,
) -> tuple[float, dict[str, float]]:
    """Compute final scalar reward and detailed diagnostic metrics.

    Every shaping term is strictly gated on `correctness >= 1.0`.
    """
    passed = correctness >= 1.0
    eff = max(0.0, min(1.0, 1.0 - telemetry.total_calls / tool_budget)) if tool_budget > 0 else 0.0

    # Only reward clean edits when `edit` was actually used cleanly (not zero-edit `cat`/`write` hacks).
    if telemetry.edit_calls > 0:
        clean_edit_score = 1.0 - telemetry.edit_error_rate
        if telemetry.batched_edit_calls > 0:
            clean_edit_score = min(1.2, clean_edit_score + 0.2)
    elif telemetry.write_calls == 0 and telemetry.cat_in_bash_calls == 0:
        clean_edit_score = 0.5
    else:
        clean_edit_score = 0.0

    # Reward using `read` rather than `cat`/`head`/`tail`/`sed -n` in `bash`.
    inspect_total = telemetry.read_calls + telemetry.cat_in_bash_calls
    read_discipline = (telemetry.read_calls / inspect_total) if inspect_total > 0 else 0.5
    cat_penalty = min(1.0, telemetry.cat_in_bash_calls * 0.25)

    if passed:
        total = (
            correctness
            + w_tool_efficiency * eff
            + w_clean_edits * clean_edit_score
            + w_read_discipline * read_discipline
            - w_cat_penalty * cat_penalty
        )
    else:
        total = correctness

    metrics = {
        "reward": float(total),
        "correctness": float(correctness),
        "test_passed": float(passed),
        "pi/efficiency_score": float(eff),
        "pi/clean_edit_score": float(clean_edit_score),
        "pi/read_discipline": float(read_discipline),
        **telemetry.to_metrics(),
    }
    return float(total), metrics


@dataclass
class PingiHarborReward:
    """Grades a Harbor task via `tests/test.sh` and applies gated pi-native shaping."""

    tests_dir: Path
    sandbox: Any
    telemetry: PiTelemetry
    grader_timeout: int = 300
    w_tool_efficiency: float = 0.10
    tool_budget: float = 20.0
    w_clean_edits: float = 0.05
    w_read_discipline: float = 0.03
    w_cat_penalty: float = 0.03

    async def __call__(self, history: list[Any]) -> tuple[float, dict[str, float]]:
        del history
        try:
            await self.sandbox.run_command(
                "mkdir -p /root /tests /logs/verifier && "
                "rm -rf /tests /logs/verifier/reward.txt /logs/verifier/reward.json && "
                "mkdir -p /tests /logs/verifier",
                workdir="/",
                timeout=30,
            )
            await self._upload_tests()
            res = await self.sandbox.run_command(
                "bash /tests/test.sh",
                workdir="/root",
                timeout=self.grader_timeout,
            )
            logger.info("test.sh exited with code=%d", res.exit_code)
            correctness = await self._parse_reward()
            return compute_shaped_reward(
                correctness,
                self.telemetry,
                w_tool_efficiency=self.w_tool_efficiency,
                tool_budget=self.tool_budget,
                w_clean_edits=self.w_clean_edits,
                w_read_discipline=self.w_read_discipline,
                w_cat_penalty=self.w_cat_penalty,
            )
        except Exception:
            logger.exception("Harbor grading failed")
            return 0.0, {
                "reward": 0.0,
                "correctness": 0.0,
                "test_passed": 0.0,
                "grading_error": 1.0,
                **self.telemetry.to_metrics(),
            }

    async def _upload_tests(self) -> None:
        await self.sandbox.run_command("mkdir -p /tests", workdir="/", timeout=30)
        for file_path in self.tests_dir.rglob("*"):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(self.tests_dir).as_posix()
            target = f"/tests/{rel}"
            parent = str(Path(target).parent)
            if parent != "/tests":
                await self.sandbox.run_command(
                    f"mkdir -p {shlex.quote(parent)}", workdir="/", timeout=30
                )
            content = file_path.read_bytes()
            await self.sandbox.write_file(
                target, content, executable=(file_path.suffix == ".sh"), timeout=60
            )

    async def _parse_reward(self) -> float:
        res = await self.sandbox.read_file("/logs/verifier/reward.txt", timeout=30)
        if res.exit_code == 0 and res.stdout.strip():
            return float(res.stdout.strip())

        res = await self.sandbox.read_file("/logs/verifier/reward.json", timeout=30)
        if res.exit_code == 0 and res.stdout.strip():
            data = json.loads(res.stdout)
            if "correctness" in data:
                return float(data["correctness"])
            return float(data.get("reward", 0.0))

        logger.warning("No verifier reward file found at /logs/verifier/")
        return 0.0
