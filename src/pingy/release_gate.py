"""Refuse a Hugging Face publish until a held-out paired eval beats the base model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GateResult:
    ok: bool
    reasons: tuple[str, ...]
    base_pass: float
    pingy_pass: float
    newly_solved: int
    lost: int
    n_tasks: int


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def evaluate_pair(base_rows: list[dict], pingy_rows: list[dict], min_tasks: int = 20) -> GateResult:
    by_base = {r["task_name"]: bool(r.get("passed")) for r in base_rows}
    by_pingy = {r["task_name"]: bool(r.get("passed")) for r in pingy_rows}
    common = sorted(set(by_base) & set(by_pingy))
    n = len(common)
    base_pass = (sum(by_base[t] for t in common) / n) if n else 0.0
    pingy_pass = (sum(by_pingy[t] for t in common) / n) if n else 0.0
    newly = sum(1 for t in common if not by_base[t] and by_pingy[t])
    lost = sum(1 for t in common if by_base[t] and not by_pingy[t])
    reasons: list[str] = []
    if n < min_tasks:
        reasons.append(f"need at least {min_tasks} paired held-out tasks, found {n}")
    if pingy_pass < base_pass:
        reasons.append(
            f"PINGY pass@1 {pingy_pass:.1%} is below base {base_pass:.1%} on the same tasks"
        )
    if newly <= lost:
        reasons.append(f"newly solved ({newly}) is not greater than lost ({lost})")
    return GateResult(
        ok=not reasons,
        reasons=tuple(reasons),
        base_pass=base_pass,
        pingy_pass=pingy_pass,
        newly_solved=newly,
        lost=lost,
        n_tasks=n,
    )


def latest_eval_dir(root: Path) -> Path | None:
    if not root.is_dir():
        return None
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "base.jsonl").is_file())
    return dirs[-1] if dirs else None


def check_release(eval_root: Path = Path("./artifacts/evals"), min_tasks: int = 20) -> GateResult:
    found = latest_eval_dir(eval_root)
    if found is None or not (found / "pingy.jsonl").is_file():
        return GateResult(
            False,
            ("no paired eval found; run `uv run pingy-eval compare_with_base=True` first",),
            0.0,
            0.0,
            0,
            0,
            0,
        )
    return evaluate_pair(
        _load_jsonl(found / "base.jsonl"),
        _load_jsonl(found / "pingy.jsonl"),
        min_tasks=min_tasks,
    )
