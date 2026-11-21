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
    pingi_pass: float
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


def evaluate_pair(base_rows: list[dict], pingi_rows: list[dict], min_tasks: int = 20) -> GateResult:
    by_base = {r["task_name"]: bool(r.get("passed")) for r in base_rows}
    by_pingi = {r["task_name"]: bool(r.get("passed")) for r in pingi_rows}
    common = sorted(set(by_base) & set(by_pingi))
    n = len(common)
    base_pass = (sum(by_base[t] for t in common) / n) if n else 0.0
    pingi_pass = (sum(by_pingi[t] for t in common) / n) if n else 0.0
    newly = sum(1 for t in common if not by_base[t] and by_pingi[t])
    lost = sum(1 for t in common if by_base[t] and not by_pingi[t])
    reasons: list[str] = []
    if n < min_tasks:
        reasons.append(f"need at least {min_tasks} paired held-out tasks, found {n}")
    if pingi_pass < base_pass:
        reasons.append(
            f"PINGI pass@1 {pingi_pass:.1%} is below base {base_pass:.1%} on the same tasks"
        )
    if newly <= lost:
        reasons.append(f"newly solved ({newly}) is not greater than lost ({lost})")
    return GateResult(
        ok=not reasons,
        reasons=tuple(reasons),
        base_pass=base_pass,
        pingi_pass=pingi_pass,
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
    if found is None or not (found / "pingi.jsonl").is_file():
        return GateResult(
            False,
            ("no paired eval found; run `uv run pingi-eval compare_with_base=True` first",),
            0.0,
            0.0,
            0,
            0,
            0,
        )
    return evaluate_pair(
        _load_jsonl(found / "base.jsonl"),
        _load_jsonl(found / "pingi.jsonl"),
        min_tasks=min_tasks,
    )
