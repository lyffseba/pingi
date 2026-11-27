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


def checkpoint_identity(path: str) -> tuple[str, str] | None:
    """Return the Tinker session and checkpoint name, ignoring weights versus sampler weights."""
    marker = ":train:0/"
    if not path.startswith("tinker://") or marker not in path:
        return None
    session = path.removeprefix("tinker://").split(marker, 1)[0]
    leaf = path.rsplit("/", 1)[-1]
    if not session or (leaf != "final" and not leaf.isdigit()):
        return None
    return session, leaf


def _refuse(reason: str) -> GateResult:
    return GateResult(False, (reason,), 0.0, 0.0, 0, 0, 0)


def check_eval_dir(eval_dir: Path, tinker_path: str, min_tasks: int = 20) -> GateResult:
    """Score one eval directory only if its evidence belongs to this checkpoint."""
    expected = checkpoint_identity(tinker_path)
    if expected is None:
        return _refuse("checkpoint path is not a Tinker training checkpoint")
    evidence_path = eval_dir / "run.json"
    if not evidence_path.is_file():
        return _refuse("paired eval is missing run.json checkpoint evidence")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if checkpoint_identity(str(evidence.get("checkpoint_path", ""))) != expected:
        return _refuse("paired eval checkpoint does not match the checkpoint being released")
    if evidence.get("compare_with_base") is False:
        return _refuse("paired eval did not compare against the base model")
    if not evidence.get("heldout_manifest_sha256"):
        return _refuse("paired eval is missing the held-out manifest hash")
    base_path, pingi_path = eval_dir / "base.jsonl", eval_dir / "pingi.jsonl"
    if not base_path.is_file() or not pingi_path.is_file():
        return _refuse("paired eval is missing base.jsonl or pingi.jsonl")
    base_rows, pingi_rows = _load_jsonl(base_path), _load_jsonl(pingi_path)
    if any(row.get("error") for row in base_rows + pingi_rows):
        return _refuse("paired eval contains task errors and cannot authorize a release")
    recorded = [row.get("task_name") for row in evidence.get("tasks", [])]
    if recorded and sorted(recorded) != sorted(row["task_name"] for row in base_rows):
        return _refuse("paired eval scores do not match the recorded task manifest")
    return evaluate_pair(base_rows, pingi_rows, min_tasks=min_tasks)


def check_release(
    tinker_path: str,
    eval_root: Path = Path("./artifacts/evals"),
    eval_dir: Path | None = None,
    min_tasks: int = 20,
) -> GateResult:
    """Release only a checkpoint named by its own paired, winning eval evidence."""
    if eval_dir is not None:
        return check_eval_dir(eval_dir, tinker_path, min_tasks)
    if checkpoint_identity(tinker_path) is None:
        return _refuse("checkpoint path is not a Tinker training checkpoint")
    if not eval_root.is_dir():
        return _refuse("no paired eval found; run `uv run pingi-eval compare_with_base=True` first")
    matches = []
    for candidate in sorted(path for path in eval_root.iterdir() if (path / "run.json").is_file()):
        evidence = json.loads((candidate / "run.json").read_text(encoding="utf-8"))
        if checkpoint_identity(str(evidence.get("checkpoint_path", ""))) == checkpoint_identity(tinker_path):
            matches.append(candidate)
    if not matches:
        return _refuse("no paired eval evidence matches this checkpoint")
    if len(matches) != 1:
        return _refuse("multiple evals match this checkpoint; choose one explicitly")
    return check_eval_dir(matches[0], tinker_path, min_tasks)
