"""Harbor task loading for PINGI (local cache + Hugging Face harness-hub datasets).

Supports all 3 Harbor task layouts used across Terminal-Bench 2.0, scale800-harbor,
and termgrade-train-harbor:
  1. `environment/Dockerfile` (e.g. terminal-bench-2.0)
  2. `task.toml` with `[environment] docker_image` + `environment/Dockerfile` (scale800-harbor)
  3. `task.toml` with `[environment] docker_image` + `environment/setup.sh` (termgrade-train-harbor)
"""

from __future__ import annotations

import logging
import random
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HARBOR_CACHE_DIR = Path.home() / ".cache" / "harbor" / "tasks"


@dataclass(frozen=True)
class PingiTask:
    """A single Harbor task directory."""

    task_name: str
    instruction: str
    task_dir: Path
    source: str = "local"
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def docker_image(self) -> str | None:
        env = self.config.get("environment", {})
        img = env.get("docker_image") if isinstance(env, dict) else None
        return str(img) if img else None

    @property
    def configured_workdir(self) -> str | None:
        env = self.config.get("environment", {})
        wd = env.get("workdir") if isinstance(env, dict) else None
        return str(wd) if wd else None

    @property
    def dockerfile_path(self) -> Path | None:
        p = self.task_dir / "environment" / "Dockerfile"
        return p if p.is_file() else None

    @property
    def setup_script_path(self) -> Path | None:
        p = self.task_dir / "environment" / "setup.sh"
        return p if p.is_file() else None

    @property
    def verifier_timeout(self) -> int | None:
        ver = self.config.get("verifier", {})
        val = ver.get("timeout_sec") if isinstance(ver, dict) else None
        return int(float(val)) if val is not None else None

    @property
    def agent_timeout(self) -> int | None:
        ag = self.config.get("agent", {})
        val = ag.get("timeout_sec") if isinstance(ag, dict) else None
        return int(float(val)) if val is not None else None


def _is_valid_harbor_task_dir(task_dir: Path) -> bool:
    if not (task_dir / "instruction.md").is_file():
        return False
    if not (task_dir / "tests").is_dir():
        return False
    if (task_dir / "environment" / "Dockerfile").is_file():
        return True
    toml_path = task_dir / "task.toml"
    if toml_path.is_file():
        try:
            cfg = tomllib.loads(toml_path.read_text(encoding="utf-8"))
            env = cfg.get("environment", {})
            if isinstance(env, dict) and env.get("docker_image"):
                return True
        except (OSError, tomllib.TOMLDecodeError):
            return False
    return False


def _discover_task_dirs(root: Path) -> list[Path]:
    """Find every valid Harbor task directory under `root`."""
    found: list[Path] = []
    if not root.exists():
        return found
    for inst in sorted(root.rglob("instruction.md")):
        task_dir = inst.parent
        if _is_valid_harbor_task_dir(task_dir):
            found.append(task_dir)
    return found


def load_tasks_from_dir(root: Path, source: str = "local") -> list[PingiTask]:
    tasks: list[PingiTask] = []
    for task_dir in _discover_task_dirs(root):
        toml_path = task_dir / "task.toml"
        cfg = tomllib.loads(toml_path.read_text(encoding="utf-8")) if toml_path.is_file() else {}
        tasks.append(
            PingiTask(
                task_name=task_dir.name,
                instruction=(task_dir / "instruction.md").read_text(encoding="utf-8"),
                task_dir=task_dir,
                source=source,
                config=cfg,
            )
        )
    tasks.sort(key=lambda t: t.task_name)
    return tasks


def load_local_harbor_dataset(dataset_name: str) -> list[PingiTask]:
    """Load tasks downloaded with `uvx harbor datasets download ...`."""
    root = HARBOR_CACHE_DIR / dataset_name
    tasks = load_tasks_from_dir(root, source=dataset_name)
    if not tasks:
        raise FileNotFoundError(
            f"No Harbor tasks found under {root}. Download them first, e.g.:\n"
            f"  uvx harbor datasets download terminal-bench@2.0 -o {HARBOR_CACHE_DIR / 'terminal-bench-2.0'}"
        )
    return tasks


def sync_hf_harbor_dataset(repo_id: str, cache_root: Path | None = None) -> list[PingiTask]:
    """Load a Hugging Face Harbor dataset from the local cache.

    Do not download here. Large Hub snapshots time out inside a training job.
    Download first with the official CLI, then rerun training:

        uv run hf download <repo_id> --repo-type dataset --max-workers 2 \
            --local-dir ~/.cache/harbor/tasks/<repo_id with / replaced by __>
    """
    target = (cache_root or HARBOR_CACHE_DIR) / repo_id.replace("/", "__")
    tasks = load_tasks_from_dir(target, source=repo_id)
    if not tasks:
        raise FileNotFoundError(
            f"No Harbor tasks in {target}. Download them first, outside training:\n"
            f"  uv run hf download {repo_id} --repo-type dataset --max-workers 2 \
"
            f"    --local-dir {target}"
        )
    logger.info("Loaded %d Harbor tasks from %s", len(tasks), target)
    return tasks


def load_datasets_spec(spec: str) -> list[list[PingiTask]]:
    """Parse a comma-separated list of local Harbor dataset names or HF dataset repo IDs."""
    groups: list[list[PingiTask]] = []
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        if "/" in item:
            groups.append(sync_hf_harbor_dataset(item))
        else:
            groups.append(load_local_harbor_dataset(item))
    if not groups:
        raise ValueError("No datasets specified. Pass e.g. datasets=terminal-bench-2.0")
    return groups


def interleave_tasks(task_groups: list[list[PingiTask]], *, seed: int = 42) -> list[PingiTask]:
    """Deterministically shuffle within each source, then round-robin interleave."""
    rng = random.Random(seed)
    shuffled: list[list[PingiTask]] = []
    for group in task_groups:
        g = list(group)
        rng.shuffle(g)
        shuffled.append(g)

    interleaved: list[PingiTask] = []
    max_len = max((len(g) for g in shuffled), default=0)
    for i in range(max_len):
        for g in shuffled:
            if i < len(g):
                interleaved.append(g[i])
    return interleaved


def split_train_dev_heldout(
    task_groups: list[list[PingiTask]],
    *,
    dev_fraction: float = 0.05,
    heldout_fraction: float = 0.05,
    max_dev_tasks: int = 16,
    max_heldout_tasks: int = 32,
    seed: int = 42,
) -> tuple[list[PingiTask], list[PingiTask], list[PingiTask]]:
    """Split into train, in-loop dev, and a held-out set never used to pick checkpoints."""
    interleaved = interleave_tasks(task_groups, seed=seed)
    if len(interleaved) < 3:
        return interleaved, [], []

    n_heldout = (
        0
        if max_heldout_tasks <= 0 or heldout_fraction <= 0
        else max(1, min(max_heldout_tasks, int(len(interleaved) * heldout_fraction)))
    )
    n_dev = (
        0
        if max_dev_tasks <= 0 or dev_fraction <= 0
        else max(1, min(max_dev_tasks, int(len(interleaved) * dev_fraction)))
    )
    if n_heldout + n_dev >= len(interleaved):
        n_heldout = 1 if n_heldout else 0
        n_dev = 1 if n_dev else 0
        if n_heldout + n_dev >= len(interleaved):
            return interleaved, [], []
    heldout = interleaved[:n_heldout]
    dev = interleaved[n_heldout : n_heldout + n_dev]
    train = interleaved[n_heldout + n_dev :]
    return train, dev, heldout


def interleave_and_split(
    task_groups: list[list[PingiTask]],
    *,
    eval_fraction: float = 0.05,
    max_eval_tasks: int = 32,
    seed: int = 42,
) -> tuple[list[PingiTask], list[PingiTask]]:
    """Train/dev split. Held-out tasks are not included; use `split_train_dev_heldout`."""
    train, dev, _heldout = split_train_dev_heldout(
        task_groups,
        dev_fraction=eval_fraction,
        heldout_fraction=0.0 if eval_fraction <= 0 else eval_fraction,
        max_dev_tasks=max_eval_tasks,
        max_heldout_tasks=0,
        seed=seed,
    )
    return train, dev
