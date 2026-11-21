"""Print the Tinker path to resume a PINGY run (`uv run pingy-resume log_dir=...`)."""

from __future__ import annotations

import logging

import chz
from tinker_cookbook.checkpoint_utils import get_last_checkpoint


@chz.chz
class ResumeConfig:
    log_dir: str
    key: str = "state_path"


def entrypoint() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = chz.entrypoint(ResumeConfig)
    record = get_last_checkpoint(cfg.log_dir, required_key=cfg.key)
    if record is None:
        raise SystemExit(f"No checkpoint with {cfg.key} in {cfg.log_dir}")
    path = getattr(record, cfg.key)
    print(path)
    print(
        "Resume with:\n"
        f"  uv run pingy-rl preset=mvp load_checkpoint_path={path}"
    )
