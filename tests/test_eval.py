"""Offline checks for the follow-up evaluator and its evidence files."""

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pingi import eval as evaluator
from pingi.datasets import PingiTask


def _task(name: str, source: str = "a") -> PingiTask:
    return PingiTask(name, "A task", Path(name), source)


def _manifest(*rows: dict[str, str]) -> bytes:
    return json.dumps(rows).encode()


def test_frozen_manifest_preserves_order_and_rejects_missing_or_changed_tasks() -> None:
    tasks = [_task("x"), _task("y"), _task("x", "b")]
    x = {"task_name": "x", "source": "b"}
    y = {
        "task_name": "y",
        "source": "a",
        "instruction_sha256": hashlib.sha256(b"A task").hexdigest(),
    }
    assert evaluator._filter_heldout(tasks, _manifest(y, x), 2) == [tasks[1], tasks[2]]
    with pytest.raises(ValueError, match="Duplicate held-out task"):
        evaluator._filter_heldout(tasks, _manifest(x, x), 2)
    with pytest.raises(ValueError, match="exactly one"):
        evaluator._filter_heldout(tasks, _manifest({"task_name": "x"}), 2)
    with pytest.raises(ValueError, match="exactly one"):
        evaluator._filter_heldout(tasks, _manifest(y, {"task_name": "gone"}), 1)
    with pytest.raises(ValueError, match="instruction changed"):
        evaluator._filter_heldout(tasks, _manifest(y | {"instruction_sha256": "invalid"}), 2)


@pytest.mark.parametrize("fail", [False, True])
def test_eval_records_provenance_and_keeps_errors_out_of_final_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    tasks = [_task("one"), _task("two")]
    manifest = _manifest(*({"task_name": t.task_name} for t in tasks))
    manifest_path = tmp_path / "heldout.json"
    manifest_path.write_bytes(manifest)
    root = tmp_path / "evals"
    checkpoint_path = "tinker://mock:train:0/sampler_weights/final"

    async def create_client(**kwargs):
        return kwargs.get("model_path", "base")

    async def eval_task(task, policy, *_):
        pilot = policy.sampling_client == checkpoint_path
        error = "sandbox down" if fail and pilot and task.task_name == "two" else None
        out = next(root.iterdir())
        if pilot:
            assert len((out / "base.jsonl.partial").read_text().splitlines()) == 2
        return evaluator.EvalTaskResult(task.task_name, not error, 1.0, 1.0, 1, 1, 0, 0, 0, error)

    monkeypatch.setattr(evaluator, "load_datasets_spec", lambda _: [tasks])
    monkeypatch.setattr(evaluator.tokenizer_utils, "get_tokenizer", lambda _: object())
    monkeypatch.setattr(evaluator.model_info, "get_recommended_renderer_name", lambda _: "renderer")
    monkeypatch.setattr(evaluator, "get_renderer", lambda *_: object())
    monkeypatch.setattr(evaluator, "PingiEffortRenderer", lambda *_, **__: object())
    monkeypatch.setattr(evaluator, "TinkerTokenCompleter", SimpleNamespace)
    monkeypatch.setattr(
        evaluator.tinker, "ServiceClient", lambda **_: SimpleNamespace(create_sampling_client=create_client)
    )
    monkeypatch.setattr(evaluator, "_eval_one_task", eval_task)
    cfg = evaluator.PingiEvalConfig(
        checkpoint_path=checkpoint_path,
        compare_with_base=True,
        heldout_file=str(manifest_path),
        output_dir=str(root),
    )
    if fail:
        with pytest.raises(RuntimeError, match="1 task errors"):
            asyncio.run(evaluator.run(cfg))
    else:
        asyncio.run(evaluator.run(cfg))
    record = next(root.iterdir())
    evidence = json.loads((record / "run.json").read_text())
    assert evidence["checkpoint_path"] == checkpoint_path
    assert evidence["heldout_manifest_sha256"] == hashlib.sha256(manifest).hexdigest()
    assert (record / "summary.md").exists() != fail
    for label in ("base", "pingi"):
        assert (record / f"{label}.jsonl").exists() != fail
        assert (record / f"{label}.jsonl.partial").exists() == fail
