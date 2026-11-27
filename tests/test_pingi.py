"""Unit tests for PINGI pi-native tools, gated reward shaping, and binary thinking policy."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from pingi.pi_tools import PiSandboxTools, apply_pi_edits
from pingi.reward import compute_shaped_reward
from pingi.thinking import (
    DEFAULT_THINKING_MODE,
    THINKING_EFFORT_HIGH,
    THINKING_EFFORT_NONE,
    PingiEffortRenderer,
    resolve_thinking_effort,
)


@dataclass
class _FakeResult:
    exit_code: int
    stdout: str
    stderr: str = ""


class _FakeSandbox:
    def __init__(self) -> None:
        self.files: dict[str, str] = {"/app/main.py": "def add(a, b):\n    return a - b\n\nX = 1\n"}

    async def run_command(self, command: str, workdir: str | None = None, timeout: int = 60, max_output_bytes: int | None = None):
        return _FakeResult(exit_code=0, stdout=f"ran:{command}@{workdir}")

    async def read_file(self, path: str, max_bytes: int | None = None, timeout: int = 60):
        if path not in self.files:
            return _FakeResult(exit_code=1, stdout="", stderr="No such file")
        return _FakeResult(exit_code=0, stdout=self.files[path])

    async def write_file(self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60):
        self.files[path] = content.decode("utf-8") if isinstance(content, bytes) else content
        return _FakeResult(exit_code=0, stdout="")


def test_binary_thinking_mode_only() -> None:
    assert DEFAULT_THINKING_MODE == "none"
    assert resolve_thinking_effort("none") == THINKING_EFFORT_NONE == 0.0
    assert resolve_thinking_effort("high") == THINKING_EFFORT_HIGH == 0.9
    for forbidden in ("low", "medium", "minimal", "xhigh", "adaptive", "auto"):
        with pytest.raises(ValueError):
            resolve_thinking_effort(forbidden)


def test_effort_renderer_injects_default_none_and_high() -> None:
    class _DummyBase:
        def build_generation_prompt(self, messages, *, effort=None):
            return {"messages": messages, "effort": effort}

        def build_supervised_example(self, messages, *, effort=None):
            return {"messages": messages, "effort": effort}

    r_default = PingiEffortRenderer(_DummyBase())
    assert r_default.build_generation_prompt([])["effort"] == 0.0
    assert r_default.build_supervised_example([])["effort"] == 0.0

    r_high = PingiEffortRenderer(_DummyBase(), thinking_mode="high")
    assert r_high.build_generation_prompt([])["effort"] == 0.9
    assert r_high.build_supervised_example([])["effort"] == 0.9


def test_apply_pi_edits_batched_and_guards() -> None:
    original = "alpha = 1\nbeta = 2\ngamma = 3\n"
    updated, err = apply_pi_edits(
        original,
        [
            {"oldText": "gamma = 3", "newText": "gamma = 30"},
            {"oldText": "alpha = 1", "newText": "alpha = 10"},
        ],
    )
    assert err is None
    assert updated == "alpha = 10\nbeta = 2\ngamma = 30\n"

    # Reject duplicate match
    _, err_dup = apply_pi_edits("x = 1\nx = 1\n", [{"oldText": "x = 1", "newText": "x = 2"}])
    assert err_dup is not None and "multiple" in err_dup

    # Reject overlapping edits
    _, err_ov = apply_pi_edits(
        "abcdef",
        [
            {"oldText": "abcd", "newText": "1"},
            {"oldText": "cdef", "newText": "2"},
        ],
    )
    assert err_ov is not None and "overlap" in err_ov


def test_pi_tools_and_gated_reward() -> None:
    async def _run() -> None:
        sb = _FakeSandbox()
        tools = PiSandboxTools(sandbox=sb, workdir="/app")

        read_out = await tools.call_tool("read", {"path": "main.py", "offset": 1, "limit": 2})
        assert "def add" in read_out
        edit_out = await tools.call_tool(
            "edit",
            {
                "path": "main.py",
                "edits": [
                    {"oldText": "return a - b", "newText": "return a + b"},
                    {"oldText": "X = 1", "newText": "X = 2"},
                ],
            },
        )
        assert "Successfully applied 2 edit(s)" in edit_out
        assert sb.files["/app/main.py"] == "def add(a, b):\n    return a + b\n\nX = 2\n"
        assert tools.telemetry.batched_edit_calls == 1
        assert tools.telemetry.edit_failures == 0

        # Passing task receives efficiency + clean-edit shaping
        pass_reward, pass_metrics = compute_shaped_reward(1.0, tools.telemetry)
        assert pass_reward > 1.0
        assert pass_metrics["test_passed"] == 1.0

        # Failing task gets NO shaping bonus (prevents reward hacking)
        fail_reward, fail_metrics = compute_shaped_reward(0.0, tools.telemetry)
        assert fail_reward == 0.0
        assert fail_metrics["test_passed"] == 0.0

    asyncio.run(_run())


def test_budget_presets_ladder() -> None:
    from pingi.train_rl import PRESETS, PingiRLConfig, resolve_preset_values

    assert set(PRESETS.keys()) == {"smoke", "pilot", "mvp", "full"}
    # Pilot = 4 * 8 * 20 = 640 rollouts (~$7 half-MVP on Tinker non-thinking)
    p = PRESETS["pilot"]
    assert p["group_size"] * p["groups_per_batch"] * p["max_steps"] == 640
    assert PRESETS["smoke"]["group_size"] * PRESETS["smoke"]["groups_per_batch"] * PRESETS["smoke"]["max_steps"] == 12
    assert PRESETS["mvp"]["group_size"] * PRESETS["mvp"]["groups_per_batch"] * PRESETS["mvp"]["max_steps"] == 1600
    assert PRESETS["full"]["group_size"] * PRESETS["full"]["groups_per_batch"] * PRESETS["full"]["max_steps"] == 5120

    # Explicit CLI overrides must ALWAYS beat preset defaults!
    cfg = PingiRLConfig(preset="pilot", sandbox="docker", max_steps=12)
    resolved = resolve_preset_values(cfg)
    assert resolved["sandbox"] == "docker"
    assert resolved["max_steps"] == 12
    assert resolved["group_size"] == 4
    assert resolved["groups_per_batch"] == 8


def test_harbor_layouts_termgrade_and_scale800(tmp_path) -> None:
    from pingi.datasets import load_tasks_from_dir
    from pingi.env import initialize_task_sandbox

    # Layout A: termgrade-train-harbor (NO Dockerfile! docker_image + setup.sh + workdir=/app)
    tg = tmp_path / "tasks" / "task_00000"
    (tg / "environment").mkdir(parents=True)
    (tg / "tests").mkdir(parents=True)
    (tg / "instruction.md").write_text("Fix service", encoding="utf-8")
    (tg / "environment" / "setup.sh").write_text("echo setup-ran\n", encoding="utf-8")
    (tg / "task.toml").write_text(
        '[environment]\ndocker_image = "docker.io/ybelkada/termgrade-base:v1"\nworkdir = "/app"\n',
        encoding="utf-8",
    )

    # Layout B: scale800-harbor (docker_image + Dockerfile)
    sc = tmp_path / "tasks" / "repo__bug"
    (sc / "environment").mkdir(parents=True)
    (sc / "tests").mkdir(parents=True)
    (sc / "instruction.md").write_text("Fix Go query", encoding="utf-8")
    (sc / "environment" / "Dockerfile").write_text("FROM alpine\nWORKDIR /testbed\n", encoding="utf-8")
    (sc / "task.toml").write_text(
        '[environment]\ndocker_image = "ghcr.io/younesbelkada/scale800:demo"\n',
        encoding="utf-8",
    )

    tasks = load_tasks_from_dir(tmp_path, source="mixed")
    assert len(tasks) == 2
    by_name = {t.task_name: t for t in tasks}
    assert by_name["task_00000"].docker_image == "docker.io/ybelkada/termgrade-base:v1"
    assert by_name["task_00000"].dockerfile_path is None
    assert by_name["task_00000"].setup_script_path is not None
    assert by_name["repo__bug"].docker_image == "ghcr.io/younesbelkada/scale800:demo"

    # Verify initialize_task_sandbox uploads & executes setup.sh in configured workdir
    sb = _FakeSandbox()
    wd = asyncio.run(initialize_task_sandbox(sb, by_name["task_00000"]))
    assert wd == "/app"
    assert "/app/setup.sh" in sb.files


def test_sft_datum_and_paired_eval_table() -> None:
    from tinker_cookbook import model_info, tokenizer_utils
    from tinker_cookbook.renderers import TrainOnWhat, get_renderer
    from tinker_cookbook.supervised.data import conversation_to_datum

    from pingi.eval import EvalTaskResult, format_summary_table
    from pingi.train_sft import normalize_trace_for_inkling

    model_name = "thinkingmachines/Inkling-2-Mini"
    tok = tokenizer_utils.get_tokenizer(model_name)
    r = PingiEffortRenderer(
        get_renderer(model_info.get_recommended_renderer_name(model_name), tok),
        thinking_mode="none",
    )
    raw = [
        {"role": "system", "content": "Custom sys"},
        {"role": "user", "content": "Read main.py and fix bug"},
        {
            "role": "assistant",
            "reasoning_content": "SHOULD BE STRIPPED IN NONE MODE",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path": "main.py"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "read", "content": "def add(a, b): return a - b"},
        {"role": "assistant", "content": "Fixed the bug in main.py."},
    ]
    norm = normalize_trace_for_inkling(raw, r, thinking_mode="none")
    assert all("reasoning_content" not in m for m in norm)
    datum = conversation_to_datum(norm, r, max_length=4096, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES)  # type: ignore[arg-type]
    assert datum.model_input.length > 0
    # Tool calls must survive rendering into Inkling's tool-call format
    rendered = tok.decode(datum.model_input.to_ints())
    assert "read" in rendered and "main.py" in rendered

    table = format_summary_table(
        "base",
        [EvalTaskResult("t1", False, 0.0, 0.0, 10, 12, 2, 1, 5.0)],
        "PINGI",
        [EvalTaskResult("t1", True, 1.0, 1.1, 4, 5, 1, 0, 2.0)],
    )
    assert "+1 newly solved / -0 lost" in table
    assert "100.0%" in table


def test_edit_schema_exposes_oldtext_and_newtext_and_verifier_timeout(tmp_path) -> None:
    import json

    from pingi.datasets import load_tasks_from_dir

    tools = PiSandboxTools(sandbox=_FakeSandbox(), workdir="/app")
    edit_tool = next(t for t in tools.all_tools() if t.name == "edit")
    schema_json = json.dumps(edit_tool.parameters_schema)
    assert '"oldText"' in schema_json
    assert '"newText"' in schema_json
    assert '"required": ["oldText", "newText"]' in schema_json

    tdir = tmp_path / "t1"
    (tdir / "environment").mkdir(parents=True)
    (tdir / "tests").mkdir(parents=True)
    (tdir / "environment" / "Dockerfile").write_text("FROM alpine\n", encoding="utf-8")
    (tdir / "instruction.md").write_text("Do task", encoding="utf-8")
    (tdir / "task.toml").write_text("[verifier]\ntimeout_sec = 900.0\n", encoding="utf-8")
    loaded = load_tasks_from_dir(tmp_path)
    assert loaded[0].verifier_timeout == 900


def test_heldout_split_and_release_gate() -> None:
    from pathlib import Path

    from pingi.datasets import PingiTask, split_train_dev_heldout
    from pingi.release_gate import evaluate_pair

    groups = [
        [
            PingiTask(task_name=f"t{i}", instruction="x", task_dir=Path(f"/tmp/t{i}"))
            for i in range(40)
        ]
    ]
    train, dev, heldout = split_train_dev_heldout(
        groups, max_dev_tasks=4, max_heldout_tasks=8, dev_fraction=0.2, heldout_fraction=0.2
    )
    names = [t.task_name for t in train + dev + heldout]
    assert len(names) == len(set(names)) == 40
    assert len(heldout) == 8
    assert len(dev) == 4
    assert not {t.task_name for t in heldout} & {t.task_name for t in train}

    base = [{"task_name": f"t{i}", "passed": i < 5} for i in range(20)]
    pingi = [{"task_name": f"t{i}", "passed": i < 12} for i in range(20)]
    gate = evaluate_pair(base, pingi, min_tasks=20)
    assert gate.ok
    assert gate.newly_solved == 7
    assert gate.lost == 0

    worse = [{"task_name": f"t{i}", "passed": False} for i in range(20)]
    assert not evaluate_pair(base, worse, min_tasks=20).ok


def test_release_requires_matching_checkpoint_and_rejects_current_pilot(tmp_path) -> None:
    import json
    from pathlib import Path

    from pingi.release_gate import check_eval_dir, check_release, checkpoint_identity

    weights = "tinker://run:train:0/weights/final"
    sampler = "tinker://run:train:0/sampler_weights/final"
    assert checkpoint_identity(weights) == checkpoint_identity(sampler) == ("run", "final")
    rows = [{"task_name": f"t{i}", "passed": i < 12, "error": None} for i in range(20)]
    base = [{"task_name": f"t{i}", "passed": i < 5, "error": None} for i in range(20)]
    evidence = {
        "checkpoint_path": sampler,
        "compare_with_base": True,
        "heldout_manifest_sha256": "abc",
        "tasks": [{"task_name": row["task_name"]} for row in rows],
    }
    (tmp_path / "run.json").write_text(json.dumps(evidence), encoding="utf-8")
    (tmp_path / "base.jsonl").write_text("".join(json.dumps(row) + "\n" for row in base), encoding="utf-8")
    (tmp_path / "pingi.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert check_eval_dir(tmp_path, weights).ok
    assert not check_eval_dir(tmp_path, "tinker://other:train:0/weights/final").ok

    followup = Path("docs/followup-eval/results")
    gate = check_release(
        "tinker://00751c13-86fd-503f-99f9-e4db8ba28e63:train:0/weights/final",
        eval_dir=followup,
    )
    assert not gate.ok
    assert gate.pingi_pass < gate.base_pass


def test_mining_provider_resolution() -> None:
    from pingi.mine_sft_modular import MineConfig, resolve_mining_provider

    url_m, model_m, env_m = resolve_mining_provider(MineConfig(provider="modular"))
    assert "modular.com" in url_m and env_m == "MODULAR_API_KEY" and model_m

    url_r, model_r, env_r = resolve_mining_provider(MineConfig(provider="river"))
    assert "river.ai" in url_r and env_r == "RIVER_API_KEY" and "DeepSeek" in model_r


def test_inkling2_mini_renderer_and_tool_env_integration() -> None:
    from tinker_cookbook import model_info, tokenizer_utils
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tool_use import build_agent_tool_env

    from pingi.env import build_pingi_rollout_config
    from pingi.pi_tools import PI_SYSTEM_PROMPT

    model_name = "thinkingmachines/Inkling-2-Mini"
    tok = tokenizer_utils.get_tokenizer(model_name)
    assert type(tok.tml_tokenizer).__name__ == "H200kBaseChatTokenizer"

    base = get_renderer(model_info.get_recommended_renderer_name(model_name), tok)
    tools = PiSandboxTools(sandbox=_FakeSandbox(), workdir="/app")
    rollout_cfg = build_pingi_rollout_config(
        max_turns=25, max_trajectory_tokens=32768, grader_timeout=300
    )

    async def _check() -> None:
        async def _reward(_h):
            return 1.0, {}

        for mode, expected_str in (("none", "Thinking effort level: 0"), ("high", "Thinking effort level: 0.9")):
            r = PingiEffortRenderer(base, thinking_mode=mode)  # type: ignore[arg-type]
            prefix = r.create_conversation_prefix_with_tools(
                tools=tools.tool_specs(), system_prompt=PI_SYSTEM_PROMPT
            )
            env = build_agent_tool_env(
                renderer=r,
                tools=tools.all_tools(),
                initial_messages=prefix + [{"role": "user", "content": "Fix bug"}],
                reward_fn=_reward,
                rollout_config=rollout_cfg,
                model_name=model_name,
                max_turns=25,
                max_trajectory_tokens=32768,
            )
            assert env.parse_error_policy is not None
            assert env.parse_error_policy.max_consecutive == 2
            assert env.message_env.tool_execution == "sequential"
            assert env.rollout_limits is not None and env.rollout_limits.max_tool_calls == 50
            obs, stop = await env.initial_observation()
            decoded = tok.decode(obs.to_ints())
            assert expected_str in decoded
            assert "tool_declare" in decoded
            assert stop == [200006]

    asyncio.run(_check())
