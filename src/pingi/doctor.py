"""Pre-flight diagnostics for PINGI (`uv run pingi-doctor`)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from pingi.datasets import HARBOR_CACHE_DIR, load_tasks_from_dir
from pingi.pi_tools import PI_SYSTEM_PROMPT, PiSandboxTools
from pingi.thinking import PingiEffortRenderer


def _check_mark(ok: bool) -> str:
    return "[OK]" if ok else "[MISSING]"


def entrypoint() -> None:
    load_dotenv()
    print("=== PINGI Pre-Flight Doctor ===\n")

    # 1. API Keys
    tinker_ok = bool(os.environ.get("TINKER_API_KEY"))
    modal_ok = bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET")) or (
        Path.home() / ".modal.toml"
    ).is_file()
    hf_ok = bool(os.environ.get("HF_TOKEN")) or (
        Path.home() / ".cache" / "huggingface" / "token"
    ).is_file()
    wandb_ok = bool(os.environ.get("WANDB_API_KEY")) or (Path.home() / ".netrc").is_file()
    modular_ok = bool(os.environ.get("MODULAR_API_KEY"))
    river_ok = bool(os.environ.get("RIVER_API_KEY"))

    print("1. Credentials:")
    print(f"  {_check_mark(tinker_ok)} TINKER_API_KEY  -> https://tinker-console.thinkingmachines.ai")
    print(f"  {_check_mark(modal_ok)} Modal auth      -> https://modal.com/settings/lyffseba/usage (or `uv run modal setup`)")
    print(f"  {_check_mark(hf_ok)} Hugging Face    -> https://huggingface.co/settings/tokens")
    print(f"  {_check_mark(wandb_ok)} W&B (optional)  -> https://wandb.ai/authorize")
    print(f"  {_check_mark(modular_ok)} Modular (opt.)  -> https://console.modular.com/")
    print(f"  {_check_mark(river_ok)} River AI (opt.) -> https://river.ai")

    # 2. Inkling-2-Mini renderer + pi tools verification
    print("\n2. Model Renderer & Pi Tool Pipeline:")
    try:
        from tinker_cookbook import model_info, tokenizer_utils
        from tinker_cookbook.renderers import get_renderer

        model_name = "thinkingmachines/Inkling-2-Mini"
        rec = model_info.get_recommended_renderer_name(model_name)
        tok = tokenizer_utils.get_tokenizer(model_name)
        base = get_renderer(rec, tok)
        r_none = PingiEffortRenderer(base, thinking_mode="none")
        tools = PiSandboxTools(sandbox=None, workdir="/app")
        prefix = r_none.create_conversation_prefix_with_tools(
            tools=tools.tool_specs(), system_prompt=PI_SYSTEM_PROMPT
        )
        mi = r_none.build_generation_prompt(prefix + [{"role": "user", "content": "ping"}])
        decoded = tok.decode(mi.to_ints())
        ok_render = (
            "Thinking effort level: 0" in decoded
            and "tool_declare" in decoded
            and len(tools.tool_specs()) == 4
        )
        print(
            f"  {_check_mark(ok_render)} {model_name} | renderer={rec} | "
            f"tokenizer={type(tok.tml_tokenizer).__name__} | prompt_tokens={len(mi.to_ints())}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [ERROR] Renderer check failed: {exc}")

    # 3. Local Docker status (for preset=smoke)
    print("\n3. Local Docker (for $0 sandbox `preset=smoke`):")
    docker_bin = shutil.which("docker")
    docker_ok = False
    docker_msg = "docker binary not found"
    if docker_bin:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            docker_ok = True
            docker_msg = f"Docker daemon v{proc.stdout.strip()} running"
        else:
            docker_msg = "Docker Desktop is not running (start Docker Desktop for `sandbox=docker`, or use `sandbox=modal`)"
    print(f"  {_check_mark(docker_ok)} {docker_msg}")

    # 4. Harbor task cache
    print(f"\n4. Local Harbor Datasets ({HARBOR_CACHE_DIR}):")
    if HARBOR_CACHE_DIR.exists():
        found_any = False
        for child in sorted(HARBOR_CACHE_DIR.iterdir()):
            if child.is_dir():
                tasks = load_tasks_from_dir(child, source=child.name)
                if tasks:
                    found_any = True
                    print(f"  [OK] {child.name}: {len(tasks)} tasks")
        if not found_any:
            print("  [MISSING] No cached Harbor tasks yet.")
    else:
        print("  [MISSING] Cache directory not created yet.")

    print("\nNext step:")
    if not tinker_ok:
        print("  1. Paste TINKER_API_KEY into pingi/.env")
        print("  2. Start Docker Desktop (for `preset=smoke`) OR run `uv run modal setup` (for `preset=pilot` / `mvp`)")
        print("  3. Run: uv run pingi-rl preset=smoke   (~$0.25) -> uv run pingi-rl preset=pilot   (~$7)")
    else:
        print("  Ready! Run: uv run pingi-rl preset=smoke   (~$0.25) -> uv run pingi-rl preset=pilot   (~$7)")


if __name__ == "__main__":
    entrypoint()
