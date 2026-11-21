"""Pi-native tool suite (`read`, `bash`, `edit`, `write`) for Harbor sandboxes.

Mirrors the exact tool names, parameter semantics, truncation rules, and prompt
conventions of the `pi` coding agent harness so PINGY learns native `pi` fluency
during Tinker RL and SFT.
"""

from __future__ import annotations

import posixpath
import re
import shlex
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

try:
    from tinker_cookbook.tool_use import ToolInput, ToolResult, simple_tool_result, tool
except ImportError:  # pragma: no cover - allows standalone unit tests before uv sync
    ToolInput = dict[str, Any]  # type: ignore[misc, assignment]
    ToolResult = dict[str, Any]  # type: ignore[misc, assignment]

    def simple_tool_result(text: str) -> Any:  # type: ignore[misc]
        return {"content": text}

    def tool(fn: Any) -> Any:  # type: ignore[misc]
        return fn


class EditOperation(TypedDict):
    """Single exact-match replacement inside a file."""

    oldText: str
    newText: str
MAX_READ_LINES = 500
MAX_READ_BYTES = 24 * 1024
MAX_BASH_LINES = 400
MAX_BASH_BYTES = 16 * 1024

_CAT_OR_SED_RE = re.compile(r"(?:^|[;&|]\s*)(?:cat|sed\s+-n|head|tail)\s+\S+", re.MULTILINE)

PI_SYSTEM_PROMPT = """You are PINGY, a coding assistant running inside pi, a coding agent harness. Help users inspect files, run commands, and modify code.

<tools>
- read: Read file contents
- bash: Execute bash commands (ls, grep, find, etc.)
- edit: Make precise file edits with exact text replacement, including multiple disjoint edits in one call
- write: Create or overwrite files
</tools>

<rules>
- Use bash for file operations like ls, rg, find
- Use read to examine files instead of cat or sed.
- Use edit for targeted changes. Batch changes to one file in a single call.
- Each edits[].oldText must exactly and uniquely match the original file. Keep matches short and non-overlapping; merge nearby changes without including large unchanged regions.
- Use write only for new files or complete rewrites.
- Be concise in your responses
- Show file paths clearly when working with files
</rules>"""


@dataclass
class PiTelemetry:
    """Tracks pi-specific tool habits across an episode for reward shaping & logging."""

    total_calls: int = 0
    read_calls: int = 0
    bash_calls: int = 0
    edit_calls: int = 0
    write_calls: int = 0
    edit_failures: int = 0
    batched_edit_calls: int = 0
    cat_in_bash_calls: int = 0
    tool_errors: int = 0
    history_events: list[str] = field(default_factory=list)

    @property
    def edit_error_rate(self) -> float:
        if self.edit_calls == 0:
            return 0.0
        return self.edit_failures / self.edit_calls

    def to_metrics(self) -> dict[str, float]:
        return {
            "pi/total_tool_calls": float(self.total_calls),
            "pi/read_calls": float(self.read_calls),
            "pi/bash_calls": float(self.bash_calls),
            "pi/edit_calls": float(self.edit_calls),
            "pi/write_calls": float(self.write_calls),
            "pi/edit_failures": float(self.edit_failures),
            "pi/edit_error_rate": self.edit_error_rate,
            "pi/batched_edit_calls": float(self.batched_edit_calls),
            "pi/cat_in_bash_calls": float(self.cat_in_bash_calls),
            "pi/tool_errors": float(self.tool_errors),
        }


def _resolve_path(path: str, workdir: str) -> str:
    if path.startswith("/"):
        return posixpath.normpath(path)
    return posixpath.normpath(posixpath.join(workdir, path))


def apply_pi_edits(
    original: str, edits: list[EditOperation] | list[dict[str, str]]
) -> tuple[str | None, str | None]:
    """Apply pi-style exact, unique, non-overlapping edits against the original file.

    Returns `(updated_content, None)` on success or `(None, error_message)` on failure.
    """
    if not isinstance(edits, list) or len(edits) == 0:
        return None, "Error: `edits` must be a non-empty list of {oldText, newText} objects."

    spans: list[tuple[int, int, str, int]] = []
    for idx, item in enumerate(edits):
        if not isinstance(item, dict) or "oldText" not in item or "newText" not in item:
            return None, f"Error: edits[{idx}] must contain both 'oldText' and 'newText'."
        old_text = item["oldText"]
        new_text = item["newText"]
        if not isinstance(old_text, str) or not isinstance(new_text, str):
            return None, f"Error: edits[{idx}].oldText and newText must be strings."
        if old_text == "":
            return None, f"Error: edits[{idx}].oldText cannot be empty."

        first = original.find(old_text)
        if first == -1:
            first_line = old_text.splitlines()[0].strip() if old_text.splitlines() else ""
            hint = ""
            if first_line:
                matches = [
                    i + 1
                    for i, ln in enumerate(original.splitlines())
                    if first_line in ln
                ][:3]
                if matches:
                    hint = f" Similar text found near line(s) {matches}."
            return None, (
                f"Error: edits[{idx}].oldText not found in file.{hint} "
                "Re-read the file with `read` and copy the exact text including indentation."
            )
        second = original.find(old_text, first + 1)
        if second != -1:
            return None, (
                f"Error: edits[{idx}].oldText matched multiple locations in file. "
                "Include a bit more surrounding context so it matches uniquely."
            )
        spans.append((first, first + len(old_text), new_text, idx))

    spans.sort(key=lambda s: s[0])
    for i in range(1, len(spans)):
        _, prev_end, _, prev_idx = spans[i - 1]
        cur_start, _, _, cur_idx = spans[i]
        if cur_start < prev_end:
            return None, (
                f"Error: edits[{prev_idx}] and edits[{cur_idx}] overlap in the original file. "
                "Merge overlapping or adjacent edits into one edit."
            )

    parts: list[str] = []
    cursor = 0
    for start, end, new_text, _ in spans:
        parts.append(original[cursor:start])
        parts.append(new_text)
        cursor = end
    parts.append(original[cursor:])
    return "".join(parts), None


def _truncate_head(text: str, max_lines: int, max_bytes: int) -> tuple[str, bool]:
    lines = text.splitlines(keepends=True)
    truncated = False
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
    joined = "".join(lines)
    encoded = joined.encode("utf-8", errors="replace")
    if len(encoded) > max_bytes:
        joined = encoded[:max_bytes].decode("utf-8", errors="ignore")
        truncated = True
    return joined, truncated


def _truncate_tail(text: str, max_lines: int, max_bytes: int) -> tuple[str, bool]:
    lines = text.splitlines(keepends=True)
    truncated = False
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
        truncated = True
    joined = "".join(lines)
    encoded = joined.encode("utf-8", errors="replace")
    if len(encoded) > max_bytes:
        joined = encoded[-max_bytes:].decode("utf-8", errors="ignore")
        truncated = True
    return joined, truncated


class PiSandboxTools:
    """Exposes pi's 4 core tools (`read`, `bash`, `edit`, `write`) over a SandboxInterface."""

    def __init__(
        self,
        sandbox: Any,
        workdir: str = "/app",
        default_command_timeout: int = 120,
    ) -> None:
        self._sandbox = sandbox
        self._workdir = workdir
        self._default_command_timeout = default_command_timeout
        self.telemetry = PiTelemetry()

    def all_tools(self) -> list[Any]:
        return [self.read, self.bash, self.edit, self.write]

    def tool_specs(self) -> list[Any]:
        return [t.to_spec() for t in self.all_tools()]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        call_id: str | None = None,
    ) -> str:
        """Execute one of the 4 pi tools through Tinker's FunctionTool.run() interface."""
        tool_map = {t.name: t for t in self.all_tools() if hasattr(t, "name")}
        if name not in tool_map:
            return f"Unknown tool: {name}"
        fn_tool = tool_map[name]
        if hasattr(fn_tool, "run"):
            res = await fn_tool.run(ToolInput(arguments=arguments, call_id=call_id))
        else:  # pragma: no cover
            res = await fn_tool(**arguments)
        messages = getattr(res, "messages", None)
        if messages:
            parts = []
            for m in messages:
                content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for chunk in content:
                        if isinstance(chunk, dict) and "text" in chunk:
                            parts.append(str(chunk["text"]))
            return "\n".join(parts)
        if isinstance(res, dict) and "content" in res:
            return str(res["content"])
        return str(res)

    @tool
    async def read(
        self,
        path: Annotated[str, "Path to the file to read (relative or absolute)"],
        offset: Annotated[int | None, "Line number to start reading from (1-indexed)"] = None,
        limit: Annotated[int | None, "Maximum number of lines to read"] = None,
    ) -> ToolResult:
        """Read the contents of a file. Output is truncated to 2000 lines or 50KB.
        Use offset/limit for large files.
        """
        self.telemetry.total_calls += 1
        self.telemetry.read_calls += 1
        resolved = _resolve_path(path, self._workdir)
        self.telemetry.history_events.append(f"read:{resolved}")

        res = await self._sandbox.read_file(resolved, max_bytes=2 * 1024 * 1024, timeout=60)
        if res.exit_code != 0:
            self.telemetry.tool_errors += 1
            err = (res.stderr or res.stdout or f"Cannot read {resolved}").strip()
            return simple_tool_result(f"Error reading {resolved}: {err}")

        lines = res.stdout.splitlines(keepends=True)
        total_lines = len(lines)
        start_idx = max(0, (offset - 1) if offset is not None else 0)
        end_idx = (start_idx + limit) if (limit is not None and limit > 0) else total_lines
        selected = "".join(lines[start_idx:end_idx])
        body, truncated = _truncate_head(selected, MAX_READ_LINES, MAX_READ_BYTES)

        if truncated or end_idx < total_lines:
            shown = len(body.splitlines())
            next_offset = start_idx + shown + 1
            body += (
                f"\n[Truncated: showing lines {start_idx + 1}-{start_idx + shown} "
                f"of {total_lines}. Use offset={next_offset} to continue.]"
            )
        return simple_tool_result(body)

    @tool
    async def bash(
        self,
        command: Annotated[str, "Shell command to execute"],
        timeout: Annotated[int | None, "Timeout in seconds (optional)"] = None,
    ) -> ToolResult:
        """Execute a bash command in the current working directory.
        Returns stdout and stderr. Output is truncated to last 2000 lines or 50KB.
        """
        self.telemetry.total_calls += 1
        self.telemetry.bash_calls += 1
        if _CAT_OR_SED_RE.search(command):
            self.telemetry.cat_in_bash_calls += 1
        self.telemetry.history_events.append(f"bash:{command[:80]}")

        effective_timeout = (
            int(timeout) if timeout is not None and timeout > 0 else self._default_command_timeout
        )
        wrapped_cmd = (
            "export DEBIAN_FRONTEND=noninteractive GIT_PAGER=cat PAGER=cat CI=true TERM=dumb; "
            f"{command}"
        )
        res = await self._sandbox.run_command(
            wrapped_cmd,
            workdir=self._workdir,
            timeout=effective_timeout,
            max_output_bytes=MAX_BASH_BYTES * 2,
        )
        combined = res.stdout
        if res.stderr:
            combined = f"{combined}\n{res.stderr}" if combined else res.stderr
        body, truncated = _truncate_tail(combined, MAX_BASH_LINES, MAX_BASH_BYTES)
        if truncated:
            body = "[Output truncated to last 2000 lines / 50KB]\n" + body
        if res.exit_code != 0:
            body = f"{body}\n[exit code: {res.exit_code}]".strip()
        return simple_tool_result(body or f"[exit code: {res.exit_code}]")

    @tool
    async def edit(
        self,
        path: Annotated[str, "Path to the file to edit (relative or absolute)"],
        edits: Annotated[
            list[EditOperation],
            "One or more targeted replacements: [{oldText, newText}]. Each oldText must uniquely match the original file and not overlap.",
        ],
    ) -> ToolResult:
        """Edit a single file using exact text replacement, supporting multiple disjoint edits in one call."""
        self.telemetry.total_calls += 1
        self.telemetry.edit_calls += 1
        if isinstance(edits, list) and len(edits) > 1:
            self.telemetry.batched_edit_calls += 1
        resolved = _resolve_path(path, self._workdir)
        self.telemetry.history_events.append(f"edit:{resolved}({len(edits) if isinstance(edits, list) else 0})")

        res = await self._sandbox.read_file(resolved, max_bytes=4 * 1024 * 1024, timeout=60)
        if res.exit_code != 0:
            self.telemetry.edit_failures += 1
            self.telemetry.tool_errors += 1
            err = (res.stderr or res.stdout or f"File not found: {resolved}").strip()
            return simple_tool_result(f"Error reading {resolved} for edit: {err}")

        updated, err = apply_pi_edits(res.stdout, edits)
        if err is not None or updated is None:
            self.telemetry.edit_failures += 1
            self.telemetry.tool_errors += 1
            return simple_tool_result(err or "Unknown edit error.")

        write_res = await self._sandbox.write_file(resolved, updated, timeout=60)
        if write_res.exit_code != 0:
            self.telemetry.edit_failures += 1
            self.telemetry.tool_errors += 1
            werr = (write_res.stderr or write_res.stdout or "write failed").strip()
            return simple_tool_result(f"Error writing edited file {resolved}: {werr}")

        return simple_tool_result(
            f"Successfully applied {len(edits)} edit(s) to {resolved}."
        )

    @tool
    async def write(
        self,
        path: Annotated[str, "Path to the file to write (relative or absolute)"],
        content: Annotated[str, "Content to write to the file"],
    ) -> ToolResult:
        """Write content to a file. Creates parent directories if needed; overwrites if it exists."""
        self.telemetry.total_calls += 1
        self.telemetry.write_calls += 1
        resolved = _resolve_path(path, self._workdir)
        self.telemetry.history_events.append(f"write:{resolved}")

        parent = posixpath.dirname(resolved)
        if parent and parent != "/":
            await self._sandbox.run_command(
                f"mkdir -p {shlex.quote(parent)}", workdir="/", timeout=30
            )
        write_res = await self._sandbox.write_file(resolved, content, timeout=60)
        if write_res.exit_code != 0:
            self.telemetry.tool_errors += 1
            werr = (write_res.stderr or write_res.stdout or "write failed").strip()
            return simple_tool_result(f"Error writing {resolved}: {werr}")

        n_bytes = len(content.encode("utf-8"))
        return simple_tool_result(f"Wrote {n_bytes} bytes to {resolved}.")
