"""Strict binary thinking-mode contract for PINGI.

PINGI only exposes two modes — no adaptive or intermediate presets:
  - "none" (DEFAULT): Thinking disabled (0.0) — fast, 3-4x fewer tokens, Qwen3.8-pi style
  - "high":           Inkling-2-Mini native high thinking effort (0.9)
"""

from __future__ import annotations

from typing import Any, Literal

ThinkingMode = Literal["none", "high"]

THINKING_EFFORT_NONE: float = 0.0
THINKING_EFFORT_HIGH: float = 0.9

DEFAULT_THINKING_MODE: ThinkingMode = "none"

_MODE_TO_EFFORT: dict[ThinkingMode, float] = {
    "none": THINKING_EFFORT_NONE,
    "high": THINKING_EFFORT_HIGH,
}


def resolve_thinking_effort(mode: str) -> float:
    """Map strictly 'none' (default) or 'high' to Inkling's scalar effort."""
    normalized = mode.strip().lower()
    if normalized not in _MODE_TO_EFFORT:
        raise ValueError(
            f"Unsupported PINGI thinking mode {mode!r}. "
            "PINGI strictly supports binary modes only: 'none' (default, 0.0) or 'high' (0.9)."
        )
    return _MODE_TO_EFFORT[normalized]  # type: ignore[index]


class PingiEffortRenderer:
    """Thin proxy around a Tinker Cookbook renderer that enforces PINGI's binary effort.

    When `thinking_mode="high"` (0.9), generation and supervised rendering both
    inject Inkling's exact `Thinking effort level: 0.9` conditioning prefix.
    When `thinking_mode="none"` (0.0), both inject `Thinking effort level: 0`.
    """

    def __init__(self, base_renderer: Any, thinking_mode: ThinkingMode = DEFAULT_THINKING_MODE):
        self._base = base_renderer
        self.thinking_mode: ThinkingMode = thinking_mode
        self.effort: float = resolve_thinking_effort(thinking_mode)

    def build_generation_prompt(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("effort", self.effort)
        try:
            return self._base.build_generation_prompt(messages, *args, **kwargs)
        except TypeError:
            # Fallback if a non-Inkling renderer is used in local tests.
            kwargs.pop("effort", None)
            return self._base.build_generation_prompt(messages, *args, **kwargs)

    def build_supervised_example(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("effort", self.effort)
        try:
            return self._base.build_supervised_example(messages, *args, **kwargs)
        except TypeError:
            kwargs.pop("effort", None)
            return self._base.build_supervised_example(messages, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)
