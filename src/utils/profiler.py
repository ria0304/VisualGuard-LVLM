# -*- coding: utf-8 -*-
"""
Runtime profiler for VisualGuard-LVLM.

Provides lightweight wall-clock and per-operation timing that works on CPU-only
hardware.  All functions are no-ops (returning neutral values) when ``torch`` is
not available, so the module can be imported without loading PyTorch in test or
docs contexts.

Typical use::

    from src.utils.profiler import get_timer

    timer = get_timer()
    with timer.section("generate"):
        result = decoder.generate(image, question)
    print(timer.summary())
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional


class _Timer:
    """Simple wall-clock section timer.

    Stores per-entry ``_start`` on the stack so nested sections each have
    their own timestamp and ``end_section`` always finds a valid start time.
    """

    def __init__(self) -> None:
        self._sections: Dict[str, List[float]] = defaultdict(list)
        self._stack: List[tuple[str, float]] = []  # (name, start_time)

    def reset(self) -> None:
        """Reset all accumulated timing data and clear the stack."""
        self._sections.clear()
        self._stack.clear()

    def start_section(self, name: str) -> None:
        self._stack.append((name, time.perf_counter()))

    def end_section(self) -> None:
        if not self._stack:
            raise RuntimeError("end_section called without a matching start_section")
        name, start_time = self._stack.pop()
        duration = time.perf_counter() - start_time
        self._sections[name].append(duration)

    def summary(self) -> str:
        lines = ["Profiler summary:"]
        for name, durations in sorted(self._sections.items()):
            count = len(durations)
            total = sum(durations)
            mean = total / count if count else 0.0
            min_d = min(durations) if durations else 0.0
            max_d = max(durations) if durations else 0.0
            lines.append(
                f"  {name:30s}  count={count:3d}  total={total:7.3f}s  "
                f"mean={mean:6.3f}s  min={min_d:6.3f}s  max={max_d:6.3f}s"
            )
        return "\n".join(lines)


_timer_instance: Optional[_Timer] = None


def get_timer() -> _Timer:
    """Return the module-level timer instance, creating it on first call."""
    global _timer_instance
    if _timer_instance is None:
        _timer_instance = _Timer()
    return _timer_instance


def reset_timer() -> None:
    """Replace the module-level instance; mainly for test isolation."""
    global _timer_instance
    _timer_instance = _Timer()


# Convenience wrappers


def timed_section(name: str):
    """Context manager for a timed section.

    Use ``with timed_section("name"):``.
    Can also be used as a decorator: ``@timed_section("name")`` on a function.
    """

    @contextmanager
    def _cm():
        timer = get_timer()
        timer.start_section(name)
        try:
            yield
        finally:
            timer.end_section()

    return _cm()


# No-torch guards: the module should never raise ImportError for missing torch.

try:
    import torch  # noqa: F401  @mypy-ignore

    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch may not be available
    HAS_TORCH = False


def assert_is_tensor(x, msg: str = "expected a torch.Tensor") -> None:
    """No-op if torch is available and x is a Tensor; no-op otherwise."""
    if HAS_TORCH and not isinstance(x, torch.Tensor):
        raise TypeError(msg)


def safe_mean(values: List[float]) -> float:
    """Return the mean of *values*, or 0.0 when the list is empty (no torch needed)."""
    if not values:
        return 0.0
    return sum(values) / len(values)


# Export

__all__ = [
    "get_timer",
    "reset_timer",
    "timed_section",
    "assert_is_tensor",
    "safe_mean",
    "HAS_TORCH",
]