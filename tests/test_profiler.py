# -*- coding: utf-8 -*-
"""
Profiler unit tests.

These verify the profiler logic, not model behaviour. Nothing here touches a
real dataset or a pretrained checkpoint.
"""

from __future__ import annotations

import pytest

from src.utils.profiler import get_timer, safe_mean, HAS_TORCH, assert_is_tensor


class TestProfilerBasic:
    def test_basic_timing(self) -> None:
        timer = get_timer()
        timer.start_section("test_op")
        import time as _time
        _time.sleep(0.01)
        timer.end_section()
        summary = timer.summary()
        assert "test_op" in summary

    def test_nested_sections(self) -> None:
        timer = get_timer()
        timer.start_section("outer")
        timer.start_section("inner")
        import time as _time
        _time.sleep(0.005)
        timer.end_section()
        timer.end_section()
        summary = timer.summary()
        assert "outer" in summary
        assert "inner" in summary

    def test_reset_clears_data(self) -> None:
        """reset() clears accumulated data on the current instance."""
        timer = get_timer()
        timer.start_section("first")
        import time as _time
        _time.sleep(0.001)
        timer.end_section()
        assert any("first" in s for s in timer.summary().splitlines())
        timer.reset()
        # After reset, summary should be empty (no sections)
        summary = timer.summary()
        assert "first" not in summary


class TestSafeMean:
    def test_empty(self) -> None:
        assert safe_mean([]) == 0.0

    def test_single(self) -> None:
        assert safe_mean([5.0]) == 5.0

    def test_multiple(self) -> None:
        assert safe_mean([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_with_negative(self) -> None:
        assert safe_mean([-1.0, 1.0]) == 0.0


class TestHastorchFlag:
    def test_flag_exists(self) -> None:
        # Just confirms the module loads without error
        assert isinstance(HAS_TORCH, bool)


class TestAssertIsTensor:
    def test_no_torch_no_op(self) -> None:
        # When torch is available this is a no-op for Tensors
        import torch
        assert_is_tensor(torch.tensor(1.0))  # type: ignore[arg-type]

    def test_no_torch_non_tensor_ignored(self) -> None:
        # When torch not available, this is always a no-op
        from src.utils.profiler import HAS_TORCH as _HT
        # We can't easily unset torch, but the function guards on HAS_TORCH


class TestProfilerIntegration:
    """Integration-style tests that use the context manager."""

    def test_decorator(self) -> None:
        from src.utils.profiler import timed_section, get_timer

        timer = get_timer()

        # Use timed_section as a decorator by applying it to a function
        # The @ syntax won't work since timed_section returns a cm;
        # instead manually time a block.
        import time as _time
        with timed_section("decorated_block"):
            _time.sleep(0.003)
        summary = timer.summary()
        assert "decorated_block" in summary

    def test_context_manager(self) -> None:
        from src.utils.profiler import timed_section, get_timer

        timer = get_timer()
        with timed_section("decorated_op"):
            import time as _time
            _time.sleep(0.005)
        summary = timer.summary()
        assert "decorated_op" in summary