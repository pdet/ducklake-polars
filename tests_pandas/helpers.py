"""Shared test helpers for ducklake-pandas tests."""

from __future__ import annotations

import math


def assert_list_equal(actual, expected):
    """Compare two lists, treating NaN and None as equivalent."""
    assert len(actual) == len(expected), f"Length mismatch: {len(actual)} != {len(expected)}"
    for i, (a, e) in enumerate(zip(actual, expected)):
        if e is None:
            assert a is None or (isinstance(a, float) and math.isnan(a)), \
                f"Index {i}: expected None/NaN, got {a!r}"
        elif isinstance(e, float) and math.isnan(e):
            assert isinstance(a, float) and math.isnan(a), \
                f"Index {i}: expected NaN, got {a!r}"
        else:
            assert a == e, f"Index {i}: {a!r} != {e!r}"
