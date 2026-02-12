"""
Tests for src/utils.py — safe_zscore, format_pct, format_large_number.
"""
import pandas as pd
import numpy as np
import pytest
from src.utils import safe_zscore, format_pct, format_large_number


class TestSafeZscore:
    """§5 — numerical stability of z-score."""

    def test_constant_series_returns_zeros(self):
        """When all values identical, std ≈ 0 → should return all zeros."""
        s = pd.Series([1.0, 1.0, 1.0, 1.0])
        z = safe_zscore(s)
        assert (z == 0.0).all()
        assert len(z) == len(s)

    def test_near_zero_std_returns_zeros(self):
        """Values that produce std < 1e-10 should also return zeros."""
        s = pd.Series([1.0, 1.0 + 1e-15, 1.0 - 1e-15])
        z = safe_zscore(s)
        assert (z == 0.0).all()

    def test_normal_series(self):
        """Normal data should produce z-scores with mean ≈ 0 and std ≈ 1."""
        s = pd.Series([1, 2, 3, 4, 5])
        z = safe_zscore(s)
        assert abs(z.mean()) < 1e-10
        assert abs(z.std() - 1.0) < 0.1  # ddof difference
        assert len(z) == 5

    def test_preserves_index(self):
        """Output index must match input index."""
        idx = pd.Index(['a', 'b', 'c'])
        s = pd.Series([10.0, 20.0, 30.0], index=idx)
        z = safe_zscore(s)
        assert list(z.index) == ['a', 'b', 'c']

    def test_single_element(self):
        """Single-element series: std is NaN → should return 0."""
        s = pd.Series([42.0])
        z = safe_zscore(s)
        assert z.iloc[0] == 0.0

    def test_handles_nan_values(self):
        """Series with NaN should not raise."""
        s = pd.Series([1.0, np.nan, 3.0, 4.0])
        z = safe_zscore(s)
        assert len(z) == 4


class TestFormatPct:
    """§5 — type guard on format_pct."""

    def test_positive_value(self):
        assert format_pct(5.123, 1) == "+5.1%"

    def test_negative_value(self):
        assert format_pct(-3.5, 1) == "-3.5%"

    def test_nan_returns_dash(self):
        assert format_pct(float('nan')) == "—"

    def test_none_returns_dash(self):
        assert format_pct(None) == "—"

    def test_string_returns_dash(self):
        assert format_pct("hello") == "—"

    def test_zero(self):
        assert format_pct(0.0) == "+0.0%"


class TestFormatLargeNumber:
    """§5 — type guard and suffix logic."""

    def test_trillions(self):
        assert format_large_number(2.5e12) == "$2.5T"

    def test_billions(self):
        assert format_large_number(1.2e9) == "$1.2B"

    def test_millions(self):
        assert format_large_number(4.8e6) == "$4.8M"

    def test_thousands(self):
        assert format_large_number(7.3e3) == "$7.3K"

    def test_small_number(self):
        assert format_large_number(42) == "$42"

    def test_negative(self):
        result = format_large_number(-5e9)
        assert result.startswith("-")
        assert "B" in result

    def test_nan_returns_dash(self):
        assert format_large_number(float('nan')) == "—"

    def test_none_returns_dash(self):
        assert format_large_number(None) == "—"

    def test_string_returns_dash(self):
        assert format_large_number("N/A") == "—"
