"""
Tests for src/formatting.py — format_pct, format_large_number.
"""
import pytest
import math
from src.formatting import format_pct, format_large_number

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
