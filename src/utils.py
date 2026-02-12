"""
Utility helpers for caching, formatting, and data operations.
"""

import os
import numbers
import pandas as pd
from datetime import datetime


def today_str() -> str:
    """Return today's date as YYYY-MM-DD string."""
    return datetime.now().strftime("%Y-%m-%d")


def format_pct(value, decimals: int = 1) -> str:
    """Format a number as percentage string. Handles NaN and non-numeric input."""
    if not isinstance(value, numbers.Number) or pd.isna(value):
        return "—"
    return f"{value:+.{decimals}f}%"


def format_large_number(value) -> str:
    """Format large numbers with K/M/B/T suffixes. Handles NaN and non-numeric input."""
    if not isinstance(value, numbers.Number) or pd.isna(value):
        return "—"
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    if abs_val >= 1e12:
        return f"{sign}${abs_val / 1e12:.1f}T"
    if abs_val >= 1e9:
        return f"{sign}${abs_val / 1e9:.1f}B"
    if abs_val >= 1e6:
        return f"{sign}${abs_val / 1e6:.1f}M"
    if abs_val >= 1e3:
        return f"{sign}${abs_val / 1e3:.1f}K"
    return f"{sign}${abs_val:.0f}"


def safe_zscore(series: pd.Series) -> pd.Series:
    """Compute z-scores, handling near-zero std gracefully."""
    std = series.std()
    if std is None or pd.isna(std) or std < 1e-10:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist, return path."""
    os.makedirs(path, exist_ok=True)
    return path
