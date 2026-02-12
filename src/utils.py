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
    """Compute z-scores, handling near-zero std gracefully.

    Note: For scoring with small cross-sections (N≤11 sectors), prefer
    ``robust_zscore`` which winsorizes before z-scoring to limit outlier
    influence.
    """
    std = series.std()
    if std is None or pd.isna(std) or std < 1e-10:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def winsorize(series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
    """Clip values to [lower, upper] percentiles to limit outlier influence.

    Parameters
    ----------
    series : pd.Series — raw feature values
    lower  : float — lower percentile (default 5th)
    upper  : float — upper percentile (default 95th)

    Returns
    -------
    pd.Series with extreme values clipped.
    """
    if series.dropna().empty or len(series.dropna()) < 3:
        return series
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lower=lo, upper=hi)


def robust_zscore(
    series: pd.Series,
    winsor_lower: float = 0.05,
    winsor_upper: float = 0.95,
) -> pd.Series:
    """Winsorize then z-score — robust to outliers in small cross-sections.

    With N=11 sectors, a single extreme value can dominate the z-score's
    mean and std. Winsorizing first caps the influence of any one sector
    to the clipped range, making rankings more stable.

    Parameters
    ----------
    series : pd.Series — raw feature values
    winsor_lower : float — lower winsorization percentile (default 5th)
    winsor_upper : float — upper winsorization percentile (default 95th)

    Returns
    -------
    pd.Series of z-scores computed on winsorized values.
    """
    clipped = winsorize(series, winsor_lower, winsor_upper)
    return safe_zscore(clipped)


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist, return path."""
    os.makedirs(path, exist_ok=True)
    return path
