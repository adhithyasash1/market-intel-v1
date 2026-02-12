"""
Formatting utilities for the Market Intelligence Dashboard.
Consolidates presentation logic used by both Streamlit and API responses.
"""

import numbers
import pandas as pd
from datetime import datetime
from typing import Union, Any

def today_str() -> str:
    """Return today's date as YYYY-MM-DD string."""
    return datetime.now().strftime("%Y-%m-%d")

def format_pct(value: Any, decimals: int = 1) -> str:
    """
    Format a number as percentage string. Handles NaN and non-numeric input.
    Example: 0.123 -> "+12.3%"
    """
    if not isinstance(value, numbers.Number) or pd.isna(value):
        return "—"
    return f"{value:+.{decimals}f}%"

def format_large_number(value: Any) -> str:
    """
    Format large numbers with K/M/B/T suffixes. Handles NaN and non-numeric input.
    Example: 1500000 -> "$1.5M"
    """
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
