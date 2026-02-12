"""
Mathematical utility helpers for the Market Intelligence Dashboard.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, Union

def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist, return path."""
    os.makedirs(path, exist_ok=True)
    return path

def sigmoid_gate(
    x: Union[float, np.ndarray, pd.Series], 
    steepness: Optional[float] = None
) -> Union[float, np.ndarray, pd.Series]:
    """
    Smooth 0→1 gate that ramps as x goes through zero.
    Used by momentum acceleration to avoid hard cutoffs.
    """
    if steepness is None:
        from config import SIGMOID_GATE_STEEPNESS
        steepness = SIGMOID_GATE_STEEPNESS
    
    # Use numpy exp for vectorization support
    return 1.0 / (1.0 + np.exp(-steepness * x))

def safe_zscore(series: pd.Series) -> pd.Series:
    """
    Compute z-scores, handling near-zero std gracefully.
    """
    std = series.std()
    if std is None or pd.isna(std) or std < 1e-10:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std

def winsorize(series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
    """
    Clip values to [lower, upper] percentiles to limit outlier influence.
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
    """
    Winsorize then z-score — robust to outliers in small cross-sections.
    """
    clipped = winsorize(series, winsor_lower, winsor_upper)
    return safe_zscore(clipped)
