"""
Composite Scorer — Z-score normalization, weighted scoring, and decision labels.

Takes sector aggregates from features.py, normalizes to cross-sectional z-scores,
computes a weighted composite score, assigns Overweight/Neutral/Avoid signals,
and generates explainability text.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from config import WEIGHT_PRESETS, DEFAULT_PRESET, OVERWEIGHT_PERCENTILE, AVOID_PERCENTILE
from src.utils import safe_zscore


# ─── Feature → Z-Score Column Mapping ────────────────────────────
# Maps weight keys to sector aggregate column names and z-score polarity
# (positive = higher is better, negative = lower is better)
FEATURE_MAP = {
    "momentum":     ("median_momentum",        +1),   # higher momentum → better
    "breadth":      ("breadth",                +1),   # higher breadth → better
    "volatility":   ("avg_volatility",         -1),   # lower volatility → better
    "liquidity":    ("liquidity_score",         +1),   # higher liquidity → better
    "acceleration": ("momentum_acceleration",  +1),   # higher accel → better
}

# Human-readable labels for explainability
_FEATURE_LABELS = {
    "momentum":     "Median 1M momentum",
    "breadth":      "Breadth (% positive members)",
    "volatility":   "Volatility (lower is better)",
    "liquidity":    "Liquidity score",
    "acceleration": "Momentum acceleration",
}


def compute_zscores(
    df: pd.DataFrame,
    feature_map: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Compute polarity-adjusted z-scores for each feature in the feature map.

    This is the shared scoring kernel used by both the live scorer
    and the backtest engine, ensuring consistent scoring logic.

    Parameters
    ----------
    df : DataFrame with raw feature columns
    feature_map : dict mapping weight_key → (col_name, polarity).
                  Defaults to FEATURE_MAP.

    Returns
    -------
    DataFrame with z_{key} columns appended.
    """
    if feature_map is None:
        feature_map = FEATURE_MAP

    result = df.copy()
    for weight_key, (col_name, polarity) in feature_map.items():
        z_col = f"z_{weight_key}"
        if col_name in result.columns:
            result[z_col] = safe_zscore(result[col_name]) * polarity
        else:
            result[z_col] = 0.0
    return result


def compute_weighted_score(
    df: pd.DataFrame,
    weights: Dict[str, float],
    feature_map: Optional[Dict] = None,
) -> pd.Series:
    """
    Compute the weighted composite score from z-score columns.

    Parameters
    ----------
    df : DataFrame that already has z_{key} columns (from compute_zscores)
    weights : dict of weight_key → weight value
    feature_map : dict for key enumeration. Defaults to FEATURE_MAP.

    Returns
    -------
    Series of composite scores, same index as df.
    """
    if feature_map is None:
        feature_map = FEATURE_MAP

    score = pd.Series(0.0, index=df.index)
    for weight_key in feature_map:
        z_col = f"z_{weight_key}"
        w = weights.get(weight_key, 0.0)
        if z_col in df.columns:
            score += w * df[z_col]
    return score


def compute_composite_scores(
    sector_aggs: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    preset: str = DEFAULT_PRESET,
) -> pd.DataFrame:
    """
    Compute composite sector scores from sector aggregates.

    1. Z-score normalizes each feature across sectors.
    2. Applies configurable weights (with polarity).
    3. Returns scored DataFrame with z-score columns + composite_score.
    """
    if weights is None:
        weights = WEIGHT_PRESETS.get(preset, WEIGHT_PRESETS[DEFAULT_PRESET])

    df = compute_zscores(sector_aggs)
    df['composite_score'] = compute_weighted_score(df, weights)

    # Rank (higher is better)
    df['score_rank'] = df['composite_score'].rank(ascending=False, method='min').astype(int)

    return df.sort_values('composite_score', ascending=False)


def assign_signals(
    scored_df: pd.DataFrame,
    overweight_pct: float = OVERWEIGHT_PERCENTILE,
    avoid_pct: float = AVOID_PERCENTILE,
) -> pd.DataFrame:
    """
    Assign Overweight / Neutral / Avoid signals based on score percentiles.

    Uses both overweight_pct (top) and avoid_pct (bottom) thresholds,
    not just overweight_pct for both as was the previous incorrect behavior.
    """
    df = scored_df.copy()

    high_threshold = df['composite_score'].quantile(overweight_pct)
    low_threshold  = df['composite_score'].quantile(avoid_pct)

    def _label(score: float) -> str:
        if score >= high_threshold:
            return "Overweight"
        if score <= low_threshold:
            return "Avoid"
        return "Neutral"

    df['signal'] = df['composite_score'].apply(_label)
    return df


def explain_signal(
    sector_row: pd.Series,
    weights: Optional[Dict[str, float]] = None,
) -> List[str]:
    """
    Generate top-3 drivers explaining why a sector got its signal.
    Returns a list of human-readable explanation strings.
    """
    if weights is None:
        weights = WEIGHT_PRESETS[DEFAULT_PRESET]

    contributions = []
    for key in FEATURE_MAP:
        z_col = f"z_{key}"
        if z_col not in sector_row.index:
            continue

        z_val = sector_row[z_col]
        w = weights.get(key, 0.0)
        contrib = w * z_val

        if z_val > 0.5:
            direction = "↑ Strong"
        elif z_val < -0.5:
            direction = "↓ Weak"
        else:
            direction = "→ Average"

        label = _FEATURE_LABELS.get(key, key)
        contributions.append((contrib, label, z_val, direction))

    # Sort by absolute contribution (descending)
    contributions.sort(key=lambda x: abs(x[0]), reverse=True)

    explanations = []
    for contrib, label, z_val, direction in contributions[:3]:
        sign = "+" if contrib > 0 else ""
        explanations.append(
            f"{direction} {label} (z={z_val:.2f}, contribution={sign}{contrib:.3f})"
        )

    return explanations


def score_pipeline(
    sector_aggs: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    preset: str = DEFAULT_PRESET,
) -> pd.DataFrame:
    """Full pipeline: score → assign signals → add explanations."""
    scored = compute_composite_scores(sector_aggs, weights=weights, preset=preset)
    labeled = assign_signals(scored)

    w = weights if weights else WEIGHT_PRESETS.get(preset, WEIGHT_PRESETS[DEFAULT_PRESET])

    # Add explanation column
    explanations = {}
    for sector in labeled.index:
        explanations[sector] = explain_signal(labeled.loc[sector], weights=w)
    labeled['explanation'] = pd.Series(explanations)

    return labeled
