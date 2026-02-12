"""
Composite Scorer — Z-score normalization, weighted scoring, and decision labels.

Takes sector aggregates from features.py, normalizes to cross-sectional z-scores,
computes a weighted composite score, assigns Overweight/Neutral/Avoid signals,
and generates explainability text.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from config import (
    WEIGHT_PRESETS, DEFAULT_PRESET, OVERWEIGHT_PERCENTILE, AVOID_PERCENTILE,
    FEATURE_MAP, FEATURE_LABELS,
)
from src.utils import safe_zscore, robust_zscore

# Alias for internal use (keeps downstream code unchanged)
_FEATURE_LABELS = FEATURE_LABELS


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
            # Winsorize then z-score: limits outlier dominance in small
            # cross-sections (N=11 sectors) where one extreme value can
            # shift mean/std dramatically, causing fragile rankings.
            result[z_col] = robust_zscore(result[col_name]) * polarity
            # Fill NaN z-scores with 0.0 so missing data contributes zero
            # to the composite score rather than propagating NaN
            result[z_col] = result[z_col].fillna(0.0)
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

    Uses vectorized np.select instead of row-wise apply for clarity.
    Mutates in-place (caller already passes a copy from compute_composite_scores).
    """
    scores = scored_df['composite_score']
    high_threshold = scores.quantile(overweight_pct)
    low_threshold = scores.quantile(avoid_pct)

    scored_df['signal'] = np.select(
        [scores >= high_threshold, scores <= low_threshold],
        ['Overweight', 'Avoid'],
        default='Neutral',
    )
    return scored_df


def contribution_breakdown(
    sector_row: pd.Series,
    weights: Optional[Dict[str, float]] = None,
    feature_map: Optional[Dict] = None,
) -> List[Dict]:
    """Return a full, auditable decomposition of the composite score.

    Each entry contains:
    - factor:       weight key name (e.g. "momentum")
    - label:        human-readable factor label
    - raw_value:    the raw feature value before z-scoring
    - z_score:      polarity-adjusted z-score  (what enters the weighted sum)
    - weight:       the weight applied to this factor
    - contribution: weight × z_score  (the additive piece of the composite)
    - pct_of_score: contribution / |composite|  (signed %, sums to ±100%)

    An investment committee can verify:
    sum(contribution) == composite_score ± rounding.
    """
    if weights is None:
        weights = WEIGHT_PRESETS[DEFAULT_PRESET]
    if feature_map is None:
        feature_map = FEATURE_MAP

    composite = 0.0
    entries = []

    for key, (col_name, polarity) in feature_map.items():
        z_col = f"z_{key}"
        w = weights.get(key, 0.0)

        raw_val = sector_row.get(col_name, np.nan)
        z_val = sector_row.get(z_col, 0.0)
        contrib = w * z_val
        composite += contrib

        entries.append({
            "factor": key,
            "label": _FEATURE_LABELS.get(key, key),
            "raw_value": raw_val,
            "z_score": round(z_val, 3),
            "weight": round(w, 3),
            "contribution": round(contrib, 4),
        })

    # Compute percentage attribution
    abs_composite = abs(composite) if abs(composite) > 1e-10 else 1.0
    for e in entries:
        e["pct_of_score"] = round(e["contribution"] / abs_composite * 100, 1)

    # Sort by absolute contribution (largest driver first)
    entries.sort(key=lambda e: abs(e["contribution"]), reverse=True)
    return entries


# ─── Human-readable unit formatters for raw values ───────────────

_RAW_FORMATTERS = {
    "momentum":      lambda v: f"{v:+.1f}%" if not np.isnan(v) else "n/a",
    "breadth":       lambda v: f"{v:.0%}" if not np.isnan(v) else "n/a",
    "volatility":    lambda v: f"{v:.1f}%" if not np.isnan(v) else "n/a",
    "liquidity":     lambda v: f"log₁ₚ={v:.1f}" if not np.isnan(v) else "n/a",
    "acceleration":  lambda v: f"{v:+.2f}" if not np.isnan(v) else "n/a",
    "concentration": lambda v: f"{v:.0%}" if not np.isnan(v) else "n/a",
}


def explain_signal(
    sector_row: pd.Series,
    weights: Optional[Dict[str, float]] = None,
) -> List[str]:
    """Generate auditable explanation strings for a sector's score.

    Improvements over the naive version:
    - Direction arrows (↑/↓) based on **contribution** sign, not z-score
      sign alone — correctly reflects factors with negative polarity.
    - Shows w×z contribution AND percentage of total composite score
      so a committee can verify the math.
    - Shows ALL factors (not just top 3) with a final verification line.
    - Includes the raw metric value so analysts can cross-reference
      against source data.
    """
    breakdown = contribution_breakdown(sector_row, weights)

    if not breakdown:
        return ["No factors available for explanation."]

    composite = sum(e["contribution"] for e in breakdown)
    explanations = []

    for e in breakdown:
        # Direction based on contribution sign (not z-score sign)
        if e["contribution"] > 0.01:
            arrow = "↑"
        elif e["contribution"] < -0.01:
            arrow = "↓"
        else:
            arrow = "→"

        raw_fmt = _RAW_FORMATTERS.get(e["factor"], lambda v: f"{v:.2f}")
        try:
            raw_str = raw_fmt(e["raw_value"])
        except (TypeError, ValueError):
            raw_str = "n/a"

        pct_str = f"{e['pct_of_score']:+.0f}%" if abs(e["pct_of_score"]) >= 0.5 else "~0%"

        explanations.append(
            f"{arrow} {e['label']}: {raw_str}  "
            f"(z={e['z_score']:+.2f}, w={e['weight']:.2f}, "
            f"w×z={e['contribution']:+.4f}, {pct_str} of score)"
        )

    # Verification line — committee can confirm sum matches composite
    explanations.append(
        f"── Composite: {composite:+.4f}  "
        f"(Σ contributions: {sum(e['contribution'] for e in breakdown):+.4f})"
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
