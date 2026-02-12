"""
Feature Engineering — Per-stock feature computation and sector aggregation.

Computes momentum, breadth, volatility, liquidity, and technical features
from screener snapshots, then aggregates to sector level.
"""

import pandas as pd
import numpy as np
from typing import Optional

from config import MIN_SECTOR_STOCKS


def _safe_column_diff(df: pd.DataFrame, col_a: str, col_b: str) -> pd.Series:
    """Compute df[col_a] - df[col_b], returning 0 if either column is missing."""
    if col_a in df.columns and col_b in df.columns:
        return df[col_a] - df[col_b]
    return pd.Series(0.0, index=df.index)


def compute_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a screener snapshot DataFrame with derived per-stock features.

    Input columns expected (from data_engine standardized names):
        price, volume, market_cap, sector, industry, country,
        perf_1w, perf_1m, perf_3m, perf_6m, perf_ytd, perf_1y,
        rsi_14, macd_level, macd_signal, sma_50, sma_200,
        atr_14, avg_vol_10d, avg_vol_30d, rel_volume,
        volatility_d, volatility_w, volatility_m, recommendation

    Returns the same DataFrame with additional feature columns.
    """
    out = df.copy()

    # ── Momentum acceleration: short-term vs medium-term momentum ──
    # Soft sigmoid gate prevents discontinuity where a tiny momentum change
    # flips acceleration from 0 to full value (hard gate instability).
    # sigmoid(-50 * x) ≈ 0 when x < -0.05, ≈ 1 when x > 0.05
    raw_accel = _safe_column_diff(out, 'perf_1m', 'perf_3m')
    if 'perf_1m' in out.columns:
        gate = 1.0 / (1.0 + np.exp(-50 * out['perf_1m']))
        out['momentum_accel'] = raw_accel * gate
    else:
        out['momentum_accel'] = 0.0

    # ── MACD histogram ──
    out['macd_histogram'] = _safe_column_diff(out, 'macd_level', 'macd_signal')
    if 'macd_level' not in out.columns or 'macd_signal' not in out.columns:
        out['macd_histogram'] = np.nan

    # ── SMA alignment: price vs moving averages ──
    if all(c in out.columns for c in ['price', 'sma_50', 'sma_200']):
        out['above_sma50']  = (out['price'] > out['sma_50']).astype(int)
        out['above_sma200'] = (out['price'] > out['sma_200']).astype(int)
        out['golden_cross'] = (out['sma_50'] > out['sma_200']).astype(int)
    else:
        out['above_sma50']  = np.nan
        out['above_sma200'] = np.nan
        out['golden_cross'] = np.nan

    # ── Liquidity proxy: log avg daily traded value ──
    # Log-transform tames the right-skew (ADTV ranges from $1M to $10B+)
    # and makes z-scores more meaningful across the distribution.
    if 'avg_vol_30d' in out.columns and 'price' in out.columns:
        raw_adtv = out['avg_vol_30d'] * out['price']
        out['adtv'] = np.log1p(raw_adtv.clip(lower=0))  # log(1+x), safe for 0
    elif 'volume' in out.columns and 'price' in out.columns:
        raw_adtv = out['volume'] * out['price']
        out['adtv'] = np.log1p(raw_adtv.clip(lower=0))
    else:
        out['adtv'] = np.nan

    # ── RSI zone ──
    if 'rsi_14' in out.columns:
        # Clip to [0, 100] so out-of-range values don't produce NaN bins
        rsi_clipped = out['rsi_14'].clip(0, 100)
        out['rsi_zone'] = pd.cut(
            rsi_clipped,
            bins=[0, 30, 45, 55, 70, 100],
            labels=['Oversold', 'Weak', 'Neutral', 'Strong', 'Overbought'],
            include_lowest=True,
        )
    else:
        out['rsi_zone'] = 'N/A'

    # ── Positive momentum flag (for breadth) ──
    if 'perf_1m' in out.columns:
        out['positive_1m'] = (out['perf_1m'] > 0).astype(int)
    else:
        out['positive_1m'] = np.nan

    return out


def _sector_concentration(grp: pd.DataFrame) -> float:
    """Compute share of market cap in the top-3 names of a sector group."""
    if 'market_cap' not in grp.columns:
        return 0.5
    total = grp['market_cap'].sum()
    if len(grp) < 3 or total == 0:
        return 1.0
    top3 = grp.nlargest(3, 'market_cap')['market_cap'].sum()
    return top3 / total


def compute_sector_aggregates(stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-stock features to sector level.

    Returns a DataFrame indexed by sector with columns:
        n_stocks, median_momentum, breadth, avg_volatility, concentration,
        liquidity_score, avg_recommendation, momentum_acceleration,
        median_rsi, pct_golden_cross, median_perf_3m, median_perf_ytd,
        total_market_cap
    """
    if 'sector' not in stock_df.columns:
        raise ValueError("DataFrame must have a 'sector' column")

    # Drop rows without sector
    df = stock_df.dropna(subset=['sector']).copy()
    grouped = df.groupby('sector')

    agg = pd.DataFrame()
    agg['n_stocks'] = grouped.size()

    # Helper to conditionally aggregate a column
    def _agg_col(col: str, func: str, default):
        if col in df.columns:
            return getattr(grouped[col], func)()
        return default

    # ── Core aggregates ──
    agg['median_momentum']        = _agg_col('perf_1m', 'median', 0.0)
    agg['breadth']                = _agg_col('positive_1m', 'mean', 0.5)
    # ── Volatility: use % vol, normalize ATR by price for unit consistency ──
    if 'volatility_d' in df.columns:
        agg['avg_volatility'] = grouped['volatility_d'].median()
    elif 'atr_14' in df.columns and 'price' in df.columns:
        # ATR is in price units; normalize to % to match volatility_d scale
        df = df.copy()
        df['_atr_pct'] = df['atr_14'] / df['price'] * 100
        agg['avg_volatility'] = df.groupby('sector')['_atr_pct'].median()
    else:
        agg['avg_volatility'] = np.nan  # NaN → z-score ignores, no bias

    agg['liquidity_score']        = _agg_col('adtv', 'median', np.nan)  # NaN, not 0.0
    agg['avg_recommendation']     = _agg_col('recommendation', 'mean', np.nan)
    agg['momentum_acceleration']  = _agg_col('momentum_accel', 'median', 0.0)
    agg['median_rsi']             = _agg_col('rsi_14', 'median', 50.0)  # diagnostic only (not in FEATURE_MAP)
    agg['pct_golden_cross']       = _agg_col('golden_cross', 'mean', 0.5)  # diagnostic only (not in FEATURE_MAP)

    # ── Concentration: top-3 market cap share ──
    if 'market_cap' in df.columns:
        agg['concentration'] = df.groupby('sector').apply(
            _sector_concentration, include_groups=False
        )
    else:
        agg['concentration'] = 0.5

    # ── Longer-term performance ──
    for col, name in [('perf_3m', 'median_perf_3m'), ('perf_ytd', 'median_perf_ytd'),
                      ('perf_1y', 'median_perf_1y')]:
        if col in df.columns:
            agg[name] = grouped[col].median()

    # ── Total market cap ──
    if 'market_cap' in df.columns:
        agg['total_market_cap'] = grouped['market_cap'].sum()

    # Filter out sectors with too few constituents
    agg = agg[agg['n_stocks'] >= MIN_SECTOR_STOCKS].copy()

    return agg.sort_values('median_momentum', ascending=False)
