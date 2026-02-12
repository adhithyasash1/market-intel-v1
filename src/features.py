"""
Feature Engineering — Per-stock feature computation and sector aggregation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from config import MIN_SECTOR_STOCKS
from src.utils import sigmoid_gate

def _safe_column_diff(df: pd.DataFrame, col_a: str, col_b: str) -> pd.Series:
    """Compute df[col_a] - df[col_b], returning 0 if either column is missing."""
    if col_a in df.columns and col_b in df.columns:
        return df[col_a] - df[col_b]
    return pd.Series(0.0, index=df.index, dtype='float32')

def _vectorized_concentration(df: pd.DataFrame) -> pd.Series:
    """
    Compute sector concentration (top 3 share) using vectorized groupby operations.
    Replaces slow .apply().
    """
    if 'market_cap' not in df.columns or 'sector' not in df.columns:
        # Return default 0.5 keyed by sector
        if 'sector' in df.columns:
            return df.groupby('sector').size() * 0.0 + 0.5
        return pd.Series(dtype='float32')

    # Sort by sector + market_cap descending
    sorted_df = df.sort_values(['sector', 'market_cap'], ascending=[True, False])
    
    # Group sum of market cap
    sector_total_mc = sorted_df.groupby('sector')['market_cap'].transform('sum')
    
    # Identify top 3 per sector
    # Since we sorted by market cap desc, the first 3 in each group are top 3
    # We can use cumcount to identify ranks 0, 1, 2
    ranks = sorted_df.groupby('sector').cumcount()
    is_top3 = ranks < 3
    
    # Compute sum of top 3
    # Use mask to zero out non-top-3 caps, then sum by sector
    top3_caps = sorted_df['market_cap'].where(is_top3, 0.0)
    sector_top3_mc = top3_caps.groupby(sorted_df['sector']).sum()
    
    # Get unique totals (groupby.first() or similar since transform repeated it)
    sector_totals = sorted_df.groupby('sector')['market_cap'].sum()
    
    # Calculate ratio (handle div by zero)
    concentration = sector_top3_mc / sector_totals.replace(0, 1.0)
    
    return concentration

def compute_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a screener snapshot DataFrame with derived per-stock features.
    """
    out = df.copy()
    
    # ── Momentum acceleration ──
    raw_accel = _safe_column_diff(out, 'perf_1m', 'perf_3m')
    if 'perf_1m' in out.columns:
        gate = sigmoid_gate(out['perf_1m'])
        out['momentum_accel'] = raw_accel * gate
    else:
        out['momentum_accel'] = 0.0

    # ── MACD histogram ──
    out['macd_histogram'] = _safe_column_diff(out, 'macd_level', 'macd_signal')

    # ── SMA alignment ──
    # Int8/bool is sufficient, saves memory
    if all(c in out.columns for c in ['price', 'sma_50', 'sma_200']):
        out['above_sma50'] = (out['price'] > out['sma_50']).astype(int)
        out['above_sma200'] = (out['price'] > out['sma_200']).astype(int)
        out['golden_cross'] = (out['sma_50'] > out['sma_200']).astype(int)
    else:
        out['above_sma50'] = 0
        out['above_sma200'] = 0
        out['golden_cross'] = 0

    # ── Liquidity proxy ──
    if 'avg_vol_30d' in out.columns and 'price' in out.columns:
        raw_adtv = out['avg_vol_30d'] * out['price']
        out['adtv'] = np.log1p(raw_adtv.clip(lower=0))
    elif 'volume' in out.columns and 'price' in out.columns:
        raw_adtv = out['volume'] * out['price']
        out['adtv'] = np.log1p(raw_adtv.clip(lower=0))
    else:
        out['adtv'] = np.nan

    # ── RSI zone ──
    if 'rsi_14' in out.columns:
        rsi_clipped = out['rsi_14'].clip(0, 100)
        out['rsi_zone'] = pd.cut(
            rsi_clipped,
            bins=[0, 30, 45, 55, 70, 100],
            labels=['Oversold', 'Weak', 'Neutral', 'Strong', 'Overbought'],
            include_lowest=True,
        )
    else:
        out['rsi_zone'] = 'N/A'

    # ── Breadth input ──
    if 'perf_1m' in out.columns:
        out['positive_1m'] = (out['perf_1m'] > 0).astype(int)
    else:
        out['positive_1m'] = 0

    return out

def compute_sector_aggregates(stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-stock features to sector level.
    Optimized to minimize slow operations.
    """
    if 'sector' not in stock_df.columns:
        raise ValueError("DataFrame must have a 'sector' column")

    # Filter
    df = stock_df.dropna(subset=['sector']).copy()
    grouped = df.groupby('sector')

    agg = pd.DataFrame()
    agg['n_stocks'] = grouped.size()

    def _agg_median(col: str, default=0.0):
        return grouped[col].median() if col in df.columns else default

    def _agg_mean(col: str, default=0.5):
        return grouped[col].mean() if col in df.columns else default

    # Aggregates
    agg['median_momentum'] = _agg_median('perf_1m')
    agg['breadth'] = _agg_mean('positive_1m')
    
    # Volatility
    if 'volatility_d' in df.columns:
        agg['avg_volatility'] = grouped['volatility_d'].median()
    elif 'atr_14' in df.columns and 'price' in df.columns:
        df['_atr_pct'] = df['atr_14'] / df['price'] * 100
        agg['avg_volatility'] = df.groupby('sector')['_atr_pct'].median()
    else:
        agg['avg_volatility'] = np.nan

    agg['liquidity_score'] = _agg_median('adtv', np.nan)
    agg['avg_recommendation'] = _agg_mean('recommendation', np.nan)
    agg['momentum_acceleration'] = _agg_median('momentum_accel')
    
    # Concentration (Vectorized)
    agg['concentration'] = _vectorized_concentration(df)

    # Longer-term
    for col, name in [('perf_3m', 'median_perf_3m'), ('perf_ytd', 'median_perf_ytd'),
                      ('perf_1y', 'median_perf_1y')]:
        if col in df.columns:
            agg[name] = grouped[col].median()
            
    # Total Market Cap
    if 'market_cap' in df.columns:
        agg['total_market_cap'] = grouped['market_cap'].sum()

    # Filter small sectors
    agg = agg[agg['n_stocks'] >= MIN_SECTOR_STOCKS].copy()
    
    return agg.sort_values('median_momentum', ascending=False)
