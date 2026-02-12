"""
Backtest Engine — Sector-tilt rebalancing simulation.
Optimized for performance with vectorized calculations.
"""

import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Any

from config import (
    SECTOR_ETFS, BENCHMARK_ETF, DEFAULT_TRANSACTION_COST,
    TILT_SIZE, MAX_TILT_PER_REBALANCE, DEFAULT_REBALANCE_FREQ,
    WEIGHT_PRESETS, DEFAULT_PRESET,
    WARMUP_DAYS, MOMENTUM_LOOKBACK, MOMENTUM_LOOKBACK_LONG,
    TRADING_DAYS_PER_YEAR, MIN_BACKTEST_DAYS,
    FEATURE_MAP, SCORE_EMA_ALPHA, MARKET_IMPACT_BPS,
    REGIME_DRAWDOWN_THRESH, REGIME_LOOKBACK_DAYS,
    N_OVERWEIGHT, N_UNDERWEIGHT,
)
from src.scorer import compute_zscores, compute_weighted_score
from src.utils import sigmoid_gate

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Container for backtest outputs."""
    portfolio_returns: pd.Series
    benchmark_returns: pd.Series
    cumulative_portfolio: pd.Series
    cumulative_benchmark: pd.Series
    weights_history: pd.DataFrame
    rebalance_dates: List[pd.Timestamp]
    metrics: Dict[str, float] = field(default_factory=dict)

def _empty_result() -> BacktestResult:
    return BacktestResult(
        portfolio_returns=pd.Series(dtype=float),
        benchmark_returns=pd.Series(dtype=float),
        cumulative_portfolio=pd.Series(dtype=float),
        cumulative_benchmark=pd.Series(dtype=float),
        weights_history=pd.DataFrame(),
        rebalance_dates=[],
        metrics={},
    )

def _compute_etf_features(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    lookback: int = MOMENTUM_LOOKBACK,
    volume: Optional[pd.DataFrame] = None,
    precomputed_returns: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute sector features from ETF price history.
    """
    etf_cols = [c for c in prices.columns if c != BENCHMARK_ETF]
    prior_mask = prices.index < date
    
    # Fast check
    if not prior_mask.any():
        return pd.DataFrame() # empty

    # Slice efficiently: only take necessary rows (e.g. last N + buffer)
    # This avoids slicing massive historical arrays if we only need the tail
    # Max lookback needed: MOMENTUM_LOOKBACK_LONG
    max_history = MOMENTUM_LOOKBACK_LONG + 5
    hist_slice = prices.loc[prior_mask].iloc[-max_history:]
    
    if len(hist_slice) < lookback + 1: # +1 because of shift for returns
        return pd.DataFrame()
        
    hist = hist_slice[etf_cols]
    features = pd.DataFrame(index=etf_cols)

    # Momentum
    current_px = hist.iloc[-1]
    lookback_px = hist.iloc[-lookback] if len(hist) >= lookback else hist.iloc[0]
    features['median_momentum'] = (current_px / lookback_px) - 1.0

    # Acceleration
    if len(hist) >= MOMENTUM_LOOKBACK_LONG:
        long_px = hist.iloc[-MOMENTUM_LOOKBACK_LONG]
        ret_long = (current_px / long_px) - 1.0
        raw_accel = features['median_momentum'] - ret_long
        gate = sigmoid_gate(features['median_momentum'])
        features['momentum_acceleration'] = raw_accel * gate
    else:
        features['momentum_acceleration'] = 0.0

    # Volatility & Breadth
    # Use precomputed returns if available to save time
    if precomputed_returns is not None:
        idx_loc = precomputed_returns.index.get_indexer([date], method='pad')[0]
        # slice [idx_loc - lookback : idx_loc]
        # Careful with indexing; date is strictly AFTER history
        # If date is in returns index, we take prior
        # Just use robust loc
        rets_slice = precomputed_returns.loc[precomputed_returns.index < date, etf_cols].iloc[-lookback:]
        daily_rets = rets_slice
    else:
        daily_rets = hist.pct_change().iloc[-lookback:]
        
    features['avg_volatility'] = daily_rets.std()
    features['breadth'] = (daily_rets > 0).mean()

    # Liquidity
    if volume is not None and len(volume) > 0:
        vol_slice = volume.loc[volume.index < date, etf_cols].iloc[-lookback:]
        if len(vol_slice) >= lookback:
            # We need prices aligned with volume for ADTV
            # Re-slice prices to match volume index if strictness needed
            # But roughly: avg vol * avg price
            avg_vol = vol_slice.mean()
            avg_px = hist.iloc[-lookback:].mean()
            features['liquidity_score'] = np.log1p((avg_vol * avg_px).clip(lower=0))
        else:
            features['liquidity_score'] = np.nan
    else:
        features['liquidity_score'] = np.nan

    features['concentration'] = np.nan
    return features

def _score_etfs(
    features: pd.DataFrame,
    weights: Dict[str, float],
    prev_scores: Optional[pd.Series] = None,
) -> pd.Series:
    """Score ETFs using shared logic."""
    scored = compute_zscores(features, feature_map=FEATURE_MAP)
    raw = compute_weighted_score(scored, weights, feature_map=FEATURE_MAP)

    if prev_scores is not None:
        common = raw.index.intersection(prev_scores.index)
        blended = raw.copy()
        # EMA blend
        blended[common] = (
            SCORE_EMA_ALPHA * raw[common]
            + (1 - SCORE_EMA_ALPHA) * prev_scores[common]
        )
        return blended
    return raw

def _precompute_spy_drawdown(prices: pd.DataFrame) -> Optional[pd.Series]:
    """Precompute rolling drawdown."""
    if BENCHMARK_ETF not in prices.columns:
        return None
    spy = prices[BENCHMARK_ETF].dropna()
    if len(spy) < REGIME_LOOKBACK_DAYS:
        return None
    
    # Vectorized rolling max
    rolling_max = spy.rolling(REGIME_LOOKBACK_DAYS, min_periods=REGIME_LOOKBACK_DAYS).max()
    return (spy - rolling_max) / rolling_max

def _compute_transaction_cost(
    target: pd.Series,
    prev: pd.Series,
    cost_rate: float,
) -> Tuple[float, float]:
    """Total cost and turnover."""
    turnover = (target - prev).abs().sum()
    linear = turnover * cost_rate
    impact = np.sqrt(turnover) * MARKET_IMPACT_BPS / 10000.0
    return linear + impact, turnover

# ─── Core Runner ───

def run_backtest(
    etf_prices: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    rebalance_freq: str = DEFAULT_REBALANCE_FREQ,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST,
    n_overweight: int = N_OVERWEIGHT,
    n_underweight: int = N_UNDERWEIGHT,
    tilt_size: float = TILT_SIZE,
    etf_volume: Optional[pd.DataFrame] = None,
) -> BacktestResult:
    
    if weights is None:
        weights = WEIGHT_PRESETS[DEFAULT_PRESET]

    # Data Validation
    if not isinstance(etf_prices.index, pd.DatetimeIndex):
        raise TypeError("etf_prices must have DatetimeIndex")
        
    volume = etf_volume
    if volume is None:
        # Optional local load
        try:
            from src.data_engine import load_etf_volume
            volume = load_etf_volume()
        except ImportError: 
            pass

    etf_cols = [c for c in etf_prices.columns if c != BENCHMARK_ETF]
    if not etf_cols:
        return _empty_result()

    # Prep Data
    # Dedupe columns
    etf_prices = etf_prices.loc[:, ~etf_prices.columns.duplicated()]
    cols = list(set(etf_cols + [BENCHMARK_ETF]))
    prices = etf_prices[[c for c in cols if c in etf_prices.columns]].dropna(how='all').copy()
    daily_returns = prices.pct_change().dropna(how='all')

    if len(daily_returns) < WARMUP_DAYS + MIN_BACKTEST_DAYS:
        return _empty_result()

    spy_drawdown = _precompute_spy_drawdown(prices)
    etf_returns = daily_returns[etf_cols]
    bench_col = daily_returns[BENCHMARK_ETF] if BENCHMARK_ETF in daily_returns.columns else None

    # Rebalance Schedule
    freq_code = 'MS' if rebalance_freq == "Monthly" else 'QS'
    rebal_dates = daily_returns.resample(freq_code).first().index
    warmup_end = daily_returns.index[min(WARMUP_DAYS, len(daily_returns)-1)]
    valid_dates = [d for d in rebal_dates if d >= warmup_end]

    if len(valid_dates) < 2:
        return _empty_result()

    # Init State
    n_sectors = len(etf_cols)
    equal_weight = 1.0 / n_sectors
    prev_weights = pd.Series(equal_weight, index=etf_cols)
    prev_scores = None
    cost_rate = transaction_cost_bps / 10000.0
    
    daily_records_list = []
    weight_log = []

    # Simulation Loop
    for i, date in enumerate(valid_dates):
        # 1. Feature Compute & Scoring
        features = _compute_etf_features(
            prices, date, 
            volume=volume, 
            precomputed_returns=etf_returns
        )
        
        # 2. Risk Check
        risk_off = False
        if spy_drawdown is not None and date in spy_drawdown.index:
            # Look up drawdown prior to rebalance? 
            # safe assumption: check drawdown on decision date
            dd_val = spy_drawdown.asof(date)
            if not np.isnan(dd_val) and dd_val < REGIME_DRAWDOWN_THRESH:
                risk_off = True

        # 3. Determine Targets
        if features.empty or risk_off:
            target = pd.Series(equal_weight, index=etf_cols)
        else:
            scores = _score_etfs(features, weights, prev_scores)
            prev_scores = scores
            ranked = scores.sort_values(ascending=False)
            
            target = pd.Series(equal_weight, index=etf_cols)
            top = ranked.index[:n_overweight]
            bot = ranked.index[-n_underweight:]
            
            # Vectorized assignment where efficient, but loop is fine for N=11
            for s in top: 
                if s in target: target[s] += tilt_size
            for s in bot:
                if s in target: target[s] = max(0.0, target[s] - tilt_size)
            
            # Normalize
            s = target.sum() 
            if s > 0: target /= s

        # 4. Tilt Constraints
        delta = (target - prev_weights).clip(-MAX_TILT_PER_REBALANCE, MAX_TILT_PER_REBALANCE)
        target = prev_weights + delta
        target = target.clip(lower=0.0)
        s = target.sum()
        if s > 0: target /= s

        # 5. Cost & Period Returns
        cost, turnover = _compute_transaction_cost(target, prev_weights, cost_rate)
        
        # Period slicing
        start_idx = daily_returns.index.get_indexer([date], method='bfill')[0]
        if i + 1 < len(valid_dates):
            end_date = valid_dates[i+1]
            end_idx = daily_returns.index.get_indexer([end_date], method='bfill')[0]
            period_slice = etf_returns.iloc[start_idx:end_idx]
            bench_slice = bench_col.iloc[start_idx:end_idx] if bench_col is not None else None
        else:
            period_slice = etf_returns.iloc[start_idx:]
            bench_slice = bench_col.iloc[start_idx:] if bench_col is not None else None
            
        if period_slice.empty:
            continue
            
        # Vectorized Period Growth
        rets_arr = period_slice.values # (T, N)
        if rets_arr.size == 0: continue
            
        # Cumprod returns -> Growth factors
        # 1+r
        cum_growth = np.cumprod(1 + rets_arr, axis=0)
        
        # Shifted growth for opening weights: row 0 is 1.0
        shift_growth = np.vstack([np.ones((1, n_sectors)), cum_growth[:-1]])
        
        # Drifted weights: w_start * growth
        w_drift = target.values * shift_growth
        # Normalize rows
        row_sums = w_drift.sum(axis=1, keepdims=True)
        row_sums[row_sums==0] = 1.0
        w_active = w_drift / row_sums
        
        # Portfolio return = sum(w * r)
        port_period_ret = (w_active * rets_arr).sum(axis=1)
        
        # Apply cost to first day
        if cost > 0:
            port_period_ret[0] = (1 + port_period_ret[0]) * (1 - cost) - 1
            
        # Benchmark return
        b_vals = bench_slice.values if bench_slice is not None else np.zeros(len(period_slice))
        
        # Store
        # Re-align dates
        p_dates = period_slice.index
        daily_records_list.append(pd.DataFrame({
            'date': p_dates,
            'port_ret': port_period_ret,
            'bench_ret': b_vals
        }))

        weight_log.append({'date': date, 'turnover': turnover, **target.to_dict()})
        
        # Next period starts with drifted weights
        # End weights = start * total_growth / sum
        end_unnorm = target.values * cum_growth[-1]
        es = end_unnorm.sum()
        w_end = end_unnorm / es if es > 0 else target.values
        prev_weights = pd.Series(w_end, index=etf_cols)

    # Compile Results
    if not daily_records_list:
        return _empty_result()
        
    records_df = pd.concat(daily_records_list).drop_duplicates('date', keep='last').set_index('date').sort_index()
    
    port_s = records_df['port_ret']
    bench_s = records_df['bench_ret']
    
    metrics = compute_metrics(port_s, bench_s)
    
    # Add turnover metrics
    if weight_log:
        ts = [x['turnover'] for x in weight_log]
        metrics['avg_turnover'] = float(round(np.mean(ts) * 100, 2))
        metrics['max_turnover'] = float(round(np.max(ts) * 100, 2))
        metrics['n_rebalances'] = len(weight_log)

    # Fingerprint
    run_meta = _build_run_metadata(
        weights, rebalance_freq, transaction_cost_bps, tilt_size, n_overweight, n_underweight
    )
    metrics['config_hash'] = run_meta['config_hash']
    metrics['run_params'] = run_meta['run_params']

    return BacktestResult(
        portfolio_returns=port_s,
        benchmark_returns=bench_s,
        cumulative_portfolio=(1 + port_s).cumprod(),
        cumulative_benchmark=(1 + bench_s).cumprod(),
        weights_history=pd.DataFrame(weight_log),
        rebalance_dates=valid_dates,
        metrics=metrics
    )

def compute_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """Compute performance metrics (vectorized)."""
    if len(portfolio_returns) < MIN_BACKTEST_DAYS:
        return {}
        
    n = len(portfolio_returns)
    years = n / TRADING_DAYS_PER_YEAR
    
    # Cumulative
    cum_p = (1 + portfolio_returns).prod() - 1
    cum_b = (1 + benchmark_returns).prod() - 1
    
    ann_p = (1 + cum_p) ** (1/years) - 1 if years > 0 else 0
    ann_b = (1 + cum_b) ** (1/years) - 1 if years > 0 else 0
    
    active = portfolio_returns - benchmark_returns
    alpha = ann_p - ann_b
    
    # Sharpe
    std_p = portfolio_returns.std()
    sharpe = (portfolio_returns.mean() / std_p * np.sqrt(TRADING_DAYS_PER_YEAR)) if std_p > 1e-9 else 0.0
    
    std_b = benchmark_returns.std()
    sharpe_b = (benchmark_returns.mean() / std_b * np.sqrt(TRADING_DAYS_PER_YEAR)) if std_b > 1e-9 else 0.0
    
    # Drawdown
    cp = (1 + portfolio_returns).cumprod()
    dd = (cp / cp.cummax()) - 1
    max_dd = dd.min()
    
    # Info Ratio
    std_act = active.std()
    ir = (active.mean() / std_act * np.sqrt(TRADING_DAYS_PER_YEAR)) if std_act > 1e-9 else 0.0
    
    # Hit Rate (Monthly)
    m_p = portfolio_returns.resample('ME').apply(lambda x: (1+x).prod()-1)
    m_b = benchmark_returns.resample('ME').apply(lambda x: (1+x).prod()-1)
    common = m_p.index.intersection(m_b.index)
    if len(common) > 0:
        hit_rate = (m_p[common] > m_b[common]).mean()
    else:
        hit_rate = 0.0

    return {
        "annualized_return": round(ann_p * 100, 2),
        "annualized_return_bench": round(ann_b * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sharpe_ratio_bench": round(sharpe_b, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "information_ratio": round(ir, 3),
        "hit_rate": round(hit_rate * 100, 1),
        "alpha": round(alpha * 100, 2),
        "total_return": round(cum_p * 100, 2),
        "total_return_bench": round(cum_b * 100, 2),
        "n_periods": len(common),
    }

def _build_run_metadata(weights, freq, cost, tilt, n_over, n_under):
    import hashlib, json
    params = {
        'weights': {k: round(v, 6) for k, v in sorted(weights.items())},
        'rebalance_freq': freq,
        'transaction_cost_bps': cost,
        'tilt_size': tilt,
        'momentum_lookback': MOMENTUM_LOOKBACK,
    }
    s = json.dumps(params, sort_keys=True)
    return {
        'run_params': params,
        'config_hash': hashlib.md5(s.encode(), usedforsecurity=False).hexdigest()[:8]
    }

# ─── Bootstrap Significance Testing ─────────────────────────────

def bootstrap_test(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    n_samples: int = 1000,
    block_size: int = 21,
    seed: Optional[int] = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Bootstrap test for significance of alpha. (Vectorized)
    Uses block bootstrap (21-day blocks) with NumPy broadcasting.

    Returns dict with: (mean, 5th percentile, 95th percentile) for each metric.
    """
    active_returns = portfolio_returns - benchmark_returns
    n = len(active_returns)

    if n < block_size * 2:
        return {}
    if n <= block_size:
        return {}

    rng = np.random.default_rng(seed)

    # 1. Generate start indices for all blocks across all samples at once
    # n_blocks needed per sample
    n_blocks = n // block_size + 1
    
    # Random start points: (n_samples, n_blocks)
    starts = rng.integers(0, n - block_size, size=(n_samples, n_blocks))

    # 2. Create offsets mask: (block_size,)
    offsets = np.arange(block_size)

    # 3. Broadcast to create (n_samples, n_blocks, block_size) indices
    # starts[..., None] -> (S, B, 1)
    # offsets[None, None, :] -> (1, 1, K)
    # indices -> (S, B, K)
    indices = starts[..., None] + offsets[None, None, :]

    # 4. Flatten the last two dims to get a stream of indices: (S, N_generated)
    indices = indices.reshape(n_samples, -1)

    # 5. Truncate to exact length n
    indices = indices[:, :n]

    # 6. Fancy indexing to fetch return values
    # active_rets_arr: (N,)
    active_rets_arr = active_returns.values.astype(np.float32)
    port_rets_arr = portfolio_returns.values.astype(np.float32)

    # sample_active: (S, N)
    sample_active = active_rets_arr[indices]
    sample_port = port_rets_arr[indices]

    # 7. Compute metrics across axis 1 (samples)
    
    # Alpha: Mean of active returns * 252
    alpha_samples = sample_active.mean(axis=1) * TRADING_DAYS_PER_YEAR

    # Sharpe: Mean / Std * sqrt(252)
    port_means = sample_port.mean(axis=1)
    port_stds = sample_port.std(axis=1)
    # Avoid div by zero
    port_stds[port_stds < 1e-10] = 1.0 # Will result in 0 sharpe if mean is 0, or just handle mask
    
    sharpe_samples = (port_means / port_stds) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe_samples[port_stds < 1e-9] = 0.0

    return {
        'alpha': (
            round(float(np.mean(alpha_samples) * 100), 2),
            round(float(np.percentile(alpha_samples, 5) * 100), 2),
            round(float(np.percentile(alpha_samples, 95) * 100), 2),
        ),
        'sharpe': (
            round(float(np.mean(sharpe_samples) * 100), 2),
            round(float(np.percentile(sharpe_samples, 5) * 100), 2),
            round(float(np.percentile(sharpe_samples, 95) * 100), 2),
        ),
    }


def sensitivity_analysis(
    etf_prices: pd.DataFrame,
    param_grid: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Run backtest across multiple parameter combinations.
    Returns a DataFrame with each config and resulting metrics.
    """
    if param_grid is None:
        param_grid = {
            'preset': list(WEIGHT_PRESETS.keys()),
            'rebalance_freq': ['Monthly', 'Quarterly'],
        }

    rows = []
    # Use standard loops as we are iterating high-level configs
    presets = param_grid.get('preset', ['Momentum-Heavy'])
    freqs = param_grid.get('rebalance_freq', ['Monthly'])
    
    for preset in presets:
        for freq in freqs:
            preset_weights = WEIGHT_PRESETS.get(preset)
            if preset_weights is None:
                logger.warning(f"Unknown preset '{preset}', skipping.")
                continue

            result = run_backtest(
                etf_prices,
                weights=preset_weights,
                rebalance_freq=freq,
                transaction_cost_bps=DEFAULT_TRANSACTION_COST,
            )
            if result.metrics:
                row = {'preset': preset, 'rebalance_freq': freq}
                row.update(result.metrics)
                rows.append(row)

    return pd.DataFrame(rows)
