"""
Backtest Engine — Sector-tilt rebalancing simulation.

Uses sector ETF daily prices to simulate a strategy that overweights
top-scored sectors and underweights bottom-scored sectors. Includes
transaction costs, performance metrics, and bootstrap significance testing.
"""

import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

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
    rebalance_dates: list
    metrics: Dict[str, float] = field(default_factory=dict)


def _empty_result() -> BacktestResult:
    """Return an empty BacktestResult for insufficient data."""
    return BacktestResult(
        portfolio_returns=pd.Series(dtype=float),
        benchmark_returns=pd.Series(dtype=float),
        cumulative_portfolio=pd.Series(dtype=float),
        cumulative_benchmark=pd.Series(dtype=float),
        weights_history=pd.DataFrame(),
        rebalance_dates=[],
        metrics={},
    )


# ─── Feature scoring from ETF prices (for historical backtest) ───

def _compute_etf_features(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    lookback: int = MOMENTUM_LOOKBACK,
    volume: Optional[pd.DataFrame] = None,
    precomputed_returns: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute sector features from ETF price history STRICTLY before ``date``.
    Uses only data available at the decision point (no lookahead).

    Parameters
    ----------
    precomputed_returns : optional pre-computed daily returns (avoids
        calling pct_change() on expanding history each rebalance).
    """
    etf_cols = [c for c in prices.columns if c != BENCHMARK_ETF]

    prior_mask = prices.index < date
    if not prior_mask.any():
        return pd.DataFrame()

    hist = prices.loc[prior_mask, etf_cols].dropna(how='all')

    if len(hist) < lookback + 5:
        return pd.DataFrame()

    features = pd.DataFrame(index=etf_cols)

    # Short-term momentum
    features['median_momentum'] = hist.iloc[-1] / hist.iloc[-lookback] - 1

    # Longer-term momentum (for acceleration)
    if len(hist) >= MOMENTUM_LOOKBACK_LONG + 5:
        ret_long = hist.iloc[-1] / hist.iloc[-MOMENTUM_LOOKBACK_LONG] - 1
        raw_accel = features['median_momentum'] - ret_long
        gate = sigmoid_gate(features['median_momentum'])
        features['momentum_acceleration'] = raw_accel * gate
    else:
        features['momentum_acceleration'] = 0.0

    # Volatility + breadth from pre-computed returns (avoids re-calling pct_change)
    if precomputed_returns is not None:
        rets_prior = precomputed_returns.loc[precomputed_returns.index < date, etf_cols]
        daily_rets = rets_prior.iloc[-lookback:]
    else:
        daily_rets = hist.pct_change().iloc[-lookback:]
    features['avg_volatility'] = daily_rets.std()
    features['breadth'] = (daily_rets > 0).mean()

    # Liquidity: log ADTV (volume × price)
    if volume is not None:
        vol_prior = volume.loc[prior_mask].reindex(columns=etf_cols)
        if len(vol_prior) >= lookback:
            avg_vol = vol_prior.iloc[-lookback:].mean()
            avg_price = hist.iloc[-lookback:].mean()
            features['liquidity_score'] = np.log1p((avg_vol * avg_price).clip(lower=0))
        else:
            features['liquidity_score'] = np.nan
    else:
        features['liquidity_score'] = np.nan

    # Concentration: not computable from single ETF (requires constituents).
    # Explicitly NaN so compute_zscores fills with 0.0 — same as missing.
    features['concentration'] = np.nan

    return features


def _score_etfs(
    features: pd.DataFrame,
    weights: Dict[str, float],
    prev_scores: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Score ETFs using the shared composite scoring logic from scorer.py.
    This ensures the backtest uses identical scoring to the live dashboard.

    If ``prev_scores`` is provided, blends with EMA for temporal stability:
    score_t = α * raw_t + (1 - α) * score_{t-1}
    """
    scored = compute_zscores(features, feature_map=FEATURE_MAP)
    raw = compute_weighted_score(scored, weights, feature_map=FEATURE_MAP)

    if prev_scores is not None:
        common = raw.index.intersection(prev_scores.index)
        blended = raw.copy()
        blended[common] = (
            SCORE_EMA_ALPHA * raw[common]
            + (1 - SCORE_EMA_ALPHA) * prev_scores[common]
        )
        return blended
    return raw


def _precompute_spy_drawdown(prices: pd.DataFrame) -> Optional[pd.Series]:
    """Precompute rolling drawdown series for SPY — O(T) once, O(1) per lookup."""
    if BENCHMARK_ETF not in prices.columns:
        return None
    spy = prices[BENCHMARK_ETF].dropna()
    if len(spy) < REGIME_LOOKBACK_DAYS:
        return None
    rolling_max = spy.rolling(REGIME_LOOKBACK_DAYS, min_periods=REGIME_LOOKBACK_DAYS).max()
    return (spy - rolling_max) / rolling_max


def _compute_transaction_cost(
    target_weights: pd.Series,
    prev_weights: pd.Series,
    cost_rate: float,
) -> Tuple[float, float]:
    """Return (total_cost, turnover) with linear + sqrt market impact."""
    turnover = (target_weights - prev_weights).abs().sum()
    linear_cost = turnover * cost_rate
    impact_cost = np.sqrt(turnover) * MARKET_IMPACT_BPS / 10_000.0
    return linear_cost + impact_cost, turnover


# ─── Core Backtest ───────────────────────────────────────────────

def _build_run_metadata(
    weights: Dict[str, float],
    rebalance_freq: str,
    transaction_cost_bps: float,
    tilt_size: float,
    n_overweight: int,
    n_underweight: int,
) -> Dict[str, object]:
    """Capture a reproducibility fingerprint of all playbook parameters.

    Stored in BacktestResult.metrics so any historical run can be audited.
    """
    import hashlib
    import json
    params = {
        'weights': {k: round(v, 6) for k, v in sorted(weights.items())},
        'rebalance_freq': rebalance_freq,
        'transaction_cost_bps': transaction_cost_bps,
        'tilt_size': tilt_size,
        'n_overweight': n_overweight,
        'n_underweight': n_underweight,
        'max_tilt_per_rebalance': MAX_TILT_PER_REBALANCE,
        'momentum_lookback': MOMENTUM_LOOKBACK,
        'momentum_lookback_long': MOMENTUM_LOOKBACK_LONG,
        'warmup_days': WARMUP_DAYS,
        'score_ema_alpha': SCORE_EMA_ALPHA,
        'market_impact_bps': MARKET_IMPACT_BPS,
        'regime_drawdown_thresh': REGIME_DRAWDOWN_THRESH,
        'regime_lookback_days': REGIME_LOOKBACK_DAYS,
        'sector_etfs': sorted(SECTOR_ETFS),
        'benchmark_etf': BENCHMARK_ETF,
    }
    param_json = json.dumps(params, sort_keys=True)
    config_hash = hashlib.sha256(param_json.encode()).hexdigest()[:12]
    return {
        'run_params': params,
        'config_hash': config_hash,
    }


# ... (Keeping imports and helper functions unchanged) ...

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
    """
    Run a sector-tilt backtest simulation. (Vectorized)

    Strategy:
    - At each rebalance, score sectors from ETF price history
    - Overweight top-N sectors by +tilt_size each
    - Underweight bottom-N sectors by -tilt_size each
    - Equal-weight the rest
    - Apply transaction costs on weight changes
    """
    if weights is None:
        weights = WEIGHT_PRESETS[DEFAULT_PRESET]

    # Validate input
    if not isinstance(etf_prices.index, pd.DatetimeIndex):
        raise TypeError(
            f"etf_prices must have a DatetimeIndex, got {type(etf_prices.index).__name__}"
        )

    # Load volume data for ADTV-based liquidity scoring
    volume = etf_volume
    if volume is None:
        try:
            from src.data_engine import load_etf_volume
            volume = load_etf_volume()
        except ImportError:
            pass
    if volume is not None:
        logger.info("Using ETF volume data for ADTV-based liquidity scoring.")

    etf_cols = [c for c in etf_prices.columns if c != BENCHMARK_ETF]
    if not etf_cols:
        logger.warning("No ETF columns found (excluding benchmark).")
        return _empty_result()

    # Deduplicate input columns robustly
    etf_prices = etf_prices.loc[:, ~etf_prices.columns.duplicated()]
    
    # Deduplicate columns to prevent Series ambiguity if benchmark is passed twice
    cols_to_use = list(dict.fromkeys(etf_cols + [BENCHMARK_ETF]))
    # Filter only available columns
    cols_to_use = [c for c in cols_to_use if c in etf_prices.columns]
    
    prices = etf_prices[cols_to_use].dropna(how='all').copy()
    daily_returns = prices.pct_change().dropna(how='all')

    if len(daily_returns) < WARMUP_DAYS + MIN_BACKTEST_DAYS:
        logger.warning("Not enough data for backtest: %d rows.", len(daily_returns))
        return _empty_result()

    # ── Pre-compute once (avoid O(N) recomputation per rebalance) ──
    spy_drawdown = _precompute_spy_drawdown(prices)  # rolling drawdown Series
    etf_returns = daily_returns[etf_cols]  # view, not copy
    bench_col = daily_returns[BENCHMARK_ETF] if BENCHMARK_ETF in daily_returns.columns else None

    # Determine rebalance dates
    freq_code = 'MS' if rebalance_freq == "Monthly" else 'QS'
    rebal_dates = daily_returns.resample(freq_code).first().index.tolist()

    warmup_idx = min(WARMUP_DAYS, len(daily_returns) - 1)
    warmup_end = daily_returns.index[warmup_idx]
    valid_dates = [d for d in rebal_dates if d >= warmup_end]

    if len(valid_dates) < 2:
        logger.warning("Too few rebalance dates after warmup: %d", len(valid_dates))
        return _empty_result()

    n_sectors = len(etf_cols)
    equal_weight = 1.0 / n_sectors

    daily_records: List[Tuple[pd.Timestamp, float, float]] = []
    weight_log = []
    prev_weights = pd.Series(equal_weight, index=etf_cols)
    prev_scores: Optional[pd.Series] = None
    cost_rate = transaction_cost_bps / 10_000.0

    for i, date in enumerate(valid_dates):
        features = _compute_etf_features(
            prices, date, volume=volume, precomputed_returns=etf_returns,
        )

        # O(1) regime guard lookup from precomputed drawdown
        regime_risk_off = False
        if spy_drawdown is not None and date in spy_drawdown.index:
            # Use the last valid drawdown at or before the decision date
            dd_val = spy_drawdown.loc[:date].iloc[-1] if len(spy_drawdown.loc[:date]) > 0 else 0.0
            if not np.isnan(dd_val) and dd_val < REGIME_DRAWDOWN_THRESH:
                regime_risk_off = True
                logger.info("Risk-off at %s: SPY %dd drawdown %.1f%%",
                            date.date(), REGIME_LOOKBACK_DAYS, dd_val * 100)

        if features.empty or regime_risk_off:
            target_weights = pd.Series(equal_weight, index=etf_cols)
        else:
            scores = _score_etfs(features, weights, prev_scores=prev_scores)
            prev_scores = scores
            ranked = scores.sort_values(ascending=False)

            target_weights = pd.Series(equal_weight, index=etf_cols)
            top_sectors = ranked.index[:n_overweight]
            bottom_sectors = ranked.index[-n_underweight:]

            for s in top_sectors:
                if s in target_weights.index:
                    target_weights[s] = equal_weight + tilt_size
            for s in bottom_sectors:
                if s in target_weights.index:
                    target_weights[s] = max(0.0, equal_weight - tilt_size)

            target_weights = target_weights / target_weights.sum()

        # ── Enforce MAX_TILT_PER_REBALANCE ──
        weight_change = target_weights - prev_weights
        capped_change = weight_change.clip(
            lower=-MAX_TILT_PER_REBALANCE,
            upper=MAX_TILT_PER_REBALANCE,
        )
        target_weights = prev_weights + capped_change
        target_weights = target_weights.clip(lower=0.0)
        if target_weights.sum() > 0:
            target_weights = target_weights / target_weights.sum()

        cost, turnover = _compute_transaction_cost(
            target_weights, prev_weights, cost_rate
        )

        # ── Period boundaries ──
        period_end = valid_dates[i + 1] if i + 1 < len(valid_dates) else daily_returns.index[-1]
        period_rets = etf_returns.loc[date:period_end]

        if len(period_rets) > 0 and period_rets.index[0] == date:
            period_rets = period_rets.iloc[1:]
        if i + 1 < len(valid_dates) and len(period_rets) > 0:
            if period_rets.index[-1] == period_end:
                period_rets = period_rets.iloc[:-1]

        if len(period_rets) == 0:
            weight_log.append({'date': date, 'turnover': turnover,
                               **target_weights.to_dict()})
            prev_weights = target_weights
            continue

        # ── Vectorized daily returns calculation ──
        # Computes growth path for the whole period at once using matrix ops.
        
        rets_arr = period_rets.values  # (T, N)
        if rets_arr.shape[0] == 0:
            continue

        # Cumulative growth starting from 1.0
        cum_growth_arr = np.cumprod(1 + rets_arr, axis=0)
        
        # Shift growth to get Start-of-Day state (for weight calculation)
        # Prepend 1.0 row for the first day
        shift_growth = np.vstack([np.ones((1, n_sectors)), cum_growth_arr[:-1]])
        
        # Unnormalized weights: w0 * growth
        w0_arr = target_weights.values  # (N,)
        w_unnorm = w0_arr * shift_growth  # (T, N) via broadcasting

        # Normalize by row sums to get actual drift-adjusted weights
        row_sums = w_unnorm.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        w_norm = w_unnorm / row_sums

        # Portfolio returns: sum(w_{t-1} * r_t)
        port_rets_arr = (w_norm * rets_arr).sum(axis=1)

        # Apply transaction cost to first day of rebalance period
        if cost > 0:
            # R_net = (1 + R_gross) * (1 - cost) - 1
            port_rets_arr[0] = (1 + port_rets_arr[0]) * (1 - cost) - 1

        # Calculate End-of-Period weights for next rebal turnover
        # Proportional to initial weights * total period growth
        w_end_unnorm = w0_arr * cum_growth_arr[-1]
        w_end_sum = w_end_unnorm.sum()
        if w_end_sum > 0:
            w_end_norm = w_end_unnorm / w_end_sum
        else:
            w_end_norm = w0_arr  # Fallback

        # Benchmark returns
        days = period_rets.index
        if bench_col is not None:
            # Align benchmark to period days using reindexing/lookup
            # Fast lookup since we have datetime index
            # .reindex is efficient enough for blocks
            b_vals = bench_col.reindex(days).fillna(0.0).values
        else:
            b_vals = np.zeros(len(days))

        daily_records.extend(
            zip(days, port_rets_arr.tolist(), b_vals.tolist())
        )

        weight_log.append({'date': date, 'turnover': turnover,
                           **target_weights.to_dict()})
        prev_weights = pd.Series(w_end_norm, index=etf_cols)

    if not daily_records:
        return _empty_result()

    # Build return series — use DataFrame to handle duplicate dates properly
    records_df = pd.DataFrame(daily_records, columns=['date', 'port_ret', 'bench_ret'])
    # If duplicate dates exist (rebalance boundary overlap), keep last value
    records_df = records_df.drop_duplicates(subset='date', keep='last')
    records_df = records_df.set_index('date').sort_index()

    port_series = records_df['port_ret']
    bench_series = records_df['bench_ret']

    # Cumulative returns
    cum_port = (1 + port_series).cumprod()
    cum_bench = (1 + bench_series).cumprod()

    # Compute metrics (with turnover tracking)
    metrics = compute_metrics(port_series, bench_series)

    # Add turnover metrics from weight log
    if weight_log:
        turnovers = [w.get('turnover', 0) for w in weight_log]
        metrics['avg_turnover'] = round(np.mean(turnovers) * 100, 2)  # as %
        metrics['max_turnover'] = round(np.max(turnovers) * 100, 2)
        metrics['n_rebalances'] = len(weight_log)

    # Attach reproducibility fingerprint
    run_meta = _build_run_metadata(
        weights, rebalance_freq, transaction_cost_bps,
        tilt_size, n_overweight, n_underweight,
    )
    metrics['config_hash'] = run_meta['config_hash']
    metrics['run_params'] = run_meta['run_params']

    return BacktestResult(
        portfolio_returns=port_series,
        benchmark_returns=bench_series,
        cumulative_portfolio=cum_port,
        cumulative_benchmark=cum_bench,
        weights_history=pd.DataFrame(weight_log),
        rebalance_dates=valid_dates,
        metrics=metrics,
    )


# ─── Performance Metrics ─────────────────────────────────────────

def compute_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """Compute key backtest performance metrics.

    Annualization uses TRADING_DAYS_PER_YEAR (252) based on the number of
    actual return observations, which is more robust than calendar-day
    counting for data with gaps or sparse trading.
    """
    if len(portfolio_returns) < MIN_BACKTEST_DAYS:
        return {}

    n_obs = len(portfolio_returns)
    years = n_obs / TRADING_DAYS_PER_YEAR
    if years < 1e-6:
        return {}

    # Cumulative / annualized returns
    cum_ret = (1 + portfolio_returns).prod() - 1
    ann_return = (1 + cum_ret) ** (1 / years) - 1

    cum_ret_bench = (1 + benchmark_returns).prod() - 1
    ann_return_bench = (1 + cum_ret_bench) ** (1 / years) - 1

    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR

    # Sharpe Ratio (portfolio)
    excess = portfolio_returns - daily_rf
    sharpe = (
        excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        if excess.std() > 1e-10 else 0.0
    )

    # Sharpe Ratio (benchmark)
    excess_b = benchmark_returns - daily_rf
    sharpe_bench = (
        excess_b.mean() / excess_b.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        if excess_b.std() > 1e-10 else 0.0
    )

    # Max Drawdown
    cum = (1 + portfolio_returns).cumprod()
    drawdowns = cum / cum.cummax() - 1
    max_dd = drawdowns.min()

    # Information Ratio
    active_ret = portfolio_returns - benchmark_returns
    ir = (
        active_ret.mean() / active_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        if active_ret.std() > 1e-10 else 0.0
    )

    # Hit Rate (% of months strategy outperforms benchmark)
    monthly_port = portfolio_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    monthly_bench = benchmark_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    common = monthly_port.index.intersection(monthly_bench.index)
    hit_rate = (monthly_port[common] > monthly_bench[common]).mean() if len(common) > 0 else 0.5

    # Alpha (simple)
    alpha = ann_return - ann_return_bench

    return {
        "annualized_return":       round(ann_return * 100, 2),
        "annualized_return_bench": round(ann_return_bench * 100, 2),
        "sharpe_ratio":            round(sharpe, 3),
        "sharpe_ratio_bench":      round(sharpe_bench, 3),
        "max_drawdown":            round(max_dd * 100, 2),
        "information_ratio":       round(ir, 3),
        "hit_rate":                round(hit_rate * 100, 1),
        "alpha":                   round(alpha * 100, 2),
        "total_return":            round(cum_ret * 100, 2),
        "total_return_bench":      round(cum_ret_bench * 100, 2),
        "n_periods":               len(common),
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
    active_rets_arr = active_returns.values
    port_rets_arr = portfolio_returns.values

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
            round(np.mean(alpha_samples) * 100, 2),
            round(np.percentile(alpha_samples, 5) * 100, 2),
            round(np.percentile(alpha_samples, 95) * 100, 2),
        ),
        'sharpe': (
            round(np.mean(sharpe_samples) * 100, 2),
            round(np.percentile(sharpe_samples, 5) * 100, 2),
            round(np.percentile(sharpe_samples, 95) * 100, 2),
        ),
    }

# ... (Keeping sensitivity_analysis unchanged) ...


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
    for preset in param_grid.get('preset', ['Momentum-Heavy']):
        for freq in param_grid.get('rebalance_freq', ['Monthly']):
            preset_weights = WEIGHT_PRESETS.get(preset)
            if preset_weights is None:
                logger.warning("Unknown preset '%s', skipping.", preset)
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
