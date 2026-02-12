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
    TILT_SIZE, DEFAULT_REBALANCE_FREQ, WEIGHT_PRESETS, DEFAULT_PRESET,
    WARMUP_DAYS, MOMENTUM_LOOKBACK, MOMENTUM_LOOKBACK_LONG,
    TRADING_DAYS_PER_YEAR, MIN_BACKTEST_DAYS,
)
from src.scorer import compute_zscores, compute_weighted_score, FEATURE_MAP
from src.utils import safe_zscore

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
) -> pd.DataFrame:
    """
    Compute sector features from ETF price history STRICTLY before `date`.
    Uses only data available at the decision point (no lookahead).

    The key change from the original: we use prices.loc[:date - 1 day]
    to prevent using the rebalance day's price for scoring while also
    using it for return computation.
    """
    etf_cols = [c for c in prices.columns if c != BENCHMARK_ETF]

    # Exclude the rebalance day itself to avoid lookahead bias:
    # scoring should only see data available BEFORE the decision point.
    prior_dates = prices.index[prices.index < date]
    if len(prior_dates) == 0:
        return pd.DataFrame()

    hist = prices.loc[prior_dates, etf_cols].dropna(how='all')

    if len(hist) < lookback + 5:
        return pd.DataFrame()

    features = pd.DataFrame(index=etf_cols)

    # Short-term momentum
    features['median_momentum'] = hist.iloc[-1] / hist.iloc[-lookback] - 1

    # Longer-term momentum (for acceleration)
    if len(hist) >= MOMENTUM_LOOKBACK_LONG + 5:
        ret_long = hist.iloc[-1] / hist.iloc[-MOMENTUM_LOOKBACK_LONG] - 1
        features['momentum_acceleration'] = features['median_momentum'] - ret_long
    else:
        features['momentum_acceleration'] = 0.0

    # Volatility (rolling window)
    daily_rets = hist.pct_change().iloc[-lookback:]
    features['avg_volatility'] = daily_rets.std()

    # Breadth proxy: fraction of days positive in lookback window
    features['breadth'] = (daily_rets > 0).mean()

    # Liquidity proxy: average price level as proxy
    features['liquidity_score'] = hist.iloc[-lookback:].mean()

    return features


def _score_etfs(features: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """
    Score ETFs using the shared composite scoring logic from scorer.py.
    This ensures the backtest uses identical scoring to the live dashboard.
    """
    scored = compute_zscores(features, feature_map=FEATURE_MAP)
    return compute_weighted_score(scored, weights, feature_map=FEATURE_MAP)


# ─── Core Backtest ───────────────────────────────────────────────

def run_backtest(
    etf_prices: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    rebalance_freq: str = DEFAULT_REBALANCE_FREQ,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST,
    n_overweight: int = 2,
    n_underweight: int = 2,
    tilt_size: float = TILT_SIZE,
) -> BacktestResult:
    """
    Run a sector-tilt backtest simulation.

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

    etf_cols = [c for c in etf_prices.columns if c != BENCHMARK_ETF]
    if not etf_cols:
        logger.warning("No ETF columns found (excluding benchmark).")
        return _empty_result()

    prices = etf_prices[etf_cols + [BENCHMARK_ETF]].dropna(how='all').copy()
    daily_returns = prices.pct_change().dropna(how='all')

    if len(daily_returns) < WARMUP_DAYS + MIN_BACKTEST_DAYS:
        logger.warning("Not enough data for backtest: %d rows.", len(daily_returns))
        return _empty_result()

    # Determine rebalance dates
    freq_code = 'MS' if rebalance_freq == "Monthly" else 'QS'
    rebal_dates = daily_returns.resample(freq_code).first().index.tolist()

    # Skip warmup period — use .iloc to avoid IndexError on short data
    warmup_idx = min(WARMUP_DAYS, len(daily_returns) - 1)
    warmup_end = daily_returns.index[warmup_idx]
    valid_dates = [d for d in rebal_dates if d >= warmup_end]

    if len(valid_dates) < 2:
        logger.warning("Too few rebalance dates after warmup: %d", len(valid_dates))
        return _empty_result()

    # Equal-weight baseline
    n_sectors = len(etf_cols)
    equal_weight = 1.0 / n_sectors

    # Run simulation — collect as list of (date, port_ret, bench_ret) tuples
    daily_records: List[Tuple[pd.Timestamp, float, float]] = []
    weight_log = []
    prev_weights = pd.Series(equal_weight, index=etf_cols)
    # Convert basis points to fraction: 10 bps → 0.001 (0.10%)
    cost_rate = transaction_cost_bps / 10_000.0

    for i, date in enumerate(valid_dates):
        # Score sectors at rebalance using data BEFORE this date
        features = _compute_etf_features(prices, date)
        if features.empty:
            target_weights = pd.Series(equal_weight, index=etf_cols)
        else:
            scores = _score_etfs(features, weights)
            ranked = scores.sort_values(ascending=False)

            target_weights = pd.Series(equal_weight, index=etf_cols)
            top_sectors = ranked.index[:n_overweight]
            bottom_sectors = ranked.index[-n_underweight:]

            for s in top_sectors:
                if s in target_weights.index:
                    target_weights[s] = equal_weight + tilt_size
            for s in bottom_sectors:
                if s in target_weights.index:
                    target_weights[s] = max(0.01, equal_weight - tilt_size)

            # Renormalize to sum to 1
            target_weights = target_weights / target_weights.sum()

        # Transaction costs
        turnover = (target_weights - prev_weights).abs().sum()
        cost = turnover * cost_rate

        # Period boundaries — from this rebalance to the next (exclusive).
        # Since scoring used data *before* date, we include date itself in
        # returns (the decision was made at open, returns are close-to-close).
        period_end = valid_dates[i + 1] if i + 1 < len(valid_dates) else daily_returns.index[-1]
        period_rets = daily_returns.loc[date:period_end, etf_cols]
        # Exclude the next rebalance date itself to prevent double-counting
        if i + 1 < len(valid_dates) and len(period_rets) > 0:
            period_rets = period_rets.iloc[:-1] if period_rets.index[-1] == period_end else period_rets

        if len(period_rets) == 0:
            continue

        # Portfolio return for each day in this period
        for day_idx, (day, day_ret) in enumerate(period_rets.iterrows()):
            port_ret = (target_weights * day_ret).sum()
            # Deduct transaction cost on first day of each period
            if day_idx == 0:
                port_ret -= cost

            bench_ret = 0.0
            if BENCHMARK_ETF in daily_returns.columns and day in daily_returns.index:
                bench_ret = daily_returns.loc[day, BENCHMARK_ETF]

            daily_records.append((day, port_ret, bench_ret))

        weight_log.append({'date': date, **target_weights.to_dict()})
        prev_weights = target_weights

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

    # Compute metrics
    metrics = compute_metrics(port_series, bench_series)

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
    Bootstrap test for significance of alpha.
    Uses block bootstrap (21-day blocks) to preserve autocorrelation.

    Returns dict with: (mean, 5th percentile, 95th percentile) for each metric.
    """
    active_returns = portfolio_returns - benchmark_returns
    n = len(active_returns)

    if n < block_size * 2:
        return {}

    # Guard: n - block_size must be > 0 for rng.integers to work
    if n <= block_size:
        logger.warning("Not enough data for block bootstrap (n=%d, block=%d)", n, block_size)
        return {}

    # Use modern numpy RNG for reproducibility
    rng = np.random.default_rng(seed)

    alpha_samples = np.empty(n_samples)
    sharpe_samples = np.empty(n_samples)

    n_blocks = n // block_size + 1

    for i in range(n_samples):
        block_starts = rng.integers(0, n - block_size, size=n_blocks)
        sample_indices = np.concatenate([
            np.arange(s, min(s + block_size, n)) for s in block_starts
        ])[:n]

        sample_active = active_returns.iloc[sample_indices].values
        sample_port = portfolio_returns.iloc[sample_indices].values

        # Annualized alpha
        alpha_samples[i] = sample_active.mean() * TRADING_DAYS_PER_YEAR

        # Sharpe
        port_std = sample_port.std()
        sharpe_samples[i] = (
            sample_port.mean() / port_std * np.sqrt(TRADING_DAYS_PER_YEAR)
            if port_std > 1e-10 else 0.0
        )

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
