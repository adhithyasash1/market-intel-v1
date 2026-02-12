"""
Shared test fixtures for Market Intelligence Dashboard tests.
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np

# Ensure project root is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture
def sample_stock_df():
    """
    A minimal stock-level DataFrame mimicking _standardize_columns output.
    Contains 10 rows across 3 sectors, with all canonical column names.
    """
    np.random.seed(42)
    n = 10
    sectors = ['Technology Services'] * 4 + ['Finance'] * 3 + ['Energy Minerals'] * 3

    return pd.DataFrame({
        'name':           [f'Stock_{i}' for i in range(n)],
        'price':          np.random.uniform(50, 500, n),
        'change_pct':     np.random.uniform(-3, 3, n),
        'volume':         np.random.randint(100_000, 10_000_000, n),
        'market_cap':     np.random.uniform(1e9, 500e9, n),
        'sector':         sectors,
        'industry':       ['Software'] * 4 + ['Banks'] * 3 + ['Oil & Gas'] * 3,
        'country':        ['united_states'] * n,
        'rsi_14':         np.random.uniform(20, 80, n),
        'macd_level':     np.random.uniform(-2, 2, n),
        'macd_signal':    np.random.uniform(-2, 2, n),
        'sma_50':         np.random.uniform(100, 400, n),
        'sma_200':        np.random.uniform(90, 380, n),
        'atr_14':         np.random.uniform(1, 10, n),
        'perf_1w':        np.random.uniform(-5, 5, n),
        'perf_1m':        np.random.uniform(-10, 15, n),
        'perf_3m':        np.random.uniform(-15, 20, n),
        'perf_6m':        np.random.uniform(-20, 30, n),
        'perf_ytd':       np.random.uniform(-25, 40, n),
        'perf_1y':        np.random.uniform(-30, 50, n),
        'recommendation': np.random.uniform(1, 5, n),
        'avg_vol_10d':    np.random.randint(100_000, 5_000_000, n),
        'avg_vol_30d':    np.random.randint(100_000, 5_000_000, n),
        'rel_volume':     np.random.uniform(0.5, 2.5, n),
        'volatility_d':   np.random.uniform(1, 5, n),
    })


@pytest.fixture
def sample_sector_aggs():
    """Sector-level aggregates with 5 sectors for testing scoring."""
    sectors = [
        'Technology Services', 'Finance', 'Energy Minerals',
        'Health Technology', 'Utilities'
    ]
    np.random.seed(99)
    return pd.DataFrame({
        'n_stocks':              [50, 40, 20, 30, 15],
        'median_momentum':       [5.0, 2.0, -1.0, 3.0, -3.0],
        'breadth':               [0.7, 0.55, 0.4, 0.6, 0.3],
        'avg_volatility':        [2.5, 3.0, 4.5, 2.0, 1.5],
        'liquidity_score':       [1e8, 5e7, 3e7, 8e7, 1e7],
        'momentum_acceleration': [2.0, 0.5, -2.0, 1.0, -1.0],
        'median_rsi':            [58, 52, 40, 55, 35],
        'pct_golden_cross':      [0.8, 0.6, 0.3, 0.7, 0.2],
        'concentration':         [0.4, 0.35, 0.5, 0.3, 0.6],
    }, index=sectors)


@pytest.fixture
def sample_etf_prices():
    """
    Synthetic daily ETF prices for 3 sector ETFs + benchmark.
    300 trading days of random-walk data.
    """
    np.random.seed(123)
    n_days = 300
    dates = pd.bdate_range('2024-01-02', periods=n_days, freq='B')
    tickers = ['XLK', 'XLF', 'XLE', 'SPY']

    prices = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    for t in tickers:
        # Start at 100, random walk with drift
        returns = np.random.normal(0.0003, 0.01, n_days)
        prices[t] = 100 * np.cumprod(1 + returns)

    return prices


@pytest.fixture
def known_returns_series():
    """
    A simple daily returns series with known cumulative return
    for verifying metrics calculations.
    100 days of 0.1% daily return → ~10.5% total.
    """
    dates = pd.bdate_range('2024-01-02', periods=100, freq='B')
    return pd.Series(0.001, index=dates)
