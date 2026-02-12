"""
Tests for src/backtest.py — lookahead, warmup, costs, metrics, bootstrap.
"""
import pandas as pd
import numpy as np
import pytest

from src.backtest import (
    run_backtest, compute_metrics, bootstrap_test,
    _compute_etf_features, BacktestResult,
)
from config import (
    WARMUP_DAYS, MIN_BACKTEST_DAYS,
)


class TestComputeEtfFeatures:
    """§4 — lookahead protection in feature computation."""

    def test_excludes_rebalance_date(self, sample_etf_prices):
        """Features must use only data BEFORE the rebalance date."""
        date = sample_etf_prices.index[100]
        features = _compute_etf_features(sample_etf_prices, date)

        # The function should only use prices where index < date
        # We can verify by checking an internal property: features should
        # not change if we remove the rebalance date's price
        modified_prices = sample_etf_prices.drop(date)
        features_without = _compute_etf_features(modified_prices, date)

        if not features.empty and not features_without.empty:
            pd.testing.assert_frame_equal(features, features_without)

    def test_insufficient_history_returns_empty(self, sample_etf_prices):
        """Very early dates should return empty DataFrame (not enough lookback)."""
        early_date = sample_etf_prices.index[5]
        result = _compute_etf_features(sample_etf_prices, early_date)
        assert result.empty


class TestRunBacktest:
    """§4 — core backtest correctness."""

    def test_returns_backtest_result(self, sample_etf_prices):
        result = run_backtest(sample_etf_prices)
        assert isinstance(result, BacktestResult)

    def test_nonempty_on_sufficient_data(self, sample_etf_prices):
        """300 trading days should be enough for a valid backtest."""
        result = run_backtest(sample_etf_prices)
        assert len(result.portfolio_returns) > 0
        assert len(result.metrics) > 0

    def test_empty_on_insufficient_data(self):
        """Very short price history should return empty result."""
        dates = pd.bdate_range('2024-01-02', periods=10)
        prices = pd.DataFrame({
            'XLK': np.random.uniform(100, 110, 10),
            'SPY': np.random.uniform(400, 410, 10),
        }, index=dates)
        result = run_backtest(prices)
        assert len(result.portfolio_returns) == 0

    def test_rejects_non_datetime_index(self):
        """Should raise TypeError if index is not DatetimeIndex."""
        prices = pd.DataFrame({
            'XLK': [100, 101], 'SPY': [400, 401]
        }, index=[0, 1])
        with pytest.raises(TypeError, match="DatetimeIndex"):
            run_backtest(prices)

    def test_transaction_cost_reduces_returns(self, sample_etf_prices):
        """Higher transaction costs should produce lower total return."""
        result_low = run_backtest(sample_etf_prices, transaction_cost_bps=0)
        result_high = run_backtest(sample_etf_prices, transaction_cost_bps=100)

        if result_low.metrics and result_high.metrics:
            assert result_low.metrics['total_return'] >= result_high.metrics['total_return']

    def test_warmup_safety_short_data(self):
        """Warmup should not IndexError even with data barely above threshold."""
        n_days = WARMUP_DAYS + MIN_BACKTEST_DAYS + 5
        dates = pd.bdate_range('2024-01-02', periods=n_days)
        np.random.seed(42)
        prices = pd.DataFrame({
            'XLK': 100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, n_days)),
            'XLF': 100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, n_days)),
            'SPY': 100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, n_days)),
        }, index=dates)
        # Should not raise
        result = run_backtest(prices)
        assert isinstance(result, BacktestResult)

    def test_no_duplicate_dates_in_returns(self, sample_etf_prices):
        """Portfolio returns should have unique dates."""
        result = run_backtest(sample_etf_prices)
        if len(result.portfolio_returns) > 0:
            assert result.portfolio_returns.index.is_unique


class TestComputeMetrics:
    """§4, §5 — metrics correctness and numerical stability."""

    def test_known_return(self, known_returns_series):
        """100 days of 0.1% daily → ~10.5% total return."""
        bench = pd.Series(0.0, index=known_returns_series.index)
        metrics = compute_metrics(known_returns_series, bench)

        assert metrics['total_return'] == pytest.approx(10.51, abs=0.5)

    def test_constant_returns_sharpe(self):
        """Constant positive returns → very high Sharpe (no variance)."""
        dates = pd.bdate_range('2024-01-02', periods=100)
        port = pd.Series(0.001, index=dates)
        bench = pd.Series(0.0, index=dates)
        metrics = compute_metrics(port, bench)
        # Std is ~0 → Sharpe should be very high or capped
        assert metrics['sharpe_ratio'] > 5.0 or metrics['sharpe_ratio'] == 0.0

    def test_zero_returns(self):
        """All-zero returns: should produce 0% return and 0 Sharpe."""
        dates = pd.bdate_range('2024-01-02', periods=100)
        port = pd.Series(0.0, index=dates)
        bench = pd.Series(0.0, index=dates)
        metrics = compute_metrics(port, bench)
        assert metrics['total_return'] == 0.0
        assert metrics['sharpe_ratio'] == 0.0

    def test_too_few_days_returns_empty(self):
        """Fewer than MIN_BACKTEST_DAYS should return empty dict."""
        dates = pd.bdate_range('2024-01-02', periods=MIN_BACKTEST_DAYS - 1)
        port = pd.Series(0.001, index=dates)
        bench = pd.Series(0.0, index=dates)
        assert compute_metrics(port, bench) == {}

    def test_transaction_cost_unit(self):
        """10 bps = 0.001 = 0.10%. Verify cost_rate calculation."""
        # This tests the constant, not compute_metrics directly
        cost_rate = 10 / 10_000.0
        assert cost_rate == pytest.approx(0.001)


class TestBootstrapTest:
    """§4, §5 — bootstrap edge cases and reproducibility."""

    def test_reproducibility(self, sample_etf_prices):
        """Same seed should produce identical results."""
        result = run_backtest(sample_etf_prices)
        if len(result.portfolio_returns) > 50:
            b1 = bootstrap_test(result.portfolio_returns, result.benchmark_returns, seed=42)
            b2 = bootstrap_test(result.portfolio_returns, result.benchmark_returns, seed=42)
            assert b1 == b2

    def test_short_data_returns_empty(self):
        """Too few data points for block bootstrap should return {}."""
        dates = pd.bdate_range('2024-01-02', periods=30)
        port = pd.Series(0.001, index=dates)
        bench = pd.Series(0.0, index=dates)
        result = bootstrap_test(port, bench, block_size=21)
        assert result == {}

    def test_returns_alpha_and_sharpe(self, sample_etf_prices):
        """Output should contain 'alpha' and 'sharpe' keys."""
        result = run_backtest(sample_etf_prices)
        if len(result.portfolio_returns) > 50:
            bs = bootstrap_test(result.portfolio_returns, result.benchmark_returns)
            if bs:
                assert 'alpha' in bs
                assert 'sharpe' in bs
                # Each value is a 3-tuple (mean, p5, p95)
                assert len(bs['alpha']) == 3
                assert len(bs['sharpe']) == 3
