"""
Tests for src/features.py — stock features, sector aggregation, edge cases.
"""
import pandas as pd
import numpy as np
import pytest

from src.features import (
    compute_stock_features, compute_sector_aggregates,
    _safe_column_diff, _sector_concentration,
)
from config import MIN_SECTOR_STOCKS


class TestSafeColumnDiff:
    """§2 — _safe_column_diff helper correctness."""

    def test_both_columns_present(self):
        df = pd.DataFrame({'a': [10.0, 20.0], 'b': [3.0, 5.0]})
        result = _safe_column_diff(df, 'a', 'b')
        assert list(result) == [7.0, 15.0]

    def test_missing_column_returns_zeros(self):
        df = pd.DataFrame({'a': [10.0, 20.0]})
        result = _safe_column_diff(df, 'a', 'missing')
        assert (result == 0.0).all()
        assert len(result) == 2

    def test_preserves_index(self):
        df = pd.DataFrame({'a': [1.0], 'b': [2.0]}, index=['x'])
        result = _safe_column_diff(df, 'a', 'b')
        assert result.index[0] == 'x'


class TestComputeStockFeatures:
    """§2 — per-stock feature computation."""

    def test_adds_expected_columns(self, sample_stock_df):
        result = compute_stock_features(sample_stock_df)
        expected_new = [
            'momentum_accel', 'macd_histogram', 'above_sma50',
            'above_sma200', 'golden_cross', 'adtv', 'rsi_zone', 'positive_1m',
        ]
        for col in expected_new:
            assert col in result.columns, f"Missing feature: {col}"

    def test_missing_perf_1m_gives_zero_momentum_accel(self):
        """When perf_1m is absent, momentum_accel should be 0.0."""
        df = pd.DataFrame({
            'name': ['A'], 'price': [100], 'sector': ['Tech'],
            'perf_3m': [5.0],  # no perf_1m
        })
        result = compute_stock_features(df)
        assert (result['momentum_accel'] == 0.0).all()

    def test_rsi_out_of_range_no_crash(self):
        """RSI values outside [0, 100] should not crash pd.cut."""
        df = pd.DataFrame({
            'name': ['A', 'B'], 'price': [100, 200],
            'sector': ['Tech', 'Fin'],
            'rsi_14': [-5.0, 105.0],  # out of range
        })
        result = compute_stock_features(df)
        # After clipping, -5 → 0 (Oversold), 105 → 100 (Overbought)
        assert result['rsi_zone'].iloc[0] == 'Oversold'
        assert result['rsi_zone'].iloc[1] == 'Overbought'

    def test_missing_sma_columns(self):
        """Without SMA columns, golden_cross etc. should be NaN."""
        df = pd.DataFrame({
            'name': ['A'], 'price': [100], 'sector': ['Tech'],
        })
        result = compute_stock_features(df)
        assert pd.isna(result['golden_cross'].iloc[0])

    def test_adtv_uses_avg_vol_30d(self, sample_stock_df):
        """ADTV should use log1p(avg_vol_30d * price) when available."""
        result = compute_stock_features(sample_stock_df)
        raw_adtv = sample_stock_df['avg_vol_30d'] * sample_stock_df['price']
        expected = np.log1p(raw_adtv.clip(lower=0))
        pd.testing.assert_series_equal(result['adtv'], expected, check_names=False)

    def test_acceleration_gated_by_positive_momentum(self):
        """Acceleration should be zero when perf_1m is negative (dead-cat bounce filter)."""
        df = pd.DataFrame({
            'name': ['A', 'B'],
            'price': [100, 200],
            'sector': ['Tech', 'Fin'],
            'perf_1m': [-5.0, 2.0],
            'perf_3m': [-20.0, -1.0],
        })
        result = compute_stock_features(df)
        # A: perf_1m=-5 <0 → gated → 0.0
        assert result['momentum_accel'].iloc[0] == 0.0
        # B: perf_1m=2 >0 → accel = 2 - (-1) = 3.0
        assert result['momentum_accel'].iloc[1] == pytest.approx(3.0)

    def test_missing_volatility_gives_nan_default(self):
        """avg_volatility should default to NaN, not 0.0, when missing."""
        n = 5
        df = pd.DataFrame({
            'name': [f'S{i}' for i in range(n)],
            'sector': ['Tech'] * n,
            'perf_1m': [1.0] * n,
            'positive_1m': [1] * n,
            # No volatility_d, no atr_14
        })
        result = compute_sector_aggregates(df)
        assert pd.isna(result.loc['Tech', 'avg_volatility'])


class TestSectorConcentration:
    """§2 — _sector_concentration edge cases."""

    def test_normal_case(self):
        df = pd.DataFrame({'market_cap': [100, 50, 30, 20]})
        result = _sector_concentration(df)
        # Top 3: 100 + 50 + 30 = 180, total = 200
        assert result == pytest.approx(0.9)

    def test_fewer_than_3_returns_1(self):
        df = pd.DataFrame({'market_cap': [100, 50]})
        assert _sector_concentration(df) == 1.0

    def test_total_zero_returns_1(self):
        df = pd.DataFrame({'market_cap': [0, 0, 0, 0]})
        assert _sector_concentration(df) == 1.0

    def test_missing_market_cap_column(self):
        df = pd.DataFrame({'price': [100, 200]})
        assert _sector_concentration(df) == 0.5


class TestComputeSectorAggregates:
    """§2 — sector aggregation logic."""

    def test_filters_small_sectors(self):
        """Sectors with fewer than MIN_SECTOR_STOCKS should be dropped."""
        df = pd.DataFrame({
            'name': ['A', 'B', 'C', 'D'],
            'sector': ['Big', 'Big', 'Big', 'Tiny'],  # Tiny has only 1
            'perf_1m': [1, 2, 3, 4],
            'positive_1m': [1, 1, 0, 1],
        })
        result = compute_sector_aggregates(df)
        assert 'Tiny' not in result.index
        if MIN_SECTOR_STOCKS <= 3:
            assert 'Big' in result.index

    def test_sector_at_threshold(self):
        """Sector with exactly MIN_SECTOR_STOCKS rows should be kept."""
        n = MIN_SECTOR_STOCKS
        df = pd.DataFrame({
            'name': [f'S{i}' for i in range(n)],
            'sector': ['Exact'] * n,
            'perf_1m': [1.0] * n,
            'positive_1m': [1] * n,
        })
        result = compute_sector_aggregates(df)
        assert 'Exact' in result.index

    def test_missing_sector_column_raises(self):
        """Must raise ValueError when 'sector' column is absent."""
        df = pd.DataFrame({'price': [100]})
        with pytest.raises(ValueError, match="sector"):
            compute_sector_aggregates(df)

    def test_output_has_expected_columns(self, sample_stock_df):
        enriched = compute_stock_features(sample_stock_df)
        result = compute_sector_aggregates(enriched)
        for col in ['n_stocks', 'median_momentum', 'breadth', 'avg_volatility']:
            assert col in result.columns
