"""
Tests for src/data_engine.py — column mapping, caching, ETF price handling.
"""
import os
import tempfile
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.data_engine import (
    _standardize_columns, save_snapshot, load_snapshot,
    load_or_fetch_snapshot, get_available_snapshots,
)


class TestStandardizeColumns:
    """§1 — column mapping completeness and correctness."""

    def test_maps_required_columns(self):
        """Core columns (price, sector, volume, perf_1m) must be mapped."""
        raw = pd.DataFrame({
            'Name': ['AAPL'],
            'Price': [150.0],
            'Volume': [1e6],
            'Market Capitalization': [2.5e12],
            'Sector': ['Technology'],
            'Monthly Performance': [3.5],
            'Relative Strength Index (14)': [55],
        })
        result = _standardize_columns(raw)
        for expected in ['name', 'price', 'volume', 'market_cap', 'sector', 'perf_1m', 'rsi_14']:
            assert expected in result.columns, f"Missing column: {expected}"

    def test_handles_variant_column_names(self):
        """3-Month vs 3 Month variants should both map to perf_3m."""
        df1 = pd.DataFrame({'3-Month Performance': [1.0]})
        df2 = pd.DataFrame({'3 Month Performance': [1.0]})
        assert 'perf_3m' in _standardize_columns(df1).columns
        assert 'perf_3m' in _standardize_columns(df2).columns

    def test_empty_df_returns_empty(self):
        """Empty input should not raise."""
        result = _standardize_columns(pd.DataFrame())
        assert len(result) == 0

    def test_unknown_columns_preserved(self):
        """Columns that don't match any mapping should remain unchanged."""
        df = pd.DataFrame({'RandomColumn': [42], 'Price': [100]})
        result = _standardize_columns(df)
        assert 'RandomColumn' in result.columns
        assert 'price' in result.columns

    def test_volatility_variants(self):
        """Weekly/monthly volatility should get separate canonical names."""
        df = pd.DataFrame({
            'Volatility': [1.0],
            'Volatility Week': [2.0],
            'Volatility Month': [3.0],
        })
        result = _standardize_columns(df)
        assert 'volatility_d' in result.columns
        assert 'volatility_w' in result.columns
        assert 'volatility_m' in result.columns


class TestSnapshotCaching:
    """§1, §6 — save/load round-trip, atomic writes, cache integrity."""

    def test_save_load_round_trip(self, tmp_path):
        """Saved snapshot should be loadable with identical content."""
        with patch('src.data_engine.SNAPSHOT_DIR', str(tmp_path)), \
             patch('src.data_engine.ensure_dirs'):
            df = pd.DataFrame({'price': [100, 200], 'sector': ['Tech', 'Fin']})
            save_snapshot(df, '2024-01-15')
            loaded = load_snapshot('2024-01-15')
            assert loaded is not None
            pd.testing.assert_frame_equal(df, loaded)

    def test_load_nonexistent_returns_none(self, tmp_path):
        """Loading a date with no cached file should return None."""
        with patch('src.data_engine.SNAPSHOT_DIR', str(tmp_path)):
            assert load_snapshot('1999-12-31') is None

    def test_get_available_snapshots_sorted(self, tmp_path):
        """Available snapshots should be returned in sorted order."""
        with patch('src.data_engine.SNAPSHOT_DIR', str(tmp_path)), \
             patch('src.data_engine.ensure_dirs'):
            df = pd.DataFrame({'x': [1]})
            for date in ['2024-03-01', '2024-01-01', '2024-02-01']:
                save_snapshot(df, date)
            result = get_available_snapshots()
            assert result == ['2024-01-01', '2024-02-01', '2024-03-01']

    def test_atomic_write_no_corruption(self, tmp_path):
        """If save_snapshot fails mid-write, no partial file should remain."""
        with patch('src.data_engine.SNAPSHOT_DIR', str(tmp_path)), \
             patch('src.data_engine.ensure_dirs'):
            # Normal write should succeed
            df = pd.DataFrame({'a': [1, 2, 3]})
            path = save_snapshot(df, '2024-06-01')
            assert os.path.exists(path)

    def test_empty_snapshot_dir(self, tmp_path):
        """Empty snapshot directory should return empty list."""
        with patch('src.data_engine.SNAPSHOT_DIR', str(tmp_path)):
            assert get_available_snapshots() == []


class TestLoadOrFetchSnapshot:
    """§1 — fallback behavior when live fetch fails."""

    def test_returns_cached_if_available(self, tmp_path):
        """Should return cached data without calling API."""
        with patch('src.data_engine.SNAPSHOT_DIR', str(tmp_path)), \
             patch('src.data_engine.ensure_dirs'):
            df = pd.DataFrame({'price': [100], 'sector': ['Tech']})
            save_snapshot(df, '2024-01-15')

            # Should NOT call fetch_screener_snapshot
            with patch('src.data_engine.fetch_screener_snapshot') as mock_fetch:
                result = load_or_fetch_snapshot('2024-01-15')
                mock_fetch.assert_not_called()
                assert len(result) == 1

    def test_fallback_to_stale_cache_on_failure(self, tmp_path):
        """When live fetch fails, should use latest cached snapshot."""
        with patch('src.data_engine.SNAPSHOT_DIR', str(tmp_path)), \
             patch('src.data_engine.ensure_dirs'):
            # Create a cached snapshot for an older date
            df = pd.DataFrame({'price': [99], 'sector': ['Fin']})
            save_snapshot(df, '2024-01-10')

            with patch('src.data_engine.fetch_screener_snapshot',
                       side_effect=RuntimeError("API down")):
                result = load_or_fetch_snapshot('2024-01-15')
                # Should fallback to the 2024-01-10 snapshot
                assert result is not None
                assert result['price'].iloc[0] == 99

    def test_raises_when_no_cache_and_fetch_fails(self, tmp_path):
        """Should raise RuntimeError when both live and cache fail."""
        with patch('src.data_engine.SNAPSHOT_DIR', str(tmp_path)), \
             patch('src.data_engine.ensure_dirs'), \
             patch('src.data_engine.fetch_screener_snapshot',
                   side_effect=RuntimeError("API down")):
            with pytest.raises(RuntimeError, match="No live data"):
                load_or_fetch_snapshot('2024-01-15')
