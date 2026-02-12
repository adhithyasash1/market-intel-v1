"""
Tests for src/scorer.py — z-scores, weighted scoring, signal assignment.
"""
import pandas as pd
import numpy as np
import pytest

from src.scorer import (
    compute_zscores, compute_weighted_score, compute_composite_scores,
    assign_signals, explain_signal, score_pipeline, FEATURE_MAP,
)
from config import WEIGHT_PRESETS, DEFAULT_PRESET


class TestComputeZscores:
    """§3 — z-score normalization correctness."""

    def test_volatility_polarity(self, sample_sector_aggs):
        """Higher volatility should produce NEGATIVE z-score (polarity = -1)."""
        result = compute_zscores(sample_sector_aggs)
        # Find sector with highest volatility
        max_vol_sector = sample_sector_aggs['avg_volatility'].idxmax()
        min_vol_sector = sample_sector_aggs['avg_volatility'].idxmin()
        assert result.loc[max_vol_sector, 'z_volatility'] < result.loc[min_vol_sector, 'z_volatility']

    def test_momentum_polarity(self, sample_sector_aggs):
        """Higher momentum should produce POSITIVE z-score (polarity = +1)."""
        result = compute_zscores(sample_sector_aggs)
        max_mom = sample_sector_aggs['median_momentum'].idxmax()
        min_mom = sample_sector_aggs['median_momentum'].idxmin()
        assert result.loc[max_mom, 'z_momentum'] > result.loc[min_mom, 'z_momentum']

    def test_missing_column_gets_zero(self):
        """If a feature column is missing, z-score should be 0.0."""
        df = pd.DataFrame({'n_stocks': [10]}, index=['Tech'])
        result = compute_zscores(df)
        assert result['z_momentum'].iloc[0] == 0.0
        assert result['z_volatility'].iloc[0] == 0.0


class TestComputeWeightedScore:
    """§3 — weighted composite score correctness."""

    def test_all_zeros_gives_zero_score(self):
        """When all z-scores are zero, composite score should be 0."""
        df = pd.DataFrame({
            'z_momentum': [0.0, 0.0],
            'z_breadth': [0.0, 0.0],
            'z_volatility': [0.0, 0.0],
            'z_liquidity': [0.0, 0.0],
            'z_acceleration': [0.0, 0.0],
        }, index=['A', 'B'])
        weights = WEIGHT_PRESETS[DEFAULT_PRESET]
        scores = compute_weighted_score(df, weights)
        assert (scores == 0.0).all()

    def test_missing_weight_key_defaults_to_zero(self):
        """Weight keys not in the dict should contribute 0."""
        df = pd.DataFrame({
            'z_momentum': [1.0],
            'z_breadth': [1.0],
            'z_volatility': [1.0],
            'z_liquidity': [1.0],
            'z_acceleration': [1.0],
        }, index=['A'])
        # Only momentum weighted
        partial_weights = {'momentum': 1.0}
        score = compute_weighted_score(df, partial_weights)
        assert score.iloc[0] == 1.0  # only z_momentum * 1.0

    def test_higher_vol_reduces_score(self, sample_sector_aggs):
        """Increasing a sector's volatility should reduce its composite score."""
        scored_base = compute_zscores(sample_sector_aggs)
        base_score = compute_weighted_score(scored_base, WEIGHT_PRESETS[DEFAULT_PRESET])

        shocked = sample_sector_aggs.copy()
        shocked.loc['Technology Services', 'avg_volatility'] *= 5  # big shock
        scored_shock = compute_zscores(shocked)
        shock_score = compute_weighted_score(scored_shock, WEIGHT_PRESETS[DEFAULT_PRESET])

        assert shock_score['Technology Services'] < base_score['Technology Services']


class TestAssignSignals:
    """§3 — signal assignment thresholds and ties."""

    def test_produces_three_signal_types(self, sample_sector_aggs):
        """With 5 sectors, should produce at least 1 Overweight and 1 Avoid."""
        scored = compute_composite_scores(sample_sector_aggs)
        labeled = assign_signals(scored)
        signals = set(labeled['signal'])
        # With default 80/20 percentiles on 5 sectors, expect all three
        assert 'Overweight' in signals or 'Avoid' in signals

    def test_custom_thresholds(self):
        """100% overweight / 0% avoid → all Overweight."""
        df = pd.DataFrame({
            'composite_score': [3.0, 2.0, 1.0, 0.0, -1.0]
        }, index=['A', 'B', 'C', 'D', 'E'])
        result = assign_signals(df, overweight_pct=0.0, avoid_pct=0.0)
        # Everything >= 0-percentile should be Overweight
        assert (result['signal'] == 'Overweight').all()

    def test_deterministic_ties(self):
        """Tied scores should produce deterministic signals."""
        df = pd.DataFrame({
            'composite_score': [1.0, 1.0, 1.0, 1.0, 1.0]
        }, index=['A', 'B', 'C', 'D', 'E'])
        result1 = assign_signals(df)
        result2 = assign_signals(df)
        pd.testing.assert_series_equal(result1['signal'], result2['signal'])


class TestExplainSignal:
    """§3 — explanation generation."""

    def test_returns_list_of_strings(self, sample_sector_aggs):
        scored = compute_composite_scores(sample_sector_aggs)
        row = scored.iloc[0]
        explanations = explain_signal(row)
        assert isinstance(explanations, list)
        assert all(isinstance(e, str) for e in explanations)
        assert len(explanations) <= 3

    def test_custom_weights_affect_explanations(self, sample_sector_aggs):
        """Different weights should produce different contribution rankings."""
        scored = compute_composite_scores(sample_sector_aggs)
        row = scored.iloc[0]

        exp_mom = explain_signal(row, weights={'momentum': 1.0, 'breadth': 0.0,
                                                'volatility': 0.0, 'liquidity': 0.0,
                                                'acceleration': 0.0})
        exp_vol = explain_signal(row, weights={'momentum': 0.0, 'breadth': 0.0,
                                                'volatility': 1.0, 'liquidity': 0.0,
                                                'acceleration': 0.0})
        # Top contributor should differ
        assert exp_mom[0] != exp_vol[0]


class TestScorePipeline:
    """§3 — full pipeline integration."""

    def test_output_has_all_columns(self, sample_sector_aggs):
        result = score_pipeline(sample_sector_aggs)
        for col in ['composite_score', 'signal', 'explanation', 'score_rank']:
            assert col in result.columns

    def test_explanations_are_lists(self, sample_sector_aggs):
        result = score_pipeline(sample_sector_aggs)
        for sector, row in result.iterrows():
            assert isinstance(row['explanation'], list)
            assert len(row['explanation']) <= 3

    def test_scores_sorted_descending(self, sample_sector_aggs):
        result = score_pipeline(sample_sector_aggs)
        scores = result['composite_score'].values
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
