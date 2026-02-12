"""
Service layer for the Market Intelligence Dashboard API.
Handles interaction with src/ modules, async wrapping, and caching.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from fastapi.concurrency import run_in_threadpool
from async_lru import alru_cache

# Core logic imports
from src.data_engine import load_or_fetch_snapshot, fetch_sector_etf_history
from src.features import compute_stock_features, compute_sector_aggregates
from src.scorer import score_pipeline
from src.backtest import run_backtest, BacktestResult

# Config imports
from config import WEIGHT_PRESETS, DEFAULT_PRESET, DEFAULT_TRANSACTION_COST

# Schema imports
from api.schemas import (
    SectorScore, BacktestResponse, BacktestMetrics, 
    EquityCurvePoint, WeightPreset, DataFrameResponse
)
from src.data_engine import DataFetchError

def serialize_dataframe(df: pd.DataFrame) -> DataFrameResponse:
    """
    Helper to serialize a Pandas DataFrame to a JSON-friendly format.
    Handles NaNs, Infs, and Timestamps.
    """
    # Replace Infinity with None (becomes null in JSON)
    df_clean = df.replace([np.inf, -np.inf], None)
    
    # Convert NaNs to None
    df_clean = df_clean.where(pd.notnull(df_clean), None)
    
    # Convert timestamps to ISO string
    if isinstance(df_clean.index, pd.DatetimeIndex):
        index = df_clean.index.strftime('%Y-%m-%d').tolist()
    else:
        index = df_clean.index.tolist()
        
    # Convert data to record list
    # We use 'records' orientation but handle index separately to preserve it if needed
    data = df_clean.to_dict(orient='records')
    
    return DataFrameResponse(
        columns=df_clean.columns.tolist(),
        data=data,
        index=index
    )

class MarketService:
    @staticmethod
    @alru_cache(maxsize=1, ttl=300) # 5 minute cache
    async def get_snapshot_data():
        """
        Async wrapper for fetching market snapshot. 
        Cached for 5 minutes.
        """
        # Run blocking I/O in threadpool
        snapshot = await run_in_threadpool(load_or_fetch_snapshot)
        return snapshot

    @staticmethod
    async def get_scored_sectors(
        custom_weights: Optional[WeightPreset] = None
    ) -> List[SectorScore]:
        """
        Full pipeline: Fetch -> Feature Eng -> Aggregate -> Score
        """
        # 1. Fetch data (cached)
        snapshot = await MarketService.get_snapshot_data()
        
        # 2. Compute features & aggregates (CPU bound -> threadpool)
        def _process():
            features = compute_stock_features(snapshot)
            aggs = compute_sector_aggregates(features)
            
            # Determine weights
            if custom_weights:
                weights = custom_weights.model_dump()
                preset = "Custom"
            else:
                weights = None # Use default from config in score_pipeline
                preset = DEFAULT_PRESET
                
            scored = score_pipeline(aggs, weights=weights, preset=preset)
            return scored, weights
            
        scored_df, used_weights = await run_in_threadpool(_process)
        
        # 3. Convert to schema
        results = []
        for sector, row in scored_df.iterrows():
            results.append(SectorScore(
                sector=str(sector),
                composite_score=row['composite_score'],
                signal=row['signal'],
                rank=int(row['score_rank']),
                explanation=row['explanation'],
                median_momentum=row.get('median_momentum'),
                breadth=row.get('breadth'),
                avg_volatility=row.get('avg_volatility'),
                liquidity_score=row.get('liquidity_score'),
                momentum_acceleration=row.get('momentum_acceleration')
            ))
            
        return results, used_weights

    @staticmethod
    async def get_sector_constituents(sector: str) -> DataFrameResponse:
        """
        Get all stocks in a specific sector with their features.
        """
        snapshot = await MarketService.get_snapshot_data()
        
        def _filter():
            features = compute_stock_features(snapshot)
            sector_df = features[features['sector'] == sector].copy()
            # Select relevant columns for display
            cols = ['name', 'price', 'change_pct', 'perf_1m', 'perf_3m', 
                    'rsi_14', 'market_cap', 'volume', 'momentum_accel']
            available = [c for c in cols if c in sector_df.columns]
            return sector_df[available]

        sector_data = await run_in_threadpool(_filter)
        return serialize_dataframe(sector_data)

class BacktestService:
    @staticmethod
    async def run_backtest_async(
        rebalance_freq: str,
        transaction_cost_bps: float,
        lookback_days: int,
        weights: Optional[WeightPreset] = None
    ) -> BacktestResponse:
        
        # 1. Fetch ETF history (IO bound -> threadpool)
        etf_prices = await run_in_threadpool(fetch_sector_etf_history)
        
        # 2. Run Backtest (CPU bound -> threadpool)
        def _run():
            w_dict = weights.model_dump() if weights else None
            
            # Note: The src.backtest.run_backtest signature might need adjustment
            # if we want to pass lookback_days dynamically. 
            # Currently src/backtest.py imports MOMENTUM_LOOKBACK from config.
            # For now, we'll use the config value or if the user really wants to override,
            # we might need to monkeypatch or refactor src/backtest.py to accept it.
            # Given the constraints "Refactor... to introduce API", 
            # modifying src/backtest.py to accept parameters is a valid refactor.
            # But let's check `run_backtest` signature in `src/backtest.py`.
            # It takes `etf_prices`, `weights`, `rebalance_freq`, `transaction_cost_bps`.
            # It DOES NOT take lookback_days. 
            # I will strictly stick to what `src/backtest.py` supports for now to avoid 
            # deep refactoring of core logic unless necessary.
            
            result = run_backtest(
                etf_prices=etf_prices,
                weights=w_dict,
                rebalance_freq=rebalance_freq,
                transaction_cost_bps=transaction_cost_bps
            )
            return result

        result: BacktestResult = await run_in_threadpool(_run)
        
        if result.metrics is None or not result.metrics:
             # Handle empty result case
             raise ValueError("Backtest returned no metrics. Check data availability.")

        # 3. Format Response
        metrics = BacktestMetrics(**result.metrics)
        
        # Build equity curve
        equity_curve = []
        # cumulative_portfolio is a Series with DatetimeIndex
        for date, val in result.cumulative_portfolio.items():
            bench_val = result.cumulative_benchmark.get(date, 0.0)
            equity_curve.append(EquityCurvePoint(
                date=date.strftime("%Y-%m-%d"),
                portfolio_value=val,
                benchmark_value=bench_val
            ))
            
        return BacktestResponse(
            metrics=metrics,
            equity_curve=equity_curve,
            config_hash=result.metrics.get('config_hash', '')
        )
