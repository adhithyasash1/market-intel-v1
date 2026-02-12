"""
Pydantic models for the Market Intelligence Dashboard API.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Dict, Optional, Literal, Any, Union
import math

class WeightPreset(BaseModel):
    """
    Custom weighting scheme for sector scoring.
    Weights must sum to 1.0 (with 0.01 tolerance).
    """
    momentum: float = Field(..., ge=0, le=1)
    breadth: float = Field(..., ge=0, le=1)
    volatility: float = Field(..., ge=0, le=1)
    liquidity: float = Field(..., ge=0, le=1)
    acceleration: float = Field(..., ge=0, le=1)
    concentration: float = Field(0.0, ge=0, le=1)

    @field_validator('*')
    @classmethod
    def check_sum(cls, v, values):
        # Validation happens after all fields are processed in model_validator for sum check
        # But we can't easily do cross-field validation in field_validator.
        # So we use a model validator.
        return v

    def model_post_init(self, __context):
        total = (self.momentum + self.breadth + self.volatility + 
                 self.liquidity + self.acceleration + self.concentration)
        if not math.isclose(total, 1.0, abs_tol=0.01):
            raise ValueError(f"Weights must sum to 1.0 (got {total:.2f})")

class BacktestRequest(BaseModel):
    """
    Parameters for running a sector rotation backtest.
    """
    tickers: Optional[List[str]] = Field(None, min_length=1, description="List of ETF tickers. Defaults to config if None.")
    lookback_days: int = Field(21, gt=10, description="Momentum lookback period in days.") # Changed gt=30 to gt=10 to allow default 21
    rebalance_freq: Literal["Monthly", "Quarterly"] = "Monthly"
    transaction_cost_bps: float = Field(10.0, ge=0)
    weights: Optional[WeightPreset] = None
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "rebalance_freq": "Monthly",
            "transaction_cost_bps": 10.0
        }
    })

class ExplanationItem(BaseModel):
    factor: str
    label: str
    raw_value: Union[float, str]
    z_score: float
    weight: float
    contribution: float
    pct_of_score: float

class SectorScore(BaseModel):
    """
    Scored sector data.
    """
    sector: str
    composite_score: float
    signal: Literal["Overweight", "Neutral", "Avoid"]
    rank: int
    explanation: List[str]
    # Raw metrics for display
    median_momentum: Optional[float] = None
    breadth: Optional[float] = None
    avg_volatility: Optional[float] = None
    liquidity_score: Optional[float] = None
    momentum_acceleration: Optional[float] = None
    
class ScoreResponse(BaseModel):
    results: List[SectorScore]
    weights_used: Dict[str, float]

class BacktestMetrics(BaseModel):
    annualized_return: float
    annualized_return_bench: float
    sharpe_ratio: float
    sharpe_ratio_bench: float
    max_drawdown: float
    information_ratio: float
    hit_rate: float
    alpha: float
    total_return: float
    total_return_bench: float
    n_periods: int

class EquityCurvePoint(BaseModel):
    date: str
    portfolio_value: float
    benchmark_value: float

class BacktestResponse(BaseModel):
    metrics: BacktestMetrics
    equity_curve: List[EquityCurvePoint]
    config_hash: str

class DataFrameResponse(BaseModel):
    """Generic container for DataFrame serialization"""
    columns: List[str]
    data: List[Dict[str, Any]]
    index: List[Any]

class MarketSnapshotRequest(BaseModel):
    pass
