from fastapi import APIRouter, HTTPException, Body
from api.services import BacktestService
from api.schemas import BacktestRequest, BacktestResponse

router = APIRouter()

@router.post("/run", response_model=BacktestResponse)
async def run_backtest(
    request: BacktestRequest = Body(...)
):
    """
    Run a sector rotation backtest simulation.
    EXECUTION: Runs in a separate thread to avoid blocking the API.
    """
    try:
        return await BacktestService.run_backtest_async(
            rebalance_freq=request.rebalance_freq,
            transaction_cost_bps=request.transaction_cost_bps,
            lookback_days=request.lookback_days,
            weights=request.weights
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
