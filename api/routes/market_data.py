from fastapi import APIRouter, HTTPException
from api.services import MarketService
from api.schemas import DataFrameResponse

router = APIRouter()

@router.get("/snapshot", response_model=DataFrameResponse)
async def get_market_snapshot():
    """
    Get the latest market snapshot from the screener.
    Cached for 5 minutes.
    """
    try:
        df = await MarketService.get_snapshot_data()
        from api.services import serialize_dataframe
        return serialize_dataframe(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
