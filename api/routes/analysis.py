from fastapi import APIRouter, HTTPException, Body
from typing import List, Optional
from api.services import MarketService
from api.schemas import SectorScore, WeightPreset, DataFrameResponse

router = APIRouter()

@router.post("/score", response_model=List[SectorScore])
async def score_sectors(
    weights: Optional[WeightPreset] = Body(None, description="Custom weights for scoring logic. Must sum to 1.0.")
):
    """
    Compute sector scores based on provided weights (or defaults if not provided).
    """
    try:
        results, _ = await MarketService.get_scored_sectors(custom_weights=weights)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/drilldown/{sector}", response_model=DataFrameResponse)
async def sector_drilldown(sector: str):
    """
    Get all constituents for a specific sector.
    """
    try:
        return await MarketService.get_sector_constituents(sector)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
