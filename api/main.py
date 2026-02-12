from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from api.routes import market_data, analysis, backtest

app = FastAPI(
    title="Market Intelligence Dashboard API",
    description="High-performance async API for sector rotation analysis.",
    version="1.0.0",
    default_response_class=ORJSONResponse
)

# CORS Configuration
# Allow all origins for now to support local development with Streamlit/Vue/etc.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(market_data.router, prefix="/market", tags=["Market Data"])
app.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])
app.include_router(backtest.router, prefix="/backtest", tags=["Backtest"])

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "ok", "service": "market-intelligence-api"}

if __name__ == "__main__":
    import uvicorn
    # For local debugging
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
