# Market Intelligence & Sector Rotation Dashboard

A decision-ready Streamlit dashboard for sector rotation analysis. Scores S&P 500 sectors using momentum, breadth, volatility, liquidity, and acceleration signals, then simulates a historical backtest using sector ETFs.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

The dashboard is available at `http://localhost:8501`.

## Project Structure

```
├── app.py              # Streamlit entry point
├── config.py           # All constants, thresholds, and mappings
├── src/
│   ├── data_engine.py  # TradingView screener fetch + yfinance ETF prices
│   ├── features.py     # Per-stock features → sector aggregation
│   ├── scorer.py       # Z-score normalization, composite score, signals
│   ├── backtest.py     # Sector-tilt rebalancing simulation
│   └── utils.py        # safe_zscore, formatters
├── tests/              # pytest test suite
├── data/
│   ├── snapshots/      # Parquet screener snapshots (auto-cached)
│   └── etf_cache/      # ETF price history (auto-cached)
└── reports/
    ├── playbook.md     # Trading & operations guide
    └── executive_summary.md
```

## Caching Behavior

- **Screener snapshots** are cached as Parquet files in `data/snapshots/`, one per day. If the live API fails, the latest cached snapshot is used as fallback.
- **ETF price history** is cached in `data/etf_cache/etf_prices.parquet`. It auto-refreshes when cached data is >1 day old.
- **Atomic writes** prevent file corruption from interrupted fetches.
- All timestamps are **timezone-naive** (local time via `datetime.now()`).

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Configuration

Key constants are in `config.py`:

| Constant | Default | Description |
|---|---|---|
| `TILT_SIZE` | 0.15 | ±15% sector over/underweight |
| `OVERWEIGHT_PERCENTILE` | 0.80 | Top 20% → "Overweight" |
| `AVOID_PERCENTILE` | 0.20 | Bottom 20% → "Avoid" |
| `DEFAULT_TRANSACTION_COST` | 10 bps | Cost per trade |
| `WARMUP_DAYS` | 60 | Trading days before first rebalance |
| `MIN_SECTOR_STOCKS` | 3 | Minimum stocks for sector aggregation |

Weight presets (Momentum-Heavy, Risk-Aware, Equal-Weight) are configurable in the sidebar.

## Development

```bash
# Smoke test all imports
python -c "from src.data_engine import *; from src.features import *; from src.scorer import *; from src.backtest import *"

# Run with verbose logging
LOG_LEVEL=DEBUG streamlit run app.py
```
