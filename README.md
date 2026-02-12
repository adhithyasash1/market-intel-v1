# Market Intelligence & Sector Rotation Dashboard

> **Elevator Pitch**: A production-ready, quantitative dashboard for real-time sector rotation analysis, backtesting, and automated market intelligence reporting.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-FF4B4B)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

This repository hosts a robust Streamlit application designed for quantitative researchers and portfolio managers. It aggregates real-time data from TradingView and Yahoo Finance to compute multi-factor sector scores (Momentum, Volatility, Breadth), providing actionable "Overweight" or "Avoid" signals backed by vectorized backtesting and rigorous sensitivity analysis.

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture & Files](#-architecture--files)
- [Quick Start](#-quick-start)
- [Development & Testing](#-development--testing)
- [Configuration](#-configuration)
- [Data Sources & Caching](#-data-sources--caching)
- [Running Backtests](#-running-backtests)
- [Exports & Reports](#-exports--reports)
- [CI/CD & Security](#-cicd--security)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

- **Live Scoring**: Real-time composite scoring of 11 GICS sectors based on customizable factors.
- **Multi-Factor Model**: Combines Momentum, Volatility, Market Breadth, Liquidity, and Acceleration.
- **Explainability**: Clear "Why?" tooltips for every signal (e.g., "High Volatility", "Breakout").
- **Vectorized Backtesting**: High-performance engine (~60ms/run) to validate strategies over 1-5 years.
- **Scenario Analysis**: Stress-test portfolios against hypothetical volatility shocks or momentum shifts.
- **Automated Reporting**: One-click generation of governance-ready investment memos and CSV exports.

---

## 📂 Architecture & Files

| File/Folder | Description |
|---|---|
| `app.py` | Main Streamlit application entry point. |
| `config.py` | Central configuration for tickers, weights, thresholds, and paths. |
| `src/data_engine.py` | Fetches and caches data from TradingView and YFinance. |
| `src/features.py` | Computes technical indicators (RSI, Volatility, Breadth). |
| `src/scorer.py` | Normalizes features and calculates composite Z-scores. |
| `src/backtest.py` | Vectorized backtesting engine (NumPy-optimized). |
| `src/observability.py` | Structured logging and Prometheus metrics hooks. |
| `reports/` | Directory for generated audit reports, memos, and performance profiles. |
| `tests/` | Unit and integration tests (pytest). |
| `k8s/` | Kubernetes manifests (Deployment, NetworkPolicy, CronJobs). |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Internet access (for TradingView/Yahoo Finance API)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/adhithyasash1/testing.git adhithyasash1-testing
cd adhithyasash1-testing

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

---

## 🛠️ Development & Testing

We use `pytest` for testing and `flake8` for linting.

### Run Test Suite
```bash
# Run all tests
pytest -q

# Run specific test file
pytest tests/test_backtest.py

# Run linters
flake8 . --max-line-length=120
mypy src --ignore-missing-imports
```

### Run Backtest Benchmark
verify performance regressions:
```bash
python perf/benchmark.py
# Outputs baseline comparison to reports/perf_baseline.txt
```

---

## ⚙️ Configuration

Start with `config.py` to customize the model:
- **`SECTOR_ETF_MAP`**: Define the universe of ETFs (e.g., XLK, XLV).
- **`PRESETS`**: Adjust factor weights (e.g., `Aggressive`: higher momentum weight).
- **`lookback_days`**: Time window for volatility/momentum calculations.

**Directories**:
- `data/etf_cache/`: Stores daily OHLCV data (parquet).
- `data/snapshots/`: Stores screener snapshots.

Ensure directories exist:
```python
from src.utils import ensure_dirs
ensure_dirs()
```

---

## 📊 Data Sources & Caching

1. **TradingView (`tvscreener`)**: Used for real-time sector metrics (P/E, volume, performance).
2. **Yahoo Finance (`yfinance`)**: Used for historical price data (backtesting).

**Caching Strategy**:
- Data is cached in `data/` as Parquet files.
- **TTL**: Screener data expires in 15 minutes; Price data expires in 18 hours.
- **Force Refresh**: Click "🔄 Refresh Screener Data" in the sidebar to clear immediate cache.

---

## 📈 Running Backtests & Reproducibility

The backtesting engine is fully vectorized for speed and determinism.

```python
from src.backtest import run_backtest
from src.data_engine import fetch_history

# Load data
prices = fetch_history(tickers, period="2y")

# Run backtest
results = run_backtest(
    prices,
    weights=target_weights, 
    rebalance_freq=21, 
    cost_bps=10
)
print(f"Sharpe: {results.sharpe_ratio:.2f}")
```

**Reproducibility**:
- `bootstrap_test` uses a fixed seed (`42`) by default for stable p-values.
- Results are visually reproducible in the "Backtest" tab.

---

## 📥 Exports & Reports

- **Investment Memo**: Auto-generated Markdown file (`reports/sector_memo_YYYY-MM-DD.md`) summarizing "Overweight" vs "Avoid" signals with reasoning.
- **Rankings CSV**: Full breakdown of Z-scores and raw metrics.
- **Audit Reports**: Released versions include `reports/release_audit.md` certifying test passes and security scans.

---

## 🔐 CI/CD & Security

### Security Hardening
- **Secrets**: NEVER commit API keys. Use environment variables.
  ```bash
  # Check for leaks
  git grep -E "(API_KEY|SECRET|TOKEN)"
  ```
- **Docker**: Images run as non-root user (`appuser`, UID 10001).
- **Network**: Kubernetes Default-Deny egress policy enabled.

### CI Pipeline (GitHub Actions)
```yaml
# .github/workflows/ci.yml snippet
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pip install -r requirements.txt
      - run: pytest
      - run: safety check
```

---

## ❓ Troubleshooting

1. **`yfinance` Network Error**: heavily rate-limited? Wait 5 mins or use a VPN/proxy.
2. **`tvscreener` Warnings**: "Column not found"? The API mapping might have changed. Check `config.py`.
3. **Streamlit Render Error**: `AttributeError: module 'streamlit' has no attribute 'dataframe'`. Update Streamlit: `pip install --upgrade streamlit`.
4. **Corrupt Cache**: "Parquet error"? Delete `data/etf_cache/*.parquet`.
5. **Backtest "Not Enough Data"**: Ensure `fetch_history` requested enough buffer (e.g., `period="2y"` for 1y backtest).
6. **Timezones**: Data is UTC-normalized. Comparisons usually align associated dates.

---

## ⚡ Performance Tips

- **Vectorize**: Use `numpy` arrays for price operations, not `pandas` rows iteration.
- **Broadcasting**: Pre-calculate weights for all rebalance dates at once.
- **Profiling**:
  ```bash
  python -m pyinstrument -m src.backtest
  ```

---

## 🤝 Contributing

1. **Fork & Branch**: Use `feat/new-factor` or `fix/cache-bug`.
2. **Test**: Ensure `pytest` passes.
3. **PR**: Submit pull request with a description of changes.

---

## 📄 License & Maintainers

**License**: MIT 

**Maintainer**: R Sashi Adhithya (Contact: [placeholder])

*This software is for educational and research purposes only. It is not financial advice.*
