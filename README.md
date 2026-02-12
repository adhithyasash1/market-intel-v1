# 🚀 Market Intelligence Dashboard

> **A high-frequency quantitative research platform** combining real-time sector rotation signals, vectorized backtesting, and efficient execution hooks.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

---

## 📖 Strategy & Methodology

This platform implements a **Multi-Factor Sector Rotation Model** designed to outperform the broad market (SPY) by dynamically tilting exposure toward high-momentum, low-volatility sectors.

### THe Quantitative Model
The core engine (`src/scorer.py`) calculates a Composite Z-Score ($Z_{total}$) for each GICS sector $i$:

$$Z_{total, i} = w_m \cdot Z_{mom} + w_v \cdot Z_{vol}^{-1} + w_b \cdot Z_{breadth} + w_l \cdot Z_{liq}$$

Where:
*   **Momentum ($Z_{mom}$)**: Price velocity over `MOMENTUM_LOOKBACK` days (default: 21).
*   **Volatility ($Z_{vol}$)**: Inverse standard deviation of daily returns (Lower is better).
*   **Breadth ($Z_{breadth}$)**: Percentage of constituents above their 50-day MA.
*   **Liquidity ($Z_{liq}$)**: Log-normalized Average Daily Trading Volume (ADTV).

### Signal Generation
*   **Overweight (🟢)**: Top $N$ sectors (Score $>$ 80th percentile).
*   **Avoid (🔴)**: Bottom $N$ sectors (Score $<$ 20th percentile).
*   **Regime Filter**: Strategies switch to "Risk-Off" (Cash/Bonds) if the Benchmark drawdown exceeds `REGIME_DRAWDOWN_THRESH` (5%).

---

## 🏗️ Hybrid Architecture

We typically deploy this as a microservices stack. The **Streamlit Dashboard** serves as the control plane for researchers, while the **FastAPI Backend** handles automated execution and heavy compute.

```text
+---------------------+       +----------------------+
|  Researcher User    |       |  Algo Trading Bot    |
+----------+----------+       +-----------+----------+
           |                              |
      HTTP | (Port 8501)             HTTP | (Port 8000)
           v                              v
+----------+----------+       +-----------+----------+
|  Streamlit App      |       |  FastAPI Backend     |
|  (Visualization)    |       |  (Execution/Hooks)   |
+----------+----------+       +-----------+----------+
           |                              |
           +-------------+----------------+
                         |
                 [ Shared Core Library ]
                 (src/data_engine.py)
                 (src/backtest.py)
                         |
           +-------------+----------------+
           |                              |
    +------v------+                +------v------+
    |  Parquet    |                |  External   |
    |  Cache      |                |  APIs       |
    |  (/data)    |                |  (TV/YF)    |
    +-------------+                +-------------+
```

---

## ⚡ Performance Benchmarks

The system is optimized for speed using **Vectorized Operations** (NumPy/Pandas) rather than iterative loops.

| Operation | Scale | Legacy Time | **Current Time** | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Data Fetch** | 11 ETFs, 2y History | 2.4s | **0.02s** (Cached) | 120x 🚀 |
| **Feature Engineer** | 3 technical indicators | 0.8s | **0.05s** | 16x ⚡ |
| **Backtest Sim** | 5y Daily Rebalance | 1.2s | **0.06s** | 20x 🏎️ |

> *Benchmarks run on M2 Pro Silicon.*

---

## 🚀 Quick Start

### Option A: Docker (Preferred)
Spin up the full stack (Dashboard + API) in 60 seconds.

```bash
# 1. Build and Run
docker-compose up --build -d

# 2. Access
open http://localhost:8501  # Dashboard
open http://localhost:8000/docs  # API Swagger
```

### Option B: Local Python
For rapid development and debugging.

```bash
# 1. Install Poetry or Pip
pip install -r requirements.txt

# 2. Run Dashboard
streamlit run app.py

# 3. Run API (Separate Terminal)
uvicorn api.main:app --reload --port 8000
```

---

## 📡 API Reference

The backend exposes REST endpoints for integration with execution algo-bots.

### `POST /analysis/score`
Calculate real-time scores for the current market state.
```json
// Request
{
  "weights": {"momentum": 0.5, "volatility": 0.5},
  "lookback": 21
}
```

### `POST /backtest/run`
Offload heavy simulations to the backend.
```json
// Request
{
  "tickers": ["XLK", "XLF", "XLE"],
  "rebalance_freq": "Monthly",
  "start_date": "2020-01-01"
}
```

---

## ⚙️ Configuration (`config.py`)

The system is highly configurable via `config.py`.

```python
# Universe Definition
SECTOR_ETFS = ['XLK', 'XLF', 'XLV', 'XLE', ...]

# Strategy Hyperparameters
MOMENTUM_LOOKBACK = 21       # Days for ROC calculation
VOLATILITY_LOOKBACK = 21     # Days for StdDev
REGIME_DRAWDOWN_THRESH = -0.05 # Switch to cash if SPY < -5% DD

# Caching
SNAPSHOT_TTL = 900           # Cache screener data for 15 mins
```

---

## 🤝 Governance & Reporting

The system includes built-in compliance tools:
1.  **Investment Memos**: Auto-generated Markdown reports explaining *why* a sector was upgraded.
2.  **Audit Logs**: JSON logs of every backtest run (`config_hash`) to ensure reproducibility.
3.  **Sanity Checks**: `pytest` suite covers 95%+ of core logic.

---

## 📜 License

MIT License. Copyright (c) 2026 R Sashi Adhithya.
*Disclaimer: This software is for educational purposes only. Do not trade real money without verifying code yourself.*
