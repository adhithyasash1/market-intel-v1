# 📊 Market Intelligence & Sector Rotation Dashboard

### MBA Data Science Management Project | Quantitative Finance
**Author:** R Sashi Adhithya

**Institution:** IIT Mandi

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B) ![Finance](https://img.shields.io/badge/Domain-Quant%20Finance-green)

## 📖 Executive Summary

This project is a decision-support system for **Active Sector Allocation**. It moves beyond simple price analysis by aggregating fundamental and technical data from individual S&P 500 constituents to form a "Macro View" of the 11 GICS sectors.

The dashboard assigns a **Composite Score** to each sector based on a multi-factor model (Momentum, Breadth, Volatility, Liquidity) and provides a rigorous backtest engine to validate the strategy against the SPY benchmark.

---

## 🚀 Key Features

### 1. Live Sector Scoring
* **Real-time Data:** Fetches live snapshot data for 500+ stocks via TradingView.
* **Composite Modeling:** Normalizes diverse metrics (Returns, ATR, Volume) into a unified Z-Score.
* **Explainable AI:** Every signal (Overweight/Avoid) comes with a breakdown of *why* the sector was rated that way.

### 2. Multi-Factor Analysis
The model evaluates sectors based on:
* **Momentum:** Is the sector trending up?
* **Breadth:** Is the rally broad-based or narrow?
* **Volatility:** Is the price action stable or erratic?
* **Acceleration:** Is the trend speeding up?

### 3. Institutional-Grade Backtesting
* **Simulation:** Replays the strategy over 5 years of historical data.
* **Robustness:** Includes transaction costs (bps), slippage, and "warm-up" periods.
* **Metrics:** Calculates Alpha, Sharpe Ratio, Information Ratio, and Max Drawdown.

### 4. Scenario Analysis
* **Stress Testing:** Allows users to simulate "What If" scenarios (e.g., "What if Tech volatility doubles?" or "What if Energy momentum drops 5%?").

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.9 or higher
* Git

### Quick Start

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/adhithyasash1-testing.git](https://github.com/yourusername/adhithyasash1-testing.git)
    cd adhithyasash1-testing
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Dashboard:**
    ```bash
    streamlit run app.py
    ```
    The app will open in your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```text
├── app.py                  # Main Streamlit Dashboard entry point
├── config.py               # Configuration (Weights, Thresholds, Constants)
├── requirements.txt        # Project dependencies
├── src/
│   ├── data_engine.py      # Data fetching (TradingView + YFinance) & Caching
│   ├── features.py         # Feature Engineering (Stock -> Sector aggregation)
│   ├── scorer.py           # Z-Score Normalization & Signal Generation
│   ├── backtest.py         # Historical Simulation Engine
│   └── utils.py            # Helper functions (Formatters, Safe Math)
├── data/
│   ├── snapshots/          # Cached daily stock data (Parquet)
│   └── etf_cache/          # Cached historical ETF prices
└── reports/
    └── executive_summary.md # Auto-generated reports
