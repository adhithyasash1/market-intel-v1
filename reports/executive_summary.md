# Executive Summary — Market Intelligence & Sector Rotation

**Date:** Auto-generated at dashboard runtime  
**Framework:** Composite momentum scoring with configurable weights

---

## Objective

Identify which equity sectors to **overweight**, **hold neutral**, or **avoid** using a transparent, backtested composite scoring system. Signals are updated daily from TradingView Screener data across S&P 500 constituents.

## Methodology

Each sector receives a **Composite Score** built from five normalized features:

| Feature | Description | Polarity |
|---------|------------|----------|
| Momentum | Median 1-month return of constituents | Higher = better |
| Breadth | % of members with positive 1M returns | Higher = better |
| Volatility | Median daily volatility | Lower = better |
| Liquidity | Median daily traded value | Higher = better |
| Acceleration | Momentum acceleration (1M − 3M return) | Higher = better |

Features are z-score normalized across sectors. Weighted sum produces the composite score. **Top 20% → Overweight** · **Middle 60% → Neutral** · **Bottom 20% → Avoid**.

Three weight presets are available: **Momentum-Heavy** (default), **Risk-Aware**, **Equal-Weight**.

## Key Capabilities

- **Daily updated** sector rankings with explainable signal drivers
- **Backtest engine** with cost-adjusted returns, Sharpe, hit-rate, and bootstrap significance
- **Scenario analysis** — stress-test sectors by adjusting volatility, momentum, breadth
- **Drilldown** into sector constituents with top movers, RSI distribution, cross-sector correlation
- **Export** CSV data and auto-generated recommendation memos

## Expected Value

| KPI | Measurement |
|-----|-------------|
| Hit-rate | % of recommended sectors outperforming benchmark |
| Strategy Sharpe | Risk-adjusted excess return |
| Cost-adjusted Alpha | Net outperformance after transaction costs |
| Decision Speed | Time-to-insight for portfolio tilt decisions |

## Risk & Governance

- Data from unofficial TradingView API — fallback to cached snapshots if rate-limited
- Backtest uses sector ETFs as proxies — subject to tracking error vs actual constituents
- **Max tilt rule**: ±20% sector weight change per rebalance period
- All signals stored with timestamped inputs for auditability
- **Not financial advice** — signals are quantitative inputs for human decision-making

---

*See dashboard (`streamlit run app.py`) for live signals and full backtest results.*
