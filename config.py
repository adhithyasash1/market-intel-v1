"""
Configuration for the Market Intelligence & Sector Rotation Dashboard.
All weight presets, thresholds, sector mappings, and operational defaults.
"""

import os

# ──────────────────────────────────────────────
# Composite Score Weight Presets
# Keys: momentum, breadth, volatility (inverted), liquidity, acceleration
# ──────────────────────────────────────────────
WEIGHT_PRESETS = {
    "Momentum-Heavy": {
        "momentum":      0.35,
        "breadth":       0.20,
        "volatility":    0.15,   # applied as negative (lower is better)
        "liquidity":     0.10,
        "acceleration":  0.15,
        "concentration": 0.05,   # applied as negative (lower = broader rally)
    },
    "Risk-Aware": {
        "momentum":      0.15,
        "breadth":       0.20,
        "volatility":    0.25,   # higher weight on low-vol
        "liquidity":     0.15,
        "acceleration":  0.15,
        "concentration": 0.10,   # broader rally matters for risk
    },
    "Equal-Weight": {
        "momentum":      0.17,
        "breadth":       0.17,
        "volatility":    0.17,
        "liquidity":     0.17,
        "acceleration":  0.16,
        "concentration": 0.16,
    },
}

DEFAULT_PRESET = "Momentum-Heavy"

# ──────────────────────────────────────────────
# Decision Thresholds (percentile-based)
# ──────────────────────────────────────────────
OVERWEIGHT_PERCENTILE = 0.80   # top 20% → "Overweight"
AVOID_PERCENTILE      = 0.20   # bottom 20% → "Avoid"
# middle 60% → "Neutral"

N_OVERWEIGHT  = 2              # sectors to overweight in backtest
N_UNDERWEIGHT = 2              # sectors to underweight in backtest

# Governance filters (from playbook.md §3 Risk Limits)
BREADTH_FILTER       = 0.60    # OW requires breadth > 60%
MIN_SCORE_MAGNITUDE  = 0.3     # |score| < 0.3 → treat as Neutral

# ──────────────────────────────────────────────
# Sector ETF Mapping (GICS → SPDR ETFs)
# Used for backtesting with real price history
# ──────────────────────────────────────────────
# WARNING: Multiple screener sectors map to the SAME ETF ticker below.
# In the backtest, this means "Technology Services" and "Electronic Technology"
# produce identical return streams, inflating apparent diversification.
# Live scoring is unaffected (it uses individual stocks, not ETFs).
# Aliases: XLK(2), XLV(2), XLI(4), XLY(2), XLP(2), XLB(2)
SECTOR_ETF_MAP = {
    "Technology Services":   "XLK",
    "Electronic Technology": "XLK",    # alias → same as Technology Services
    "Finance":               "XLF",
    "Energy Minerals":       "XLE",
    "Health Technology":     "XLV",
    "Health Services":       "XLV",    # alias → same as Health Technology
    "Producer Manufacturing":"XLI",
    "Industrial Services":   "XLI",    # alias
    "Communications":        "XLC",
    "Consumer Durables":     "XLY",
    "Retail Trade":          "XLY",    # alias
    "Consumer Non-Durables": "XLP",
    "Distribution Services": "XLP",    # alias
    "Utilities":             "XLU",
    "Non-Energy Minerals":   "XLB",
    "Process Industries":    "XLB",    # alias
    "Transportation":        "XLI",    # alias
    "Commercial Services":   "XLI",    # alias
    "Miscellaneous":         "SPY",
}

# Canonical ETF list for backtest
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE", "XLB"]
BENCHMARK_ETF = "SPY"

# ──────────────────────────────────────────────
# Market Regions for Filtering
# ──────────────────────────────────────────────
MARKET_REGIONS = {
    "All":      None,
    "Americas": ["united_states", "canada", "brazil", "mexico"],
    "EMEA":     ["united_kingdom", "germany", "france", "switzerland",
                 "netherlands", "sweden", "spain", "italy", "south_africa"],
    "APAC":     ["japan", "china", "hong_kong", "australia", "india",
                 "south_korea", "taiwan", "singapore"],
}

# ──────────────────────────────────────────────
# Backtest Defaults
# ──────────────────────────────────────────────
DEFAULT_REBALANCE_FREQ   = "Monthly"     # Monthly | Quarterly
DEFAULT_TRANSACTION_COST = 10            # basis points (0.10%)
TILT_SIZE                = 0.10          # ±10% overweight / underweight
                                          # (was 15%; reduced for 11-sector universe
                                          #  where each sector is ~9% equal-weight,
                                          #  so ±15% was nearly 2× the neutral weight)
MAX_TILT_PER_REBALANCE   = 0.20          # max single-period weight change
BACKTEST_HISTORY_YEARS   = 5             # years of ETF history to fetch
WARMUP_DAYS              = 60            # trading days before first rebalance
MOMENTUM_LOOKBACK        = 42            # days for short-term momentum
                                          # (was 20; too short → noise-dominated.
                                          #  42 days ≈ 2 months, captures one
                                          #  full earnings cycle.)
MOMENTUM_LOOKBACK_LONG   = 60            # days for longer-term momentum
TRADING_DAYS_PER_YEAR    = 252           # annualization factor
MIN_BACKTEST_DAYS        = 10            # minimum days for valid metrics
MIN_SECTOR_STOCKS        = 3             # minimum stocks for sector aggregation

# ──────────────────────────────────────────────
# Signal Stability Tuning
# ──────────────────────────────────────────────
SIGMOID_GATE_STEEPNESS   = 50            # ramp speed for acceleration gate
                                          # sigmoid ≈ 0 when momentum < -5%
                                          # sigmoid ≈ 1 when momentum > +5%
SCORE_EMA_ALPHA          = 0.3           # blend: 30% new signal + 70% prior
MARKET_IMPACT_BPS        = 5             # sqrt-impact cost per unit turnover
REGIME_DRAWDOWN_THRESH   = -0.15         # SPY drawdown to trigger risk-off
REGIME_LOOKBACK_DAYS     = 60            # window for drawdown measurement

# ──────────────────────────────────────────────
# Factor Definitions (single source of truth)
# Used by scorer.py (live) and backtest.py (historical)
# ──────────────────────────────────────────────
# Maps weight keys → (column_name, polarity)
# polarity: +1 = higher is better, -1 = lower is better
FEATURE_MAP = {
    "momentum":      ("median_momentum",        +1),
    "breadth":       ("breadth",                +1),
    "volatility":    ("avg_volatility",         -1),
    "liquidity":     ("liquidity_score",         +1),
    "acceleration":  ("momentum_acceleration",  +1),
    "concentration": ("concentration",           -1),
}

# Human-readable labels for explainability
FEATURE_LABELS = {
    "momentum":      "Median 1M momentum",
    "breadth":       "Breadth (% positive members)",
    "volatility":    "Volatility (lower is better)",
    "liquidity":     "Liquidity score",
    "acceleration":  "Momentum acceleration",
    "concentration": "Concentration (lower is better)",
}

# ──────────────────────────────────────────────
# Data Paths
# ──────────────────────────────────────────────
PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(PROJECT_ROOT, "data")
SNAPSHOT_DIR  = os.path.join(DATA_DIR, "snapshots")
ETF_CACHE_DIR = os.path.join(DATA_DIR, "etf_cache")
REPORTS_DIR   = os.path.join(PROJECT_ROOT, "reports")


def ensure_dirs():
    """Create data directories if they don't exist. Call lazily, not at import."""
    for d in [DATA_DIR, SNAPSHOT_DIR, ETF_CACHE_DIR, REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)


# ──────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────
SIGNAL_COLORS = {
    "Overweight": "#00c853",   # green
    "Neutral":    "#ffa726",   # amber
    "Avoid":      "#ef5350",   # red
}
