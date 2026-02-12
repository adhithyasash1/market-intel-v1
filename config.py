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
        "momentum":     0.40,
        "breadth":      0.20,
        "volatility":   0.15,   # applied as negative (lower is better)
        "liquidity":    0.10,
        "acceleration": 0.15,
    },
    "Risk-Aware": {
        "momentum":     0.20,
        "breadth":      0.20,
        "volatility":   0.30,   # higher weight on low-vol
        "liquidity":    0.15,
        "acceleration": 0.15,
    },
    "Equal-Weight": {
        "momentum":     0.20,
        "breadth":      0.20,
        "volatility":   0.20,
        "liquidity":    0.20,
        "acceleration": 0.20,
    },
}

DEFAULT_PRESET = "Momentum-Heavy"

# ──────────────────────────────────────────────
# Decision Thresholds (percentile-based)
# ──────────────────────────────────────────────
OVERWEIGHT_PERCENTILE = 0.80   # top 20% → "Overweight"
AVOID_PERCENTILE      = 0.20   # bottom 20% → "Avoid"
# middle 60% → "Neutral"

# ──────────────────────────────────────────────
# Sector ETF Mapping (GICS → SPDR ETFs)
# Used for backtesting with real price history
# ──────────────────────────────────────────────
SECTOR_ETF_MAP = {
    "Technology Services":   "XLK",
    "Electronic Technology": "XLK",
    "Finance":               "XLF",
    "Energy Minerals":       "XLE",
    "Health Technology":     "XLV",
    "Health Services":       "XLV",
    "Producer Manufacturing":"XLI",
    "Industrial Services":   "XLI",
    "Communications":        "XLC",
    "Consumer Durables":     "XLY",
    "Retail Trade":          "XLY",
    "Consumer Non-Durables": "XLP",
    "Distribution Services": "XLP",
    "Utilities":             "XLU",
    "Non-Energy Minerals":   "XLB",
    "Process Industries":    "XLB",
    "Transportation":        "XLI",
    "Commercial Services":   "XLI",
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
TILT_SIZE                = 0.15          # ±15% overweight / underweight
MAX_TILT_PER_REBALANCE   = 0.20          # max single-period weight change
BACKTEST_HISTORY_YEARS   = 5             # years of ETF history to fetch
WARMUP_DAYS              = 60            # trading days before first rebalance
MOMENTUM_LOOKBACK        = 20            # days for short-term momentum
MOMENTUM_LOOKBACK_LONG   = 60            # days for longer-term momentum
TRADING_DAYS_PER_YEAR    = 252           # annualization factor
MIN_BACKTEST_DAYS        = 10            # minimum days for valid metrics
MIN_SECTOR_STOCKS        = 3             # minimum stocks for sector aggregation

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
