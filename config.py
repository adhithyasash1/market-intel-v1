
"""
Configuration settings for the Market Intelligence Dashboard.
Centralizes constants, paths, and tunable parameters.
"""
import os

# ─── Data Engine Settings ───
# ETFs to track for sector rotation
SECTOR_ETFS = [
    'SPY',   # S&P 500 (Benchmark / Market)
    'XLB',   # Materials
    'XLE',   # Energy
    'XLF',   # Financials
    'XLI',   # Industrials
    'XLK',   # Technology
    'XLP',   # Consumer Staples
    'XLU',   # Utilities
    'XLV',   # Health Care
    'XLY',   # Consumer Discretionary
    'XLC',   # Communication Services
    'XLRE',  # Real Estate
]

BENCHMARK_ETF = 'SPY'

# Feature aggregation
MIN_SECTOR_STOCKS = 3  # Minimum stocks required to compute sector aggregate
# (Not used if we only use ETF price features, but kept for future expansion)

# Data fetching
HISTORY_PERIOD = "2y"  # Default history to fetch (1y, 2y, 5y, max)
CACHE_EXPIRY_SECONDS = 3600 * 4  # 4 hours


# ─── Scorer Weight Presets ───
# Defines how we mix different factors to get a final score.
# Weights should sum to 1.0 roughly, but the scorer handles normalization.

WEIGHT_PRESETS = {
    "Balanced": {
        "momentum": 0.3,
        "breadth": 0.2,
        "volatility": 0.2,
        "liquidity": 0.1,
        "acceleration": 0.1,
        "concentration": 0.1,
    },
    "Momentum-Heavy": {
        "momentum": 0.6,
        "breadth": 0.1,
        "volatility": 0.1,
        "liquidity": 0.0,
        "acceleration": 0.2,
        "concentration": 0.0,
    },
    "Defensive": {
        "momentum": 0.1,
        "breadth": 0.2,
        "volatility": 0.5,
        "liquidity": 0.2,
        "acceleration": 0.0,
        "concentration": 0.0,
    },
}

DEFAULT_PRESET = "Momentum-Heavy"


# ─── Feature Mapping ───
# Maps raw column names or internal keys to display names
FEATURE_LABELS = {
    'momentum': 'Momentum (1M)',
    'breadth': 'Breadth (% > SMA50)',
    'volatility': 'Volatility (Inv)',
    'liquidity': 'Liquidity (Log)',
    'acceleration': 'Momentum Accel',
    'concentration': 'Concentration (Inv)',
}

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


# ─── Backtest Settings ───

# 1. Trading Params
DEFAULT_TRANSACTION_COST = 10.0  # Basis points (bps) per trade turnover
# 10 bps = 0.10% round-trip (conservative for ETFs + slippage)

TILT_SIZE = 0.05                 # Weight shift per sector (+/- 5%)
MAX_TILT_PER_REBALANCE = 0.20    # Max turnover constraint per rebalance (20%)
DEFAULT_REBALANCE_FREQ = "Monthly"

# 2. Strategy Logic
N_OVERWEIGHT = 3                 # Top N sectors to overweight
N_UNDERWEIGHT = 3                # Bottom N sectors to underweight

# 3. Lookback Windows (Days)
MOMENTUM_LOOKBACK = 21           # 21 days = ~1 month
MOMENTUM_LOOKBACK_LONG = 60      # days for longer-term momentum
TRADING_DAYS_PER_YEAR = 252      # annualization factor
MIN_BACKTEST_DAYS = 10           # minimum days for valid metrics
MIN_SECTOR_STOCKS = 3            # minimum stocks for sector aggregation (repeat)
WARMUP_DAYS = 60                 # trading days before first rebalance
BACKTEST_HISTORY_YEARS = 5       # years of ETF history to fetch
OVERWEIGHT_PERCENTILE = 0.80     # top 20% → "Overweight"
AVOID_PERCENTILE = 0.20          # bottom 20% → "Avoid"

# 4. Advanced Logic
SIGMOID_GATE_STEEPNESS = 50      # ramp speed for acceleration gate
SIGMOID_GATE_CENTER = 0.0        # center of gate (0% return)

SCORE_EMA_ALPHA = 0.3            # blend: 30% new signal + 70% prior
MARKET_IMPACT_BPS = 5            # sqrt-impact cost per unit turnover
REGIME_DRAWDOWN_THRESH = -0.15   # SPY drawdown to trigger risk-off
REGIME_LOOKBACK_DAYS = 60        # window for drawdown measurement


# ─── UI Settings ───
COLOR_MAP = {
    'Bullish': '#00FA9A',  # MediumSpringGreen
    'Bearish': '#FF4500',  # OrangeRed
    'Neutral': '#808080',  # Gray
}


# ─── Visualization ───
CHARTS_THEME = "streamlit"  # or "plotly_dark"


# ─── Data Paths ───
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SNAPSHOT_DIR = os.path.join(DATA_DIR, "snapshots")
ETF_CACHE_DIR = os.path.join(DATA_DIR, "etf_cache")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


def ensure_dirs():
    """Create data directories if they don't exist. Call lazily, not at import."""
    for d in [DATA_DIR, SNAPSHOT_DIR, ETF_CACHE_DIR, REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)
