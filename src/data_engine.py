"""
Data Engine — Snapshot pipeline and historical price fetcher.

Fetches live screener snapshots from TradingView via tvscreener,
caches as Parquet, and downloads sector ETF history via yfinance.
"""

import os
import tempfile
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List

from config import (
    SNAPSHOT_DIR, ETF_CACHE_DIR,
    SECTOR_ETFS, BENCHMARK_ETF, BACKTEST_HISTORY_YEARS,
    ensure_dirs,
)

logger = logging.getLogger(__name__)


# ─── Column Name Mapping ─────────────────────────────────────────
# Maps tvscreener's verbose column names to short canonical names.
# We use substring matching because tv API output names vary slightly
# between versions (e.g., "3-Month Performance" vs "3 Month Performance").

_COLUMN_MAP = {
    'Name':                     'name',
    'Price':                    'price',
    'Change':                   'change_pct',
    'Volume':                   'volume',
    'Market Cap':               'market_cap',
    'Sector':                   'sector',
    'Industry':                 'industry',
    'Country':                  'country',
    'Relative Strength Index':  'rsi_14',
    'MACD Level':               'macd_level',
    'MACD Signal':              'macd_signal',
    'Simple Moving Average (50':'sma_50',
    'Simple Moving Average (200':'sma_200',
    'Average True Range':       'atr_14',
    'Weekly Perf':              'perf_1w',
    'Monthly Perf':             'perf_1m',
    '3-Month Perf':             'perf_3m',
    '3 Month Perf':             'perf_3m',
    '6-Month Perf':             'perf_6m',
    '6 Month Perf':             'perf_6m',
    'YTD Perf':                 'perf_ytd',
    'Yearly Perf':              'perf_1y',
    '1-Year Perf':              'perf_1y',
    'Analyst Ra':               'recommendation',
    'Recommend':                'recommendation',
    'Average Volume (10':       'avg_vol_10d',
    '10 Day Average':           'avg_vol_10d',
    'Average Volume (30':       'avg_vol_30d',
    '30 Day Average':           'avg_vol_30d',
    'Relative Volume':          'rel_volume',
    'Volatility':               'volatility_d',
}


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename raw tvscreener columns to canonical short names using substring matching.

    Strategy: iterate the mapping in order, match each prefix against raw column
    names. First match wins; already-matched columns are skipped. Volatility
    weekly/monthly variants get special handling.
    """
    mapped = {}
    used_new_names: set = set()

    for prefix, canonical in _COLUMN_MAP.items():
        if canonical in used_new_names:
            continue
        for col in df.columns:
            if col in mapped:
                continue
            if prefix.lower() in col.lower():
                mapped[col] = canonical
                used_new_names.add(canonical)
                break

    # Separate daily/weekly/monthly volatility variants
    for col in df.columns:
        if col in mapped:
            continue
        cl = col.lower()
        if 'volatility' in cl and 'week' in cl:
            mapped[col] = 'volatility_w'
        elif 'volatility' in cl and 'month' in cl:
            mapped[col] = 'volatility_m'

    return df.rename(columns=mapped)


# ─── TradingView Screener Snapshot ───────────────────────────────

def fetch_screener_snapshot(max_results: int = 1000) -> pd.DataFrame:
    """
    Fetch a live snapshot from TradingView StockScreener.
    Returns a DataFrame with key fields for S&P 500 stocks.

    Raises RuntimeError if the API call fails after logging the error.

    Note
    ----
    Timestamps (snapshot_date, snapshot_ts) use ``datetime.now()`` which
    returns timezone-naive local time.  All downstream code should treat
    snapshot timestamps as timezone-naive or convert explicitly.
    """
    from tvscreener import StockScreener, StockField, IndexSymbol

    ss = StockScreener()
    ss.set_index(IndexSymbol.SP500)
    ss.set_range(0, max_results)

    ss.select(
        StockField.NAME,
        StockField.PRICE,
        StockField.CHANGE_PERCENT,
        StockField.VOLUME,
        StockField.MARKET_CAPITALIZATION,
        StockField.SECTOR,
        StockField.INDUSTRY,
        StockField.COUNTRY,
        # Technical
        StockField.RELATIVE_STRENGTH_INDEX_14,
        StockField.MACD_LEVEL_12_26,
        StockField.MACD_SIGNAL_12_26,
        StockField.SIMPLE_MOVING_AVERAGE_50,
        StockField.SIMPLE_MOVING_AVERAGE_200,
        StockField.AVERAGE_TRUE_RANGE_14,
        # Performance
        StockField.WEEKLY_PERFORMANCE,
        StockField.MONTHLY_PERFORMANCE,
        StockField.MONTH_PERFORMANCE_3,
        StockField.MONTH_PERFORMANCE_6,
        StockField.YTD_PERFORMANCE,
        StockField.YEARLY_PERFORMANCE,
        # Analyst
        StockField.RECOMMENDATION_MARK,
        # Volume
        StockField.AVERAGE_VOLUME_10_DAY,
        StockField.AVERAGE_VOLUME_30_DAY,
        StockField.RELATIVE_VOLUME,
        # Volatility
        StockField.VOLATILITY,
        StockField.VOLATILITY_WEEK,
        StockField.VOLATILITY_MONTH,
    )

    ss.sort_by(StockField.MARKET_CAPITALIZATION, ascending=False)

    try:
        df = ss.get()
    except Exception as e:
        logger.error("TradingView screener API call failed: %s", e)
        raise RuntimeError(
            f"Failed to fetch screener snapshot: {e}. "
            "Check network connectivity and API rate limits."
        ) from e

    if df.empty:
        raise RuntimeError("Screener returned an empty DataFrame — possible API issue.")

    df = _standardize_columns(df)

    # Fail-fast if critical columns are missing after mapping
    _required = {'sector', 'price', 'volume'}
    _missing = _required - set(df.columns)
    if _missing:
        logger.warning("Column mapping missed required columns: %s", _missing)

    # Add snapshot metadata (timezone-naive local time)
    now = datetime.now()
    df['snapshot_date'] = now.strftime("%Y-%m-%d")
    df['snapshot_ts'] = now.isoformat()

    return df


# ─── Snapshot Caching ────────────────────────────────────────────

def save_snapshot(df: pd.DataFrame, date_str: Optional[str] = None) -> str:
    """Save a snapshot DataFrame to Parquet via atomic write (tmp + rename)."""
    ensure_dirs()
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    path = os.path.join(SNAPSHOT_DIR, f"snapshot_{date_str}.parquet")
    # Atomic write: write to temp file then rename to prevent corruption
    fd, tmp_path = tempfile.mkstemp(suffix='.parquet', dir=SNAPSHOT_DIR)
    try:
        os.close(fd)
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    logger.info("Saved snapshot to %s (%d rows)", path, len(df))
    return path


def load_snapshot(date_str: str) -> Optional[pd.DataFrame]:
    """Load a cached snapshot for a given date, or None if not found."""
    path = os.path.join(SNAPSHOT_DIR, f"snapshot_{date_str}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def _get_latest_cached_snapshot() -> Optional[pd.DataFrame]:
    """Fallback: load the most recent cached snapshot regardless of date."""
    available = get_available_snapshots()
    if not available:
        return None
    latest_date = available[-1]  # sorted ascending, take newest
    logger.info("Falling back to cached snapshot from %s", latest_date)
    return load_snapshot(latest_date)


def load_or_fetch_snapshot(date_str: Optional[str] = None) -> pd.DataFrame:
    """
    Load from cache if available, otherwise fetch live and cache.
    Falls back to latest cached snapshot if live fetch fails.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    # Try cached first
    cached = load_snapshot(date_str)
    if cached is not None:
        return cached

    # Try live fetch
    try:
        df = fetch_screener_snapshot()
        save_snapshot(df, date_str)
        return df
    except Exception as e:
        logger.warning("Live fetch failed (%s), trying cache fallback...", e)
        fallback = _get_latest_cached_snapshot()
        if fallback is not None:
            return fallback
        raise RuntimeError(
            f"No live data and no cached snapshots available. Original error: {e}"
        ) from e


def get_available_snapshots() -> List[str]:
    """Return sorted list of available snapshot dates (YYYY-MM-DD)."""
    if not os.path.exists(SNAPSHOT_DIR):
        return []
    files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.parquet')]
    dates = sorted(f.replace('snapshot_', '').replace('.parquet', '') for f in files)
    return dates


# ─── Sector ETF Historical Prices ───────────────────────────────

def fetch_sector_etf_history(
    period: Optional[str] = None,
    tickers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Download daily Close prices for sector ETFs + benchmark from yfinance.
    Returns a DataFrame with columns = tickers, index = dates.
    Caches result as Parquet (refreshed if >18 hours old).

    Also caches Volume data alongside prices (etf_volume.parquet) for
    use by the backtest engine's ADTV computation.
    """
    import yfinance as yf

    ensure_dirs()

    if tickers is None:
        tickers = SECTOR_ETFS + [BENCHMARK_ETF]

    if period is None:
        period = f"{BACKTEST_HISTORY_YEARS}y"

    cache_file = os.path.join(ETF_CACHE_DIR, "etf_prices.parquet")

    # Check cache freshness (use if < 18 hours old and has all tickers)
    if os.path.exists(cache_file):
        mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if (datetime.now() - mod_time) < timedelta(hours=18):
            try:
                df = pd.read_parquet(cache_file)
                if set(tickers).issubset(set(df.columns)):
                    return df
            except Exception as e:
                logger.warning("Cache read failed (%s), re-downloading.", e)

    # Download fresh data
    try:
        data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    except Exception as e:
        logger.error("yfinance download failed: %s", e)
        # Try reading stale cache as fallback
        if os.path.exists(cache_file):
            logger.info("Using stale cache as fallback.")
            return pd.read_parquet(cache_file)
        raise RuntimeError(f"Failed to download ETF data and no cache exists: {e}") from e

    # Extract Close prices and Volume
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
        volume = data['Volume'] if 'Volume' in data.columns.get_level_values(0) else pd.DataFrame()
    else:
        prices = data[['Close']].rename(columns={'Close': tickers[0]})
        volume = data[['Volume']].rename(columns={'Volume': tickers[0]}) if 'Volume' in data.columns else pd.DataFrame()

    prices = prices.dropna(how='all')

    if prices.empty:
        raise RuntimeError("yfinance returned empty price data — check ticker symbols.")

    # Cache prices
    prices.to_parquet(cache_file)

    # Cache volume alongside prices
    if not volume.empty:
        vol_cache = os.path.join(ETF_CACHE_DIR, "etf_volume.parquet")
        volume.dropna(how='all').to_parquet(vol_cache)

    return prices


def load_etf_volume() -> Optional[pd.DataFrame]:
    """Load cached ETF volume data, or None if not available."""
    vol_cache = os.path.join(ETF_CACHE_DIR, "etf_volume.parquet")
    if os.path.exists(vol_cache):
        return pd.read_parquet(vol_cache)
    return None


def fetch_single_ticker_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch OHLCV history for a single ticker via yfinance."""
    import yfinance as yf

    try:
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch {ticker} history: {e}") from e
    return data
