"""
Data Engine — Snapshot pipeline and historical price fetcher.
Fetches live data from TradingView/yfinance with robust error handling.
"""

import os
import tempfile
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import yfinance as yf
from tvscreener import StockScreener, StockField, IndexSymbol

from config import (
    SNAPSHOT_DIR, ETF_CACHE_DIR,
    SECTOR_ETFS, BENCHMARK_ETF, BACKTEST_HISTORY_YEARS,
    ensure_dirs,
)

logger = logging.getLogger(__name__)

class DataFetchError(Exception):
    """Custom exception for data retrieval failures."""
    pass

# ─── Column Name Mapping ─────────────────────────────────────────

_COLUMN_MAP: Dict[str, str] = {
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
    'Simple Moving Average (50': 'sma_50',
    'Simple Moving Average (200': 'sma_200',
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
    """
    mapped = {}
    used_new_names = set()

    # Standard mapping
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

    # Volatility variants
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
    """
    try:
        ss = StockScreener()
        ss.set_index(IndexSymbol.SP500)
        ss.set_range(0, max_results)
        
        # Select fields (kept same as original)
        ss.select(
            StockField.NAME, StockField.PRICE, StockField.CHANGE_PERCENT, 
            StockField.VOLUME, StockField.MARKET_CAPITALIZATION, 
            StockField.SECTOR, StockField.INDUSTRY, StockField.COUNTRY,
            StockField.RELATIVE_STRENGTH_INDEX_14, StockField.MACD_LEVEL_12_26, 
            StockField.MACD_SIGNAL_12_26, StockField.SIMPLE_MOVING_AVERAGE_50, 
            StockField.SIMPLE_MOVING_AVERAGE_200, StockField.AVERAGE_TRUE_RANGE_14,
            StockField.WEEKLY_PERFORMANCE, StockField.MONTHLY_PERFORMANCE, 
            StockField.MONTH_PERFORMANCE_3, StockField.MONTH_PERFORMANCE_6, 
            StockField.YTD_PERFORMANCE, StockField.YEARLY_PERFORMANCE,
            StockField.RECOMMENDATION_MARK, StockField.AVERAGE_VOLUME_10_DAY, 
            StockField.AVERAGE_VOLUME_30_DAY, StockField.RELATIVE_VOLUME, 
            StockField.VOLATILITY, StockField.VOLATILITY_WEEK, StockField.VOLATILITY_MONTH,
        )
        ss.sort_by(StockField.MARKET_CAPITALIZATION, ascending=False)
        
        df = ss.get()
        
    except Exception as e:
        logger.error(f"TradingView API failed: {e}")
        # Use fallback if appropriate, but here we raise DataFetchError
        raise DataFetchError(f"Failed to fetch screener snapshot: {str(e)}") from e

    if df.empty:
        raise DataFetchError("TradingView returned empty snapshot.")

    df = _standardize_columns(df)
    
    # Metadata
    now = datetime.now()
    df['snapshot_date'] = now.strftime("%Y-%m-%d")
    df['snapshot_ts'] = now.isoformat()
    
    return df

# ─── Snapshot Caching ────────────────────────────────────────────

def save_snapshot(df: pd.DataFrame, date_str: Optional[str] = None) -> str:
    """Save a snapshot DataFrame to Parquet via atomic write."""
    ensure_dirs()
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    path = os.path.join(SNAPSHOT_DIR, f"snapshot_{date_str}.parquet")
    
    # Atomic write
    fd, tmp_path = tempfile.mkstemp(suffix='.parquet', dir=SNAPSHOT_DIR)
    try:
        os.close(fd)
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise IOError(f"Failed to save snapshot: {e}") from e
        
    logger.info(f"Saved snapshot to {path}")
    return path 

def load_snapshot(date_str: str) -> Optional[pd.DataFrame]:
    """Load a cached snapshot for a given date."""
    path = os.path.join(SNAPSHOT_DIR, f"snapshot_{date_str}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None

def _get_latest_cached_snapshot() -> Optional[pd.DataFrame]:
    """Fallback: load the most recent cached snapshot."""
    available = get_available_snapshots()
    if not available:
        return None
    latest_date = available[-1]
    logger.info(f"Using fallback cache from {latest_date}")
    return load_snapshot(latest_date)

def load_or_fetch_snapshot(date_str: Optional[str] = None) -> pd.DataFrame:
    """
    Load from cache if available, otherwise fetch live and cache.
    Falls back to cache on failure.
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
    except DataFetchError as e:
        logger.warning(f"Live fetch failed: {e}")
        fallback = _get_latest_cached_snapshot()
        if fallback is not None:
            return fallback
        raise e  # Propagate error if no fallback

def get_available_snapshots() -> List[str]:
    """Return sorted list of available snapshot dates."""
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
    Download daily Close prices + Volume for sector ETFs.
    """
    ensure_dirs()
    if tickers is None:
        tickers = SECTOR_ETFS + [BENCHMARK_ETF]
    if period is None:
        period = f"{BACKTEST_HISTORY_YEARS}y"

    cache_file = os.path.join(ETF_CACHE_DIR, "etf_prices.parquet")
    vol_cache = os.path.join(ETF_CACHE_DIR, "etf_volume.parquet")

    # Check cache validity (18h)
    if os.path.exists(cache_file):
        mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if (datetime.now() - mod_time) < timedelta(hours=18):
            try:
                df = pd.read_parquet(cache_file)
                if set(tickers).issubset(set(df.columns)):
                    return df
            except Exception:
                pass # corrupted cache, re-fetch

    # Live Fetch
    try:
        data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    except Exception as e:
        logger.error(f"yfinance failed: {e}")
        # Try stale cache
        if os.path.exists(cache_file):
            logger.warning("Using stale ETF cache.")
            return pd.read_parquet(cache_file)
        raise DataFetchError(f"Failed to fetch ETF history: {e}") from e

    if data.empty:
        raise DataFetchError("yfinance returned empty data.")

    # Extract Close and Volume
    # Handle both MultiIndex and single ticker cases
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
        volume = data['Volume'] if 'Volume' in data.columns.get_level_values(0) else pd.DataFrame()
    else:
        # Fallback for single ticker or flat columns (yfinance structure varies)
        if 'Close' in data.columns:
            prices = data[['Close']]
        elif len(tickers) == 1:
            prices = data # assume single series
        else:
            # Try to infer or fail
             prices = data

        if 'Volume' in data.columns:
            volume = data[['Volume']]
        else:
            volume = pd.DataFrame()

    prices = prices.dropna(how='all')
    
    # Save caches
    try:
        prices.to_parquet(cache_file)
        if not volume.empty:
            volume.dropna(how='all').to_parquet(vol_cache)
    except Exception as e:
        logger.warning(f"Failed to write ETF cache: {e}")

    return prices

def load_etf_volume() -> Optional[pd.DataFrame]:
    """Load cached ETF volume data."""
    vol_cache = os.path.join(ETF_CACHE_DIR, "etf_volume.parquet")
    if os.path.exists(vol_cache):
        return pd.read_parquet(vol_cache)
    return None

def fetch_single_ticker_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        return yf.download(ticker, period=period, auto_adjust=True, progress=False)
    except Exception as e:
        raise DataFetchError(f"Failed to fetch history for {ticker}: {e}") from e
