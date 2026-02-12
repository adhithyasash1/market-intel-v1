import sys
import os
import time
import pandas as pd
import numpy as np
import logging

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest import run_backtest, bootstrap_test
from config import SECTOR_ETFS, BENCHMARK_ETF

logging.basicConfig(level=logging.WARNING)

def generate_data(n_days=300):
    np.random.seed(123)
    dates = pd.bdate_range('2024-01-01', periods=n_days, freq='B')
    tickers = SECTOR_ETFS + [BENCHMARK_ETF]
    prices = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    for t in tickers:
        # Start at 100, random walk with drift
        returns = np.random.normal(0.0003, 0.015, n_days)
        prices[t] = 100 * np.cumprod(1 + returns)
    return prices

def benchmark_backtest(n_loops=10, n_days=300):
    prices = generate_data(n_days)
    print(f"Benchmarking run_backtest over {n_days} days ({n_loops} loops)...")
    
    times = []
    results = []
    
    # Warmup
    try:
        run_backtest(prices)
    except Exception as e:
        print(f"Warmup failed: {e}")
        return None

    for i in range(n_loops):
        start = time.perf_counter()
        res = run_backtest(prices)
        end = time.perf_counter()
        times.append(end - start)
        results.append(res)
        
    avg = np.mean(times)
    std = np.std(times)
    print(f"  Avg: {avg:.4f}s ± {std:.4f}s")
    return results[0]

def benchmark_bootstrap(result, n_samples=1000, n_loops=10):
    print(f"Benchmarking bootstrap_test ({n_samples} samples, {n_loops} loops)...")
    
    port_ret = result.portfolio_returns
    bench_ret = result.benchmark_returns
    
    times = []
    
    # Warmup
    bootstrap_test(port_ret, bench_ret, n_samples=10)
    
    for i in range(n_loops):
        start = time.perf_counter()
        bootstrap_test(port_ret, bench_ret, n_samples=n_samples)
        end = time.perf_counter()
        times.append(end - start)
        
    avg = np.mean(times)
    std = np.std(times)
    print(f"  Avg: {avg:.4f}s ± {std:.4f}s")

if __name__ == "__main__":
    res = benchmark_backtest(n_loops=10, n_days=300)
    if res and not res.portfolio_returns.empty:
        benchmark_bootstrap(res, n_samples=1000, n_loops=10)
    else:
        print("Backtest failed or produced no results.")
