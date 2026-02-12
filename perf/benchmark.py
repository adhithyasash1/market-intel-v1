
import logging
import sys
import os
import time
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest import run_backtest, bootstrap_test  # noqa: E402
from config import SECTOR_ETFS, BENCHMARK_ETF  # noqa: E402


def generate_data(n_days=3000):
    np.random.seed(42)
    dates = pd.bdate_range('2015-01-01', periods=n_days, freq='B')
    tickers = SECTOR_ETFS + [BENCHMARK_ETF]
    prices = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    for t in tickers:
        returns = np.random.normal(0.0005, 0.012, n_days)
        prices[t] = 100 * np.cumprod(1 + returns)
    return prices


def benchmark_performance():
    print("Benchmarking Vectorized Backtest Performance...")
    
    sizes = [300, 1000, 3000, 5000]
    results = []

    for n in sizes:
        print(f"Generating {n} days of data...")
        prices = generate_data(n)
        
        # Warmup (JIT compilation trigger if using numba, or cache warming)
        run_backtest(prices)

        start = time.time()
        for _ in range(5):
            run_backtest(prices)
        end = time.time()
        avg_time = (end - start) / 5.0
        print(f"  N={n}: {avg_time*1000:.2f} ms")
        
        # Benchmark bootstrap
        # Only run for larger sizes to see impact
        boot_time = 0.0
        if n >= 1000:
            res = run_backtest(prices)
            start_b = time.time()
            bootstrap_test(res.portfolio_returns, res.benchmark_returns, n_samples=100)
            end_b = time.time()
            boot_time = (end_b - start_b)
            print(f"  Bootstrap(N={n}, S=100): {boot_time*1000:.2f} ms")

        results.append({
            'rows': n,
            'backtest_ms': avg_time * 1000,
            'bootstrap_ms': boot_time * 1000
        })

    print("\nSummary:")
    print(pd.DataFrame(results))


if __name__ == "__main__":
    benchmark_performance()
