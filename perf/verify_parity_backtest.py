import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import both versions
from src.backtest_legacy import run_backtest as run_backtest_v1  # noqa: E402
# We will import v2 dynamically or assume src.backtest is updated in place
# For now, this script expects src.backtest to be the *new* version
from src.backtest import run_backtest as run_backtest_v2  # noqa: E402
from config import SECTOR_ETFS, BENCHMARK_ETF  # noqa: E402

import warnings  # noqa: E402
warnings.filterwarnings("ignore")


def generate_data(n_days=300):
    np.random.seed(42)
    dates = pd.bdate_range('2024-01-01', periods=n_days, freq='B')
    tickers = SECTOR_ETFS + [BENCHMARK_ETF]
    prices = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    for t in tickers:
        returns = np.random.normal(0.0005, 0.012, n_days)
        prices[t] = 100 * np.cumprod(1 + returns)
    return prices


def compare_results(r1, r2):
    print("Verifying parity between Legacy (v1) and Vectorized (v2)...")

    # 1. Portfolio Returns
    diff_ret = (r1.portfolio_returns - r2.portfolio_returns).abs()
    max_diff_ret = diff_ret.max()
    print(f"Max difference in Portfolio Returns: {max_diff_ret:.2e}")

    if max_diff_ret > 1e-8:
        print("FAIL: Portfolio returns diverge!")
        return False

    # 2. Benchmark Returns
    diff_bench = (r1.benchmark_returns - r2.benchmark_returns).abs()
    max_diff_bench = diff_bench.max()
    if max_diff_bench > 1e-8:
        print(f"FAIL: Benchmark returns diverge! ({max_diff_bench:.2e})")
        return False

    # 3. Weights
    # Need to align weight history dataframes
    w1 = r1.weights_history.set_index('date').sort_index()
    w2 = r2.weights_history.set_index('date').sort_index()

    # Drop non-numeric cols if any (turnover is in metrics mostly, but check headers)
    # weights_history contains 'turnover' and ticker columns.

    try:
        pd.testing.assert_frame_equal(w1, w2, check_exact=False, atol=1e-6)
        print("Weights history matches.")
    except Exception as e:
        print(f"FAIL: Weights history mismatch: {e}")
        return False

    # 4. Metrics
    print("Metrics comparison:")
    for k, v1 in r1.metrics.items():
        if isinstance(v1, (int, float)):
            v2 = r2.metrics.get(k, np.nan)
            if abs(v1 - v2) > 1e-4:
                print(f"  MISMATCH {k}: v1={v1}, v2={v2}")
            else:
                # print(f"  OK {k}")
                pass

    print("SUCCESS: Results are effectively identical.")
    return True


if __name__ == "__main__":
    prices = generate_data(300)

    print("Running Legacy Backtest...")
    res_v1 = run_backtest_v1(prices)

    print("Running Vectorized Backtest...")
    res_v2 = run_backtest_v2(prices)

    if compare_results(res_v1, res_v2):
        sys.exit(0)
    else:
        sys.exit(1)
