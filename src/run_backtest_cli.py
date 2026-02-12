
import argparse
import logging
import sys
import os
from datetime import datetime

# Ensure project root is in path
sys.path.insert(0, os.getcwd())

from src.backtest import run_backtest  # noqa: E402
from src.data_engine import fetch_sector_etf_history  # noqa: E402
from config import SECTOR_ETFS, BENCHMARK_ETF  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run backtest from CLI")
    parser.add_argument("--days", type=int, default=300, help="Number of days to backtest (default: 300)")
    parser.add_argument("--sample", action="store_true", help="Run with synthetic data for smoke testing")
    args = parser.parse_args()

    logger.info("Starting backtest worker...")

    try:
        if args.sample:
            logger.info("Running in SAMPLE mode with synthetic data.")
            # Use benchmark generation logic or simple mock
            import pandas as pd
            import numpy as np
            dates = pd.bdate_range(end=datetime.now(), periods=args.days, freq='B')
            tickers = SECTOR_ETFS + [BENCHMARK_ETF]
            prices = pd.DataFrame(index=dates, columns=tickers, dtype=float)
            for t in tickers:
                prices[t] = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.015, args.days))

            res = run_backtest(prices)
            logger.info("Backtest complete. Return: %.2f%%", res.metrics['total_return'])
        else:
            logger.info("Fetching market data...")
            prices = fetch_sector_etf_history(period=f"{args.days // 252 + 1}y")
            logger.info("Running backtest on %d rows...", len(prices))
            res = run_backtest(prices)
            logger.info("Backtest complete. Total Return: %.2f%%", res.metrics['total_return'])
            logger.info("Sharpe Ratio: %.2f", res.metrics['sharpe_ratio'])

        # In a real worker, we would save artifacts to S3/GCS here
        logger.info("Worker finished successfully.")

    except Exception as e:
        logger.error("Backtest failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
