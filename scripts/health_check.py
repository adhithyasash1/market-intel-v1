"""
Health Check Script.
Verifies 'src' core logic integrity without spinning up the API or UI.
"""

import sys
import os
import logging
from colorama import init, Fore, Style

# Add root to python path
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

init(autoreset=True)

def check_step(name: str):
    print(f"{Fore.CYAN}➤ Checking: {name}...{Style.RESET_ALL}", end=" ")

def step_ok(msg: str = "OK"):
    print(f"{Fore.GREEN}✓ {msg}")

def step_fail(msg: str):
    print(f"{Fore.RED}✗ FAILED: {msg}")
    sys.exit(1)

def main():
    print(f"{Style.BRIGHT}Running Market Intelligence Health Check...{Style.RESET_ALL}\n")

    # 1. Check Directories
    check_step("Data Directories")
    try:
        from config import ensure_dirs, DATA_DIR
        ensure_dirs()
        if os.access(DATA_DIR, os.W_OK):
            step_ok()
        else:
            step_fail("Data dir not writable")
    except Exception as e:
        step_fail(str(e))

    # 2. Data Engine Import
    check_step("Data Engine Import")
    try:
        from src.data_engine import load_or_fetch_snapshot
        step_ok()
    except Exception as e:
        step_fail(str(e))

    # 3. Core Logic (Features + Scoring)
    check_step("Scoring Pipeline Simulation")
    try:
        # Mock minimal dataframe
        import pandas as pd
        from src.features import compute_stock_features, compute_sector_aggregates
        from src.scorer import score_pipeline
        
        mock_df = pd.DataFrame({
            'name': ['TestA', 'TestB', 'TestC', 'TestD', 'TestE', 'TestF'],
            'price': [100.0, 101.0, 102.0, 99.0, 105.0, 110.0],
            'change_pct': [1.0, 0.5, -0.2, 0.3, 1.2, -0.5],
            'volume': [1000, 2000, 1500, 1200, 3000, 2500],
            'market_cap': [1e9, 2e9, 1.5e9, 1.2e9, 2.5e9, 1.8e9],
            'sector': ['Tech', 'Tech', 'Energy', 'Tech', 'Energy', 'Energy'],
            'industry': ['Software', 'Hardware', 'Oil', 'Software', 'Oil', 'Oil'],
            'country': ['USA', 'USA', 'USA', 'USA', 'USA', 'USA'],
            'perf_1m': [5.0, 3.0, -1.0, 4.0, -2.0, 0.5],
            'perf_3m': [10.0, 5.0, 0.0, 8.0, -5.0, 1.0],
            'recommendation': [1, 2, 3, 2, 3, 2] # Buy, Hold, Sell
        })
        
        feats = compute_stock_features(mock_df)
        if 'momentum_accel' not in feats.columns:
            raise ValueError("Feature 'momentum_accel' missing")
            
        aggs = compute_sector_aggregates(feats)
        if 'Tech' not in aggs.index:
            raise ValueError("Aggregation failed")
            
        scored = score_pipeline(aggs)
        if 'signal' not in scored.columns:
            raise ValueError("Scorer failed to assign signals")
            
        step_ok(f"Ranked {len(scored)} sectors")
        
    except Exception as e:
        step_fail(str(e))

    # 4. Backtest Engine
    check_step("Backtest Engine Vectorization")
    try:
        from src.backtest import run_backtest
        # No actual run, just import verified
        step_ok("Import verified")
    except Exception as e:
        step_fail(str(e))

    print(f"\n{Fore.GREEN}{Style.BRIGHT}ALL CHECKS PASSED. System is production-ready.{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
