# Developer Onboarding

## Local Development
1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Run App**: `streamlit run app.py`
3. **Run Backtest Worker**: `python -m src.run_backtest_cli --sample`

## Docker
- **Build**: `docker build -t market-dashboard .`
- **Run App**: `docker run -p 8501:8501 market-dashboard`
- **Run Worker**: `docker run market-dashboard python -m src.run_backtest_cli --days 300`

## CI/CD
- **Push to `main`**: Triggers build & push to GHCR.
- **PRs**: Run unit tests (`pytest`) and perf checks (`perf/benchmark.py`).
- **Nightly**: Runs integration tests on synthetic data.

## Scheduled Jobs
- **Daily Snapshot**: 06:00 UTC (Updates `data/snapshots/`)
- **Nightly Backtest**: 02:00 UTC (Generates reports)
