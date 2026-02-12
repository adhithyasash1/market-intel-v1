# Incident Playbook

## 1. App Crash / Loop
**Symptoms**: Pod restarts, `CrashLoopBackOff`, 503 errors.
**Diagnosis**:
```bash
kubectl logs -l app=market-dashboard --tail=100
kubectl describe pod -l app=market-dashboard
```
**Resolution**:
- **OOM (Exit Code 137)**: Increase memory limit in `k8s/deployment-app.yaml`.
- **Config Error**: Rollback: `kubectl rollout undo deployment/market-dashboard-app`.

## 2. Backtest Worker Failure
**Symptoms**: `market-nightly-backtest` Job fails, `HighJobFailureRate` alert.
**Diagnosis**:
```bash
kubectl get jobs
kubectl logs job/market-nightly-backtest-xxxxx
```
**Resolution**:
- Check `src/observability.py` JSON logs for stack trace.
- Retry manually: `kubectl create job --from=cronjob/market-nightly-backtest manual-retry-1`

## 3. Stale Data
**Symptoms**: `StaleSnapshot` alert fires (>26h old).
**Resolution**:
- Check `market-daily-snapshot` cronjob logs.
- Verify TradingView API connectivity.
- Fetch manually locally if needed: `python -c "from src.data_engine import load_or_fetch_snapshot; load_or_fetch_snapshot()"`
