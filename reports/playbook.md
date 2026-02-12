# Sector Rotation Playbook — Trading & Operations Guide

**Version:** 1.0  |  **Last Updated:** Auto-generated

---

## 1. Signal Interpretation

| Signal | Action | Max Position Δ |
|--------|--------|----------------|
| **Overweight** | Increase sector exposure vs benchmark | +15% per rebalance |
| **Neutral** | Maintain benchmark weight | No action required |
| **Avoid** | Reduce sector exposure below benchmark | −15% per rebalance |

**Decision rule:** Only act on signals with composite score magnitude > 0.3 (z-score units). Marginal signals near thresholds should be treated as Neutral.

---

## 2. Rebalance Cadence & Sizing

- **Default cadence:** Monthly (first trading day)
- **Alternative:** Quarterly for lower turnover / lower-frequency mandates
- **Maximum single-period weight change:** ±20% per sector
- **Minimum holding period:** 1 month (avoid whipsaws)
- **Transaction cost budget:** 10 bps per trade (configurable in dashboard)

### Sizing Formula

```
Target_Weight(sector) = Benchmark_Weight + Tilt
  where Tilt ∈ {−15%, 0%, +15%} based on signal
  
Re-normalize all weights to sum to 100%
```

---

## 3. Risk Limits

| Limit | Value | Rationale |
|-------|-------|-----------|
| Max single-sector overweight | +20% vs benchmark | Diversification |
| Max total active risk | ±40% (sum of tilts) | Portfolio-level cap |
| Stop-loss on tilted sector | −10% from entry | Momentum reversal protection |
| Breadth filter | Require breadth > 60% for Overweight | Avoid narrow rallies |
| Correlation check | Don't overweight two sectors with ρ > 0.85 | Avoid hidden concentration |

---

## 4. Approval & Governance

### Decision Authority

| Tilt Size | Authority |
|-----------|-----------|
| ≤ 5% | Analyst self-serve |
| 5–15% | Portfolio Manager sign-off |
| > 15% | Investment Committee approval |

### Audit Trail

- All signals and inputs are timestamped and stored as Parquet snapshots
- Recommendation memos can be exported from the dashboard
- Monthly performance attribution report to be generated from backtest engine

---

## 5. Monitoring Checklist

**Daily:**
- [ ] Check dashboard for signal changes
- [ ] Verify data freshness (snapshot timestamp)
- [ ] Review any sector crossing the "Avoid" threshold

**Weekly:**
- [ ] Review sector correlation shifts
- [ ] Check breadth deterioration in Overweight sectors
- [ ] Monitor volatility regime (VIX proxy)

**Monthly:**
- [ ] Run backtest to validate signal performance
- [ ] Review hit-rate and alpha metrics
- [ ] Adjust weights if regime shift detected (risk-on → risk-off)
- [ ] Generate recommendation memo for Investment Committee

---

## 6. Regime Toggle

When market stress indicators activate (e.g., VIX > 25, cross-sector correlation spike):

1. **Switch to Risk-Aware preset** (higher volatility weight)
2. **Halve tilt sizes** (±7.5% instead of ±15%)
3. **Increase breadth threshold** to > 70%
4. **Review weekly** instead of monthly

---

*⚠️ Advisory disclaimer: Signals are quantitative inputs. All portfolio decisions require human judgment and must comply with applicable investment guidelines and regulatory requirements.*
