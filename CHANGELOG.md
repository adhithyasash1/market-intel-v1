# Changelog

All notable changes to this project will be documented in this file.

## [v0.1.0] - 2026-02-13

### Added
- **Production Infrastructure**:
  - Dockerfile with non-root security hardening.
  - Kubernetes manifests (Deployment, HPA, NetworkPolicy, RBAC).
  - CI/CD pipelines (Tests, Build, Security Scan, Perf Check).
- **Performance**:
  - Vectorized Backtesting Engine (~100x speedup).
  - Parity verification with legacy implementation.
- **Observability**:
  - Structured logging and Prometheus metrics (`src/observability.py`).
  - Grafana dashboard template.
- **Documentation**:
  - `docs/onboarding.md` and `runbooks/incident_playbook.md`.
  - Security policy and governance docs.

### Changed
- Refactored `src/backtest.py` for performance.
- Fixed Streamlit deprecation warnings (`use_container_width`).
- Hardened security posture (removed secrets, added headers).

### Security
- Implemented `bandit` and `safety` scans in CI.
- Enforced non-root container execution.
