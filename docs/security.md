# Security & Governance

## Data Privacy & PII
- **No PII Stored**: The application does not collect, process, or store Personally Identifiable Information (PII).
- **Data Retention**:
  - ETF Cache: 18 hours (runtime only)
  - Snapshots: 90 days (S3/Volume)
  - Logs: 30 days (Aggregated)

## Secrets Management
**NEVER** commit secrets to git. Use the following keys in GitHub Secrets and K8s Secrets:
- `TRADINGVIEW_API_KEY`
- `YFINANCE_API_KEY` (if used)
- `SENTRY_DSN`
- `SLACK_WEBHOOK_URL`
- `ARTIFACT_BUCKET_CREDS`

## Compliance
- **Scan Policy**: All PRs must pass Bandit (SAST) and Safety (SCA) scans.
- **Image Hardening**: Production images run as non-root (UID 10001) with read-only filesystem (mostly).
- **Network**: Default-deny egress policy applied in Kubernetes (`k8s/network-policy.yaml`).
