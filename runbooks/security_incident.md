# Security Incident Response

## 1. Leaked Secret
**Detect**: Secret found in git history or logs.
**Response**:
1. **Revoke** the credential immediately at the provider (e.g., TradingView, AWS).
2. **Rotate** the secret in GitHub Secrets and K8s Secrets.
3. **Purge** history if committed:
   ```bash
   # Use BFG Repo-Cleaner or git filter-repo
   bfg --delete-files .env
   git reflog expire --expire=now --all && git gc --prune=now --aggressive
   ```

## 2. Vulnerable Dependency
**Detect**: Dependabot alert or Safety check failure.
**Response**:
1. Create a hotfix branch.
2. Update `requirements.txt`.
3. Verify with `safety check`.
4. Deploy immediately.

## 3. Compromised Container
**Detect**: Unexpected egress traffic (blocked by NetworkPolicy) or file modification alerts.
**Response**:
1. **Isolate**: Label pod to cut traffic (if using isolation policy) or just kill it.
   ```bash
   kubectl label pod <pod-name> isolation=true
   ```
2. **Capture Evidence**:
   ```bash
   kubectl logs <pod-name> > evidence.log
   ```
3. **Kill**: `kubectl delete pod <pod-name>`.
