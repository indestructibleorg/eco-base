## Summary

<!-- One-sentence description of what this PR does -->

## Type of Change

- [ ] `feat` — New feature
- [ ] `fix` — Bug fix
- [ ] `refactor` — Code refactor (no functional change)
- [ ] `chore` — Build, CI, or dependency update
- [ ] `docs` — Documentation only
- [ ] `security` — Security fix or hardening
- [ ] `infra` — Infrastructure / Kubernetes / GitOps change

## Motivation & Context

<!-- Why is this change needed? Link to issue if applicable. Closes #<issue> -->

## Changes Made

<!-- List the key files/components changed and why -->

## Governance Checklist

> All items must be checked before requesting review.

### Code Quality
- [ ] `ci-validator` passes locally (`python3 tools/ci-validator/validate.py`)
- [ ] All `.qyaml` files include required governance blocks (`document_metadata`, `governance_info`, `registry_binding`, `vector_alignment_map`)
- [ ] No hardcoded secrets, tokens, or credentials
- [ ] No `:latest` Docker image tags

### Security
- [ ] OPA policy check passes (`conftest test --policy policy/`)
- [ ] Dockerfile does not run as `root`
- [ ] New dependencies reviewed for known CVEs

### Supply Chain
- [ ] New GitHub Actions use pinned SHA (not tag/branch)
- [ ] SBOM updated if new dependencies added (`sbom.json`)
- [ ] If new container image: image is signed with cosign

### Observability
- [ ] New services expose `/metrics` endpoint (OpenMetrics format)
- [ ] Structured logs include `traceId`, `spanId`, `service`, `action`
- [ ] SLO impact assessed (availability, P95 latency, error rate)

### Infrastructure
- [ ] Kubernetes manifests validated (`kubectl apply --dry-run=client`)
- [ ] ArgoCD app health will not be degraded
- [ ] Resource limits and requests defined for new workloads

## Test Evidence

<!-- Screenshot, log output, or test results demonstrating the change works -->

## Rollback Plan

<!-- How to revert this change if it causes issues in production -->

---

> **Auto-generated PRs** from `github-actions[bot]` (auto-fix, canary-deploy, drift-detection) are exempt from the full checklist above. They are validated by the CI pipeline automatically.
