#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Enforce Branch Protection
# =============================================================================
# 套用分支保护规则到 GitHub Repo
# 需要: gh (GitHub CLI) authenticated with repo admin rights
# =============================================================================

BRANCH_JSON="${1:-.github/branch-protection.json}"
CHECKS_JSON="${2:-.github/required-checks.json}"

if ! command -v gh >/dev/null 2>&1; then
  echo "❌ gh CLI not found"
  exit 2
fi

if [[ ! -f "$BRANCH_JSON" ]]; then
  echo "❌ missing: $BRANCH_JSON"
  exit 2
fi

if [[ ! -f "$CHECKS_JSON" ]]; then
  echo "❌ missing: $CHECKS_JSON"
  exit 2
fi

owner_repo="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
branch="$(jq -r .branch "$BRANCH_JSON")"

echo "== Enforcing branch protection for ${owner_repo}:${branch} =="

# 1) Apply base branch protection settings
gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  "/repos/${owner_repo}/branches/${branch}/protection" \
  -f enforce_admins="$(jq -r .enforce_admins "$BRANCH_JSON")" \
  -F required_linear_history="$(jq -r .required_linear_history "$BRANCH_JSON")" \
  -F allow_force_pushes="$(jq -r .allow_force_pushes "$BRANCH_JSON")" \
  -F allow_deletions="$(jq -r .allow_deletions "$BRANCH_JSON")" \
  -F required_conversation_resolution="$(jq -r .required_conversation_resolution "$BRANCH_JSON")" \
  -F required_signatures="$(jq -r .required_signatures "$BRANCH_JSON")" \
  -F lock_branch="$(jq -r .lock_branch "$BRANCH_JSON")" \
  -F allow_fork_syncing="$(jq -r .allow_fork_syncing "$BRANCH_JSON")" \
  -F required_status_checks.strict="$(jq -r .required_status_checks.strict "$BRANCH_JSON")" \
  -F required_pull_request_reviews.dismiss_stale_reviews="$(jq -r .required_pull_request_reviews.dismiss_stale_reviews "$BRANCH_JSON")" \
  -F required_pull_request_reviews.require_code_owner_reviews="$(jq -r .required_pull_request_reviews.require_code_owner_reviews "$BRANCH_JSON")" \
  -F required_pull_request_reviews.required_approving_review_count="$(jq -r .required_pull_request_reviews.required_approving_review_count "$BRANCH_JSON")" \
  -F required_pull_request_reviews.require_last_push_approval="$(jq -r .required_pull_request_reviews.require_last_push_approval "$BRANCH_JSON")" \
  -F restrictions="$(jq -c .restrictions "$BRANCH_JSON")"

echo "== Enforcing required checks =="

# 2) Apply required checks
body="$(jq -n \
  --argjson strict "$(jq '.required_status_checks.strict' "$BRANCH_JSON")" \
  --arg branch "$branch" \
  --arg owner_repo "$owner_repo" \
  --argfile cfg "$CHECKS_JSON" \
  '{
    strict: true,
    checks: ($cfg.required_status_checks | map({context: .}))
  }')"

gh api \
  --method PATCH \
  -H "Accept: application/vnd.github+json" \
  "/repos/${owner_repo}/branches/${branch}/protection/required_status_checks" \
  --input <(echo "$body")

echo "✅ Branch protection enforced for ${owner_repo}:${branch}"
