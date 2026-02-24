#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Verify Branch Protection Drift
# =============================================================================
# 验证分支保护规则是否被篡改（漂移检测）
# 如果检测到漂移，脚本会 fail，CI 会阻断
# =============================================================================

BRANCH_JSON="${1:-.github/branch-protection.json}"
CHECKS_JSON="${2:-.github/required-checks.json}"

if ! command -v gh >/dev/null 2>&1; then
  echo "❌ gh CLI not found"
  exit 2
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "❌ jq not found"
  exit 2
fi

owner_repo="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
branch="$(jq -r .branch "$BRANCH_JSON")"

echo "== Verifying branch protection drift for ${owner_repo}:${branch} =="

current="$(gh api -H "Accept: application/vnd.github+json" "/repos/${owner_repo}/branches/${branch}/protection")"

# Check enforce_admins
want_enforce_admins="$(jq -r .enforce_admins "$BRANCH_JSON")"
got_enforce_admins="$(echo "$current" | jq -r '.enforce_admins.enabled')"
[[ "$want_enforce_admins" == "$got_enforce_admins" ]] || { echo "❌ drift: enforce_admins want=$want_enforce_admins got=$got_enforce_admins"; exit 1; }

# Check PR review requirements
want_reviews="$(jq -r '.required_pull_request_reviews.required_approving_review_count' "$BRANCH_JSON")"
got_reviews="$(echo "$current" | jq -r '.required_pull_request_reviews.required_approving_review_count')"
[[ "$want_reviews" == "$got_reviews" ]] || { echo "❌ drift: required_approving_review_count want=$want_reviews got=$got_reviews"; exit 1; }

want_codeowners="$(jq -r '.required_pull_request_reviews.require_code_owner_reviews' "$BRANCH_JSON")"
got_codeowners="$(echo "$current" | jq -r '.required_pull_request_reviews.require_code_owner_reviews')"
[[ "$want_codeowners" == "$got_codeowners" ]] || { echo "❌ drift: require_code_owner_reviews want=$want_codeowners got=$got_codeowners"; exit 1; }

want_last_push="$(jq -r '.required_pull_request_reviews.require_last_push_approval' "$BRANCH_JSON")"
got_last_push="$(echo "$current" | jq -r '.required_pull_request_reviews.require_last_push_approval')"
[[ "$want_last_push" == "$got_last_push" ]] || { echo "❌ drift: require_last_push_approval want=$want_last_push got=$got_last_push"; exit 1; }

# Check required status checks list
required_now="$(gh api -H "Accept: application/vnd.github+json" "/repos/${owner_repo}/branches/${branch}/protection/required_status_checks" | jq -r '.checks[]?.context' | sort)"
required_want="$(jq -r '.required_status_checks[]' "$CHECKS_JSON" | sort)"

diff_out="$(diff -u <(echo "$required_want") <(echo "$required_now") || true)"
if [[ -n "$diff_out" ]]; then
  echo "❌ drift: required checks mismatch"
  echo "$diff_out"
  exit 1
fi

echo "✅ No drift detected for ${owner_repo}:${branch}"
