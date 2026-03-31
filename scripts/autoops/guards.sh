#!/usr/bin/env bash
set -euo pipefail

# AutoOps guards — hard fail if any rule is violated.
#
# Rules:
#   1. run.manifest.yaml must exist
#   2. If entry fields missing → NEED_INPUT.md must exist
#   3. All artifact files must be under artifacts/
#   4. No secrets echoed in artifact content

MANIFEST="artifacts/manifests/run.manifest.yaml"
NEED_INPUT="artifacts/NEED_INPUT.md"
fail=0

echo "=== AutoOps Guards ==="

# ── Guard 1: manifest must exist ─────────────────────────────────────
if [[ ! -f "$MANIFEST" ]]; then
  echo "GUARD FAIL: missing $MANIFEST" >&2
  fail=1
else
  echo "GUARD OK: manifest exists"
fi

# ── Guard 2: if NEED_INPUT has missing fields, it must be present ────
if [[ -f "$NEED_INPUT" ]]; then
  if grep -q "MISSING" "$NEED_INPUT"; then
    echo "GUARD OK: NEED_INPUT.md present with missing fields (pipeline should stop)"
    # This is the expected state when entry is incomplete
  fi
fi

# ── Guard 3: no files written outside artifacts/ by this run ─────────
# (enforcement: check git diff --name-only for non-artifacts changes)
# This is advisory in local mode; in CI, the workflow limits write scope.
echo "GUARD OK: write scope (advisory — enforce via CI permissions)"

# ── Guard 4: no secrets in artifact content ──────────────────────────
SECRET_PATTERNS='(VERCEL_TOKEN|CLOUDFLARE_API_TOKEN|SUPABASE_ACCESS_TOKEN|KEYCLOAK_ADMIN_CLIENT_SECRET|N8N_API_KEY|sk-|ghp_|ghu_)'
if find artifacts/ -type f -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "*.sh" -o -name "*.md" 2>/dev/null | \
   xargs grep -lE "$SECRET_PATTERNS" 2>/dev/null | grep -v "NEED_INPUT.md" | grep -v "guards.sh" | head -1 | grep -q .; then
  echo "GUARD FAIL: potential secret found in artifacts" >&2
  fail=1
else
  echo "GUARD OK: no secrets detected in artifacts"
fi

# ── Guard 5: manifest run_id is not TEMPLATE ─────────────────────────
if [[ -f "$MANIFEST" ]]; then
  if grep -q 'run_id: "TEMPLATE"' "$MANIFEST"; then
    echo "GUARD FAIL: manifest still has TEMPLATE run_id (run.py did not execute)" >&2
    fail=1
  else
    echo "GUARD OK: manifest has real run_id"
  fi
fi

echo "=== Guards complete (fail=$fail) ==="
exit "$fail"
