#!/usr/bin/env bash
set -euo pipefail

# IMPORT-GUARD: Enforces cross-domain import boundaries
# Usage: bash scripts/import_guard.sh
# Domain boundaries: apps ↔ ops ↔ infra

echo "IMPORT-GUARD: checking cross-domain import boundaries..."

# Define forbidden import patterns
# apps should not import from ops or infra
# ops should not import from infra
# infra should not import from apps or ops

violations=0

# Check apps/ → ops/ or infra/ imports
if [[ -d "apps" ]]; then
  if grep -r "from ops\." apps/ 2>/dev/null || grep -r "import.*ops\." apps/ 2>/dev/null; then
    echo "IMPORT-GUARD: forbidden: apps/ importing from ops/"
    violations=$((violations + 1))
  fi
  if grep -r "from infra\." apps/ 2>/dev/null || grep -r "import.*infra\." apps/ 2>/dev/null; then
    echo "IMPORT-GUARD: forbidden: apps/ importing from infra/"
    violations=$((violations + 1))
  fi
fi

# Check ops/ → infra/ imports
if [[ -d "ops" ]]; then
  if grep -r "from infra\." ops/ 2>/dev/null || grep -r "import.*infra\." ops/ 2>/dev/null; then
    echo "IMPORT-GUARD: forbidden: ops/ importing from infra/"
    violations=$((violations + 1))
  fi
fi

# Check infra/ → apps/ or ops/ imports
if [[ -d "infra" ]]; then
  if grep -r "from apps\." infra/ 2>/dev/null || grep -r "import.*apps\." infra/ 2>/dev/null; then
    echo "IMPORT-GUARD: forbidden: infra/ importing from apps/"
    violations=$((violations + 1))
  fi
  if grep -r "from ops\." infra/ 2>/dev/null || grep -r "import.*ops\." infra/ 2>/dev/null; then
    echo "IMPORT-GUARD: forbidden: infra/ importing from ops/"
    violations=$((violations + 1))
  fi
fi

if [[ $violations -gt 0 ]]; then
  echo "IMPORT-GUARD: found $violations boundary violation(s)"
  exit 1
fi

echo "IMPORT-GUARD: ✓ no cross-domain import violations"
exit 0
