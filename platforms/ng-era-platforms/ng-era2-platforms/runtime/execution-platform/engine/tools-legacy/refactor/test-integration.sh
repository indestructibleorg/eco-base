#!/bin/bash
# Integration test for Auto Refactor & Evolution System
# 自動重構與演化系統整合測試

set -e  # Exit on error

echo "═══════════════════════════════════════════════════════════════════"
echo "  Auto Refactor & Evolution System - Integration Test"
echo "  自動重構與演化系統 - 整合測試"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BASE_DIR"

# Test 1: Check files exist
echo "Test 1: Checking file structure..."
FILES=(
    "config/refactor-evolution.yaml"
    "config/pipelines/refactor-evolution-pipeline.yaml"
    "tools/refactor/auto_refactor.py"
    "tools/refactor/refactor_evolution_workflow.py"
    "docs/AUTO_REFACTOR_EVOLUTION.md"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} Found: $file"
    else
        echo -e "${RED}✗${NC} Missing: $file"
        exit 1
    fi
done
echo ""

# Test 2: Check CLI help
echo "Test 2: Testing CLI help output..."
if python tools/refactor/auto_refactor.py --help > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} auto_refactor.py CLI working"
else
    echo -e "${RED}✗${NC} auto_refactor.py CLI failed"
    exit 1
fi

if python tools/refactor/refactor_evolution_workflow.py --help > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} refactor_evolution_workflow.py CLI working"
else
    echo -e "${RED}✗${NC} refactor_evolution_workflow.py CLI failed"
    exit 1
fi
echo ""

# Test 3: Quick scan
echo "Test 3: Running quick scan..."
if python tools/refactor/auto_refactor.py quick-scan > /tmp/quick_scan_output.txt 2>&1; then
    echo -e "${GREEN}✓${NC} Quick scan completed"
    
    # Check if analysis report was generated
    if ls reports/refactor-evolution/analysis_*.yaml > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Analysis report generated"
    else
        echo -e "${RED}✗${NC} Analysis report not generated"
        exit 1
    fi
else
    echo -e "${RED}✗${NC} Quick scan failed"
    cat /tmp/quick_scan_output.txt
    exit 1
fi
echo ""

# Test 4: Configuration validation
echo "Test 4: Validating YAML configuration..."
if python -c "import yaml; yaml.safe_load(open('config/refactor-evolution.yaml'))" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} refactor-evolution.yaml is valid"
else
    echo -e "${RED}✗${NC} refactor-evolution.yaml is invalid"
    exit 1
fi

if python -c "import yaml; yaml.safe_load(open('config/pipelines/refactor-evolution-pipeline.yaml'))" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} pipeline YAML is valid"
else
    echo -e "${RED}✗${NC} pipeline YAML is invalid"
    exit 1
fi
echo ""

# Test 5: Check report structure
echo "Test 5: Validating report structure..."
LATEST_REPORT=$(ls -t reports/refactor-evolution/analysis_*.yaml 2>/dev/null | head -1)
if [ -n "$LATEST_REPORT" ]; then
    if python -c "import yaml; d=yaml.safe_load(open('$LATEST_REPORT')); assert 'summary' in d and 'targets' in d" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Report structure is valid"
        echo "  Report: $LATEST_REPORT"
    else
        echo -e "${RED}✗${NC} Report structure is invalid"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠${NC} No reports found to validate"
fi
echo ""

# Summary
echo "═══════════════════════════════════════════════════════════════════"
echo -e "${GREEN}✅ All integration tests passed!${NC}"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  • Run full workflow: python tools/refactor/auto_refactor.py start"
echo "  • Check status: python tools/refactor/auto_refactor.py status"
echo "  • View documentation: docs/AUTO_REFACTOR_EVOLUTION.md"
echo ""
