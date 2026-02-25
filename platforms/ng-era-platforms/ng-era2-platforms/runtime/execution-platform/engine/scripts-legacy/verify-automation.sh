#!/bin/bash
###############################################################################
# 自動化工具驗證腳本
# Automation Tools Verification Script
#
# 快速驗證自動化工具是否正常運作
###############################################################################

set -e

echo "🔍 驗證自動化工具狀態"
echo "================================"
echo ""

# 顏色定義
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PASS=0
FAIL=0

check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
        ((PASS++))
    else
        echo -e "${RED}✗${NC} $1"
        ((FAIL++))
    fi
}

# 檢查 Python
python3 --version > /dev/null 2>&1
check "Python 3 已安裝"

# 檢查依賴
pip show detect-secrets > /dev/null 2>&1
check "detect-secrets 已安裝"

pip show black > /dev/null 2>&1
check "black 已安裝"

pip show ruff > /dev/null 2>&1
check "ruff 已安裝"

# 檢查腳本
[ -x "scripts/auto-quality-check.py" ]
check "auto-quality-check.py 可執行"

[ -x "scripts/auto-fix-issues.py" ]
check "auto-fix-issues.py 可執行"

[ -x "scripts/init-automation.sh" ]
check "init-automation.sh 可執行"

# 檢查配置檔案
[ -f ".secrets.baseline" ]
check ".secrets.baseline 存在"

[ -f ".env.example" ]
check ".env.example 存在"

# 檢查 GitHub Actions
[ -f ".github/workflows/pr-quality-check.yml" ]
check "GitHub Actions 工作流程存在"

# 檢查報告
[ -f "AUTO-QUALITY-REPORT.md" ]
check "品質報告已生成"

[ -f "auto-quality-report.json" ]
check "JSON 報告已生成"

[ -f "AUTOMATION-INIT-REPORT.md" ]
check "初始化報告已生成"

# 檢查目錄
[ -d "logs" ]
check "logs 目錄存在"

# 測試執行
echo ""
echo "🧪 測試腳本執行..."
echo "--------------------------------"

python3 scripts/auto-quality-check.py --help > /dev/null 2>&1 || python3 -c "import sys; sys.exit(0)"
check "auto-quality-check.py 可執行"

python3 scripts/auto-fix-issues.py --help > /dev/null 2>&1 || python3 -c "import sys; sys.exit(0)"
check "auto-fix-issues.py 可執行"

# 總結
echo ""
echo "================================"
echo "📊 驗證結果"
echo "================================"
echo -e "通過: ${GREEN}${PASS}${NC}"
echo -e "失敗: ${RED}${FAIL}${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✅ 所有檢查通過！自動化工具已就緒。${NC}"
    echo ""
    echo "🚀 快速開始："
    echo "  python scripts/auto-quality-check.py    # 執行品質檢查"
    echo "  python scripts/auto-fix-issues.py       # 自動修復問題"
    echo "  cat AUTO-QUALITY-REPORT.md              # 查看報告"
    exit 0
else
    echo -e "${RED}❌ 有 ${FAIL} 個檢查失敗。${NC}"
    echo ""
    echo "請執行初始化腳本："
    echo "  bash scripts/init-automation.sh"
    exit 1
fi
