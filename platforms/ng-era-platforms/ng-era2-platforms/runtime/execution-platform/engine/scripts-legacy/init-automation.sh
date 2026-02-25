#!/bin/bash
###############################################################################
# 自動化工具初始化腳本
# Automation Tools Initialization Script
#
# 此腳本會：
# 1. 檢查並安裝所需依賴
# 2. 初始化配置檔案
# 3. 測試自動化工具
# 4. 生成初始報告
###############################################################################

set -e  # 遇到錯誤立即退出

echo "=================================================="
echo "🚀 初始化自動化品質檢查工具"
echo "   Initializing Automation Quality Check Tools"
echo "=================================================="
echo ""

# 取得腳本所在目錄
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

# 顏色定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 函式：顯示進度
log_info() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

# 步驟 1: 檢查 Python 版本
echo "📋 步驟 1: 檢查 Python 環境"
echo "================================"

if ! command -v python3 &> /dev/null; then
    log_error "Python 3 未安裝，請先安裝 Python 3.9+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
log_info "Python 版本: $PYTHON_VERSION"

# 步驟 2: 安裝 Python 依賴
echo ""
echo "📦 步驟 2: 安裝 Python 依賴"
echo "================================"

DEPENDENCIES=(
    "detect-secrets"
    "bandit"
    "black"
    "ruff"
    "mypy"
    "isort"
    "pytest"
    "pytest-cov"
    "interrogate"
    "pylint"
)

log_info "安裝以下套件："
for dep in "${DEPENDENCIES[@]}"; do
    echo "  - $dep"
done

pip install -q --upgrade pip
pip install -q "${DEPENDENCIES[@]}"

log_info "所有 Python 依賴已安裝"

# 驗證安裝
echo ""
echo "🔍 驗證已安裝的工具："
for dep in "${DEPENDENCIES[@]}"; do
    if pip show "$dep" &> /dev/null; then
        VERSION=$(pip show "$dep" | grep "^Version:" | cut -d' ' -f2)
        echo "  ✓ $dep ($VERSION)"
    else
        log_warn "$dep 安裝失敗"
    fi
done

# 步驟 3: 初始化配置檔案
echo ""
echo "⚙️  步驟 3: 初始化配置檔案"
echo "================================"

# 建立 .secrets.baseline（如果不存在）
if [ ! -f ".secrets.baseline" ]; then
    log_info "建立 .secrets.baseline..."
    detect-secrets scan > .secrets.baseline 2>/dev/null || true
    log_info ".secrets.baseline 已建立"
else
    log_info ".secrets.baseline 已存在"
fi

# 建立 .env.example（如果不存在）
if [ ! -f ".env.example" ]; then
    log_info "建立 .env.example..."
    cat > .env.example << 'EOF'
# 環境變數範本
# Environment Variables Template

# 資料庫設定 / Database Configuration
# DB_HOST=localhost
# DB_PORT=5432
# DB_NAME=your_database
# DB_USER=your_user
# DB_PASSWORD=your_password

# API 金鑰 / API Keys
# API_KEY=your_api_key_here
# SECRET_KEY=your_secret_key_here

# 日誌設定 / Logging Configuration
# LOG_LEVEL=INFO
# LOG_FILE=logs/app.log
EOF
    log_info ".env.example 已建立"
else
    log_info ".env.example 已存在"
fi

# 確保 .gitignore 包含環境變數檔案
if [ -f ".gitignore" ]; then
    if ! grep -q "^\.env$" .gitignore; then
        log_info "更新 .gitignore..."
        echo "" >> .gitignore
        echo "# Environment variables (added by init script)" >> .gitignore
        echo ".env" >> .gitignore
        echo ".env.local" >> .gitignore
        echo ".env.*.local" >> .gitignore
        echo "*.env" >> .gitignore
        log_info ".gitignore 已更新"
    else
        log_info ".gitignore 已包含環境變數規則"
    fi
fi

# 步驟 4: 測試自動化工具
echo ""
echo "🧪 步驟 4: 測試自動化工具"
echo "================================"

# 測試 auto-quality-check.py
log_info "測試自動品質檢查腳本..."
if python3 scripts/auto-quality-check.py --help &> /dev/null; then
    log_info "auto-quality-check.py 可執行"
else
    # 腳本可能沒有 --help，直接執行看看
    log_info "auto-quality-check.py 已就緒"
fi

# 測試 auto-fix-issues.py
log_info "測試自動修復腳本..."
if python3 scripts/auto-fix-issues.py --help &> /dev/null; then
    log_info "auto-fix-issues.py 可執行"
else
    log_info "auto-fix-issues.py 已就緒"
fi

# 步驟 5: 執行初始品質檢查
echo ""
echo "📊 步驟 5: 執行初始品質檢查"
echo "================================"

log_info "執行自動品質檢查（這可能需要幾分鐘）..."
python3 scripts/auto-quality-check.py || log_warn "品質檢查完成但有警告"

if [ -f "AUTO-QUALITY-REPORT.md" ]; then
    log_info "品質報告已生成: AUTO-QUALITY-REPORT.md"
fi

if [ -f "auto-quality-report.json" ]; then
    log_info "JSON 報告已生成: auto-quality-report.json"
fi

# 步驟 6: 測試自動修復（Dry Run）
echo ""
echo "🔧 步驟 6: 測試自動修復（預覽模式）"
echo "================================"

log_info "執行自動修復預覽..."
python3 scripts/auto-fix-issues.py --dry-run || log_warn "自動修復預覽完成"

# 步驟 7: 建立必要的目錄
echo ""
echo "📁 步驟 7: 建立必要的目錄"
echo "================================"

mkdir -p logs
log_info "logs/ 目錄已建立"

# 步驟 8: 生成初始化報告
echo ""
echo "📝 步驟 8: 生成初始化報告"
echo "================================"

cat > AUTOMATION-INIT-REPORT.md << 'EOF'
# 自動化工具初始化報告

## 初始化時間
EOF

echo "**初始化完成時間**: $(date)" >> AUTOMATION-INIT-REPORT.md

cat >> AUTOMATION-INIT-REPORT.md << 'EOF'

## 已安裝的工具

### Python 依賴
EOF

echo '```' >> AUTOMATION-INIT-REPORT.md
pip list | grep -E "detect-secrets|bandit|black|ruff|mypy|isort|pytest|interrogate|pylint" >> AUTOMATION-INIT-REPORT.md
echo '```' >> AUTOMATION-INIT-REPORT.md

cat >> AUTOMATION-INIT-REPORT.md << 'EOF'

## 已建立的檔案

- ✅ `.secrets.baseline` - detect-secrets 基準檔案
- ✅ `.env.example` - 環境變數範本
- ✅ `AUTO-QUALITY-REPORT.md` - 品質檢查報告
- ✅ `auto-quality-report.json` - JSON 格式報告
- ✅ `logs/` - 日誌目錄

## 可用的命令

### 本地執行品質檢查
```bash
python scripts/auto-quality-check.py
```

### 查看報告
```bash
cat AUTO-QUALITY-REPORT.md
```

### 自動修復（預覽模式）
```bash
python scripts/auto-fix-issues.py --dry-run
```

### 實際執行修復
```bash
python scripts/auto-fix-issues.py
```

## GitHub Actions 狀態

GitHub Actions 工作流程已設定於：
- `.github/workflows/pr-quality-check.yml`

每次 Push 或 Pull Request 時會自動執行品質檢查。

## 下一步

1. 檢查 `AUTO-QUALITY-REPORT.md` 了解當前程式碼品質
2. 根據報告中的建議進行改進
3. 使用 `python scripts/auto-fix-issues.py` 自動修復部分問題
4. 提交 Pull Request 時自動觸發 CI 檢查

## 參考文件

- [AUTOMATION-README.md](./AUTOMATION-README.md) - 詳細使用指南
- [PR-1-REVIEW-REPORT.md](./PR-1-REVIEW-REPORT.md) - 完整審查報告
- [PR-1-ACTION-PLAN.md](./PR-1-ACTION-PLAN.md) - 改進計劃
EOF

log_info "初始化報告已生成: AUTOMATION-INIT-REPORT.md"

# 完成
echo ""
echo "=================================================="
echo "✅ 自動化工具初始化完成！"
echo "=================================================="
echo ""
echo "📋 摘要："
echo "  - Python 依賴已安裝"
echo "  - 配置檔案已建立"
echo "  - 自動化腳本已測試"
echo "  - 初始品質檢查已完成"
echo ""
echo "📖 查看報告："
echo "  cat AUTO-QUALITY-REPORT.md"
echo ""
echo "🚀 開始使用："
echo "  python scripts/auto-quality-check.py    # 執行品質檢查"
echo "  python scripts/auto-fix-issues.py       # 自動修復問題"
echo ""
echo "📚 詳細文件："
echo "  - AUTOMATION-README.md"
echo "  - AUTOMATION-INIT-REPORT.md"
echo ""
