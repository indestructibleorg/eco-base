#!/bin/bash
# MachineNativeOps 重構腳本 - 安全執行版本
# 使用方法：./scripts/safe-refactor.sh [phase]
# 例如：./scripts/safe-refactor.sh phase1

set -e  # 遇到錯誤立即退出

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日誌函數
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 創建備份
create_backup() {
    log_info "創建備份..."
    BACKUP_DIR="backup/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # 創建 Git 標籤
    TAG_NAME="pre-refactor-$(date +%Y%m%d_%H%M%S)"
    git tag -a "$TAG_NAME" -m "Pre-refactoring backup: $(date)"
    log_info "Git 標籤：$TAG_NAME"
    
    # 備份關鍵目錄
    if [ -d "workspace/src/core" ]; then
        cp -r workspace/src/core "$BACKUP_DIR/"
        log_info "已備份：workspace/src/core/"
    fi
    
    if [ -d "workspace/src/contracts" ]; then
        cp -r workspace/src/contracts "$BACKUP_DIR/"
        log_info "已備份：workspace/src/contracts/"
    fi
    
    if [ -d "workspace/src/web" ]; then
        cp -r workspace/src/web "$BACKUP_DIR/"
        log_info "已備份：workspace/src/web/"
    fi
    
    # 備份配置文件
    cp package.json "$BACKUP_DIR/" 2>/dev/null || true
    cp .gitignore "$BACKUP_DIR/" 2>/dev/null || true
    
    log_info "備份完成：$BACKUP_DIR"
    echo "$BACKUP_DIR" > .last-backup
}

# Phase 1: 低風險清理
phase1() {
    log_info "開始 Phase 1：低風險清理..."
    
    # 1. 重命名中文目錄
    if [ -d "workspace/src/代碼聖殿" ]; then
        log_step "重命名中文目錄..."
        
        # 檢查引用
        log_info "檢查引用..."
        grep -r "代碼聖殿" workspace/src/ > "$BACKUP_DIR/chinese-dir-references.txt" || true
        
        if [ -s "$BACKUP_DIR/chinese-dir-references.txt" ]; then
            log_warn "發現 $(wc -l < "$BACKUP_DIR/chinese-dir-references.txt") 個引用"
        fi
        
        # 重命名
        mv "workspace/src/代碼聖殿" "workspace/src/sacred-modules"
        log_info "✅ 已重命名：代码圣殿 → sacred-modules"
        
        # 更新引用
        log_info "更新引用..."
        find workspace/src/ -name "*.py" -exec sed -i 's/代碼聖殿/sacred-modules/g' {} \;
        find workspace/src/ -name "*.md" -exec sed -i 's/代碼聖殿/sacred-modules/g' {} \;
        find workspace/src/ -name "*.json" -exec sed -i 's/代碼聖殿/sacred-modules/g' {} \;
        
        log_info "✅ 引用已更新"
    else
        log_warn "中文目錄不存在，跳過"
    fi
    
    # 2. 清理構建產物
    if [ -d "workspace/src/machinenativeops.egg-info" ]; then
        log_step "清理構建產物..."
        rm -rf workspace/src/machinenativeops.egg-info
        
        # 更新 .gitignore
        if ! grep -q "*.egg-info/" .gitignore; then
            echo "*.egg-info/" >> .gitignore
        fi
        if ! grep -q "__pycache__/" .gitignore; then
            echo "__pycache__/" >> .gitignore
        fi
        if ! grep -q "*.pyc" .gitignore; then
            echo "*.pyc" >> .gitignore
        fi
        
        log_info "✅ 已清理構建產物並更新 .gitignore"
    fi
    
    # 3. 處理 _scratch/ 目錄
    if [ -d "workspace/src/_scratch" ]; then
        log_step "重命名 _scratch → _sandbox..."
        mv "workspace/src/_scratch" "workspace/src/_sandbox"
        
        # 創建說明文件
        cat > workspace/src/_sandbox/README.md << 'EOF'
# 🚧 Sandbox Environment

此目錄用於存放實驗性代碼和臨時測試。

## 使用規則
- 定期清理（每季度）
- 禁止從此目錄導入代碼到生產環境
- 敏感信息不應在此目錄中
EOF
        
        log_info "✅ 已重命名：_scratch → _sandbox"
    else
        log_warn "_scratch/ 不存在，跳過"
    fi
    
    log_info "Phase 1 完成！"
}

# Phase 2: 解決重複
phase2() {
    log_info "開始 Phase 2：解決重複..."
    
    # 1. 合併 contracts/
    if [ -d "workspace/src/core/contracts" ]; then
        log_warn "發現重複的 contracts/ 目錄"
        
        # 檢查引用
        log_info "檢查 contracts 引用..."
        grep -r "from core\.contracts" workspace/src/ > "$BACKUP_DIR/contracts-imports.txt" || true
        grep -r "import.*core\.contracts" workspace/src/ >> "$BACKUP_DIR/contracts-imports.txt" || true
        
        if [ -s "$BACKUP_DIR/contracts-imports.txt" ]; then
            log_info "發現 $(wc -l < "$BACKUP_DIR/contracts-imports.txt") 個引用"
        fi
        
        # 比較差異
        log_info "比較 contracts/ 目錄差異..."
        diff -r workspace/src/contracts/ workspace/src/core/contracts/ > "$BACKUP_DIR/contracts-diff.txt" || true
        
        # 備份並刪除
        log_info "備份舊目錄..."
        mv workspace/src/core/contracts workspace/src/core/contracts.backup
        
        # 更新引用
        log_info "更新 contracts 引用..."
        find workspace/src/ -name "*.py" -exec sed -i 's/from core\.contracts/from contracts/g' {} \;
        find workspace/src/ -name "*.py" -exec sed -i 's/import core\.contracts/import contracts/g' {} \;
        
        log_info "✅ 已備份並刪除 core/contracts/"
        log_warn "請人工審查並更新 contracts 相關的導入引用"
        log_info "引用清單已保存到：$BACKUP_DIR/contracts-imports.txt"
    else
        log_warn "core/contracts/ 不存在，跳過"
    fi
    
    # 2. 整合 contract_service/
    if [ -d "workspace/src/core/contract_service" ]; then
        log_step "整合 contract_service/ 到 services/..."
        
        # 檢查服務引用
        grep -r "contract_service" workspace/src/ > "$BACKUP_DIR/service-imports.txt" || true
        grep -r "contract-service" workspace/src/ >> "$BACKUP_DIR/service-imports.txt" || true
        
        # 創建新位置
        mkdir -p workspace/src/services/contract-service
        cp -r workspace/src/core/contract_service/* workspace/src/services/contract-service/
        
        # 備份舊位置
        mv workspace/src/core/contract_service workspace/src/core/contract_service.backup
        
        log_info "✅ 已整合 contract_service/ 到 services/contract-service/"
        log_warn "請更新服務發現配置"
    else
        log_warn "core/contract_service/ 不存在，跳過"
    fi
    
    # 3. 整合前端
    if [ -d "workspace/src/web" ]; then
        log_step "整合 web/ 到 apps/..."
        
        mkdir -p workspace/src/apps/web
        cp -r workspace/src/web/* workspace/src/apps/web/
        
        mv workspace/src/web workspace/src/web.backup
        
        log_info "✅ 已整合 web/ 到 apps/web/"
        log_warn "請更新構建腳本和部署配置"
    else
        log_warn "web/ 不存在，跳過"
    fi
    
    log_info "Phase 2 完成！請檢查並更新相關配置。"
}

# Phase 3: 整理散落檔案
phase3() {
    log_info "開始 Phase 3：整理散落檔案..."
    
    # 創建目標目錄
    mkdir -p workspace/src/core/ai_engine
    mkdir -p workspace/src/core/automation
    mkdir -p workspace/src/core/engine
    
    # 移動檔案
    if [ -f "workspace/src/core/ai_decision_engine.py" ]; then
        mv workspace/src/core/ai_decision_engine.py workspace/src/core/ai_engine/
        log_info "已移動：ai_decision_engine.py → ai_engine/"
    fi
    
    if ls workspace/src/core/auto_*.py 1> /dev/null 2>&1; then
        mv workspace/src/core/auto_*.py workspace/src/core/automation/
        log_info "已移動：auto_*.py → automation/"
    fi
    
    if [ -f "workspace/src/core/context_understanding_engine.py" ]; then
        mv workspace/src/core/context_understanding_engine.py workspace/src/core/ai_engine/
        log_info "已移動：context_understanding_engine.py → ai_engine/"
    fi
    
    if [ -f "workspace/src/core/contract_engine.py" ]; then
        mv workspace/src/core/contract_engine.py workspace/src/core/engine/
        log_info "已移動：contract_engine.py → engine/"
    fi
    
    # 創建 __init__.py 檔案
    touch workspace/src/core/ai_engine/__init__.py
    touch workspace/src/core/automation/__init__.py
    touch workspace/src/core/engine/__init__.py
    
    log_info "✅ 已整理核心檔案"
    log_warn "請更新 Python 導入路徑"
    
    # 保存導入更新建議
    cat > "$BACKUP_DIR/import-updates.txt" << 'EOF'
# 需要更新的導入路徑

# 舊導入 → 新導入
from core.ai_decision_engine import AIDecisionEngine → from core.ai_engine.ai_decision_engine import AIDecisionEngine
from core.auto_* import * → from core.automation.auto_* import *
from core.context_understanding_engine import ContextUnderstandingEngine → from core.ai_engine.context_understanding_engine import ContextUnderstandingEngine
from core.contract_engine import ContractEngine → from core.engine.contract_engine import ContractEngine
EOF
    
    log_info "Phase 3 完成！"
}

# 驗證步驟
validate() {
    log_info "運行驗證..."
    
    # 檢查關鍵模組
    if [ -f "workspace/scripts/validate-structure.sh" ]; then
        log_info "運行結構驗證..."
        bash workspace/scripts/validate-structure.sh
    fi
    
    # 檢查目錄結構
    log_info "檢查目錄結構..."
    echo ""
    echo "=== workspace/src/ 結構 ==="
    ls -la workspace/src/ | head -20
    echo ""
    
    # 檢查 Git 狀態
    log_info "檢查 Git 狀態..."
    git status --short
    
    log_info "驗證完成！請檢查上述輸出。"
}

# 回滾腳本
rollback() {
    log_warn "開始回滾..."
    
    # 讀取最後的備份目錄
    if [ -f ".last-backup" ]; then
        BACKUP_DIR=$(cat .last-backup)
        log_info "使用備份：$BACKUP_DIR"
    else
        log_error "未找到備份目錄"
        exit 1
    fi
    
    # 恢復備份
    if [ -d "$BACKUP_DIR/core" ]; then
        log_info "恢復 workspace/src/core/..."
        rm -rf workspace/src/core
        cp -r "$BACKUP_DIR/core" workspace/src/
    fi
    
    if [ -d "$BACKUP_DIR/contracts" ]; then
        log_info "恢復 workspace/src/contracts/..."
        rm -rf workspace/src/contracts
        cp -r "$BACKUP_DIR/contracts" workspace/src/
    fi
    
    if [ -d "$BACKUP_DIR/web" ]; then
        log_info "恢復 workspace/src/web/..."
        rm -rf workspace/src/web
        cp -r "$BACKUP_DIR/web" workspace/src/
    fi
    
    # Git reset
    log_warn "執行 Git reset..."
    git reset --hard HEAD
    
    log_warn "回滾完成！"
}

# 主函數
main() {
    case "$1" in
        phase1)
            create_backup
            phase1
            validate
            ;;
        phase2)
            create_backup
            phase2
            validate
            ;;
        phase3)
            create_backup
            phase3
            validate
            ;;
        validate)
            validate
            ;;
        rollback)
            rollback
            ;;
        all)
            create_backup
            phase1
            validate
            read -p "Phase 1 完成，是否繼續？(y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                phase2
                validate
                read -p "Phase 2 完成，是否繼續？(y/n) " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    phase3
                    validate
                fi
            fi
            ;;
        *)
            echo "使用方法：$0 {phase1|phase2|phase3|validate|rollback|all}"
            echo ""
            echo "選項："
            echo "  phase1   - 執行 Phase 1：低風險清理"
            echo "  phase2   - 執行 Phase 2：解決重複"
            echo "  phase3   - 執行 Phase 3：整理散落檔案"
            echo "  validate - 運行驗證"
            echo "  rollback - 回滾到備份狀態"
            echo "  all      - 執行所有階段（交互式）"
            exit 1
            ;;
    esac
}

main "$@"