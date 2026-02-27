#!/bin/bash

# MachineNativeOps 目錄重構自動化執行腳本
# 這個腳本提供了一站式的重構執行體驗

set -e  # 遇到錯誤立即退出

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日誌函數
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 檢查先決條件
check_prerequisites() {
    log_info "檢查先決條件..."
    
    # 檢查 Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 未安裝"
        exit 1
    fi
    
    # 檢查項目根目錄
    if [[ ! -f "package.json" && ! -f "pyproject.toml" ]]; then
        log_error "請在項目根目錄執行此腳本"
        exit 1
    fi
    
    # 檢查工具文件
    if [[ ! -f "tools/automated_directory_restructure.py" ]]; then
        log_error "找不到重構工具文件"
        exit 1
    fi
    
    log_success "先決條件檢查通過"
}

# 顯示菜單
show_menu() {
    echo -e "\n${BLUE}MachineNativeOps 目錄重構工具${NC}"
    echo "=================================="
    echo "1. 試運行重構（預覽變更）"
    echo "2. 執行完整重構"
    echo "3. 只重構 src 目錄"
    echo "4. 只重構 config 目錄"
    echo "5. 驗證重構結果"
    echo "6. 修復導入路徑"
    echo "7. 查看幫助文檔"
    echo "8. 退出"
    echo "=================================="
}

# 試運行重構
dry_run_restructure() {
    log_info "開始試運行重構..."
    python3 tools/automated_directory_restructure.py --dry-run
    log_success "試運行完成"
}

# 執行完整重構
full_restructure() {
    log_warning "即將執行完整重構，這將修改大量文件"
    read -p "確認繼續？(y/N): " -n 1 -r
    echo
    
    if [[ "${REPLY}" =~ ^[Yy]$ ]]; then
        log_info "開始執行完整重構..."
        python3 tools/automated_directory_restructure.py
        log_success "重構完成"
        
        log_info "自動驗證重構結果..."
        python3 tools/validate_restructure.py --detailed
    else
        log_info "已取消重構"
    fi
}

# 重構 src 目錄
restructure_src() {
    log_info "開始重構 src 目錄..."
    python3 tools/automated_directory_restructure.py --phase src
    log_success "src 目錄重構完成"
}

# 重構 config 目錄
restructure_config() {
    log_info "開始重構 config 目錄..."
    python3 tools/automated_directory_restructure.py --phase config
    log_success "config 目錄重構完成"
}

# 驗證重構
validate_restructure() {
    log_info "開始驗證重構結果..."
    python3 tools/validate_restructure.py --detailed
    log_success "驗證完成"
}

# 修復導入路徑
fix_imports() {
    log_info "開始修復導入路徑..."
    python3 tools/validate_restructure.py --fix-imports
    log_success "導入路徑修復完成"
}

# 顯示幫助
show_help() {
    log_info "打開自動化指南文檔..."
    if command -v xdg-open &> /dev/null; then
        xdg-open docs/DIRECTORY_RESTRUCTURE_AUTOMATION_GUIDE.md
    elif command -v open &> /dev/null; then
        open docs/DIRECTORY_RESTRUCTURE_AUTOMATION_GUIDE.md
    else
        echo "請手動查看: docs/DIRECTORY_RESTRUCTURE_AUTOMATION_GUIDE.md"
    fi
}

# 主程序
main() {
    check_prerequisites
    
    while true; do
        show_menu
        read -p "請選擇操作 (1-8): " choice
        echo
        
        case $choice in
            1)
                dry_run_restructure
                ;;
            2)
                full_restructure
                ;;
            3)
                restructure_src
                ;;
            4)
                restructure_config
                ;;
            5)
                validate_restructure
                ;;
            6)
                fix_imports
                ;;
            7)
                show_help
                ;;
            8)
                log_info "退出程序"
                exit 0
                ;;
            *)
                log_error "無效選擇，請輸入 1-8"
                ;;
        esac
        
        echo
        read -p "按 Enter 繼續..."
        clear
    done
}

# 處理命令行參數
if [[ $# -gt 0 ]]; then
    case $1 in
        --dry-run)
            dry_run_restructure
            ;;
        --full)
            full_restructure
            ;;
        --src)
            restructure_src
            ;;
        --config)
            restructure_config
            ;;
        --validate)
            validate_restructure
            ;;
        --fix-imports)
            fix_imports
            ;;
        --help)
            echo "用法: $0 [選項]"
            echo "選項:"
            echo "  --dry-run      試運行重構"
            echo "  --full         執行完整重構"
            echo "  --src          只重構 src 目錄"
            echo "  --config       只重構 config 目錄"
            echo "  --validate     驗證重構結果"
            echo "  --fix-imports  修復導入路徑"
            echo "  --help         顯示此幫助"
            exit 0
            ;;
        *)
            log_error "未知參數: $1"
            echo "使用 --help 查看可用選項"
            exit 1
            ;;
    esac
else
    main
fi
