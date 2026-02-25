#!/bin/bash

# MachineNativeOps Branch Manager
# 用於管理 Git 分支系統的輔助腳本

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 顯示使用說明
show_usage() {
    echo -e "${GREEN}MachineNativeOps Branch Manager${NC}"
    echo "用法: $0 [命令] [參數]"
    echo ""
    echo "命令:"
    echo "  feature <name>     建立新功能分支"
    echo "  fix <name>         建立修復分支"
    echo "  hotfix <name>      建立緊急修復分支"
    echo "  refactor <name>    建立重構分支"
    echo "  test <name>        建立測試分支"
    echo "  integration <name> 建立整合分支"
    echo "  docs <name>        建立文件分支"
    echo "  release <version>  建立發布分支"
    echo "  experiment <name>  建立實驗分支"
    echo "  research <name>    建立研究分支"
    echo "  list               列出所有分支"
    echo "  sync               同步所有主要分支"
    echo "  help               顯示此說明"
    echo ""
    echo "範例:"
    echo "  $0 feature user-authentication"
    echo "  $0 fix login-validation"
    echo "  $0 release v2.0.0"
}

# 建立分支
create_branch() {
    local branch_type=$1
    local branch_name=$2
    local base_branch="develop"
    
    if [ "$branch_type" = "hotfix" ]; then
        base_branch="main"
    fi
    
    # 檢查是否已經在該分支
    current_branch=$(git branch --show-current)
    if [ "$current_branch" != "$base_branch" ]; then
        echo -e "${YELLOW}切換到 $base_branch 分支...${NC}"
        git checkout "$base_branch"
        git pull origin "$base_branch"
    fi
    
    # 建立新分支
    local new_branch="${branch_type}/${branch_name}"
    echo -e "${GREEN}建立分支: $new_branch${NC}"
    git checkout -b "$new_branch"
    
    echo -e "${GREEN}✓ 分支 $new_branch 已建立${NC}"
    echo -e "${BLUE}當前分支: $(git branch --show-current)${NC}"
}

# 列出所有分支
list_branches() {
    echo -e "${GREEN}=== 分支列表 ===${NC}"
    echo ""
    
    echo -e "${YELLOW}主要分支:${NC}"
    git branch | grep -E "^(main|develop|staging)" | sed 's/^/  /'
    
    echo ""
    echo -e "${YELLOW}功能分支:${NC}"
    git branch | grep "^feature/" | sed 's/^/  /'
    
    echo ""
    echo -e "${YELLOW}修復分支:${NC}"
    git branch | grep -E "^(fix/|hotfix/)" | sed 's/^/  /'
    
    echo ""
    echo -e "${YELLOW}其他分支:${NC}"
    git branch | grep -v -E "^(main|develop|staging|feature/|fix/|hotfix/)" | sed 's/^/  /'
}

# 同步主要分支
sync_branches() {
    echo -e "${GREEN}同步主要分支...${NC}"
    
    for branch in main develop staging; do
        echo -e "${YELLOW}同步 $branch 分支...${NC}"
        git checkout "$branch"
        git pull origin "$branch"
    done
    
    echo -e "${GREEN}✓ 所有主要分支已同步${NC}"
}

# 主程式
main() {
    case "$1" in
        feature|fix|hotfix|refactor|test|integration|docs|release|experiment|research)
            if [ -z "$2" ]; then
                echo -e "${RED}錯誤: 請提供分支名稱${NC}"
                show_usage
                exit 1
            fi
            create_branch "$1" "$2"
            ;;
        list)
            list_branches
            ;;
        sync)
            sync_branches
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            echo -e "${RED}未知命令: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
}

# 執行主程式
main "$@"