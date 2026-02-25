#!/bin/bash
# 環境檢查與自動修復腳本

set -e

# ============================================
# 顏色定義
# ============================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================
# 檢查函數
# ============================================
check_command() {
    local cmd=$1
    local install_guide=$2
    
    if command -v "$cmd" &> /dev/null; then
        echo -e "${GREEN}✓${NC} $cmd 已安裝：$($cmd --version 2>&1 | head -n1)"
        return 0
    else
        echo -e "${RED}✗${NC} $cmd 未安裝"
        echo -e "${YELLOW}  安裝指南：$install_guide${NC}"
        return 1
    fi
}

# ============================================
# 主檢查流程
# ============================================
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     CI 環境檢查與自動修復工具          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}\n"

FAILED=0

# 檢查 Docker
echo -e "${BLUE}[1/4]${NC} 檢查 Docker..."
if ! check_command "docker" "https://docs.docker.com/get-docker/"; then
    FAILED=$((FAILED + 1))
fi

# 檢查 Docker Compose
echo -e "\n${BLUE}[2/4]${NC} 檢查 Docker Compose..."
if command -v docker-compose &> /dev/null; then
    echo -e "${GREEN}✓${NC} docker-compose 已安裝：$(docker-compose --version)"
elif docker compose version &> /dev/null; then
    echo -e "${GREEN}✓${NC} docker compose (plugin) 已安裝：$(docker compose version)"
else
    echo -e "${RED}✗${NC} Docker Compose 未安裝"
    echo -e "${YELLOW}  安裝指南：https://docs.docker.com/compose/install/${NC}"
    FAILED=$((FAILED + 1))
fi

# 檢查 Node.js
echo -e "\n${BLUE}[3/4]${NC} 檢查 Node.js..."
if ! check_command "node" "https://nodejs.org/"; then
    FAILED=$((FAILED + 1))
else
    # 檢查 Node.js 版本
    NODE_VERSION=$(node --version | sed 's/v//' | cut -d'.' -f1)
    if [ "$NODE_VERSION" -lt 18 ]; then
        echo -e "${YELLOW}⚠️  Node.js 版本過舊（需要 >= 18）${NC}"
        echo -e "   當前版本：$(node --version)"
        echo -e "   安裝指南：https://nodejs.org/"
        FAILED=$((FAILED + 1))
    fi
fi

# 檢查 Git
echo -e "\n${BLUE}[4/4]${NC} 檢查 Git..."
if ! check_command "git" "https://git-scm.com/"; then
    FAILED=$((FAILED + 1))
fi

# ============================================
# 磁盤空間檢查
# ============================================
echo -e "\n${BLUE}[磁盤空間]${NC}"
# 使用 df 不帶 -h 選項以獲得更可靠的輸出格式，然後計算百分比
DISK_INFO=$(df . 2>/dev/null | awk 'NR==2 {printf "%.0f", ($3/$2)*100}' || echo "0")
DISK_USAGE=${DISK_INFO:-0}
if [ "$DISK_USAGE" -gt 80 ]; then
    echo -e "${YELLOW}⚠️  磁盤使用率：${DISK_USAGE}%（建議清理）${NC}"
    echo -e "   執行：${YELLOW}docker system prune -a${NC}"
else
    echo -e "${GREEN}✓${NC} 磁盤空間充足：${DISK_USAGE}%"
fi

# ============================================
# 結果總結
# ============================================
echo -e "\n${BLUE}╔════════════════════════════════════════╗${NC}"
if [ $FAILED -eq 0 ]; then
    echo -e "${BLUE}║${NC}${GREEN}     環境檢查通過 - 準備就緒！        ${NC}${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
    exit 0
else
    echo -e "${BLUE}║${NC}${RED}   環境檢查失敗 - 缺少 $FAILED 個依賴    ${NC}${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
    echo -e "\n${YELLOW}建議操作：${NC}"
    echo "1. 按上述指南安裝缺失的工具"
    echo "2. 重新運行此腳本驗證"
    echo "3. 若問題持續，請聯繫 DevOps 團隊"
    exit 1
fi
