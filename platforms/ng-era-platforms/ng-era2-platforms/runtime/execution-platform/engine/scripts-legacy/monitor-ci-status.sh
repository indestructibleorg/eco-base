#!/bin/bash
# CI 驗證狀態持續監控腳本
# 用於追蹤 PR 的 CI 狀態並識別阻斷性問題

# 注意：不使用全局 set -e，以避免在預期的暫時性失敗（例如 CI 尚未開始、檢查為 pending）時提前退出腳本。

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日誌函數
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# 顯示幫助信息
show_help() {
    cat << EOF
使用方法: $0 <PR_NUMBER> [CHECK_INTERVAL]

參數:
  PR_NUMBER       Pull Request 編號（必需）
  CHECK_INTERVAL  檢查間隔（秒），默認 30

示例:
  $0 123
  $0 123 60

功能:
  - 持續監控 PR 的 CI 狀態
  - 識別阻斷性問題
  - 顯示檢查詳細信息
  - 提供修復建議
EOF
}

# 檢查參數
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

if [ -z "$1" ]; then
    log_error "錯誤: PR_NUMBER 為必需參數"
    echo ""
    show_help
    exit 1
fi

PR_NUMBER=$1
CHECK_INTERVAL=${2:-30}

log "=== 監控 PR #$PR_NUMBER 的 CI 狀態 ==="
log "檢查間隔: $CHECK_INTERVAL 秒"
log ""

# 獲取 PR 信息
log "獲取 PR 信息..."
PR_INFO=$(gh pr view "$PR_NUMBER" --json title,state,headRefName 2>/dev/null)

if [ -z "$PR_INFO" ]; then
    log_error "無法獲取 PR #$PR_NUMBER 的信息"
    exit 1
fi

PR_TITLE=$(echo "$PR_INFO" | jq -r '.title')
PR_STATE=$(echo "$PR_INFO" | jq -r '.state')
PR_BRANCH=$(echo "$PR_INFO" | jq -r '.headRefName')

log_success "PR: #$PR_NUMBER - $PR_TITLE"
log "分支: $PR_BRANCH"
log "狀態: $PR_STATE"
log ""

# 檢查 PR 狀態
if [ "$PR_STATE" = "closed" ]; then
    log_warning "PR 已關閉，無需監控"
    exit 0
fi

if [ "$PR_STATE" = "merged" ]; then
    log_success "PR 已合併"
    exit 0
fi

# 主監控循環
log "開始監控..."
log ""

iteration=0
while true; do
    iteration=$((iteration + 1))
    log "--- 檢查 #$iteration ($(date '+%Y-%m-%d %H:%M:%S')) ---"
    
    # 獲取 PR 的檢查狀態
    CHECKS=$(gh pr view "$PR_NUMBER" --json statusCheckRollup --jq '.statusCheckRollup | .[] | select(.conclusion != null) | {name, conclusion, status, startedAt, completedAt, detailsUrl}' 2>/dev/null)
    
    if [ -z "$CHECKS" ]; then
        log "⏳ 等待 CI 開始..."
    else
        # 統計檢查狀態
        TOTAL=$(echo "$CHECKS" | jq -s 'length')
        PASSED=$(echo "$CHECKS" | jq -s '[.[] | select(.conclusion == "success")] | length')
        FAILED=$(echo "$CHECKS" | jq -s '[.[] | select(.conclusion == "failure" or .conclusion == "timed_out")] | length')
        PENDING=$(echo "$CHECKS" | jq -s '[.[] | select(.conclusion == null or .conclusion == "pending")] | length')
        
        log "總檢查數: $TOTAL"
        log_success "✅ 通過: $PASSED"
        
        if [ "$FAILED" -gt 0 ]; then
            log_error "❌ 失敗: $FAILED"
        fi
        
        if [ "$PENDING" -gt 0 ]; then
            log_warning "⏳ 進行中: $PENDING"
        fi
        
        log ""
        
        # 顯示失敗的檢查
        if [ "$FAILED" -gt 0 ]; then
            log_error "=== 失敗的 CI 檢查 ==="
            echo "$CHECKS" | jq -r '.[] | select(.conclusion == "failure" or .conclusion == "timed_out") | "\(.name): \(.conclusion)"' | while read -r line; do
                log_error "  - $line"
            done
            log ""
            
            # 提供修復建議
            log "=== 修復建議 ==="
            
            # 檢查是否是測試失敗
            if echo "$CHECKS" | jq -r '.[] | select(.conclusion == "failure") | .name' | grep -qi "test"; then
                log "📝 測試失敗:"
                log "  1. 查看失敗的測試日誌"
                log "     gh run list --limit 1 --json databaseId | jq -r '.[0].databaseId'"
                log "     gh run view RUN_ID --log-failed"
                log ""
                log "  2. 本地重現問題"
                log "     npm test -- --grep '失敗的測試名稱'"
                log ""
                log "  3. 修復並重新運行測試"
                log "     npm test"
            fi
            
            # 檢查是否是 Linting 錯誤
            if echo "$CHECKS" | jq -r '.[] | select(.conclusion == "failure") | .name' | grep -qi "lint\|style\|format"; then
                log "📝 Linting 錯誤處理:"
                log "  1. 自動修復（如果支持）"
                log "     npm run lint:fix"
                log ""
                log "  2. 手動修復錯誤"
                log "     npm run lint"
            fi
            
            log ""
        fi
        
        # 檢查是否全部通過
        if [ "$PENDING" -eq 0 ] && [ "$FAILED" -eq 0 ]; then
            log_success "✅ 所有 CI 檢查已通過！"
            log ""
            log "=== 詳細檢查結果 ==="
            echo "$CHECKS" | jq -r '.[] | "✅ \(.name): \(.conclusion)"'
            log ""
            break
        fi
    fi
    
    # 詢問是否繼續監控
    if [ "$iteration" -gt 1 ]; then
        read -p "是否繼續監控? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "監控已停止"
            break
        fi
    fi
    
    log "等待 $CHECK_INTERVAL 秒後再次檢查..."
    sleep "$CHECK_INTERVAL"
done

log ""
log_success "=== 監控完成 ==="
log "PR #$PR_NUMBER 的 CI 狀態: 全部通過"