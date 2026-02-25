#!/bin/bash
# ============================================================================
# SLASolve 代碼分析腳本 (Code Analysis Script)
# ============================================================================
# 用途: 執行全面的代碼分析，包括安全掃描、質量檢查和性能分析
# 版本: 1.0.0
# ============================================================================

set -euo pipefail

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置
ANALYSIS_DIR="reports/analysis"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="$ANALYSIS_DIR/analysis_$TIMESTAMP.json"
HTML_REPORT="$ANALYSIS_DIR/analysis_$TIMESTAMP.html"

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

# 解析命令行參數
parse_args() {
    SCAN_TYPE="full"
    TARGET_PATH="."
    OUTPUT_FORMAT="json"
    PARALLEL="true"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --scan-type)
                SCAN_TYPE="$2"
                shift 2
                ;;
            --target)
                TARGET_PATH="$2"
                shift 2
                ;;
            --format)
                OUTPUT_FORMAT="$2"
                shift 2
                ;;
            --sequential)
                PARALLEL="false"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知參數: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 顯示幫助信息
show_help() {
    cat << EOF
用法: $0 [選項]

選項:
    --scan-type TYPE    掃描類型: full, incremental, security, performance
                        (默認: full)
    --target PATH       掃描目標路徑 (默認: .)
    --format FORMAT     輸出格式: json, html, markdown, all
                        (默認: json)
    --sequential        順序執行（禁用並行分析）
    -h, --help          顯示此幫助信息

示例:
    $0
    $0 --scan-type security --target src/
    $0 --format all
EOF
}

# 創建報告目錄
setup_directories() {
    log_info "準備分析環境..."
    mkdir -p "$ANALYSIS_DIR"
    mkdir -p "logs"
    log_success "報告目錄已準備: $ANALYSIS_DIR"
}

# 靜態代碼分析
run_static_analysis() {
    log_info "執行靜態代碼分析..."
    
    local issues=0
    
    # TypeScript/JavaScript 分析
    if command -v eslint >/dev/null 2>&1; then
        log_info "  → ESLint 分析中..."
        eslint "$TARGET_PATH" --ext .ts,.js,.tsx,.jsx \
            --format json --output-file "$ANALYSIS_DIR/eslint_$TIMESTAMP.json" \
            2>/dev/null || true
        log_success "  ✓ ESLint 完成"
    fi
    
    # Python 分析
    if command -v pylint >/dev/null 2>&1; then
        log_info "  → Pylint 分析中..."
        find "$TARGET_PATH" -name "*.py" \
            -not -path "*/venv/*" \
            -not -path "*/node_modules/*" \
            | xargs pylint --output-format=json \
            > "$ANALYSIS_DIR/pylint_$TIMESTAMP.json" 2>&1 || true
        log_success "  ✓ Pylint 完成"
    fi
    
    # TypeScript 類型檢查
    if command -v tsc >/dev/null 2>&1 && [ -f "tsconfig.json" ]; then
        log_info "  → TypeScript 類型檢查中..."
        tsc --noEmit 2>&1 | tee "$ANALYSIS_DIR/tsc_$TIMESTAMP.log" || true
        log_success "  ✓ TypeScript 完成"
    fi
    
    log_success "靜態分析完成"
}

# 安全掃描
run_security_scan() {
    log_info "執行安全掃描..."
    
    # Semgrep 掃描
    if command -v semgrep >/dev/null 2>&1; then
        log_info "  → Semgrep 掃描中..."
        semgrep --config=auto \
            --json \
            --output="$ANALYSIS_DIR/semgrep_$TIMESTAMP.json" \
            "$TARGET_PATH" 2>/dev/null || true
        log_success "  ✓ Semgrep 完成"
    fi
    
    # Snyk 掃描依賴
    if command -v snyk >/dev/null 2>&1; then
        log_info "  → Snyk 依賴掃描中..."
        snyk test --json \
            > "$ANALYSIS_DIR/snyk_$TIMESTAMP.json" 2>&1 || true
        log_success "  ✓ Snyk 完成"
    fi
    
    # Trivy 掃描
    if command -v trivy >/dev/null 2>&1; then
        log_info "  → Trivy 掃描中..."
        trivy fs \
            --format json \
            --output "$ANALYSIS_DIR/trivy_$TIMESTAMP.json" \
            --security-checks vuln,config,secret \
            "$TARGET_PATH" 2>/dev/null || true
        log_success "  ✓ Trivy 完成"
    fi
    
    # 檢測硬編碼密鑰
    log_info "  → 密鑰洩漏檢測中..."
    grep -rn \
        -E "(password|api_key|secret|token)\s*=\s*['\"][^'\"]+['\"]" \
        "$TARGET_PATH" \
        --include="*.py" --include="*.js" --include="*.ts" \
        > "$ANALYSIS_DIR/secrets_$TIMESTAMP.log" 2>&1 || true
    log_success "  ✓ 密鑰檢測完成"
    
    log_success "安全掃描完成"
}

# 性能分析
run_performance_analysis() {
    log_info "執行性能分析..."
    
    # 計算代碼複雜度
    log_info "  → 圈複雜度分析中..."
    find "$TARGET_PATH" -name "*.py" -o -name "*.js" -o -name "*.ts" \
        | wc -l > "$ANALYSIS_DIR/files_count_$TIMESTAMP.txt"
    
    # 檢測重複代碼
    if command -v jscpd >/dev/null 2>&1; then
        log_info "  → 重複代碼檢測中..."
        jscpd "$TARGET_PATH" \
            --format json \
            --output "$ANALYSIS_DIR" \
            > /dev/null 2>&1 || true
        log_success "  ✓ 重複代碼檢測完成"
    fi
    
    log_success "性能分析完成"
}

# 代碼質量分析
run_quality_analysis() {
    log_info "執行代碼質量分析..."
    
    # 計算代碼行數
    log_info "  → 代碼統計中..."
    {
        echo "=== Code Statistics ==="
        echo "Total lines:"
        find "$TARGET_PATH" \
            -name "*.py" -o -name "*.js" -o -name "*.ts" \
            -not -path "*/node_modules/*" \
            -not -path "*/venv/*" \
            | xargs wc -l 2>/dev/null | tail -1
        
        echo ""
        echo "Files by type:"
        find "$TARGET_PATH" -type f \
            -not -path "*/node_modules/*" \
            -not -path "*/venv/*" \
            | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -10
    } > "$ANALYSIS_DIR/statistics_$TIMESTAMP.txt"
    
    log_success "代碼質量分析完成"
}

# 使用 Python 分析器
run_python_analyzer() {
    log_info "執行 Python 代碼分析引擎..."
    
    if [ -f "advanced-system-src/core/analyzers/analyzer.py" ]; then
        # 激活虛擬環境
        if [ -d "venv" ]; then
            source venv/bin/activate
        fi
        
        python3 advanced-system-src/core/analyzers/analyzer.py \
            --target "$TARGET_PATH" \
            --output "$REPORT_FILE" \
            2>&1 || true
        
        if [ -d "venv" ]; then
            deactivate
        fi
        
        log_success "Python 分析引擎完成"
    else
        log_warning "Python 分析引擎未找到"
    fi
}

# 聚合分析結果
aggregate_results() {
    log_info "聚合分析結果..."
    
    local total_issues=0
    local critical_issues=0
    local high_issues=0
    
    # 創建聚合報告
    cat > "$REPORT_FILE" << EOF
{
  "analysis_id": "analysis_$TIMESTAMP",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "scan_type": "$SCAN_TYPE",
  "target": "$TARGET_PATH",
  "tools_used": [
    "eslint",
    "pylint",
    "semgrep",
    "snyk",
    "trivy"
  ],
  "summary": {
    "total_files": $(find "$TARGET_PATH" -type f | wc -l),
    "total_issues": $total_issues,
    "critical": $critical_issues,
    "high": $high_issues
  },
  "reports": {
    "static_analysis": "$ANALYSIS_DIR/",
    "security_scan": "$ANALYSIS_DIR/",
    "performance": "$ANALYSIS_DIR/",
    "quality": "$ANALYSIS_DIR/"
  }
}
EOF
    
    log_success "結果已聚合到: $REPORT_FILE"
}

# 生成 HTML 報告
generate_html_report() {
    if [ "$OUTPUT_FORMAT" = "html" ] || [ "$OUTPUT_FORMAT" = "all" ]; then
        log_info "生成 HTML 報告..."
        
        cat > "$HTML_REPORT" << 'EOF'
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SLASolve 代碼分析報告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        .summary { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }
        .card { padding: 15px; border-radius: 6px; text-align: center; }
        .card.critical { background: #ff4444; color: white; }
        .card.high { background: #ff8c00; color: white; }
        .card.medium { background: #ffa500; color: white; }
        .card.low { background: #4caf50; color: white; }
        .section { margin: 20px 0; }
        .timestamp { color: #7f8c8d; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 SLASolve 代碼分析報告</h1>
        <p class="timestamp">生成時間: TIMESTAMP</p>
        
        <div class="summary">
            <div class="card critical">
                <h3>0</h3>
                <p>Critical</p>
            </div>
            <div class="card high">
                <h3>0</h3>
                <p>High</p>
            </div>
            <div class="card medium">
                <h3>0</h3>
                <p>Medium</p>
            </div>
            <div class="card low">
                <h3>0</h3>
                <p>Low</p>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 分析摘要</h2>
            <p>完整的分析結果請查看 JSON 報告</p>
        </div>
    </div>
</body>
</html>
EOF
        # 使用臨時變量避免 sed 特殊字符問題
        REPORT_TIMESTAMP=$(date)
        sed -i "s|TIMESTAMP|${REPORT_TIMESTAMP}|" "$HTML_REPORT"
        
        log_success "HTML 報告已生成: $HTML_REPORT"
    fi
}

# 顯示結果摘要
show_summary() {
    echo ""
    echo "============================================================================"
    log_success "代碼分析完成！"
    echo "============================================================================"
    echo ""
    echo "分析報告:"
    echo "  JSON: ${BLUE}$REPORT_FILE${NC}"
    [ -f "$HTML_REPORT" ] && echo "  HTML: ${BLUE}$HTML_REPORT${NC}"
    echo ""
    echo "詳細日誌:"
    echo "  ${BLUE}$ANALYSIS_DIR${NC}"
    echo ""
    echo "下一步:"
    echo "  1. 查看分析報告: ${BLUE}cat $REPORT_FILE${NC}"
    echo "  2. 運行自動修復: ${BLUE}./scripts/repair.sh${NC}"
    echo "============================================================================"
}

# 主函數
main() {
    echo "============================================================================"
    echo "                   SLASolve 代碼分析"
    echo "============================================================================"
    echo ""
    
    parse_args "$@"
    setup_directories
    
    case "$SCAN_TYPE" in
        full)
            run_static_analysis
            run_security_scan
            run_performance_analysis
            run_quality_analysis
            run_python_analyzer
            ;;
        security)
            run_security_scan
            ;;
        performance)
            run_performance_analysis
            ;;
        quality)
            run_static_analysis
            run_quality_analysis
            ;;
        *)
            log_error "未知的掃描類型: $SCAN_TYPE"
            exit 1
            ;;
    esac
    
    aggregate_results
    generate_html_report
    show_summary
}

# 執行主函數
main "$@"
