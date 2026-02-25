#!/bin/bash
#══════════════════════════════════════════════════════════════════════════════
#                  Autonomous Cleanup Automation Script
#                     自主清理自動化腳本
#══════════════════════════════════════════════════════════════════════════════
#
# This script replicates Claude's cleanup workflow for autonomous execution.
# 此腳本複製 Claude 的清理工作流程以實現自主執行。
#
# Usage:
#   ./tools/run_full_cleanup.sh              # Interactive mode
#   ./tools/run_full_cleanup.sh --auto       # Automatic mode (dry-run)
#   ./tools/run_full_cleanup.sh --execute    # Execute all cleanups
#
#══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TOOLS_DIR="$REPO_ROOT/tools"
LOG_DIR="$REPO_ROOT/.automation_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Log file
LOGFILE="$LOG_DIR/cleanup_$TIMESTAMP.log"

# Function to print with color and log
log() {
    local color=$1
    shift
    echo -e "${color}$*${NC}" | tee -a "$LOGFILE"
}

log_info() { log "$CYAN" "ℹ️  $*"; }
log_success() { log "$GREEN" "✅ $*"; }
log_warning() { log "$YELLOW" "⚠️  $*"; }
log_error() { log "$RED" "❌ $*"; }

# Banner
print_banner() {
    log_info "══════════════════════════════════════════════════════════════"
    log_info "       🤖 Autonomous Cleanup Toolkit"
    log_info "          Replicating Claude Code Capabilities"
    log_info "══════════════════════════════════════════════════════════════"
    log_info "Repository: $REPO_ROOT"
    log_info "Timestamp: $(date +'%Y-%m-%d %H:%M:%S')"
    log_info "Log File: $LOGFILE"
    log_info "══════════════════════════════════════════════════════════════"
}

# Parse arguments
AUTO_MODE=false
EXECUTE_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --auto)
            AUTO_MODE=true
            shift
            ;;
        --execute)
            EXECUTE_MODE=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Start
print_banner

#══════════════════════════════════════════════════════════════════════════════
# Phase 1: P0 Safety Verification
#══════════════════════════════════════════════════════════════════════════════

log_info ""
log_info "┌─────────────────────────────────────────────────────────────┐"
log_info "│ Phase 1: P0 Safety Verification 🔒                          │"
log_info "└─────────────────────────────────────────────────────────────┘"

if [[ -f "$TOOLS_DIR/verify_p0_safety.py" ]]; then
    log_info "Running P0 safety verification..."
    python3 "$TOOLS_DIR/verify_p0_safety.py" | tee -a "$LOGFILE"
    log_success "P0 safety verification complete"
else
    log_warning "P0 safety verification tool not found, skipping..."
fi

#══════════════════════════════════════════════════════════════════════════════
# Phase 2: Duplicate File Detection and Cleanup
#══════════════════════════════════════════════════════════════════════════════

log_info ""
log_info "┌─────────────────────────────────────────────────────────────┐"
log_info "│ Phase 2: Duplicate File Cleanup 📂                          │"
log_info "└─────────────────────────────────────────────────────────────┘"

if [[ -f "$TOOLS_DIR/find_duplicate_scripts.py" ]]; then
    log_info "Scanning for duplicate files..."
    python3 "$TOOLS_DIR/find_duplicate_scripts.py" | tee -a "$LOGFILE"

    if [[ -f "$TOOLS_DIR/cleanup_duplicates.py" ]]; then
        if [[ "$EXECUTE_MODE" == true ]]; then
            log_warning "Executing duplicate cleanup..."
            python3 "$TOOLS_DIR/cleanup_duplicates.py" --execute | tee -a "$LOGFILE"
            log_success "Duplicate cleanup executed"
        elif [[ "$AUTO_MODE" == false ]]; then
            log_info "Run cleanup? (yes/no)"
            read -r response
            if [[ "$response" == "yes" ]]; then
                python3 "$TOOLS_DIR/cleanup_duplicates.py" --execute | tee -a "$LOGFILE"
                log_success "Duplicate cleanup executed"
            else
                log_info "Skipping duplicate cleanup"
            fi
        else
            log_info "Dry-run mode - run with --execute to apply changes"
            python3 "$TOOLS_DIR/cleanup_duplicates.py" | tee -a "$LOGFILE"
        fi
    fi
else
    log_warning "Duplicate detection tool not found, skipping..."
fi

#══════════════════════════════════════════════════════════════════════════════
# Phase 6: Technical Debt Scan
#══════════════════════════════════════════════════════════════════════════════

log_info ""
log_info "┌─────────────────────────────────────────────────────────────┐"
log_info "│ Phase 6: Technical Debt Scanning 📊                         │"
log_info "└─────────────────────────────────────────────────────────────┘"

if [[ -f "$TOOLS_DIR/scan_tech_debt.py" ]]; then
    log_info "Scanning technical debt..."
    python3 "$TOOLS_DIR/scan_tech_debt.py" | tee -a "$LOGFILE"
    log_success "Technical debt scan complete"

    # Check if report was generated
    if [[ -f "TECH_DEBT_SCAN_REPORT.json" ]]; then
        log_success "Report: TECH_DEBT_SCAN_REPORT.json"

        # Parse and display summary
        if command -v jq &> /dev/null; then
            log_info ""
            log_info "Summary:"
            jq -r '.summary' TECH_DEBT_SCAN_REPORT.json 2>/dev/null | while read -r line; do
                log_info "  $line"
            done
        fi
    fi
else
    log_warning "Technical debt scanner not found, skipping..."
fi

#══════════════════════════════════════════════════════════════════════════════
# Comprehensive Analysis Report
#══════════════════════════════════════════════════════════════════════════════

log_info ""
log_info "┌─────────────────────────────────────────────────────────────┐"
log_info "│ Comprehensive Analysis Report 📋                             │"
log_info "└─────────────────────────────────────────────────────────────┘"

if [[ -f "$TOOLS_DIR/autonomous_cleanup_toolkit.py" ]]; then
    log_info "Generating comprehensive report..."
    python3 "$TOOLS_DIR/autonomous_cleanup_toolkit.py" analyze \
        --output "$LOG_DIR/cleanup_analysis_$TIMESTAMP.json" | tee -a "$LOGFILE"
    log_success "Report generated: $LOG_DIR/cleanup_analysis_$TIMESTAMP.json"
else
    log_warning "Autonomous cleanup toolkit not found, skipping..."
fi

#══════════════════════════════════════════════════════════════════════════════
# Summary and Next Steps
#══════════════════════════════════════════════════════════════════════════════

log_info ""
log_info "┌─────────────────────────────────────────────────────────────┐"
log_info "│ Cleanup Summary ✨                                           │"
log_info "└─────────────────────────────────────────────────────────────┘"

log_success "Autonomous cleanup workflow complete!"
log_info ""
log_info "📄 Log file: $LOGFILE"
log_info ""
log_info "Next Steps:"
log_info "  1. Review generated reports"
log_info "  2. Address high-priority TODOs manually"
log_info "  3. Implement NotImplementedError stubs"
log_info "  4. Refactor high-complexity functions"
log_info ""

if [[ "$EXECUTE_MODE" == false ]]; then
    log_warning "Note: This was a dry-run. Use --execute to apply changes."
fi

log_info "══════════════════════════════════════════════════════════════"
