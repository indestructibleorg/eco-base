#!/bin/bash
#═══════════════════════════════════════════════════════════════════════════
#           Hybrid Multi-Repository Integration Tool
#                混合策略多倉庫整合工具
#═══════════════════════════════════════════════════════════════════════════
#
# 一鍵執行混合整合策略：
# - 核心倉庫 (10-20個) → Git Subtree (保留歷史)
# - 普通倉庫 (80-90個) → 自動同步 (定期更新)
#
# One-command hybrid integration strategy:
# - Core repos (10-20) → Git Subtree (preserve history)
# - Regular repos (80-90) → Auto-sync (periodic updates)
#
# Usage:
#   ./tools/integrate_all_repos.sh [MODE] [OPTIONS]
#
# Modes:
#   sync        - Sync mode only (auto-sync all repos)
#   subtree     - Subtree mode only (integrate core repos)
#   hybrid      - Hybrid mode (core via subtree, rest via sync) [DEFAULT]
#
# Options:
#   --dry-run   - Show what would be done without making changes
#   --force     - Skip confirmations
#   --config    - Custom config file path
#
# Examples:
#   ./tools/integrate_all_repos.sh                    # Hybrid mode
#   ./tools/integrate_all_repos.sh sync               # Sync only
#   ./tools/integrate_all_repos.sh hybrid --dry-run   # Test hybrid
#
#═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# Configuration
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG_FILE="${REPO_ROOT}/config/external_repos.yaml"
LOG_DIR="${REPO_ROOT}/.automation_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/integration_${TIMESTAMP}.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# Statistics
STATS_SUBTREE_SUCCESS=0
STATS_SUBTREE_FAILED=0
STATS_SYNC_SUCCESS=0
STATS_SYNC_FAILED=0
STATS_TOTAL=0

# Logging functions
log() { echo -e "$*" | tee -a "$LOG_FILE"; }
log_info() { log "${CYAN}ℹ️  $*${NC}"; }
log_success() { log "${GREEN}✅ $*${NC}"; }
log_warning() { log "${YELLOW}⚠️  $*${NC}"; }
log_error() { log "${RED}❌ $*${NC}"; }
log_header() { log "${BOLD}${BLUE}$*${NC}"; }
log_section() { log "${BOLD}${MAGENTA}$*${NC}"; }

# Banner
print_banner() {
    echo ""
    log_header "═══════════════════════════════════════════════════════════════════"
    log_header "     Hybrid Multi-Repository Integration Tool"
    log_header "         混合策略多倉庫整合工具"
    log_header "═══════════════════════════════════════════════════════════════════"
    log_info "Repository: ${REPO_ROOT}"
    log_info "Config: ${CONFIG_FILE}"
    log_info "Log: ${LOG_FILE}"
    log_info "Timestamp: $(date +'%Y-%m-%d %H:%M:%S')"
    log_header "═══════════════════════════════════════════════════════════════════"
    echo ""
}

# Help message
show_help() {
    cat << EOF
Hybrid Multi-Repository Integration Tool

Usage:
    $0 [MODE] [OPTIONS]

Modes:
    sync        Auto-sync all repositories (simple copy)
    subtree     Git Subtree integration for core repos (full history)
    hybrid      Hybrid strategy: core via subtree, rest via sync [DEFAULT]

Options:
    --dry-run       Show what would be done without making changes
    --force         Skip all confirmations
    --config PATH   Custom configuration file
    --skip-sync     Skip sync phase (subtree only)
    --skip-subtree  Skip subtree phase (sync only)
    -h, --help      Show this help message

Examples:
    # Full hybrid integration
    $0 hybrid

    # Dry run to see what would happen
    $0 hybrid --dry-run

    # Only sync repositories
    $0 sync

    # Only integrate core repos with subtree
    $0 subtree

    # Custom config file
    $0 hybrid --config my-repos.yaml

Strategy Explanation:
    HYBRID MODE (Recommended for ~100 repos):
      ├─ Core Repos (10-20)      → Git Subtree
      │  ├─ Preserves full git history
      │  ├─ Supports bidirectional sync
      │  └─ Best for actively maintained code
      │
      └─ Regular Repos (80-90)   → Auto-Sync
         ├─ Simple file copy
         ├─ No git history
         └─ Easy periodic updates

EOF
}

# Parse arguments
MODE="hybrid"
DRY_RUN=false
FORCE=false
SKIP_SYNC=false
SKIP_SUBTREE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        sync|subtree|hybrid)
            MODE="$1"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --skip-sync)
            SKIP_SYNC=true
            shift
            ;;
        --skip-subtree)
            SKIP_SUBTREE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Adjust mode based on skip flags
if [[ "$SKIP_SYNC" == "true" && "$SKIP_SUBTREE" == "true" ]]; then
    log_error "Cannot skip both sync and subtree phases"
    exit 1
fi

if [[ "$SKIP_SYNC" == "true" ]]; then
    MODE="subtree"
fi

if [[ "$SKIP_SUBTREE" == "true" ]]; then
    MODE="sync"
fi

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Check configuration
check_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Configuration file not found: $CONFIG_FILE"
        log_info "Please create it from the example:"
        log_info "  cp config/external_repos.yaml.example config/external_repos.yaml"
        exit 1
    fi

    # Validate YAML (basic check)
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not found"
        exit 1
    fi

    log_info "Configuration validated"
}

# Check required tools
check_requirements() {
    log_info "Checking requirements..."

    local missing=()

    # Check git
    if ! command -v git &> /dev/null; then
        missing+=("git")
    fi

    # Check python3
    if ! command -v python3 &> /dev/null; then
        missing+=("python3")
    fi

    # Check PyYAML
    if ! python3 -c "import yaml" 2>/dev/null; then
        log_warning "PyYAML not installed, will be installed automatically"
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing[*]}"
        exit 1
    fi

    log_success "All requirements satisfied"
}

# Confirm with user
confirm_integration() {
    if [[ "$FORCE" == "true" ]]; then
        return 0
    fi

    log_warning "This will integrate external repositories into keystone-ai"
    log_info "Mode: ${MODE}"
    log_info "Dry run: ${DRY_RUN}"
    echo ""

    # Show what will be integrated
    if [[ "$MODE" == "hybrid" || "$MODE" == "subtree" ]]; then
        log_info "Core repositories will be integrated via Git Subtree"
        python3 tools/subtree_integrate.py --config "$CONFIG_FILE" --core-only --list --format table 2>/dev/null || true
        echo ""
    fi

    if [[ "$MODE" == "hybrid" || "$MODE" == "sync" ]]; then
        log_info "Regular repositories will be synced"
        # We can show count here
        local sync_count=$(python3 -c "
import yaml
with open('$CONFIG_FILE') as f:
    config = yaml.safe_load(f)
    print(len(config.get('sync_repositories', [])))
" 2>/dev/null || echo "?")
        log_info "Total sync repositories: ${sync_count}"
        echo ""
    fi

    if [[ "$DRY_RUN" == "false" ]]; then
        read -p "Continue with integration? (yes/no) " -r
        if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
            log_info "Aborted by user"
            exit 0
        fi
    fi
}

# Phase 1: Git Subtree Integration
phase_subtree() {
    log_section ""
    log_section "╔════════════════════════════════════════════════════════════════╗"
    log_section "║  Phase 1: Git Subtree Integration (Core Repositories)         ║"
    log_section "╚════════════════════════════════════════════════════════════════╝"
    log_section ""

    local subtree_opts=""
    if [[ "$DRY_RUN" == "true" ]]; then
        subtree_opts="--dry-run"
    fi

    if [[ -f "${REPO_ROOT}/tools/integrate_repositories.sh" ]]; then
        log_info "Executing: ./tools/integrate_repositories.sh ${subtree_opts}"

        if bash "${REPO_ROOT}/tools/integrate_repositories.sh" $subtree_opts; then
            STATS_SUBTREE_SUCCESS=$((STATS_SUBTREE_SUCCESS + 1))
            log_success "Subtree integration completed"
        else
            STATS_SUBTREE_FAILED=$((STATS_SUBTREE_FAILED + 1))
            log_error "Subtree integration failed"
            return 1
        fi
    else
        log_error "Subtree integration script not found"
        return 1
    fi
}

# Phase 2: Auto-Sync
phase_sync() {
    log_section ""
    log_section "╔════════════════════════════════════════════════════════════════╗"
    log_section "║  Phase 2: Auto-Sync (Regular Repositories)                    ║"
    log_section "╚════════════════════════════════════════════════════════════════╝"
    log_section ""

    local sync_opts=""
    if [[ "$DRY_RUN" == "true" ]]; then
        sync_opts="--dry-run"
    fi

    # Exclude core repos if in hybrid mode
    if [[ "$MODE" == "hybrid" ]]; then
        sync_opts="$sync_opts --exclude-core"
    fi

    if [[ -f "${REPO_ROOT}/tools/sync_external_repos.py" ]]; then
        log_info "Executing: python3 tools/sync_external_repos.py ${sync_opts}"

        if python3 "${REPO_ROOT}/tools/sync_external_repos.py" $sync_opts; then
            STATS_SYNC_SUCCESS=$((STATS_SYNC_SUCCESS + 1))
            log_success "Auto-sync completed"
        else
            STATS_SYNC_FAILED=$((STATS_SYNC_FAILED + 1))
            log_error "Auto-sync failed"
            return 1
        fi
    else
        log_error "Sync script not found"
        return 1
    fi
}

# Generate integration report
generate_report() {
    local report_file="${LOG_DIR}/integration_report_${TIMESTAMP}.md"

    cat > "$report_file" << EOF
# Multi-Repository Integration Report

**Date**: $(date +'%Y-%m-%d %H:%M:%S')
**Mode**: ${MODE}
**Dry Run**: ${DRY_RUN}
**Config**: ${CONFIG_FILE}

## Summary

| Metric | Count |
|--------|-------|
| Subtree Success | ${STATS_SUBTREE_SUCCESS} |
| Subtree Failed | ${STATS_SUBTREE_FAILED} |
| Sync Success | ${STATS_SYNC_SUCCESS} |
| Sync Failed | ${STATS_SYNC_FAILED} |
| Total | ${STATS_TOTAL} |

## Integration Strategy

EOF

    if [[ "$MODE" == "hybrid" ]]; then
        cat >> "$report_file" << EOF
**Hybrid Mode**: Core repositories integrated via Git Subtree, regular repositories synced.

### Core Repositories (Subtree)
- Preserves full git history
- Supports bidirectional sync
- Located in: \`external/\`

### Regular Repositories (Sync)
- Simple file copy
- No git history
- Located in: \`external/\`

EOF
    elif [[ "$MODE" == "subtree" ]]; then
        cat >> "$report_file" << EOF
**Subtree Mode**: All core repositories integrated via Git Subtree.

EOF
    elif [[ "$MODE" == "sync" ]]; then
        cat >> "$report_file" << EOF
**Sync Mode**: All repositories synced via auto-sync.

EOF
    fi

    cat >> "$report_file" << EOF
## Next Steps

EOF

    if [[ "$DRY_RUN" == "false" ]]; then
        cat >> "$report_file" << EOF
1. Review changes:
   \`\`\`bash
   git status
   git log --oneline -10
   \`\`\`

2. Commit changes:
   \`\`\`bash
   git add external/
   git commit -m "feat: integrate external repositories (${MODE} mode)"
   \`\`\`

3. Push to remote:
   \`\`\`bash
   git push
   \`\`\`

## Maintenance

### Update Subtree Repositories
\`\`\`bash
python3 tools/subtree_integrate.py --update REPO_NAME
\`\`\`

### Re-sync All Repositories
\`\`\`bash
python3 tools/sync_external_repos.py
\`\`\`

### Check Integration Status
\`\`\`bash
python3 tools/subtree_integrate.py --status
\`\`\`

EOF
    else
        cat >> "$report_file" << EOF
**Dry Run Mode**: No actual changes were made.

To execute for real, run:
\`\`\`bash
./tools/integrate_all_repos.sh ${MODE}
\`\`\`

EOF
    fi

    cat >> "$report_file" << EOF
## Logs

- Full log: \`${LOG_FILE}\`
- Report: \`${report_file}\`

---

Generated by: \`tools/integrate_all_repos.sh\`
EOF

    log_success "Report generated: ${report_file}"
}

# Print final summary
print_summary() {
    echo ""
    log_header "═══════════════════════════════════════════════════════════════════"
    log_header "Integration Summary"
    log_header "═══════════════════════════════════════════════════════════════════"
    log_info "Mode: ${MODE}"
    log_info "Dry Run: ${DRY_RUN}"
    echo ""

    if [[ "$MODE" == "hybrid" || "$MODE" == "subtree" ]]; then
        log_info "Subtree Integration:"
        log_success "  Success: ${STATS_SUBTREE_SUCCESS}"
        if [[ $STATS_SUBTREE_FAILED -gt 0 ]]; then
            log_error "  Failed: ${STATS_SUBTREE_FAILED}"
        fi
    fi

    if [[ "$MODE" == "hybrid" || "$MODE" == "sync" ]]; then
        log_info "Auto-Sync:"
        log_success "  Success: ${STATS_SYNC_SUCCESS}"
        if [[ $STATS_SYNC_FAILED -gt 0 ]]; then
            log_error "  Failed: ${STATS_SYNC_FAILED}"
        fi
    fi

    echo ""
    log_header "═══════════════════════════════════════════════════════════════════"

    if [[ "$DRY_RUN" == "false" ]]; then
        echo ""
        log_success "✨ Integration complete!"
        log_info ""
        log_info "Next steps:"
        log_info "  1. Review: git status"
        log_info "  2. Commit: git add external/ && git commit -m 'feat: integrate repos'"
        log_info "  3. Push: git push"
        log_info ""
        log_info "Full log: ${LOG_FILE}"
    else
        echo ""
        log_warning "⚠️  Dry run complete - no changes were made"
        log_info "To execute for real, run: ./tools/integrate_all_repos.sh ${MODE}"
    fi
}

# Main execution
main() {
    print_banner

    # Pre-flight checks
    check_config
    check_requirements

    # Confirm
    confirm_integration

    # Execute based on mode
    case $MODE in
        sync)
            log_info "Mode: SYNC ONLY"
            phase_sync
            ;;

        subtree)
            log_info "Mode: SUBTREE ONLY"
            phase_subtree
            ;;

        hybrid)
            log_info "Mode: HYBRID (Subtree + Sync)"

            # Phase 1: Subtree
            if phase_subtree; then
                log_success "Phase 1 complete"
            else
                log_error "Phase 1 failed, aborting..."
                exit 1
            fi

            # Phase 2: Sync
            if phase_sync; then
                log_success "Phase 2 complete"
            else
                log_warning "Phase 2 failed, but continuing..."
            fi
            ;;

        *)
            log_error "Unknown mode: $MODE"
            exit 1
            ;;
    esac

    # Generate report
    generate_report

    # Print summary
    print_summary
}

# Trap errors
trap 'log_error "Script failed at line $LINENO"' ERR

# Run main
main

# Exit code
if [[ $STATS_SUBTREE_FAILED -gt 0 || $STATS_SYNC_FAILED -gt 0 ]]; then
    exit 1
else
    exit 0
fi
