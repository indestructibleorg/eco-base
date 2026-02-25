#!/bin/bash
#═══════════════════════════════════════════════════════════════════════════
#                Git Subtree Batch Integration Tool
#                     Git Subtree 批量整合工具
#═══════════════════════════════════════════════════════════════════════════
#
# 使用 Git Subtree 將多個倉庫完全整合到 keystone-ai
# 保留完整的 Git 歷史記錄
#
# Usage:
#   ./tools/integrate_repositories.sh
#   ./tools/integrate_repositories.sh --config config/external_repos.yaml
#   ./tools/integrate_repositories.sh --repo repo-name
#
#═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG_FILE="${REPO_ROOT}/config/external_repos.yaml"
TARGET_PREFIX="external"
SQUASH=true

# Statistics
TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0

# Logging functions
log_info() { echo -e "${CYAN}ℹ️  $*${NC}"; }
log_success() { echo -e "${GREEN}✅ $*${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $*${NC}"; }
log_error() { echo -e "${RED}❌ $*${NC}"; }
log_header() { echo -e "${BOLD}${BLUE}$*${NC}"; }

# Banner
print_banner() {
    echo ""
    log_header "═══════════════════════════════════════════════════════════════"
    log_header "     Git Subtree Batch Integration Tool"
    log_header "        批量 Subtree 整合工具"
    log_header "═══════════════════════════════════════════════════════════════"
    echo ""
}

# Help message
show_help() {
    cat << EOF
Git Subtree Batch Integration Tool

Usage:
    $0 [OPTIONS]

Options:
    --config PATH       Configuration file path (default: config/external_repos.yaml)
    --repo NAME         Integrate single repository by name
    --prefix PATH       Target directory prefix (default: external)
    --no-squash         Don't squash commits (preserve full history)
    --dry-run           Show what would be done without making changes
    -h, --help          Show this help message

Examples:
    # Integrate all core repositories
    $0

    # Integrate single repository
    $0 --repo authentication-service

    # Use custom config
    $0 --config my-repos.yaml

    # Don't squash commits (preserve all history)
    $0 --no-squash

EOF
}

# Parse arguments
DRY_RUN=false
SINGLE_REPO=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --repo)
            SINGLE_REPO="$2"
            shift 2
            ;;
        --prefix)
            TARGET_PREFIX="$2"
            shift 2
            ;;
        --no-squash)
            SQUASH=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
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

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    log_error "Configuration file not found: $CONFIG_FILE"
    log_info "Please create it from the example:"
    log_info "  cp config/external_repos.yaml.example config/external_repos.yaml"
    exit 1
fi

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    log_error "Not in a git repository"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    log_warning "You have uncommitted changes"
    log_info "It's recommended to commit or stash them before integrating"
    read -p "Continue anyway? (yes/no) " -r
    if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
        log_info "Aborted"
        exit 0
    fi
fi

# Function to add git remote
add_remote() {
    local name=$1
    local url=$2

    if git remote | grep -q "^${name}$"; then
        log_info "Remote '${name}' already exists"
        return 0
    fi

    log_info "Adding remote: ${name}"
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would add remote: git remote add ${name} ${url}"
        return 0
    fi

    if git remote add "$name" "$url" 2>&1; then
        log_success "Remote '${name}' added"
        return 0
    else
        log_error "Failed to add remote '${name}'"
        return 1
    fi
}

# Function to fetch remote
fetch_remote() {
    local name=$1
    local branch=$2

    log_info "Fetching: ${name}/${branch}"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would fetch: git fetch ${name} ${branch}"
        return 0
    fi

    if git fetch "$name" "$branch" 2>&1; then
        log_success "Fetched ${name}/${branch}"
        return 0
    else
        log_error "Failed to fetch ${name}/${branch}"
        return 1
    fi
}

# Function to add subtree
add_subtree() {
    local name=$1
    local remote=$2
    local branch=$3
    local target_dir="${TARGET_PREFIX}/${name}"

    log_info "Adding subtree: ${target_dir}"

    # Check if directory already exists
    if [[ -d "$target_dir" ]]; then
        log_warning "Directory already exists: ${target_dir}"
        read -p "Replace it? (yes/no) " -r
        if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
            log_info "Skipped ${name}"
            return 2
        fi

        log_info "Removing existing directory..."
        if [[ "$DRY_RUN" == "false" ]]; then
            rm -rf "$target_dir"
        fi
    fi

    # Build subtree add command
    local cmd="git subtree add --prefix=${target_dir} ${remote} ${branch}"
    if [[ "$SQUASH" == "true" ]]; then
        cmd="${cmd} --squash"
    fi

    log_info "Command: ${cmd}"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would execute: ${cmd}"
        return 0
    fi

    # Execute subtree add
    if eval "$cmd" 2>&1; then
        log_success "Subtree added: ${target_dir}"
        return 0
    else
        log_error "Failed to add subtree: ${target_dir}"
        return 1
    fi
}

# Function to integrate a single repository
integrate_repo() {
    local name=$1
    local url=$2
    local branch=$3

    log_header "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_header "Integrating: ${name}"
    log_header "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    ((TOTAL++))

    # Step 1: Add remote
    if ! add_remote "$name" "$url"; then
        ((FAILED++))
        return 1
    fi

    # Step 2: Fetch remote
    if ! fetch_remote "$name" "$branch"; then
        ((FAILED++))
        return 1
    fi

    # Step 3: Add subtree
    local result
    add_subtree "$name" "$name" "$branch"
    result=$?

    if [[ $result -eq 0 ]]; then
        ((SUCCESS++))
        log_success "✨ Successfully integrated: ${name}"
        return 0
    elif [[ $result -eq 2 ]]; then
        ((SKIPPED++))
        return 0
    else
        ((FAILED++))
        return 1
    fi
}

# Function to print summary
print_summary() {
    echo ""
    log_header "═══════════════════════════════════════════════════════════════"
    log_header "Integration Summary"
    log_header "═══════════════════════════════════════════════════════════════"
    log_info "Total:    ${TOTAL}"
    log_success "Success:  ${SUCCESS}"
    log_error "Failed:   ${FAILED}"
    log_warning "Skipped:  ${SKIPPED}"
    log_header "═══════════════════════════════════════════════════════════════"

    if [[ $SUCCESS -gt 0 ]]; then
        echo ""
        log_success "✨ ${SUCCESS} repositories integrated successfully!"
        log_info ""
        log_info "Next steps:"
        log_info "  1. Review changes: git log --oneline -10"
        log_info "  2. Push to remote: git push"
        log_info ""
        log_info "To update a subtree later:"
        log_info "  git subtree pull --prefix=external/REPO_NAME REMOTE_NAME BRANCH --squash"
    fi

    if [[ $FAILED -gt 0 ]]; then
        echo ""
        log_error "❌ ${FAILED} repositories failed to integrate"
        log_info "Please check the errors above and try again"
    fi
}

# Main execution
main() {
    print_banner

    # Use Python helper to parse YAML and get repositories
    log_info "Loading configuration from: ${CONFIG_FILE}"

    if [[ ! -f "${REPO_ROOT}/tools/subtree_integrate.py" ]]; then
        log_error "Helper script not found: tools/subtree_integrate.py"
        log_info "Please ensure all tools are installed"
        exit 1
    fi

    # Get repository list from Python helper
    local repo_data
    if [[ -n "$SINGLE_REPO" ]]; then
        log_info "Integrating single repository: ${SINGLE_REPO}"
        repo_data=$(python3 "${REPO_ROOT}/tools/subtree_integrate.py" \
            --config "$CONFIG_FILE" \
            --repo "$SINGLE_REPO" \
            --list)
    else
        log_info "Integrating all core repositories"
        repo_data=$(python3 "${REPO_ROOT}/tools/subtree_integrate.py" \
            --config "$CONFIG_FILE" \
            --core-only \
            --list)
    fi

    if [[ -z "$repo_data" ]]; then
        log_error "No repositories found in configuration"
        exit 1
    fi

    # Parse and integrate repositories
    while IFS='|' read -r name url branch; do
        if [[ -n "$name" && -n "$url" && -n "$branch" ]]; then
            integrate_repo "$name" "$url" "$branch"
        fi
    done <<< "$repo_data"

    # Print summary
    print_summary
}

# Run main function
main

# Exit with appropriate code
if [[ $FAILED -gt 0 ]]; then
    exit 1
else
    exit 0
fi
