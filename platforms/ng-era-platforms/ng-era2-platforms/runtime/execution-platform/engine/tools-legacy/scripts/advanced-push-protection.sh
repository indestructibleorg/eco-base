#!/bin/bash

# Advanced Push Protection Script
# 高級推播保護腳本 - 用於在推送前檢測潛在的秘密洩露

set -e

REPO_OWNER=${1:-""}
REPO_NAME=${2:-""}
CHECK_MODE=${3:-"strict"}

echo "=========================================="
echo "Advanced Push Protection"
echo "=========================================="
echo "Repository: ${REPO_OWNER}/${REPO_NAME}"
echo "Check Mode: ${CHECK_MODE}"
echo "=========================================="

# Color codes for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Secret patterns to check
declare -A SECRET_PATTERNS=(
    ["GitHub Token"]='github_pat_[0-9A-Za-z]{36}|ghp_[0-9A-Za-z]{36}|gho_[0-9A-Za-z]{36}'
    ["OpenAI API Key"]='sk-[0-9A-Za-z]{48}'
    ["AWS Access Key"]='AKIA[0-9A-Z]{16}'
    ["AWS Secret Key"]='[0-9a-zA-Z/+]{40}'
    ["Azure Client Secret"]='[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
    ["Generic API Key"]='api[_-]?key['"'"'"]?[\s]*[:=][\s]*['"'"'"]?[0-9a-zA-Z]{32,}'
    ["Private Key"]='-----BEGIN (RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----'
    ["JWT Token"]='eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}'
    ["Database Password"]='(?i)(password|passwd|pwd)['"'"'"]?[\s]*[:=][\s]*['"'"'"]?[a-zA-Z0-9!@#$%^&*]{8,}'
    ["Connection String"]='(mongodb|postgresql|mysql|redis)://[^:]+:[^@]+@'
    ["Slack Token"]='xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24,}'
    ["Stripe Key"]='sk_(live|test)_[0-9a-zA-Z]{24,}'
)

# Check if running in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}❌ Not a git repository${NC}"
    exit 1
fi

# Function to check a single file
check_file() {
    local file=$1
    local secrets_found=0
    
    if [[ ! -f "$file" ]]; then
        return 0
    fi
    
    # Skip binary files
    if file "$file" | grep -q "binary"; then
        return 0
    fi
    
    # Skip large files (>1MB)
    if [[ $(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null) -gt 1048576 ]]; then
        echo -e "${YELLOW}⚠️  Skipping large file: $file${NC}"
        return 0
    fi
    
    for pattern_name in "${!SECRET_PATTERNS[@]}"; do
        pattern="${SECRET_PATTERNS[$pattern_name]}"
        
        if grep -E "$pattern" "$file" > /dev/null 2>&1; then
            echo -e "${RED}❌ $pattern_name detected in: $file${NC}"
            
            # Show context (redacted)
            echo "   Context:"
            grep -n -E "$pattern" "$file" | head -3 | sed 's/\(.\{50\}\).*/\1.../' | sed 's/^/   /'
            echo ""
            
            secrets_found=$((secrets_found + 1))
        fi
    done
    
    return $secrets_found
}

# Function to check commit content
check_commit_content() {
    local commit_sha=$1
    local total_secrets=0
    
    echo "Checking commit: $commit_sha"
    
    # Get list of changed files in commit
    git diff-tree --no-commit-id --name-only -r "$commit_sha" | while read -r file; do
        if [[ -f "$file" ]]; then
            check_file "$file"
            total_secrets=$((total_secrets + $?))
        fi
    done
    
    return $total_secrets
}

# Function to check staged changes
check_staged_changes() {
    echo "Checking staged changes..."
    local total_secrets=0
    
    # Get list of staged files
    git diff --cached --name-only | while read -r file; do
        if [[ -f "$file" ]]; then
            check_file "$file"
            total_secrets=$((total_secrets + $?))
        fi
    done
    
    return $total_secrets
}

# Function to check recent commits
check_recent_commits() {
    local num_commits=${1:-5}
    local total_secrets=0
    
    echo "Checking last $num_commits commits..."
    
    git log --oneline -"$num_commits" --format="%H" | while read -r commit; do
        check_commit_content "$commit"
        total_secrets=$((total_secrets + $?))
    done
    
    return $total_secrets
}

# Function to scan entire repository
scan_repository() {
    echo "Scanning entire repository..."
    local total_secrets=0
    
    # Find all text files
    find . -type f ! -path "*/\.git/*" ! -path "*/node_modules/*" ! -path "*/vendor/*" | while read -r file; do
        check_file "$file"
        total_secrets=$((total_secrets + $?))
    done
    
    return $total_secrets
}

# Main execution logic
main() {
    local exit_code=0
    local secrets_found=0
    
    case "$CHECK_MODE" in
        "staged")
            check_staged_changes
            secrets_found=$?
            ;;
        "commits")
            check_recent_commits 10
            secrets_found=$?
            ;;
        "full")
            scan_repository
            secrets_found=$?
            ;;
        "strict"|*)
            # Check both staged and recent commits
            check_staged_changes
            secrets_found=$?
            
            if [[ $secrets_found -eq 0 ]]; then
                check_recent_commits 5
                secrets_found=$?
            fi
            ;;
    esac
    
    echo "=========================================="
    if [[ $secrets_found -gt 0 ]]; then
        echo -e "${RED}❌ PUSH BLOCKED: $secrets_found potential secret(s) detected${NC}"
        echo ""
        echo "Action required:"
        echo "1. Remove all detected secrets from your code"
        echo "2. Rotate/revoke any exposed credentials"
        echo "3. Use environment variables or secret management systems"
        echo "4. Consider using git-secrets or similar tools"
        echo ""
        echo "To bypass this check (NOT RECOMMENDED):"
        echo "  git push --no-verify"
        echo ""
        exit_code=1
    else
        echo -e "${GREEN}✅ No secrets detected - Push allowed${NC}"
        exit_code=0
    fi
    echo "=========================================="
    
    exit $exit_code
}

# Install as git pre-push hook
install_hook() {
    local hook_path=".git/hooks/pre-push"
    
    echo "Installing pre-push hook..."
    
    cat > "$hook_path" << 'EOF'
#!/bin/bash
# Pre-push hook to check for secrets

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROTECTION_SCRIPT="$SCRIPT_DIR/scripts/advanced-push-protection.sh"

if [[ -f "$PROTECTION_SCRIPT" ]]; then
    bash "$PROTECTION_SCRIPT" "" "" "staged"
else
    echo "Warning: Push protection script not found"
fi
EOF
    
    chmod +x "$hook_path"
    echo "✅ Pre-push hook installed at $hook_path"
}

# Command line options
case "${1:-check}" in
    "install")
        install_hook
        ;;
    "scan")
        main
        ;;
    *)
        main
        ;;
esac
