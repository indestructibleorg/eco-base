#!/bin/bash
# MachineNativeOps Actions Migration Script
# Replaces third-party GitHub Actions with official MN Actions
# Version: 1.0.0

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
WORKFLOWS_DIR=".github/workflows"
ACTIONS_DIR=".github/actions"
DRY_RUN=false
VERBOSE=false
SPECIFIC_FILE=""
BACKUP=true

# Action mappings (third-party -> MN action)
declare -A ACTION_MAP=(
    # Core Actions
    ["actions/checkout@v4"]="./.github/actions/mn-checkout"
    ["actions/checkout@v6"]="./.github/actions/mn-checkout"
    ["actions/upload-artifact@v4"]="./.github/actions/mn-upload-artifact"
    ["actions/upload-artifact@v6"]="./.github/actions/mn-upload-artifact"
    ["actions/download-artifact@v4"]="./.github/actions/mn-download-artifact"
    ["actions/download-artifact@v7"]="./.github/actions/mn-download-artifact"
    ["actions/setup-python@v5"]="./.github/actions/mn-setup-python"
    ["actions/setup-python@v6"]="./.github/actions/mn-setup-python"
    ["actions/setup-node@v4"]="./.github/actions/mn-setup-node"
    ["actions/setup-node@v6"]="./.github/actions/mn-setup-node"
    ["actions/github-script@v7"]="./.github/actions/mn-github-script"
    ["actions/github-script@v8"]="./.github/actions/mn-github-script"
    ["actions/cache@v4"]="./.github/actions/mn-cache"
    ["actions/cache@v5"]="./.github/actions/mn-cache"
    
    # Security Actions
    ["github/codeql-action/init@v3"]="./.github/actions/mn-codeql"
    ["github/codeql-action/init@v4"]="./.github/actions/mn-codeql"
    ["github/codeql-action/autobuild@v3"]="./.github/actions/mn-codeql"
    ["github/codeql-action/autobuild@v4"]="./.github/actions/mn-codeql"
    ["github/codeql-action/analyze@v3"]="./.github/actions/mn-codeql"
    ["github/codeql-action/analyze@v4"]="./.github/actions/mn-codeql"
    ["github/codeql-action/upload-sarif@v3"]="./.github/actions/mn-codeql"
    ["github/codeql-action/upload-sarif@v4"]="./.github/actions/mn-codeql"
    ["aquasecurity/trivy-action@master"]="./.github/actions/mn-trivy-scan"
    ["aquasecurity/trivy-action@0.18.0"]="./.github/actions/mn-trivy-scan"
    ["trufflesecurity/trufflehog@main"]="./.github/actions/mn-secret-scan"
    ["gitleaks/gitleaks-action@v2"]="./.github/actions/mn-secret-scan"
    # Note: returntocorp/semgrep-action is not migrated as mn-sast-scan doesn't exist yet
    
    # Docker Actions
    ["docker/setup-buildx-action@v3"]="./.github/actions/mn-docker-build"
    ["docker/metadata-action@v5"]="./.github/actions/mn-docker-build"
    ["docker/login-action@v3"]="./.github/actions/mn-docker-build"
    ["docker/build-push-action@v5"]="./.github/actions/mn-docker-build"
    
    # Notification Actions
    ["8398a7/action-slack@v3"]="./.github/actions/mn-slack-notify"
    
    # PR/Issue Actions
    ["peter-evans/create-pull-request@v6"]="./.github/actions/mn-create-pr"
    ["peter-evans/create-pull-request@v8.0.0"]="./.github/actions/mn-create-pr"
    
    # Tool Actions
    ["open-policy-agent/setup-opa@v2"]="./.github/actions/mn-setup-opa"
    ["super-linter/super-linter/slim@v7"]="./.github/actions/mn-super-linter"
)

# Print usage
usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Migrate third-party GitHub Actions to MachineNativeOps official Actions.

OPTIONS:
    -h, --help          Show this help message
    -d, --dry-run       Preview changes without modifying files
    -v, --verbose       Enable verbose output
    -f, --file FILE     Only process specific workflow file
    -n, --no-backup     Don't create backup files
    --list              List all action mappings
    --verify            Verify MN actions exist before migration

EXAMPLES:
    $(basename "$0")                    # Migrate all workflows
    $(basename "$0") --dry-run          # Preview changes
    $(basename "$0") -f ci.yml          # Migrate specific file
    $(basename "$0") --list             # Show action mappings

EOF
}

# Print action mappings
list_mappings() {
    echo -e "${BLUE}Action Mappings:${NC}"
    echo "==============================================="
    for key in "${!ACTION_MAP[@]}"; do
        echo -e "${YELLOW}$key${NC}"
        echo -e "  -> ${GREEN}${ACTION_MAP[$key]}${NC}"
    done
    echo "==============================================="
    echo "Total: ${#ACTION_MAP[@]} mappings"
}

# Verify MN actions exist
verify_actions() {
    echo -e "${BLUE}Verifying MN Actions...${NC}"
    local missing=0
    
    declare -A checked
    for mn_action in "${ACTION_MAP[@]}"; do
        if [[ -z "${checked[$mn_action]}" ]]; then
            checked[$mn_action]=1
            action_path="${mn_action#./}"
            if [[ -f "$action_path/action.yml" ]]; then
                echo -e "  ${GREEN}✓${NC} $mn_action"
            else
                echo -e "  ${RED}✗${NC} $mn_action (missing)"
                ((missing++))
            fi
        fi
    done
    
    if [[ $missing -gt 0 ]]; then
        echo -e "${RED}Warning: $missing MN actions are missing${NC}"
        return 1
    fi
    
    echo -e "${GREEN}All MN actions verified${NC}"
    return 0
}

# Process a single workflow file
process_file() {
    local file="$1"
    local changes=0
    
    if [[ ! -f "$file" ]]; then
        echo -e "${RED}File not found: $file${NC}"
        return 1
    fi
    
    echo -e "${BLUE}Processing: $file${NC}"
    
    # Create backup
    if [[ "$BACKUP" == "true" && "$DRY_RUN" == "false" ]]; then
        cp "$file" "${file}.bak"
    fi
    
    # Read file content
    local content
    content=$(cat "$file")
    local original_content="$content"
    
    # Process each action mapping
    for third_party in "${!ACTION_MAP[@]}"; do
        local mn_action="${ACTION_MAP[$third_party]}"
        
        # Check if this action is used in the file
        if echo "$content" | grep -q "uses: $third_party"; then
            ((changes++))
            
            if [[ "$VERBOSE" == "true" ]]; then
                echo -e "  ${YELLOW}Found:${NC} $third_party"
                echo -e "  ${GREEN}Replace with:${NC} $mn_action"
            fi
            
            # Replace the action
            content=$(echo "$content" | sed "s|uses: $third_party|uses: $mn_action|g")
        fi
    done
    
    # Handle special cases for CodeQL (multiple steps -> single step)
    if echo "$content" | grep -q "mn-codeql"; then
        # Add mode parameter based on original action
        if echo "$original_content" | grep -q "codeql-action/init"; then
            # Build line-by-line mapping so we only add mode: init for steps that
            # were originally codeql-action/init, and avoid invalid YAML.
            mapfile -t content_lines <<< "$content"
            mapfile -t original_lines <<< "$original_content"

            new_content=""
            for i in "${!content_lines[@]}"; do
                line="${content_lines[$i]}"
                orig_line="${original_lines[$i]}"

                # Always keep the current line
                new_content+="$line"$'\n'

                # Only consider lines where the action was migrated to mn-codeql from init
                if [[ "$line" == *"uses: ./.github/actions/mn-codeql"* && "$orig_line" == *"codeql-action/init"* ]]; then
                    # Determine indentation from the part before 'uses:'
                    indent="${line%%uses:*}"

                    # Check if the next line already has a 'with:' block
                    next_index=$((i + 1))
                    if [[ $next_index -lt ${#content_lines[@]} ]]; then
                        next_line="${content_lines[$next_index]}"
                        if [[ "$next_line" =~ ^[[:space:]]*with: ]]; then
                            # with: block exists, inject mode: init as first parameter
                            new_content+="${indent}with:"$'\n'
                            new_content+="${indent}  mode: init"$'\n'
                            # Skip the original 'with:' line, we'll add remaining params
                            i=$((i + 1))
                            continue
                        fi
                    fi

                    # No with: block exists, inject one with mode: init
                    new_content+="${indent}with:"$'\n'
                    new_content+="${indent}  mode: init"$'\n'
                fi
            done

            content="$new_content"
        fi
    fi
    
    # Write changes
    if [[ "$changes" -gt 0 ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            echo -e "  ${YELLOW}Would make $changes replacement(s)${NC}"
            if [[ "$VERBOSE" == "true" ]]; then
                echo "--- Diff ---"
                diff <(echo "$original_content") <(echo "$content") || true
                echo "------------"
            fi
        else
            echo "$content" > "$file"
            echo -e "  ${GREEN}Made $changes replacement(s)${NC}"
        fi
    else
        echo -e "  ${GREEN}No changes needed${NC}"
    fi
    
    return 0
}

# Main migration function
migrate() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}MachineNativeOps Actions Migration${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo -e "${YELLOW}DRY RUN MODE - No files will be modified${NC}"
    fi
    
    # Verify we're in the right directory
    if [[ ! -d "$WORKFLOWS_DIR" ]]; then
        echo -e "${RED}Error: $WORKFLOWS_DIR not found${NC}"
        echo "Please run this script from the repository root"
        exit 1
    fi
    
    # Verify MN actions exist
    if [[ ! -d "$ACTIONS_DIR" ]]; then
        echo -e "${RED}Error: $ACTIONS_DIR not found${NC}"
        echo "MN Actions have not been created yet"
        exit 1
    fi
    
    local total_files=0
    local processed_files=0
    
    if [[ -n "$SPECIFIC_FILE" ]]; then
        # Process specific file
        if [[ "$SPECIFIC_FILE" == *.yml || "$SPECIFIC_FILE" == *.yaml ]]; then
            process_file "$WORKFLOWS_DIR/$SPECIFIC_FILE"
        else
            process_file "$SPECIFIC_FILE"
        fi
        processed_files=1
    else
        # Process all workflow files
        for file in "$WORKFLOWS_DIR"/*.yml "$WORKFLOWS_DIR"/*.yaml; do
            if [[ -f "$file" ]]; then
                ((total_files++))
                if process_file "$file"; then
                    ((processed_files++))
                fi
            fi
        done
    fi
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}Migration complete!${NC}"
    echo -e "Processed: $processed_files file(s)"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo -e "${YELLOW}Run without --dry-run to apply changes${NC}"
    elif [[ "$BACKUP" == "true" ]]; then
        echo -e "Backup files created with .bak extension"
    fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -f|--file)
            SPECIFIC_FILE="$2"
            shift 2
            ;;
        -n|--no-backup)
            BACKUP=false
            shift
            ;;
        --list)
            list_mappings
            exit 0
            ;;
        --verify)
            verify_actions
            exit $?
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Run migration
migrate