#!/bin/bash
# Conditional Deployment Script
# 條件式部署腳本
#
# Purpose: Install dependencies based on module-environment-matrix
# Principle: No hardcoded dependencies, no one-size-fits-all templates

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GOVERNANCE_FILE="$ROOT_DIR/governance/environment-matrix/module-environment-matrix.yml"

echo "=== Conditional Deployment System ==="
echo "條件式部署系統"
echo ""

# Function to check if a module requires specific runtime
check_module_requirement() {
    local module_path=$1
    local requirement=$2
    
    # For now, use simple file-based detection
    # In production, parse module-environment-matrix.yml
    
    case $requirement in
        "python")
            [ -f "$module_path/requirements.txt" ] || [ -f "$module_path/pyproject.toml" ]
            ;;
        "nodejs")
            [ -f "$module_path/package.json" ]
            ;;
        "rust")
            [ -f "$module_path/Cargo.toml" ]
            ;;
        "go")
            [ -f "$module_path/go.mod" ]
            ;;
        *)
            return 1
            ;;
    esac
}

# Function to install Python dependencies conditionally
install_python_deps() {
    local module_path=$1
    
    if check_module_requirement "$module_path" "python"; then
        echo "✓ Python module detected: $module_path"
        
        # Check for Python version flexibility
        if [ -f "$module_path/pyproject.toml" ]; then
            echo "  Using pyproject.toml for dependency management"
            pip install -e "$module_path"
        elif [ -f "$module_path/requirements.txt" ]; then
            echo "  Installing from requirements.txt"
            pip install -r "$module_path/requirements.txt"
        fi
    else
        echo "⊘ No Python dependencies required for: $module_path"
    fi
}

# Function to install Node.js dependencies conditionally
install_nodejs_deps() {
    local module_path=$1
    
    if check_module_requirement "$module_path" "nodejs"; then
        echo "✓ Node.js module detected: $module_path"
        
        if [ -f "$module_path/package.json" ]; then
            echo "  Installing npm dependencies"
            (cd "$module_path" && npm install)
        fi
    else
        echo "⊘ No Node.js dependencies required for: $module_path"
    fi
}

# Main deployment logic
main() {
    echo "Scanning modules for dependency requirements..."
    echo ""
    
    # Check intelligent-automation (Python - OPTIONAL dependencies)
    if [ -d "$ROOT_DIR/intelligent-automation" ]; then
        echo "--- Intelligent Automation Module ---"
        echo "NOTE: This module has OPTIONAL dependencies"
        echo "      Core functionality works without AI/ML services"
        
        if command -v python3 &> /dev/null; then
            install_python_deps "$ROOT_DIR/intelligent-automation"
        else
            echo "⚠ Python not available, skipping intelligent-automation"
            echo "  Module can still be used via API if deployed separately"
        fi
        echo ""
    fi
    
    # Check core services (Node.js)
    if [ -d "$ROOT_DIR/core" ]; then
        echo "--- Core Services ---"
        install_nodejs_deps "$ROOT_DIR/core"
        echo ""
    fi
    
    # Check mcp-servers (Node.js)
    if [ -d "$ROOT_DIR/mcp-servers" ]; then
        echo "--- MCP Servers ---"
        install_nodejs_deps "$ROOT_DIR/mcp-servers"
        echo ""
    fi
    
    # Check root package.json (if exists)
    if [ -f "$ROOT_DIR/package.json" ]; then
        echo "--- Root Dependencies ---"
        echo "✓ Installing workspace root dependencies"
        (cd "$ROOT_DIR" && npm install)
        echo ""
    fi
    
    echo "=== Deployment Complete ==="
    echo ""
    echo "Summary:"
    echo "  - Dependencies installed conditionally based on module type"
    echo "  - No hardcoded dependency templates applied"
    echo "  - Refer to governance/environment-matrix/module-environment-matrix.yml for details"
}

# Run main deployment
main "$@"
