#!/usr/bin/env bash
# run-v2.sh - Quick script to switch to and run v2-multi-islands
#
# Usage:
#   ./run-v2.sh              # Run in auto mode (default)
#   ./run-v2.sh --island=python  # Run specific island
#   ./run-v2.sh --all        # Run all islands
#   ./run-v2.sh --help       # Show help
#
set -e

# Move to the script's directory (repo root) then to v2-multi-islands
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_DIR="$SCRIPT_DIR/v2-multi-islands"

# Check if v2-multi-islands exists
if [ ! -d "$V2_DIR" ]; then
    echo "❌ Error: v2-multi-islands directory not found at $V2_DIR"
    exit 1
fi

cd "$V2_DIR"

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found in $V2_DIR"
    exit 1
fi

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed or not in PATH"
    exit 1
fi

echo "🏝️ Switching to v2-multi-islands..."
echo "📁 Working directory: $(pwd)"
echo ""

# Pass all arguments to the Python script
exec python3 main.py "$@"
