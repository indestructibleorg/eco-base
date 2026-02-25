#!/bin/bash
# Production Bug Log Collection Script

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/bug_analysis_$TIMESTAMP"
mkdir -p "$LOG_DIR"

echo "📊 Collecting logs for analysis..."

# 1. GitHub CI/CD Logs
echo "Collecting GitHub workflow logs..."
gh run list --limit 10 > "$LOG_DIR/recent_runs.txt" 2>/dev/null || echo "No GitHub CLI available"
for run_id in $(gh run list --limit 10 --json databaseId 2>/dev/null | jq -r '.[].databaseId'); do
    echo "Downloading logs for run $run_id..."
    gh run view "$run_id" --log > "$LOG_DIR/run_$run_id.log" 2>/dev/null || echo "Could not download logs for run $run_id"
done

# 2. Application Logs
echo "Collecting application logs..."
find . -name "*.log" -type f -exec cp {} "$LOG_DIR/" \; 2>/dev/null || echo "No log files found"

# 3. Error Reports
echo "Collecting error reports..."
find . -name "*error*" \( -name "*.txt" -o -name "*.md" \) -exec cp {} "$LOG_DIR/" \; 2>/dev/null || echo "No error reports found"

# 4. Recent Git History
echo "Collecting recent git history..."
git log --oneline -20 > "$LOG_DIR/recent_commits.txt" 2>/dev/null || echo "No git history available"

# 5. System Status
echo "Collecting system status..."
docker ps > "$LOG_DIR/docker_status.txt" 2>/dev/null || echo "Docker not available"
docker-compose ps > "$LOG_DIR/docker_compose_status.txt" 2>/dev/null || echo "Docker Compose not available"

echo "✅ Logs collected in: $LOG_DIR"
echo "📋 Log analysis file list:"
ls -lh "$LOG_DIR"

# Generate summary
cat > "$LOG_DIR/analysis_summary.txt" << EOF
=== Bug Analysis Summary ===
Collected: $(date)
Repository: MachineNativeOps/machine-native-ops
Logs Directory: $LOG_DIR

Files Collected:
$(ls -1 "$LOG_DIR" | wc -l) total files

Recommendation: Review failed runs first, then error patterns
EOF

cat "$LOG_DIR/analysis_summary.txt"