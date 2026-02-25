#!/bin/bash

set -e

KEEP_DAYS=7
BLUE_BACKUP_DIR="/var/backups/blue"
GREEN_BACKUP_DIR="/var/backups/green"
LOG_FILE="/var/log/cleanup.log"

echo "======================================"
echo "Cleanup Old Deployments Script"
echo "======================================"
echo "Keep deployments from last: ${KEEP_DAYS} days"
echo ""

# Log cleanup start
echo "[$(date)] Starting cleanup of old deployments" >> ${LOG_FILE}

# Cleanup blue backups
echo "Cleaning up blue backups..."
if [ -d "${BLUE_BACKUP_DIR}" ]; then
    find ${BLUE_BACKUP_DIR} -type d -name "backup-*" -mtime +${KEEP_DAYS} -exec rm -rf {} \;
    echo "✓ Blue backups cleaned up"
else
    echo "✗ Blue backup directory not found"
fi

# Cleanup green backups
echo "Cleaning up green backups..."
if [ -d "${GREEN_BACKUP_DIR}" ]; then
    find ${GREEN_BACKUP_DIR} -type d -name "backup-*" -mtime +${KEEP_DAYS} -exec rm -rf {} \;
    echo "✓ Green backups cleaned up"
else
    echo "✗ Green backup directory not found"
fi

# Cleanup old Docker images (if using Docker)
echo "Cleaning up old Docker images..."
docker system prune -f --filter "until=${KEEP_DAYS}d" 2>/dev/null || echo "No Docker cleanup needed"

# Cleanup old logs
echo "Cleaning up old logs..."
find /var/log -name "deployment-*.log" -mtime +${KEEP_DAYS} -delete 2>/dev/null || echo "No old logs to clean"

# Log cleanup completion
echo "[$(date)] Cleanup completed successfully" >> ${LOG_FILE}

echo "======================================"
echo "Cleanup completed successfully"
echo "======================================"