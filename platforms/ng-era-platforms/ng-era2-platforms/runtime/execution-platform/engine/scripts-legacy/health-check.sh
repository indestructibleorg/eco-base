#!/bin/bash

set -e

HEALTH_URL=$1
MAX_RETRIES=5
RETRY_INTERVAL=10

echo "======================================"
echo "Health Check Script"
echo "======================================"
echo "URL: ${HEALTH_URL}"
echo "Max retries: ${MAX_RETRIES}"
echo "Retry interval: ${RETRY_INTERVAL}s"
echo ""

# Validate input
if [ -z "${HEALTH_URL}" ]; then
    echo "Error: Health check URL is required"
    echo "Usage: ./health-check.sh <URL>"
    exit 1
fi

# Perform health check
for i in $(seq 1 ${MAX_RETRIES}); do
    echo "Attempt ${i}/${MAX_RETRIES}: Checking ${HEALTH_URL}..."
    
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" ${HEALTH_URL} || echo "000")
    
    if [ "${RESPONSE}" == "200" ] || [ "${RESPONSE}" == "204" ]; then
        echo "✓ Health check passed (HTTP ${RESPONSE})"
        echo "======================================"
        exit 0
    else
        echo "✗ Health check failed (HTTP ${RESPONSE})"
        
        if [ ${i} -lt ${MAX_RETRIES} ]; then
            echo "Waiting ${RETRY_INTERVAL}s before retry..."
            sleep ${RETRY_INTERVAL}
        fi
    fi
done

echo "======================================"
echo "Health check failed after ${MAX_RETRIES} attempts"
echo "======================================"
exit 1