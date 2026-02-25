#!/bin/bash
# GL Unified Charter Activated
# Green Environment Deployment Script

set -euo pipefail

# Configuration
DEPLOY_SHA="${1:-$(git rev-parse HEAD)}"
ENVIRONMENT="green"
DEPLOY_DIR="/opt/deployments/production/${ENVIRONMENT}"
BACKUP_DIR="/opt/deployments/backups"
LOG_FILE="/var/log/deployments/${ENVIRONMENT}-$(date +%Y%m%d%H%M%S).log"
PORT="${GREEN_PORT:-3002}"
HEALTH_CHECK_URL="http://localhost:${PORT}/health"
MAX_RETRIES=10
RETRY_INTERVAL=5

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOG_FILE}"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "${YELLOW}WARN${NC}" "$@"; }
log_error() { log "${RED}ERROR${NC}" "$@"; }
log_success() { log "${GREEN}SUCCESS${NC}" "$@"; }

# Create log directory
mkdir -p "$(dirname ${LOG_FILE})"
mkdir -p "${BACKUP_DIR}"

log_info "Starting ${ENVIRONMENT} deployment for SHA: ${DEPLOY_SHA}"

# Step 1: Create backup of current deployment
backup_current() {
    log_info "Creating backup of current deployment..."
    
    if [ -d "${DEPLOY_DIR}/current" ]; then
        BACKUP_NAME="backup-${ENVIRONMENT}-$(date +%Y%m%d%H%M%S)"
        cp -r "${DEPLOY_DIR}/current" "${BACKUP_DIR}/${BACKUP_NAME}"
        echo "${BACKUP_NAME}" > "${BACKUP_DIR}/latest-${ENVIRONMENT}"
        log_success "Backup created: ${BACKUP_NAME}"
    else
        log_warn "No current deployment to backup"
    fi
}

# Step 2: Prepare new deployment
prepare_deployment() {
    log_info "Preparing new deployment..."
    
    NEW_DEPLOY_DIR="${DEPLOY_DIR}/${DEPLOY_SHA}"
    mkdir -p "${NEW_DEPLOY_DIR}"
    
    # Copy deployment artifacts
    if [ -d "./deploy-package" ]; then
        cp -r ./deploy-package/* "${NEW_DEPLOY_DIR}/"
    elif [ -d "./production-package" ]; then
        cp -r ./production-package/* "${NEW_DEPLOY_DIR}/"
    else
        cp -r . "${NEW_DEPLOY_DIR}/"
    fi
    
    # Create environment file
    cat > "${NEW_DEPLOY_DIR}/.env" << EOF
NODE_ENV=production
PORT=${PORT}
ENVIRONMENT=${ENVIRONMENT}
DEPLOY_SHA=${DEPLOY_SHA}
DEPLOY_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF
    
    log_success "Deployment prepared at: ${NEW_DEPLOY_DIR}"
    echo "${NEW_DEPLOY_DIR}"
}

# Step 3: Stop current service
stop_service() {
    log_info "Stopping current ${ENVIRONMENT} service..."
    
    # Try Docker first
    if docker compose -p "${ENVIRONMENT}" ps 2>/dev/null | grep -q "Up"; then
        docker compose -p "${ENVIRONMENT}" down --timeout 30
        log_success "Docker services stopped"
    fi
    
    # Try PM2
    if pm2 list 2>/dev/null | grep -q "${ENVIRONMENT}-app"; then
        pm2 stop "${ENVIRONMENT}-app" || true
        pm2 delete "${ENVIRONMENT}-app" || true
        log_success "PM2 process stopped"
    fi
    
    # Wait for port to be released
    for i in $(seq 1 10); do
        if ! lsof -i:${PORT} > /dev/null 2>&1; then
            log_success "Port ${PORT} is free"
            return 0
        fi
        log_info "Waiting for port ${PORT} to be released... (${i}/10)"
        sleep 2
    done
    
    log_warn "Port ${PORT} may still be in use"
}

# Step 4: Start new service
start_service() {
    local deploy_dir=$1
    log_info "Starting new ${ENVIRONMENT} service..."
    
    cd "${deploy_dir}"
    
    # Install production dependencies
    if [ -f "package.json" ]; then
        npm ci --production --ignore-scripts || npm install --production
    fi
    
    # Try Docker first
    if [ -f "docker-compose.yml" ] || [ -f "docker-compose.yaml" ]; then
        docker compose -p "${ENVIRONMENT}" up -d --build
        log_success "Docker services started"
    elif [ -f "Dockerfile" ]; then
        docker build -t "app:${ENVIRONMENT}" .
        docker run -d --name "${ENVIRONMENT}-app" -p "${PORT}:${PORT}" --env-file .env "app:${ENVIRONMENT}"
        log_success "Docker container started"
    else
        # Use PM2
        pm2 start npm --name "${ENVIRONMENT}-app" -- start
        pm2 save
        log_success "PM2 process started"
    fi
}

# Step 5: Health check
health_check() {
    log_info "Running health checks..."
    
    for i in $(seq 1 ${MAX_RETRIES}); do
        log_info "Health check attempt ${i}/${MAX_RETRIES}..."
        
        HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${HEALTH_CHECK_URL}" --max-time 10 || echo "000")
        
        if [ "${HTTP_STATUS}" == "200" ]; then
            log_success "Health check passed! (HTTP ${HTTP_STATUS})"
            return 0
        fi
        
        log_warn "Health check returned HTTP ${HTTP_STATUS}, retrying in ${RETRY_INTERVAL}s..."
        sleep ${RETRY_INTERVAL}
    done
    
    log_error "Health check failed after ${MAX_RETRIES} attempts"
    return 1
}

# Step 6: Update symlink
update_symlink() {
    local deploy_dir=$1
    log_info "Updating current symlink..."
    
    rm -f "${DEPLOY_DIR}/current"
    ln -sf "${deploy_dir}" "${DEPLOY_DIR}/current"
    
    log_success "Symlink updated: ${DEPLOY_DIR}/current -> ${deploy_dir}"
}

# Step 7: Cleanup old deployments
cleanup_old() {
    log_info "Cleaning up old deployments..."
    
    # Keep last 5 deployments
    cd "${DEPLOY_DIR}"
    ls -dt */ 2>/dev/null | tail -n +6 | xargs -r rm -rf
    
    # Keep last 10 backups
    cd "${BACKUP_DIR}"
    ls -dt backup-${ENVIRONMENT}-* 2>/dev/null | tail -n +11 | xargs -r rm -rf
    
    log_success "Cleanup complete"
}

# Rollback function
rollback() {
    log_error "Deployment failed, initiating rollback..."
    
    LATEST_BACKUP=$(cat "${BACKUP_DIR}/latest-${ENVIRONMENT}" 2>/dev/null || echo "")
    
    if [ -n "${LATEST_BACKUP}" ] && [ -d "${BACKUP_DIR}/${LATEST_BACKUP}" ]; then
        log_info "Rolling back to: ${LATEST_BACKUP}"
        
        stop_service
        
        ROLLBACK_DIR="${DEPLOY_DIR}/rollback-$(date +%Y%m%d%H%M%S)"
        cp -r "${BACKUP_DIR}/${LATEST_BACKUP}" "${ROLLBACK_DIR}"
        
        start_service "${ROLLBACK_DIR}"
        
        if health_check; then
            update_symlink "${ROLLBACK_DIR}"
            log_success "Rollback successful"
        else
            log_error "Rollback also failed! Manual intervention required."
            exit 2
        fi
    else
        log_error "No backup available for rollback"
        exit 2
    fi
}

# Main execution
main() {
    trap rollback ERR
    
    backup_current
    
    NEW_DEPLOY_DIR=$(prepare_deployment)
    
    stop_service
    
    start_service "${NEW_DEPLOY_DIR}"
    
    if health_check; then
        update_symlink "${NEW_DEPLOY_DIR}"
        cleanup_old
        
        log_success "=========================================="
        log_success "${ENVIRONMENT} deployment completed successfully!"
        log_success "SHA: ${DEPLOY_SHA}"
        log_success "URL: ${HEALTH_CHECK_URL}"
        log_success "=========================================="
        
        # Create deployment record
        cat > "${DEPLOY_DIR}/current/deployment-info.json" << EOF
{
    "environment": "${ENVIRONMENT}",
    "sha": "${DEPLOY_SHA}",
    "deployed_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "port": ${PORT},
    "status": "active"
}
EOF
        
        exit 0
    else
        rollback
        exit 1
    fi
}

main "$@"