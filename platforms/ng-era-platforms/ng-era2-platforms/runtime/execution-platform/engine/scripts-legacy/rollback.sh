#!/bin/bash
# GL Unified Charter Activated
# Enhanced Rollback Script with Multiple Strategies

set -euo pipefail

# Configuration
ROLLBACK_TARGET="${1:-}"
ENVIRONMENT="${2:-production}"
ROLLBACK_STRATEGY="${3:-auto}"  # auto, backup, git, docker
DEPLOY_DIR="/opt/deployments/${ENVIRONMENT}"
BACKUP_DIR="/opt/deployments/backups"
STATE_FILE="/var/lib/deployment/active-environment"
LOG_FILE="/var/log/deployments/rollback-$(date +%Y%m%d%H%M%S).log"

BLUE_PORT="${BLUE_PORT:-3001}"
GREEN_PORT="${GREEN_PORT:-3002}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging
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

# Create directories
mkdir -p "$(dirname ${LOG_FILE})"

# Print usage
usage() {
    cat << EOF
Usage: $0 <target> [environment] [strategy]

Arguments:
    target      - Version/tag/commit to rollback to, or 'previous' for last deployment
    environment - Target environment: staging, production (default: production)
    strategy    - Rollback strategy: auto, backup, git, docker (default: auto)

Examples:
    $0 previous                     # Rollback to previous deployment
    $0 v1.2.3 production           # Rollback to specific version
    $0 abc123 staging git          # Rollback using git checkout

Strategies:
    auto   - Automatically determine best rollback method
    backup - Restore from backup directory
    git    - Checkout specific git commit/tag
    docker - Use Docker image from registry
EOF
    exit 1
}

# Validate input
if [ -z "${ROLLBACK_TARGET}" ]; then
    usage
fi

log_info "=========================================="
log_info "Rollback Request"
log_info "Target: ${ROLLBACK_TARGET}"
log_info "Environment: ${ENVIRONMENT}"
log_info "Strategy: ${ROLLBACK_STRATEGY}"
log_info "=========================================="

# Get current active environment (for blue-green)
get_active_env() {
    if [ -f "${STATE_FILE}" ]; then
        cat "${STATE_FILE}"
    else
        echo "blue"
    fi
}

# Find backup by target
find_backup() {
    local target=$1
    
    if [ "${target}" == "previous" ]; then
        # Get most recent backup
        ls -dt "${BACKUP_DIR}"/backup-* 2>/dev/null | head -1
    else
        # Find backup matching target
        ls -d "${BACKUP_DIR}"/*"${target}"* 2>/dev/null | head -1
    fi
}

# Find git reference
find_git_ref() {
    local target=$1
    
    if [ "${target}" == "previous" ]; then
        # Get previous commit
        git rev-parse HEAD~1 2>/dev/null
    else
        # Try to resolve as tag, branch, or commit
        git rev-parse "${target}" 2>/dev/null || \
        git rev-parse "refs/tags/${target}" 2>/dev/null || \
        git rev-parse "origin/${target}" 2>/dev/null || \
        echo ""
    fi
}

# Find Docker image
find_docker_image() {
    local target=$1
    local registry="${DOCKER_REGISTRY:-ghcr.io}"
    local image="${DOCKER_IMAGE:-machine-native-ops}"
    
    if [ "${target}" == "previous" ]; then
        # Get second most recent image
        docker images "${registry}/${image}" --format "{{.Tag}}" | head -2 | tail -1
    else
        # Check if image exists
        docker images "${registry}/${image}:${target}" --format "{{.Tag}}" | head -1
    fi
}

# Strategy: Rollback from backup
rollback_from_backup() {
    local backup_path=$1
    local env_dir=$2
    
    log_info "Rolling back from backup: ${backup_path}"
    
    if [ ! -d "${backup_path}" ]; then
        log_error "Backup not found: ${backup_path}"
        return 1
    fi
    
    # Create rollback directory
    local rollback_dir="${env_dir}/rollback-$(date +%Y%m%d%H%M%S)"
    mkdir -p "${rollback_dir}"
    
    # Copy backup
    cp -r "${backup_path}"/* "${rollback_dir}/"
    
    echo "${rollback_dir}"
}

# Strategy: Rollback from git
rollback_from_git() {
    local git_ref=$1
    local env_dir=$2
    
    log_info "Rolling back from git: ${git_ref}"
    
    if [ -z "${git_ref}" ]; then
        log_error "Invalid git reference"
        return 1
    fi
    
    # Create rollback directory
    local rollback_dir="${env_dir}/rollback-$(date +%Y%m%d%H%M%S)"
    mkdir -p "${rollback_dir}"
    
    # Clone/checkout specific version
    git archive "${git_ref}" | tar -x -C "${rollback_dir}"
    
    # Build if necessary
    cd "${rollback_dir}"
    if [ -f "package.json" ]; then
        npm ci --production || npm install --production
        npm run build 2>/dev/null || true
    fi
    
    echo "${rollback_dir}"
}

# Strategy: Rollback from Docker
rollback_from_docker() {
    local image_tag=$1
    local env=$2
    
    log_info "Rolling back from Docker image: ${image_tag}"
    
    local registry="${DOCKER_REGISTRY:-ghcr.io}"
    local image="${DOCKER_IMAGE:-machine-native-ops}"
    local full_image="${registry}/${image}:${image_tag}"
    
    # Pull image if not present
    docker pull "${full_image}" || {
        log_error "Failed to pull Docker image: ${full_image}"
        return 1
    }
    
    # Stop current container
    docker stop "${env}-app" 2>/dev/null || true
    docker rm "${env}-app" 2>/dev/null || true
    
    # Determine port
    local port
    if [ "${env}" == "blue" ]; then
        port="${BLUE_PORT}"
    else
        port="${GREEN_PORT}"
    fi
    
    # Start new container
    docker run -d \
        --name "${env}-app" \
        --restart unless-stopped \
        -p "${port}:${port}" \
        -e NODE_ENV=production \
        -e PORT="${port}" \
        "${full_image}"
    
    echo "docker:${full_image}"
}

# Stop current service
stop_current_service() {
    local env=$1
    
    log_info "Stopping current ${env} service..."
    
    # Docker compose
    docker compose -p "${env}" down 2>/dev/null || true
    
    # Docker container
    docker stop "${env}-app" 2>/dev/null || true
    docker rm "${env}-app" 2>/dev/null || true
    
    # PM2
    pm2 stop "${env}-app" 2>/dev/null || true
    pm2 delete "${env}-app" 2>/dev/null || true
    
    log_success "Current service stopped"
}

# Start service from directory
start_service_from_dir() {
    local deploy_dir=$1
    local env=$2
    
    log_info "Starting service from: ${deploy_dir}"
    
    cd "${deploy_dir}"
    
    # Determine port
    local port
    if [ "${env}" == "blue" ]; then
        port="${BLUE_PORT}"
    else
        port="${GREEN_PORT}"
    fi
    
    # Create/update .env
    cat >> .env << EOF
PORT=${port}
NODE_ENV=production
ENVIRONMENT=${env}
EOF
    
    # Try Docker compose first
    if [ -f "docker-compose.yml" ]; then
        docker compose -p "${env}" up -d --build
        log_success "Started with Docker Compose"
    elif [ -f "Dockerfile" ]; then
        docker build -t "app:${env}" .
        docker run -d --name "${env}-app" -p "${port}:${port}" --env-file .env "app:${env}"
        log_success "Started with Docker"
    else
        # PM2
        pm2 start npm --name "${env}-app" -- start
        pm2 save
        log_success "Started with PM2"
    fi
}

# Health check
health_check() {
    local env=$1
    local port
    
    if [ "${env}" == "blue" ]; then
        port="${BLUE_PORT}"
    else
        port="${GREEN_PORT}"
    fi
    
    log_info "Running health check on port ${port}..."
    
    for i in $(seq 1 10); do
        HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/health" --max-time 5 || echo "000")
        
        if [ "${HTTP_STATUS}" == "200" ]; then
            log_success "Health check passed!"
            return 0
        fi
        
        log_info "Health check attempt ${i}/10: HTTP ${HTTP_STATUS}"
        sleep 5
    done
    
    log_error "Health check failed!"
    return 1
}

# Update symlink
update_symlink() {
    local deploy_dir=$1
    local env_dir=$2
    
    rm -f "${env_dir}/current"
    ln -sf "${deploy_dir}" "${env_dir}/current"
    
    log_success "Symlink updated"
}

# Auto-detect best rollback strategy
auto_detect_strategy() {
    local target=$1
    
    # Check backup first (fastest)
    if find_backup "${target}" > /dev/null 2>&1; then
        echo "backup"
        return
    fi
    
    # Check Docker
    if find_docker_image "${target}" > /dev/null 2>&1; then
        echo "docker"
        return
    fi
    
    # Check git
    if find_git_ref "${target}" > /dev/null 2>&1; then
        echo "git"
        return
    fi
    
    echo "unknown"
}

# Main rollback execution
main() {
    local active_env=$(get_active_env)
    local target_env="${active_env}"  # Rollback to same environment
    local env_dir="${DEPLOY_DIR}/${target_env}"
    
    mkdir -p "${env_dir}"
    
    # Determine strategy
    local strategy="${ROLLBACK_STRATEGY}"
    if [ "${strategy}" == "auto" ]; then
        strategy=$(auto_detect_strategy "${ROLLBACK_TARGET}")
        log_info "Auto-detected strategy: ${strategy}"
    fi
    
    if [ "${strategy}" == "unknown" ]; then
        log_error "Could not find rollback target: ${ROLLBACK_TARGET}"
        log_error "Please specify a valid backup, git reference, or Docker image"
        exit 1
    fi
    
    # Execute rollback based on strategy
    local rollback_result
    case "${strategy}" in
        backup)
            local backup_path=$(find_backup "${ROLLBACK_TARGET}")
            stop_current_service "${target_env}"
            rollback_result=$(rollback_from_backup "${backup_path}" "${env_dir}")
            start_service_from_dir "${rollback_result}" "${target_env}"
            ;;
        git)
            local git_ref=$(find_git_ref "${ROLLBACK_TARGET}")
            stop_current_service "${target_env}"
            rollback_result=$(rollback_from_git "${git_ref}" "${env_dir}")
            start_service_from_dir "${rollback_result}" "${target_env}"
            ;;
        docker)
            local image_tag=$(find_docker_image "${ROLLBACK_TARGET}")
            rollback_result=$(rollback_from_docker "${image_tag}" "${target_env}")
            ;;
        *)
            log_error "Unknown strategy: ${strategy}"
            exit 1
            ;;
    esac
    
    # Verify rollback
    sleep 10
    
    if health_check "${target_env}"; then
        # Update symlink if applicable
        if [[ "${rollback_result}" != docker:* ]]; then
            update_symlink "${rollback_result}" "${env_dir}"
        fi
        
        log_success "=========================================="
        log_success "Rollback completed successfully!"
        log_success "Target: ${ROLLBACK_TARGET}"
        log_success "Strategy: ${strategy}"
        log_success "Environment: ${target_env}"
        log_success "=========================================="
        
        # Create rollback record
        cat > "${env_dir}/rollback-record.json" << EOF
{
    "rollback_target": "${ROLLBACK_TARGET}",
    "strategy": "${strategy}",
    "environment": "${target_env}",
    "result": "${rollback_result}",
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "status": "success"
}
EOF
        
        exit 0
    else
        log_error "Rollback verification failed!"
        log_error "Manual intervention may be required"
        exit 1
    fi
}

main "$@"