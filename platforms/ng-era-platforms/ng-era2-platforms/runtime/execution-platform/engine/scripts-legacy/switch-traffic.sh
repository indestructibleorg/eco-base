#!/bin/bash
# GL Unified Charter Activated
# Traffic Switching Script for Blue-Green Deployment

set -euo pipefail

# Configuration
TARGET_ENV="${1:-}"
TRAFFIC_PERCENTAGE="${2:-100}"
NGINX_CONF="/etc/nginx/conf.d/upstream.conf"
HAPROXY_CONF="/etc/haproxy/haproxy.cfg"
STATE_FILE="/var/lib/deployment/active-environment"
LOG_FILE="/var/log/deployments/traffic-switch-$(date +%Y%m%d%H%M%S).log"

BLUE_HOST="${BLUE_HOST:-localhost}"
BLUE_PORT="${BLUE_PORT:-3001}"
GREEN_HOST="${GREEN_HOST:-localhost}"
GREEN_PORT="${GREEN_PORT:-3002}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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
mkdir -p "$(dirname ${STATE_FILE})"

# Validate input
if [ -z "${TARGET_ENV}" ]; then
    log_error "Usage: $0 <blue|green> [traffic_percentage]"
    log_error "Example: $0 green 100"
    exit 1
fi

if [[ ! "${TARGET_ENV}" =~ ^(blue|green)$ ]]; then
    log_error "Invalid environment: ${TARGET_ENV}. Must be 'blue' or 'green'"
    exit 1
fi

if ! [[ "${TRAFFIC_PERCENTAGE}" =~ ^[0-9]+$ ]] || [ "${TRAFFIC_PERCENTAGE}" -lt 0 ] || [ "${TRAFFIC_PERCENTAGE}" -gt 100 ]; then
    log_error "Invalid traffic percentage: ${TRAFFIC_PERCENTAGE}. Must be 0-100"
    exit 1
fi

log_info "=========================================="
log_info "Traffic Switch Request"
log_info "Target: ${TARGET_ENV}"
log_info "Traffic: ${TRAFFIC_PERCENTAGE}%"
log_info "=========================================="

# Get current active environment
get_current_active() {
    if [ -f "${STATE_FILE}" ]; then
        cat "${STATE_FILE}"
    else
        echo "blue"  # Default
    fi
}

# Verify target environment is healthy
verify_target_health() {
    local env=$1
    local host port
    
    if [ "${env}" == "blue" ]; then
        host="${BLUE_HOST}"
        port="${BLUE_PORT}"
    else
        host="${GREEN_HOST}"
        port="${GREEN_PORT}"
    fi
    
    log_info "Verifying ${env} environment health at ${host}:${port}..."
    
    for i in $(seq 1 5); do
        HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://${host}:${port}/health" --max-time 5 || echo "000")
        
        if [ "${HTTP_STATUS}" == "200" ]; then
            log_success "${env} environment is healthy"
            return 0
        fi
        
        log_warn "Health check attempt ${i}/5 returned ${HTTP_STATUS}"
        sleep 2
    done
    
    log_error "${env} environment is not healthy!"
    return 1
}

# Update Nginx configuration
update_nginx() {
    local target=$1
    local percentage=$2
    
    log_info "Updating Nginx configuration..."
    
    if [ ! -d "$(dirname ${NGINX_CONF})" ]; then
        log_warn "Nginx configuration directory not found, skipping..."
        return 0
    fi
    
    local blue_weight green_weight
    
    if [ "${target}" == "blue" ]; then
        blue_weight="${percentage}"
        green_weight=$((100 - percentage))
    else
        blue_weight=$((100 - percentage))
        green_weight="${percentage}"
    fi
    
    # Generate upstream configuration
    cat > "${NGINX_CONF}" << EOF
# GL Unified Charter - Auto-generated configuration
# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# Active: ${target} (${percentage}%)

upstream app_backend {
    # Blue environment
    server ${BLUE_HOST}:${BLUE_PORT} weight=${blue_weight};
    
    # Green environment
    server ${GREEN_HOST}:${GREEN_PORT} weight=${green_weight};
    
    keepalive 32;
}

# Health check upstream (for monitoring)
upstream blue_backend {
    server ${BLUE_HOST}:${BLUE_PORT};
}

upstream green_backend {
    server ${GREEN_HOST}:${GREEN_PORT};
}
EOF
    
    # Test configuration
    if nginx -t 2>/dev/null; then
        nginx -s reload
        log_success "Nginx configuration updated and reloaded"
    else
        log_error "Nginx configuration test failed!"
        return 1
    fi
}

# Update HAProxy configuration
update_haproxy() {
    local target=$1
    local percentage=$2
    
    log_info "Updating HAProxy configuration..."
    
    if [ ! -f "${HAPROXY_CONF}" ]; then
        log_warn "HAProxy configuration not found, skipping..."
        return 0
    fi
    
    local blue_weight green_weight
    
    if [ "${target}" == "blue" ]; then
        blue_weight="${percentage}"
        green_weight=$((100 - percentage))
    else
        blue_weight=$((100 - percentage))
        green_weight="${percentage}"
    fi
    
    # Backup current configuration
    cp "${HAPROXY_CONF}" "${HAPROXY_CONF}.bak"
    
    # Update weights using sed
    sed -i "s/server blue .* weight [0-9]*/server blue ${BLUE_HOST}:${BLUE_PORT} weight ${blue_weight}/" "${HAPROXY_CONF}"
    sed -i "s/server green .* weight [0-9]*/server green ${GREEN_HOST}:${GREEN_PORT} weight ${green_weight}/" "${HAPROXY_CONF}"
    
    # Test and reload
    if haproxy -c -f "${HAPROXY_CONF}" 2>/dev/null; then
        systemctl reload haproxy || haproxy -f "${HAPROXY_CONF}" -sf $(cat /var/run/haproxy.pid)
        log_success "HAProxy configuration updated and reloaded"
    else
        log_error "HAProxy configuration test failed, restoring backup..."
        mv "${HAPROXY_CONF}.bak" "${HAPROXY_CONF}"
        return 1
    fi
}

# Update state file
update_state() {
    local target=$1
    local percentage=$2
    
    echo "${target}" > "${STATE_FILE}"
    
    # Create detailed state record
    cat > "${STATE_FILE}.json" << EOF
{
    "active_environment": "${target}",
    "traffic_percentage": ${percentage},
    "switched_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "switched_by": "${USER:-system}",
    "blue": {
        "host": "${BLUE_HOST}",
        "port": ${BLUE_PORT},
        "weight": $( [ "${target}" == "blue" ] && echo "${percentage}" || echo "$((100 - percentage))" )
    },
    "green": {
        "host": "${GREEN_HOST}",
        "port": ${GREEN_PORT},
        "weight": $( [ "${target}" == "green" ] && echo "${percentage}" || echo "$((100 - percentage))" )
    }
}
EOF
    
    log_success "State file updated: ${target} (${percentage}%)"
}

# Verify traffic switch
verify_switch() {
    local target=$1
    
    log_info "Verifying traffic switch..."
    
    # Wait for changes to propagate
    sleep 5
    
    # Make test requests
    local success_count=0
    local total_requests=10
    
    for i in $(seq 1 ${total_requests}); do
        HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost/health" --max-time 5 || echo "000")
        
        if [ "${HTTP_STATUS}" == "200" ]; then
            ((success_count++))
        fi
    done
    
    local success_rate=$((success_count * 100 / total_requests))
    
    if [ ${success_rate} -ge 80 ]; then
        log_success "Traffic switch verified: ${success_rate}% success rate"
        return 0
    else
        log_error "Traffic switch verification failed: ${success_rate}% success rate"
        return 1
    fi
}

# Main execution
main() {
    CURRENT_ACTIVE=$(get_current_active)
    log_info "Current active environment: ${CURRENT_ACTIVE}"
    
    # Verify target is healthy before switching
    if ! verify_target_health "${TARGET_ENV}"; then
        log_error "Cannot switch traffic to unhealthy environment"
        exit 1
    fi
    
    # Update load balancer configurations
    update_nginx "${TARGET_ENV}" "${TRAFFIC_PERCENTAGE}" || true
    update_haproxy "${TARGET_ENV}" "${TRAFFIC_PERCENTAGE}" || true
    
    # Update state
    update_state "${TARGET_ENV}" "${TRAFFIC_PERCENTAGE}"
    
    # Verify the switch
    if verify_switch "${TARGET_ENV}"; then
        log_success "=========================================="
        log_success "Traffic switch completed successfully!"
        log_success "Active: ${TARGET_ENV} (${TRAFFIC_PERCENTAGE}%)"
        log_success "=========================================="
        
        # Create switch record
        cat >> "/var/log/deployments/traffic-switches.log" << EOF
$(date -u +"%Y-%m-%dT%H:%M:%SZ") | ${CURRENT_ACTIVE} -> ${TARGET_ENV} | ${TRAFFIC_PERCENTAGE}% | SUCCESS
EOF
        
        exit 0
    else
        log_error "Traffic switch verification failed!"
        
        # Attempt to rollback
        log_warn "Attempting to rollback to ${CURRENT_ACTIVE}..."
        update_nginx "${CURRENT_ACTIVE}" "100" || true
        update_haproxy "${CURRENT_ACTIVE}" "100" || true
        update_state "${CURRENT_ACTIVE}" "100"
        
        exit 1
    fi
}

main "$@"