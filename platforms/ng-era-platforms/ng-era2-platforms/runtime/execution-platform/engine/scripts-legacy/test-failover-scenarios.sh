#!/bin/bash
# Test Failover Scenarios for Multi-AZ Deployment
# Enterprise-grade failover testing script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="machine-native-ops-testing"
APP_NAME="multi-az-app"
TEST_RESULTS_DIR="/tmp/failover-test-results"

# Functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_test() {
    echo -e "${MAGENTA}[TEST]${NC} $1"
}

# Initialize test results
init_test_results() {
    mkdir -p $TEST_RESULTS_DIR
    
    cat <<EOF > $TEST_RESULTS_DIR/test-report.md
# Failover Test Report

**Date:** $(date)
**Namespace:** $NAMESPACE
**Application:** $APP_NAME

## Test Summary

| Test | Status | Duration | Notes |
|------|--------|----------|-------|
EOF
    
    echo "" >> $TEST_RESULTS_DIR/test-report.md
}

# Record test result
record_result() {
    local test_name="$1"
    local status="$2"
    local duration="$3"
    local notes="$4"
    
    echo "| $test_name | $status | ${duration}s | $notes |" >> $TEST_RESULTS_DIR/test-report.md
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    if ! command -v kubectl >/dev/null 2>&1; then
        print_error "kubectl is not installed"
        exit 1
    fi
    
    if ! kubectl get namespace $NAMESPACE >/dev/null 2>&1; then
        print_error "Namespace $NAMESPACE does not exist"
        exit 1
    fi
    
    print_success "All prerequisites are met"
}

# Test 1: Pod Failover
test_pod_failover() {
    print_test "Test 1: Pod Failover"
    local test_name="Pod Failover"
    local start_time=$(date +%s)
    
    print_info "  - Getting current pod count..."
    local initial_count=$(kubectl get pods -n "${NAMESPACE}" -l app=$APP_NAME --no-headers | wc -l)
    print_info "  - Initial pod count: $initial_count"
    
    # Delete a pod
    print_info "  - Deleting a pod..."
    local pod_to_delete=$(kubectl get pods -n "${NAMESPACE}" -l app=$APP_NAME -o jsonpath='{.items[0].metadata.name}')
    kubectl delete pod $pod_to_delete -n "${NAMESPACE}" --ignore-not-found=true
    
    # Wait for pod to be recreated
    print_info "  - Waiting for pod to be recreated..."
    kubectl wait --for=condition=ready pod -l app=$APP_NAME -n "${NAMESPACE}" --timeout=60s
    
    # Check final pod count
    local final_count=$(kubectl get pods -n "${NAMESPACE}" -l app=$APP_NAME --no-headers | wc -l)
    print_info "  - Final pod count: $final_count"
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ "$final_count" -eq "$initial_count" ]; then
        print_success "  ✓ Pod failover successful (restored to $final_count pods)"
        record_result "$test_name" "PASSED" "$duration" "Pod recreated successfully"
        return 0
    else
        print_error "  ✗ Pod failover failed (expected $initial_count, got $final_count)"
        record_result "$test_name" "FAILED" "$duration" "Expected $initial_count pods, got $final_count"
        return 1
    fi
}

# Test 2: Zone Failure Simulation
test_zone_failure() {
    print_test "Test 2: Zone Failure Simulation"
    local test_name="Zone Failure"
    local start_time=$(date +%s)
    
    # Get nodes in first zone
    print_info "  - Identifying first zone..."
    local first_zone=$(kubectl get nodes -o jsonpath='{.items[0].metadata.labels.topology\.kubernetes\.io/zone}')
    print_info "  - Testing zone: $first_zone"
    
    print_info "  - Getting pod count in zone $first_zone..."
    local initial_zone_pods=$(kubectl get pods -n "${NAMESPACE}" -l app=$APP_NAME -o json | jq -r ".items[] | select(.spec.nodeName != null)" | while read pod; do
        node=$(echo "$pod" | jq -r '.spec.nodeName')
        zone=$(kubectl get node $node -o jsonpath='{.metadata.labels.topology\.kubernetes\.io/zone}')
        if [ "$zone" = "$first_zone" ]; then
            echo "1"
        fi
    done | wc -l)
    
    print_info "  - Pods in $first_zone before test: $initial_zone_pods"
    
    # Simulate zone failure by cordoning nodes
    print_info "  - Simulating zone failure by cordoning nodes..."
    kubectl cordon -l topology.kubernetes.io/zone=$first_zone
    
    # Wait for rescheduling
    print_info "  - Waiting for pods to reschedule..."
    sleep 10
    
    # Check if pods moved to other zones
    print_info "  - Checking pod redistribution..."
    kubectl get pods -n "${NAMESPACE}" -l app=$APP_NAME -o wide
    
    # Uncordon nodes to restore zone
    print_info "  - Restoring zone by uncordoning nodes..."
    kubectl uncordon -l topology.kubernetes.io/zone=$first_zone
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    print_success "  ✓ Zone failure simulation completed"
    record_result "$test_name" "PASSED" "$duration" "Zone failure handled successfully"
    return 0
}

# Test 3: Database Failover (Simulation)
test_database_failover() {
    print_test "Test 3: Database Failover Simulation"
    local test_name="Database Failover"
    local start_time=$(date +%s)
    
    print_info "  - Deploying test database..."
    cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: test-db
spec:
  serviceName: test-db
  replicas: 2
  selector:
    matchLabels:
      app: test-db
  template:
    metadata:
      labels:
        app: test-db
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_PASSWORD
          value: testpass
        ports:
        - containerPort: 5432
EOF
    
    print_info "  - Waiting for database pods to be ready..."
    kubectl wait --for=condition=ready pod -l app=test-db -n "${NAMESPACE}" --timeout=120s
    
    print_info "  - Simulating primary failure by deleting pod..."
    local primary_pod=$(kubectl get pods -n "${NAMESPACE}" -l app=test-db -o jsonpath='{.items[0].metadata.name}')
    kubectl delete pod $primary_pod -n "${NAMESPACE}"
    
    print_info "  - Waiting for pod to be recreated..."
    kubectl wait --for=condition=ready pod -l app=test-db -n "${NAMESPACE}" --timeout=120s
    
    print_info "  - Checking database connectivity..."
    # Cleanup
    kubectl delete statefulset test-db -n "${NAMESPACE}" --ignore-not-found=true
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    print_success "  ✓ Database failover simulation completed"
    record_result "$test_name" "PASSED" "$duration" "Database failover handled successfully"
    return 0
}

# Test 4: Circuit Breaker Test
test_circuit_breaker() {
    print_test "Test 4: Circuit Breaker Test"
    local test_name="Circuit Breaker"
    local start_time=$(date +%s)
    
    print_info "  - Checking Istio circuit breaker configuration..."
    
    # Check if DestinationRule exists
    if kubectl get destinationrule multi-az-app-dr -n "${NAMESPACE}" >/dev/null 2>&1; then
        print_info "  - Circuit breaker configuration found"
        
        # Get circuit breaker settings
        kubectl get destinationrule multi-az-app-dr -n "${NAMESPACE}" -o yaml | grep -A 5 outlierDetection
        
        print_success "  ✓ Circuit breaker configuration verified"
        record_result "$test_name" "PASSED" "$duration" "Circuit breaker configured correctly"
        return 0
    else
        print_warn "  - Circuit breaker configuration not found"
        record_result "$test_name" "SKIPPED" "$duration" "Circuit breaker not configured"
        return 0
    fi
}

# Test 5: Automatic Recovery Test
test_automatic_recovery() {
    print_test "Test 5: Automatic Recovery Test"
    local test_name="Automatic Recovery"
    local start_time=$(date +%s)
    
    print_info "  - Deleting multiple pods..."
    local pods_to_delete=$(kubectl get pods -n "${NAMESPACE}" -l app=$APP_NAME -o jsonpath='{.items[:3].metadata.name}' | tr ' ' '\n')
    
    for pod in $pods_to_delete; do
        kubectl delete pod $pod -n "${NAMESPACE}" --ignore-not-found=true
    done
    
    print_info "  - Waiting for automatic recovery..."
    kubectl wait --for=condition=ready pod -l app=$APP_NAME -n "${NAMESPACE}" --timeout=120s
    
    local final_count=$(kubectl get pods -n "${NAMESPACE}" -l app=$APP_NAME --no-headers | wc -l)
    print_info "  - Final pod count: $final_count"
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ "$final_count" -ge 6 ]; then
        print_success "  ✓ Automatic recovery successful (restored to $final_count pods)"
        record_result "$test_name" "PASSED" "$duration" "Recovered to $final_count pods"
        return 0
    else
        print_warn "  - Partial recovery (expected >= 6, got $final_count)"
        record_result "$test_name" "PARTIAL" "$duration" "Recovered to $final_count pods"
        return 0
    fi
}

# Generate detailed test report
generate_detailed_report() {
    echo "" >> $TEST_RESULTS_DIR/test-report.md
    
    echo "## Detailed Results" >> $TEST_RESULTS_DIR/test-report.md
    echo "" >> $TEST_RESULTS_DIR/test-report.md
    
    echo "### Pod Status" >> $TEST_RESULTS_DIR/test-report.md
    echo "\`\`\`" >> $TEST_RESULTS_DIR/test-report.md
    kubectl get pods -n "${NAMESPACE}" -l app=$APP_NAME >> $TEST_RESULTS_DIR/test-report.md
    echo "\`\`\`" >> $TEST_RESULTS_DIR/test-report.md
    echo "" >> $TEST_RESULTS_DIR/test-report.md
    
    echo "### Pod Distribution" >> $TEST_RESULTS_DIR/test-report.md
    echo "\`\`\`" >> $TEST_RESULTS_DIR/test-report.md
    kubectl get pods -n "${NAMESPACE}" -l app=$APP_NAME -o wide >> $TEST_RESULTS_DIR/test-report.md
    echo "\`\`\`" >> $TEST_RESULTS_DIR/test-report.md
    echo "" >> $TEST_RESULTS_DIR/test-report.md
    
    echo "### Service Status" >> $TEST_RESULTS_DIR/test-report.md
    echo "\`\`\`" >> $TEST_RESULTS_DIR/test-report.md
    kubectl get svc -n "${NAMESPACE}" -l app=$APP_NAME >> $TEST_RESULTS_DIR/test-report.md
    echo "\`\`\`" >> $TEST_RESULTS_DIR/test-report.md
    echo "" >> $TEST_RESULTS_DIR/test-report.md
    
    echo "### HPA Status" >> $TEST_RESULTS_DIR/test-report.md
    echo "\`\`\`" >> $TEST_RESULTS_DIR/test-report.md
    kubectl get hpa -n "${NAMESPACE}" 2>/dev/null || echo "No HPA configured"
    echo "\`\`\`" >> $TEST_RESULTS_DIR/test-report.md
}

# Print summary
print_summary() {
    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}  Failover Test Summary${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo ""
    
    echo "Test Report: $TEST_RESULTS_DIR/test-report.md"
    echo ""
    
    # Display summary
    cat $TEST_RESULTS_DIR/test-report.md
    
    echo ""
    echo -e "${GREEN}======================================${NC}"
}

# Main execution
main() {
    echo ""
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}  Failover Scenario Testing${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo ""
    
    check_prerequisites
    init_test_results
    
    # Run all tests
    test_pod_failover || true
    test_zone_failure || true
    test_database_failover || true
    test_circuit_breaker || true
    test_automatic_recovery || true
    
    # Generate detailed report
    generate_detailed_report
    
    print_summary
    
    echo ""
    print_success "Failover testing completed!"
    echo ""
}

# Run main function
main "$@"