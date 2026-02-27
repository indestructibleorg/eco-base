#!/bin/bash

# MachineNativeOps Governance CRDs Deployment Script
# This script deploys all governance CRDs to the Kubernetes cluster

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="governance"
CRD_DIR="src/governance"
EXAMPLES_DIR="examples"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "Kubernetes connection verified"
}

create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    
    if kubectl get namespace $NAMESPACE &> /dev/null; then
        log_warning "Namespace $NAMESPACE already exists"
    else
        kubectl create namespace $NAMESPACE
        log_success "Namespace $NAMESPACE created"
    fi
}

deploy_crds() {
    log_info "Deploying Governance CRDs"
    
    # List of CRD files to deploy
    local crd_files=(
        "00-vision-strategy/k8s/crd/risk-register-crd.yaml"
        "00-vision-strategy/k8s/crd/implementation-roadmap-crd.yaml"
        "02-decision/k8s/crd/decision-authority-matrix-crd.yaml"
        "09-performance/k8s/crd/performance-targets-crd.yaml"
    )
    
    for crd_file in "${crd_files[@]}"; do
        local full_path="$CRD_DIR/$crd_file"
        
        if [[ ! -f "$full_path" ]]; then
            log_error "CRD file not found: $full_path"
            continue
        fi
        
        log_info "Deploying CRD: $crd_file"
        
        # Validate CRD syntax first
        if kubectl apply --dry-run=client -f "$full_path"; then
            kubectl apply -f "$full_path"
            log_success "CRD deployed: $crd_file"
        else
            log_error "CRD validation failed: $crd_file"
            exit 1
        fi
    done
}

verify_crds() {
    log_info "Verifying CRD deployment"
    
    local expected_crds=(
        "riskregisters.governance.machinenativeops.io"
        "implementationroadmaps.governance.machinenativeops.io"
        "decisionauthoritymatrices.governance.machinenativeops.io"
        "performancetargets.governance.machinenativeops.io"
    )
    
    local all_found=true
    
    for crd in "${expected_crds[@]}"; do
        if kubectl get crd $crd &> /dev/null; then
            log_success "CRD found: $crd"
        else
            log_error "CRD not found: $crd"
            all_found=false
        fi
    done
    
    if [[ "$all_found" == "true" ]]; then
        log_success "All CRDs deployed successfully"
    else
        log_error "Some CRDs are missing"
        exit 1
    fi
}

deploy_examples() {
    log_info "Deploying example resources"
    
    # List of example files to deploy
    local example_files=(
        "00-vision-strategy/k8s/examples/risk-register-example.yaml"
        "00-vision-strategy/k8s/examples/implementation-roadmap-example.yaml"
        "02-decision/k8s/examples/decision-authority-matrix-example.yaml"
        "09-performance/k8s/examples/performance-targets-example.yaml"
    )
    
    for example_file in "${example_files[@]}"; do
        local full_path="$CRD_DIR/$example_file"
        
        if [[ ! -f "$full_path" ]]; then
            log_error "Example file not found: $full_path"
            continue
        fi
        
        log_info "Deploying example: $example_file"
        
        # Validate example syntax first
        if kubectl apply --dry-run=client -f "$full_path"; then
            kubectl apply -f "$full_path"
            log_success "Example deployed: $example_file"
        else
            log_error "Example validation failed: $example_file"
        fi
    done
}

verify_examples() {
    log_info "Verifying example resources"
    
    local expected_resources=(
        "riskregister/security-risk-001"
        "implementationroadmap/digital-transformation-2025"
        "decisionauthoritymatrix/enterprise-decision-matrix"
        "performancetargets/enterprise-performance-targets-2025"
    )
    
    for resource in "${expected_resources[@]}"; do
        local resource_type=$(echo $resource | cut -d'/' -f1)
        local resource_name=$(echo $resource | cut -d'/' -f2)
        
        if kubectl get $resource_type $resource_name -n "${NAMESPACE}" &> /dev/null; then
            log_success "Resource found: $resource"
        else
            log_warning "Resource not found: $resource"
        fi
    done
}

show_status() {
    log_info "Showing deployment status"
    
    echo ""
    echo "=== CRDs ==="
    kubectl get crd | grep governance.machinenativeops.io
    
    echo ""
    echo "=== Resources in $NAMESPACE namespace ==="
    kubectl get all -n "${NAMESPACE}"
    
    echo ""
    echo "=== Governance Resources ==="
    kubectl get riskregisters -n "${NAMESPACE}" 2>/dev/null || echo "No risk registers found"
    kubectl get implementationroadmaps -n "${NAMESPACE}" 2>/dev/null || echo "No implementation roadmaps found"
    kubectl get decisionauthoritymatrices -n "${NAMESPACE}" 2>/dev/null || echo "No decision authority matrices found"
    kubectl get performancetargets -n "${NAMESPACE}" 2>/dev/null || echo "No performance targets found"
}

cleanup() {
    log_warning "Cleaning up deployed resources"
    
    read -p "Are you sure you want to delete all governance resources? (y/N): " -n 1 -r
    echo
    
    if [[ "${REPLY}" =~ ^[Yy]$ ]]; then
        log_info "Deleting example resources"
        kubectl delete -f $CRD_DIR/00-vision-strategy/k8s/examples/ --ignore-not-found=true
        kubectl delete -f $CRD_DIR/02-decision/k8s/examples/ --ignore-not-found=true
        kubectl delete -f $CRD_DIR/09-performance/k8s/examples/ --ignore-not-found=true
        
        log_info "Deleting CRDs"
        kubectl delete -f $CRD_DIR/00-vision-strategy/k8s/crd/ --ignore-not-found=true
        kubectl delete -f $CRD_DIR/02-decision/k8s/crd/ --ignore-not-found=true
        kubectl delete -f $CRD_DIR/09-performance/k8s/crd/ --ignore-not-found=true
        
        log_info "Deleting namespace"
        kubectl delete namespace $NAMESPACE --ignore-not-found=true
        
        log_success "Cleanup completed"
    else
        log_info "Cleanup cancelled"
    fi
}

show_help() {
    echo "MachineNativeOps Governance CRDs Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy     Deploy all CRDs and examples (default)"
    echo "  verify     Verify deployment status"
    echo "  status     Show current status"
    echo "  cleanup    Remove all deployed resources"
    echo "  help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 deploy    # Deploy all CRDs and examples"
    echo "  $0 verify    # Verify deployment"
    echo "  $0 status    # Show status"
    echo "  $0 cleanup   # Clean up everything"
}

# Main script logic
main() {
    local command=${1:-deploy}
    
    case $command in
        "deploy")
            log_info "Starting deployment of Governance CRDs"
            check_kubectl
            create_namespace
            deploy_crds
            verify_crds
            deploy_examples
            verify_examples
            show_status
            log_success "Deployment completed successfully"
            ;;
        "verify")
            log_info "Verifying deployment"
            check_kubectl
            verify_crds
            verify_examples
            ;;
        "status")
            log_info "Showing deployment status"
            check_kubectl
            show_status
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
