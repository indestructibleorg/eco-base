#!/bin/bash
# Deploy Istio to Testing Environment
# Enterprise-grade Istio deployment script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="machine-native-ops-testing"
ISTIO_VERSION="1.20.0"

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

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    if ! command -v kubectl &amp;&gt; /dev/null; then
        print_error "kubectl is not installed"
        exit 1
    fi
    
    if ! command -v istioctl &amp;&gt; /dev/null; then
        print_error "istioctl is not installed"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &amp;&gt; /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    print_success "All prerequisites are met"
}

# Create testing namespace
create_namespace() {
    print_info "Creating testing namespace: $NAMESPACE"
    
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Enable Istio sidecar injection
    kubectl label namespace $NAMESPACE istio-injection=enabled --overwrite
    
    print_success "Namespace created with Istio injection enabled"
}

# Deploy Istio configuration
deploy_istio_config() {
    print_info "Deploying Istio configuration..."
    
    # Apply namespace configuration
    print_info "  - Applying namespace configuration..."
    kubectl apply -f k8s/istio/namespace.yaml || true
    
    # Update namespace for testing environment
    print_info "  - Updating namespace for testing environment..."
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | \
        yq '.metadata.labels.istio-injection = "enabled"' | \
        yq '.metadata.labels.environment = "testing"' | \
        kubectl apply -f -
    
    # Apply gateway (update namespace)
    print_info "  - Deploying gateway..."
    yq 'del .metadata.namespace' k8s/istio/gateway.yaml | \
        yq '.metadata.namespace = "'$NAMESPACE'"' | \
        kubectl apply -f -
    
    # Apply virtual services
    print_info "  - Deploying virtual services..."
    yq 'del .metadata.namespace' k8s/istio/virtualservices/canary-deployment.yaml | \
        yq '.metadata.namespace = "'$NAMESPACE'"' | \
        kubectl apply -f -
    
    yq 'del .metadata.namespace' k8s/istio/virtualservices/resilience-features.yaml | \
        yq '.metadata.namespace = "'$NAMESPACE'"' | \
        kubectl apply -f -
    
    # Apply destination rules
    print_info "  - Deploying destination rules..."
    yq 'del .metadata.namespace' k8s/istio/destination-rules/circuit-breaking.yaml | \
        yq '.metadata.namespace = "'$NAMESPACE'"' | \
        kubectl apply -f -
    
    # Apply security policies
    print_info "  - Applying security policies..."
    yq 'del .metadata.namespace' k8s/istio/security/peer-authentication.yaml | \
        yq '.metadata.namespace = "'$NAMESPACE'"' | \
        kubectl apply -f -
    
    yq 'del .metadata.namespace' k8s/istio/security/service-entry.yaml | \
        yq '.metadata.namespace = "'$NAMESPACE'"' | \
        kubectl apply -f -
    
    print_success "Istio configuration deployed successfully"
}

# Verify Istio installation
verify_istio() {
    print_info "Verifying Istio installation..."
    
    # Check namespace
    print_info "  - Checking namespace..."
    kubectl get namespace $NAMESPACE
    
    # Check sidecar injection
    print_info "  - Checking sidecar injection..."
    INJECTION_ENABLED=$(kubectl get namespace $NAMESPACE -o jsonpath='{.metadata.labels.istio-injection}')
    if [ "$INJECTION_ENABLED" = "enabled" ]; then
        print_success "Sidecar injection is enabled"
    else
        print_error "Sidecar injection is not enabled"
        exit 1
    fi
    
    # Check Istio components
    print_info "  - Checking Istio components in istio-system..."
    kubectl get pods -n istio-system
    
    # Check gateways
    print_info "  - Checking gateways..."
    kubectl get gateway -n $NAMESPACE
    
    # Check virtual services
    print_info "  - Checking virtual services..."
    kubectl get virtualservice -n $NAMESPACE
    
    # Check destination rules
    print_info "  - Checking destination rules..."
    kubectl get destinationrule -n $NAMESPACE
    
    # Check security policies
    print_info "  - Checking security policies..."
    kubectl get peerauthentication -n $NAMESPACE
    kubectl get authorizationpolicy -n $NAMESPACE
    
    print_success "Istio verification completed successfully"
}

# Deploy sample application for testing
deploy_sample_app() {
    print_info "Deploying sample application for testing..."
    
    # Create a simple HTTP server deployment
    cat &lt;&lt;EOF | kubectl apply -n $NAMESPACE -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sample-app
  labels:
    app: sample-app
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sample-app
  template:
    metadata:
      labels:
        app: sample-app
        version: v1
    spec:
      containers:
      - name: sample-app
        image: nginx:alpine
        ports:
        - containerPort: 80
        livenessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: sample-app
  labels:
    app: sample-app
spec:
  ports:
  - port: 80
    targetPort: 80
    protocol: TCP
  selector:
    app: sample-app
---
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: sample-app
spec:
  hosts:
  - sample-app
  http:
  - route:
    - destination:
        host: sample-app
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: sample-app
spec:
  host: sample-app
  trafficPolicy:
    loadBalancer:
      simple: ROUND_ROBIN
EOF
    
    print_success "Sample application deployed"
    
    # Wait for pods to be ready
    print_info "Waiting for sample app pods to be ready..."
    kubectl wait --for=condition=ready pod -l app=sample-app -n $NAMESPACE --timeout=60s
    
    print_success "Sample application is ready"
}

# Test Istio routing
test_istio_routing() {
    print_info "Testing Istio routing..."
    
    # Get a sample pod
    SAMPLE_POD=$(kubectl get pod -l app=sample-app -n $NAMESPACE -o jsonpath='{.items[0].metadata.name}')
    
    # Test intra-cluster routing
    print_info "  - Testing intra-cluster routing..."
    kubectl exec -n $NAMESPACE $SAMPLE_POD -- wget -qO- http://sample-app/ | head -n 5
    
    # Check Istio proxy
    print_info "  - Checking Istio sidecar proxy..."
    kubectl exec -n $NAMESPACE $SAMPLE_POD -c istio-proxy -- curl -s localhost:15000/clusters | head -n 10
    
    print_success "Istio routing test completed"
}

# Print summary
print_summary() {
    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}  Istio Deployment Summary${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo ""
    echo "Namespace: $NAMESPACE"
    echo "Istio Sidecar Injection: Enabled"
    echo "Sample App: Deployed (3 replicas)"
    echo ""
    echo "Next Steps:"
    echo "  1. View pods: kubectl get pods -n $NAMESPACE"
    echo "  2. View services: kubectl get svc -n $NAMESPACE"
    echo "  3. View Istio resources: kubectl get all -n $NAMESPACE"
    echo "  4. Check sidecar: kubectl describe pod <pod-name> -n $NAMESPACE"
    echo "  5. Test routing: kubectl exec -it <pod-name> -n $NAMESPACE -- sh"
    echo ""
    echo -e "${GREEN}======================================${NC}"
}

# Main execution
main() {
    echo ""
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}  Istio Deployment to Testing Env${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo ""
    
    check_prerequisites
    create_namespace
    deploy_istio_config
    verify_istio
    deploy_sample_app
    test_istio_routing
    print_summary
    
    echo ""
    print_success "Istio deployment to testing environment completed successfully!"
    echo ""
}

# Run main function
main "$@"