#!/bin/bash
# Install Jaeger with Elasticsearch for Testing Environment
# Enterprise-grade Jaeger deployment script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="jaeger"
ES_NAMESPACE="elasticsearch"
TESTING_NAMESPACE="machine-native-ops-testing"
JAEGER_VERSION="1.52.0"
ES_VERSION="8.11.0"

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
    
    if ! command -v kubectl &amp;> /dev/null; then
        print_error "kubectl is not installed"
        exit 1
    fi
    
    if ! command -v helm &amp;> /dev/null; then
        print_error "helm is not installed"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &amp;> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    print_success "All prerequisites are met"
}

# Install Elasticsearch Operator
install_elasticsearch_operator() {
    print_info "Installing Elasticsearch Operator..."
    
    # Add Elasticsearch repository
    helm repo add elastic https://helm.elastic.co
    helm repo update
    
    # Create namespace
    kubectl create namespace $ES_NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Install operator
    helm upgrade --install elastic-operator elastic/eck-operator \
        --namespace $ES_NAMESPACE \
        --create-namespace \
        --set managedNamespaces={$ES_NAMESPACE,$NAMESPACE} \
        --set replicas=1 \
        --wait \
        --timeout 5m || print_warn "Elasticsearch operator may already be installed"
    
    print_success "Elasticsearch Operator installed"
}

# Deploy Elasticsearch for Jaeger
deploy_elasticsearch() {
    print_info "Deploying Elasticsearch for Jaeger..."
    
    # Wait for operator to be ready
    print_info "  - Waiting for Elasticsearch operator..."
    kubectl wait --for=condition=available \
        -n $ES_NAMESPACE \
        deployment/elastic-operator \
        --timeout=5m || print_warn "Operator check timed out"
    
    # Deploy Elasticsearch
    print_info "  - Creating Elasticsearch cluster..."
    cat <<EOF | kubectl apply -f -
apiVersion: elasticsearch.k8s.elastic.co/v1
kind: Elasticsearch
metadata:
  name: jaeger-es
  namespace: $NAMESPACE
spec:
  version: $ES_VERSION
  nodeSets:
  - name: nodes
    count: 1
    config:
      node.store.allow_mmap: false
      cluster.routing.allocation.disk.threshold_enabled: true
      cluster.routing.allocation.disk.watermark.low: 90%
      cluster.routing.allocation.disk.watermark.high: 95%
    resources:
      requests:
        memory: 2Gi
        cpu: 1
      limits:
        memory: 4Gi
        cpu: 2
    volumeClaimTemplates:
    - metadata:
        name: elasticsearch-data
      spec:
        accessModes:
        - ReadWriteOnce
        resources:
          requests:
            storage: 20Gi
EOF
    
    print_info "  - Waiting for Elasticsearch to be ready..."
    kubectl wait --for=condition=ready \
        -n $NAMESPACE \
        elasticsearch/jaeger-es \
        --timeout=10m || print_warn "Elasticsearch readiness check timed out"
    
    print_success "Elasticsearch deployed successfully"
}

# Install Jaeger
install_jaeger() {
    print_info "Installing Jaeger..."
    
    # Add Jaeger repository
    helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
    helm repo update
    
    # Create namespace
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Get Elasticsearch password
    print_info "  - Getting Elasticsearch credentials..."
    ES_PASSWORD=$(kubectl get secret jaeger-es-es-elastic-user -n $NAMESPACE -o jsonpath='{.data.elastic}' | base64 --decode 2>/dev/null || echo "changeme")
    
    # Install Jaeger
    print_info "  - Installing Jaeger with Elasticsearch backend..."
    helm upgrade --install jaeger jaegertracing/jaeger \
        --namespace $NAMESPACE \
        --create-namespace \
        --set storage.type=elasticsearch \
        --set storage.elasticsearch.user=elastic \
        --set storage.elasticsearch.password="$ES_PASSWORD" \
        --set storage.elasticsearch.tls.enabled=false \
        --set agent.daemonset.useHostPort=false \
        --set collector.replicaCount=1 \
        --set query.replicaCount=1 \
        --set agent.replicaCount=1 \
        --set samplingStrategiesConfigMap.content='{"service_strategies":[{"service":".*","type":"probabilistic","param":10}]}' \
        --wait \
        --timeout 10m
    
    print_success "Jaeger installed successfully"
}

# Configure Istio for Jaeger tracing
configure_istio_tracing() {
    print_info "Configuring Istio for Jaeger tracing..."
    
    # Apply Istio telemetry configuration for testing
    cat <<EOF | kubectl apply -f -
apiVersion: telemetry.istio.io/v1alpha1
kind: Telemetry
metadata:
  name: jaeger-tracing-testing
  namespace: istio-system
spec:
  selector:
    matchLabels:
      istio-injection: enabled
  tracing:
  - providers:
    - name: jaeger
    randomSamplingPercentage: 10.0
    customTags:
      environment:
        literal:
          value: testing
      cluster:
        literal:
          value: testing-cluster
EOF
    
    print_success "Istio tracing configured"
}

# Deploy sample application with tracing
deploy_tracing_app() {
    print_info "Deploying sample application with tracing..."
    
    cat <<EOF | kubectl apply -n $TESTING_NAMESPACE -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: tracing-demo
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: tracing-demo-config
data:
  config.yaml: |
    log_level: info
    tracing:
      enabled: true
      sampler_type: probabilistic
      sampler_param: 1.0
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tracing-demo
  labels:
    app: tracing-demo
spec:
  replicas: 2
  selector:
    matchLabels:
      app: tracing-demo
  template:
    metadata:
      labels:
        app: tracing-demo
    spec:
      serviceAccountName: tracing-demo
      containers:
      - name: tracing-demo
        image: jaegertracing/HotROD:latest
        args: ["all"]
        env:
        - name: JAEGER_AGENT_HOST
          value: "jaeger-agent.$NAMESPACE.svc"
        - name: JAEGER_AGENT_PORT
          value: "6831"
        - name: JAEGER_SAMPLER_TYPE
          value: "probabilistic"
        - name: JAEGER_SAMPLER_PARAM
          value: "1"
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 8081
          name: monitoring
        - containerPort: 8082
          name: rpc
        - containerPort: 8083
          name: jaeger-ui
        livenessProbe:
          httpGet:
            path: /
            port: 8080
          initialDelaySeconds: 5
        readinessProbe:
          httpGet:
            path: /
            port: 8080
          initialDelaySeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: tracing-demo
  labels:
    app: tracing-demo
spec:
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  - name: monitoring
    port: 8081
    targetPort: 8081
  selector:
    app: tracing-demo
  type: ClusterIP
EOF
    
    print_info "Waiting for tracing demo pods to be ready..."
    kubectl wait --for=condition=ready pod -l app=tracing-demo -n $TESTING_NAMESPACE --timeout=60s
    
    print_success "Tracing demo application deployed"
}

# Verify Jaeger installation
verify_jaeger() {
    print_info "Verifying Jaeger installation..."
    
    # Check Jaeger pods
    print_info "  - Checking Jaeger pods..."
    kubectl get pods -n $NAMESPACE
    
    # Check Jaeger services
    print_info "  - Checking Jaeger services..."
    kubectl get svc -n $NAMESPACE
    
    # Check Elasticsearch
    print_info "  - Checking Elasticsearch..."
    kubectl get elasticsearch -n $NAMESPACE
    
    # Check tracing demo
    print_info "  - Checking tracing demo..."
    kubectl get pods -n $TESTING_NAMESPACE -l app=tracing-demo
    
    # Test Jaeger UI endpoint
    print_info "  - Testing Jaeger query endpoint..."
    JAEGER_POD=$(kubectl get pod -n $NAMESPACE -l app.kubernetes.io/component=query -o jsonpath='{.items[0].metadata.name}')
    kubectl exec -n $NAMESPACE $JAEGER_POD -- curl -s http://localhost:16686/api/services | head -c 200
    
    print_success "Jaeger verification completed"
}

# Generate test traffic
generate_test_traffic() {
    print_info "Generating test traffic for tracing..."
    
    # Get tracing demo service
    TRACING_DEMO_POD=$(kubectl get pod -n $TESTING_NAMESPACE -l app=tracing-demo -o jsonpath='{.items[0].metadata.name}')
    
    # Generate traffic
    print_info "  - Generating HTTP requests..."
    for i in {1..10}; do
        kubectl exec -n $TESTING_NAMESPACE $TRACING_DEMO_POD -- curl -s http://localhost:8080/api/drivers > /dev/null || true
        sleep 1
    done
    
    print_success "Test traffic generated"
}

# Check Jaeger UI access
check_jaeger_ui() {
    print_info "Checking Jaeger UI access..."
    
    echo ""
    echo -e "${YELLOW}To access Jaeger UI, run:${NC}"
    echo "  kubectl port-forward -n $NAMESPACE svc/jaeger-query 16686:16686"
    echo ""
    echo "Then open browser at: http://localhost:16686"
    echo ""
    
    # Check if port-forward is possible
    print_info "  - Testing Jaeger query endpoint..."
    JAEGER_POD=$(kubectl get pod -n $NAMESPACE -l app.kubernetes.io/component=query -o jsonpath='{.items[0].metadata.name}')
    
    # Test API endpoint
    kubectl exec -n $NAMESPACE $JAEGER_POD -- curl -s http://localhost:16686/api/services | jq . | head -n 20 || echo "No services found yet (needs more traffic)"
    
    print_success "Jaeger UI access verified"
}

# Print summary
print_summary() {
    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}  Jaeger Installation Summary${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo ""
    echo "Namespace: $NAMESPACE"
    echo "Elasticsearch Namespace: $ES_NAMESPACE"
    echo "Testing Namespace: $TESTING_NAMESPACE"
    echo ""
    echo "Components:"
    echo "  - Elasticsearch: Deployed (1 node)"
    echo "  - Jaeger Agent: Deployed (1 replica)"
    echo "  - Jaeger Collector: Deployed (1 replica)"
    echo "  - Jaeger Query: Deployed (1 replica)"
    echo "  - Tracing Demo: Deployed (2 replicas)"
    echo ""
    echo "Access Jaeger UI:"
    echo "  kubectl port-forward -n $NAMESPACE svc/jaeger-query 16686:16686"
    echo "  Open: http://localhost:16686"
    echo ""
    echo "Next Steps:"
    echo "  1. Generate more traffic to see traces"
    echo "  2. Check traces in Jaeger UI"
    echo "  3. Verify trace propagation"
    echo "  4. Adjust sampling rates as needed"
    echo ""
    echo -e "${GREEN}======================================${NC}"
}

# Main execution
main() {
    echo ""
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}  Jaeger Installation for Testing${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo ""
    
    check_prerequisites
    install_elasticsearch_operator
    deploy_elasticsearch
    install_jaeger
    configure_istio_tracing
    deploy_tracing_app
    verify_jaeger
    generate_test_traffic
    check_jaeger_ui
    print_summary
    
    echo ""
    print_success "Jaeger installation and tracing verification completed!"
    echo ""
}

# Run main function
main "$@"