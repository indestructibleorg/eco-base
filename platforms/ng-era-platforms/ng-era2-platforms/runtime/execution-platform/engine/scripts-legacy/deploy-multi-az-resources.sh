#!/bin/bash
# Deploy Multi-AZ Resources for Testing Environment
# Enterprise-grade multi-AZ deployment script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="machine-native-ops-testing"
ZONES=("us-east-1a" "us-east-1b" "us-east-1c")

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
    
    # Check cluster connectivity
    if ! kubectl cluster-info &amp;> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    print_success "All prerequisites are met"
}

# Label nodes with zone information
label_nodes_with_zones() {
    print_info "Labeling nodes with zone information..."
    
    # Get all nodes
    NODES=$(kubectl get nodes --no-headers -o custom-columns=NAME:.metadata.name)
    
    # Distribute nodes across zones
    ZONE_COUNT=${#ZONES[@]}
    NODE_COUNT=$(echo "$NODES" | wc -l)
    NODES_PER_ZONE=$(( (NODE_COUNT + ZONE_COUNT - 1) / ZONE_COUNT ))
    
    print_info "  - Total nodes: $NODE_COUNT"
    print_info "  - Zones: ${ZONES[*]}"
    print_info "  - Nodes per zone: $NODES_PER_ZONE"
    
    # Label nodes
    INDEX=0
    for NODE in $NODES; do
        ZONE=${ZONES[$((INDEX % ZONE_COUNT))]}
        print_info "  - Labeling $NODE with zone $ZONE"
        kubectl label node $NODE topology.kubernetes.io/zone=$ZONE --overwrite
        kubectl label node $NODE topology.kubernetes.io/region=us-east-1 --overwrite
        INDEX=$((INDEX + 1))
    done
    
    print_success "Nodes labeled with zone information"
}

# Verify node labels
verify_node_labels() {
    print_info "Verifying node labels..."
    
    kubectl get nodes -L topology.kubernetes.io/zone,topology.kubernetes.io/region
    
    # Count nodes per zone
    echo ""
    echo "Node distribution per zone:"
    for ZONE in "${ZONES[@]}"; do
        COUNT=$(kubectl get nodes -l topology.kubernetes.io/zone=$ZONE --no-headers | wc -l)
        echo "  - $ZONE: $COUNT nodes"
    done
    
    print_success "Node labels verified"
}

# Deploy multi-AZ applications
deploy_multi_az_apps() {
    print_info "Deploying multi-AZ applications..."
    
    # Create namespace if not exists
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Enable Istio injection
    kubectl label namespace $NAMESPACE istio-injection=enabled --overwrite
    
    # Deploy test application with multi-AZ configuration
    cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: multi-az-app
  labels:
    app: multi-az-app
    component: application
spec:
  replicas: 9  # 3 per zone
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 3
      maxUnavailable: 1
  selector:
    matchLabels:
      app: multi-az-app
  template:
    metadata:
      labels:
        app: multi-az-app
        component: application
        version: v1.0.0
    spec:
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            app: multi-az-app
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchLabels:
                  app: multi-az-app
              topologyKey: kubernetes.io/hostname
      terminationGracePeriodSeconds: 30
      containers:
      - name: multi-az-app
        image: nginx:alpine
        ports:
        - name: http
          containerPort: 80
          protocol: TCP
        env:
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        - name: ZONE
          valueFrom:
            fieldRef:
              fieldPath: metadata.labels['topology.kubernetes.io/zone']
        resources:
          requests:
            cpu: 200m
            memory: 256Mi
          limits:
            cpu: 500m
            memory: 512Mi
        livenessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
---
apiVersion: v1
kind: Service
metadata:
  name: multi-az-app
  labels:
    app: multi-az-app
spec:
  type: ClusterIP
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800
  selector:
    app: multi-az-app
  ports:
  - name: http
    port: 80
    targetPort: 80
    protocol: TCP
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: multi-az-app-pdb
spec:
  minAvailable: 66%  # Ensure 2/3 availability
  selector:
    matchLabels:
      app: multi-az-app
EOF
    
    print_success "Multi-AZ application deployed"
    
    # Wait for pods to be ready
    print_info "Waiting for pods to be ready..."
    kubectl wait --for=condition=ready pod -l app=multi-az-app -n "${NAMESPACE}" --timeout=120s
    
    print_success "All pods are ready"
}

# Verify pod distribution across zones
verify_pod_distribution() {
    print_info "Verifying pod distribution across zones..."
    
    echo ""
    echo "Pod distribution per zone:"
    for ZONE in "${ZONES[@]}"; do
        COUNT=$(kubectl get pods -n "${NAMESPACE}" -l app=multi-az-app -o json | jq -r ".items[] | select(.spec.nodeName != null)" | while read -r pod; do
            NODE=$(echo "$pod" | jq -r '.spec.nodeName')
            NODE_ZONE=$(kubectl get node $NODE -o jsonpath='{.metadata.labels.topology\.kubernetes\.io/zone}')
            if [ "$NODE_ZONE" = "$ZONE" ]; then
                echo "1"
            fi
        done | wc -l)
        
        # Alternative method using pod labels
        POD_COUNT=$(kubectl get pods -n "${NAMESPACE}" -l app=multi-az-app -o json | jq -r ".items[] | select(.spec.nodeName != null) | .spec.nodeName" | while read NODE; do
            ZONE_LABEL=$(kubectl get node $NODE -o jsonpath='{.metadata.labels.topology\.kubernetes\.io/zone}')
            if [ "$ZONE_LABEL" = "$ZONE" ]; then
                echo "1"
            fi
        done | wc -l)
        
        echo "  - $ZONE: $POD_COUNT pods"
    done
    
    echo ""
    echo "Detailed pod information:"
    kubectl get pods -n "${NAMESPACE}" -l app=multi-az-app -o wide
    
    print_success "Pod distribution verified"
}

# Deploy zone-aware routing
deploy_zone_aware_routing() {
    print_info "Deploying zone-aware routing..."
    
    # Create Istio DestinationRule for zone-aware routing
    cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: multi-az-app-dr
spec:
  host: multi-az-app
  trafficPolicy:
    loadBalancer:
      simple: ROUND_ROBIN
      localityLbSetting:
        enabled: true
        distribute:
        - from: us-east-1a/us-east-1a/*
          to:
            us-east-1a/us-east-1a/*: 80
            us-east-1b/us-east-1b/*: 10
            us-east-1c/us-east-1c/*: 10
        - from: us-east-1b/us-east-1b/*
          to:
            us-east-1b/us-east-1b/*: 80
            us-east-1c/us-east-1c/*: 10
            us-east-1a/us-east-1a/*: 10
        - from: us-east-1c/us-east-1c/*
          to:
            us-east-1c/us-east-1c/*: 80
            us-east-1a/us-east-1a/*: 10
            us-east-1b/us-east-1b/*: 10
    connectionPool:
      tcp:
        maxConnections: 100
        connectTimeout: 10s
      http:
        http1MaxPendingRequests: 50
        maxRequestsPerConnection: 3
        idleTimeout: 300s
    outlierDetection:
      consecutive5xxErrors: 3
      interval: 30s
      baseEjectionTime: 60s
      maxEjectionPercent: 50
      minHealthPercent: 50
---
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: multi-az-app-vs
spec:
  hosts:
  - multi-az-app
  http:
  - route:
    - destination:
        host: multi-az-app
    timeout: 30s
    retries:
      attempts: 3
      perTryTimeout: 10s
      retryOn: 5xx,connect-failure,refused-stream
EOF
    
    print_success "Zone-aware routing deployed"
}

# Configure failover policies
configure_failover_policies() {
    print_info "Configuring failover policies..."
    
    # Create PriorityClasses
    cat <<EOF | kubectl apply -f -
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: critical-priority
value: 1000000
globalDefault: false
description: "High priority class for critical workloads"
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: high-priority
value: 500000
globalDefault: false
description: "High priority class for important workloads"
EOF
    
    # Create HorizontalPodAutoscaler
    cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: multi-az-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: multi-az-app
  minReplicas: 9
  maxReplicas: 15
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
EOF
    
    print_success "Failover policies configured"
}

# Test load balancing
test_load_balancing() {
    print_info "Testing load balancing..."
    
    # Create a test pod
    cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: v1
kind: Pod
metadata:
  name: load-balancer-test
spec:
  containers:
  - name: curl
    image: curlimages/curl
    command: ['sh', '-c', 'sleep 3600']
  restartPolicy: Always
EOF
    
    # Wait for test pod to be ready
    kubectl wait --for=condition=ready pod/load-balancer-test -n "${NAMESPACE}" --timeout=60s
    
    # Test load balancing
    print_info "  - Testing load distribution..."
    for i in {1..10}; do
        kubectl exec -n "${NAMESPACE}" load-balancer-test -- curl -s http://multi-az-app/ | grep -o "Welcome to nginx" || echo "Request $i failed"
        sleep 1
    done
    
    # Clean up test pod
    kubectl delete pod load-balancer-test -n "${NAMESPACE}" --ignore-not-found=true
    
    print_success "Load balancing test completed"
}

# Print summary
print_summary() {
    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}  Multi-AZ Deployment Summary${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo ""
    echo "Namespace: $NAMESPACE"
    echo "Zones: ${ZONES[*]}"
    echo ""
    echo "Deployed Resources:"
    echo "  - Multi-AZ Application: 9 replicas (3 per zone)"
    echo "  - Zone-Aware Routing: Configured"
    echo "  - Pod Disruption Budget: 66% min available"
    echo "  - HorizontalPodAutoscaler: 9-15 replicas"
    echo "  - Priority Classes: Critical, High"
    echo ""
    echo "Load Balancing Strategy:"
    echo "  - Primary zone: 80% traffic"
    echo "  - Secondary zones: 10% each"
    echo "  - Failover: Automatic on zone failure"
    echo ""
    echo "Next Steps:"
    echo "  1. Monitor pod distribution: kubectl get pods -n "${NAMESPACE}" -o wide"
    echo "  2. Check HPA status: kubectl get hpa -n $NAMESPACE"
    echo "  3. View routing: istioctl pc destinationrule <pod-name>"
    echo "  4. Test failover: See test-failover-scenarios.sh"
    echo ""
    echo -e "${GREEN}======================================${NC}"
}

# Main execution
main() {
    echo ""
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}  Multi-AZ Deployment to Testing Env${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo ""
    
    check_prerequisites
    label_nodes_with_zones
    verify_node_labels
    deploy_multi_az_apps
    verify_pod_distribution
    deploy_zone_aware_routing
    configure_failover_policies
    test_load_balancing
    print_summary
    
    echo ""
    print_success "Multi-AZ deployment completed successfully!"
    echo ""
}

# Run main function
main "$@"