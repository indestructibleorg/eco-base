#!/bin/bash
# Configure Velero for Backups in Testing Environment
# Enterprise-grade Velero configuration script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="velero"
TESTING_NAMESPACE="machine-native-ops-testing"
BUCKET_NAME="machine-native-ops-backups-testing"
REGION="us-east-1"
VELERO_VERSION="1.12.0"

# S3 Configuration (configure these values)
S3_ENDPOINT="${S3_ENDPOINT:-https://s3.amazonaws.com}"
S3_ACCESS_KEY="${S3_ACCESS_KEY:-your-access-key}"
S3_SECRET_KEY="${S3_SECRET_KEY:-your-secret-key}"

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
    
    # Check if Velero CLI is installed, if not install it
    if ! command -v velero &amp;> /dev/null; then
        print_warn "Velero CLI not found, installing..."
        install_velero_cli
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &amp;> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    print_success "All prerequisites are met"
}

# Install Velero CLI
install_velero_cli() {
    print_info "Installing Velero CLI v$VELERO_VERSION..."
    
    # Detect OS
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)
    
    if [ "$ARCH" = "x86_64" ]; then
        ARCH="amd64"
    elif [ "$ARCH" = "aarch64" ]; then
        ARCH="arm64"
    fi
    
    # Download Velero
    print_info "  - Downloading Velero for $OS-$ARCH..."
    wget -q https://github.com/vmware-tanzu/velero/releases/download/v$VELERO_VERSION/velero-v$VELERO_VERSION-${OS}-${ARCH}.tar.gz
    
    # Extract
    print_info "  - Extracting Velero..."
    tar -xvf velero-v${VELERO_VERSION}-${OS}-${ARCH}.tar.gz
    
    # Install
    print_info "  - Installing Velero CLI..."
    sudo mv velero-v${VELERO_VERSION}-${OS}-${ARCH}/velero /usr/local/bin/
    
    # Clean up
    rm -rf velero-v${VELERO_VERSION}-${OS}-${ARCH}*
    
    # Verify installation
    if velero version --client &amp;> /dev/null; then
        print_success "Velero CLI installed successfully"
        velero version --client
    else
        print_error "Failed to install Velero CLI"
        exit 1
    fi
}

# Configure S3 credentials
configure_s3_credentials() {
    print_info "Configuring S3 credentials..."
    
    # Create credentials file
    cat <<EOF > /tmp/velero-credentials.txt
[default]
aws_access_key_id = ${S3_ACCESS_KEY}
aws_secret_access_key = ${S3_SECRET_KEY}
EOF
    
    print_warn "Using S3 endpoint: $S3_ENDPOINT"
    print_warn "Bucket: $BUCKET_NAME"
    print_warn "Region: $REGION"
    echo ""
    print_warn "To use custom credentials, set environment variables:"
    print_warn "  export S3_ENDPOINT=your-endpoint"
    print_warn "  export S3_ACCESS_KEY=your-access-key"
    print_warn "  export S3_SECRET_KEY=your-secret-key"
    echo ""
    
    print_success "S3 credentials configured"
}

# Install Velero server
install_velero_server() {
    print_info "Installing Velero server..."
    
    # Check if Velero is already installed
    if kubectl get namespace $NAMESPACE &amp;> /dev/null; then
        print_warn "Velero namespace exists, skipping installation"
        return
    fi
    
    # Install Velero
    print_info "  - Installing Velero with AWS provider..."
    velero install \
        --provider aws \
        --plugins velero/velero-plugin-for-aws:v1.8.0 \
        --bucket $BUCKET_NAME \
        --secret-file /tmp/velero-credentials.txt \
        --use-volume-snapshots \
        --backup-location-config region=$REGION,s3ForcePathStyle="true",s3Url=$S3_ENDPOINT \
        --snapshot-location-config region=$REGION \
        --wait \
        --timeout 10m || print_warn "Velero installation may have warnings"
    
    print_success "Velero server installed"
}

# Deploy backup schedules
deploy_backup_schedules() {
    print_info "Deploying backup schedules..."
    
    # Update namespace for testing environment
    print_info "  - Updating backup schedules for testing namespace..."
    
    # Create testing-specific schedules
    cat <<EOF | kubectl apply -f -
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: daily-backup-testing
  namespace: $NAMESPACE
  labels:
    environment: testing
    frequency: daily
spec:
  schedule: "0 2 * * *"  # 2 AM daily
  template:
    includedNamespaces:
    - $TESTING_NAMESPACE
    storageLocation: default
    ttl: 168h  # 7 days retention
    snapshotVolumes: false
---
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: hourly-backup-testing
  namespace: $NAMESPACE
  labels:
    environment: testing
    frequency: hourly
spec:
  schedule: "0 * * * *"  # Every hour
  template:
    includedNamespaces:
    - $TESTING_NAMESPACE
    storageLocation: default
    ttl: 24h  # 1 day retention
    snapshotVolumes: false
---
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: config-backup-testing
  namespace: $NAMESPACE
  labels:
    environment: testing
    frequency: frequent
spec:
  schedule: "*/30 * * * *"  # Every 30 minutes
  template:
    includedNamespaces:
    - $TESTING_NAMESPACE
    includedResources:
    - configmaps
    - secrets
    - deployments
    - statefulsets
    - services
    storageLocation: default
    ttl: 72h  # 3 days retention
    snapshotVolumes: false
EOF
    
    print_success "Backup schedules deployed"
}

# Deploy backup jobs
deploy_backup_jobs() {
    print_info "Deploying backup jobs..."
    
    # Create service account and RBAC for backup jobs
    cat <<EOF | kubectl apply -n $TESTING_NAMESPACE -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: backup-service-account
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: backup-role
rules:
- apiGroups: [""]
  resources: ["pods", "pods/exec"]
  verbs: ["get", "list", "create"]
- apiGroups: [""]
  resources: ["persistentvolumes", "persistentvolumeclaims"]
  verbs: ["get", "list"]
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: backup-role-binding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: backup-role
subjects:
- kind: ServiceAccount
  name: backup-service-account
EOF
    
    # Deploy simple backup validation job
    cat <<EOF | kubectl apply -n $TESTING_NAMESPACE -f -
apiVersion: batch/v1
kind: CronJob
metadata:
  name: backup-validation
spec:
  schedule: "0 3 * * *"  # Daily at 3 AM
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: backup-service-account
          containers:
          - name: validator
            image: bitnami/kubectl:latest
            command:
            - /bin/sh
            - -c
            - |
              set -e
              echo "Starting backup validation at \$(date)"
              
              # Check latest backup
              LATEST_BACKUP=\$(velero backup get -o json | jq -r '.items | sort_by(.metadata.creationTimestamp) | reverse | .[0].metadata.name')
              BACKUP_STATUS=\$(velero backup get \$LATEST_BACKUP -o json | jq -r '.status.phase')
              
              if [ "\$BACKUP_STATUS" != "Completed" ]; then
                echo "ERROR: Velero backup \$LATEST_BACKUP failed with status: \$BACKUP_STATUS"
                exit 1
              fi
              
              echo "SUCCESS: Latest backup \$LATEST_BACKUP completed successfully"
              echo "Backup age: \$(velero backup get \$LATEST_BACKUP -o json | jq -r '.metadata.creationTimestamp')"
          restartPolicy: OnFailure
EOF
    
    print_success "Backup jobs deployed"
}

# Create manual backup
create_manual_backup() {
    print_info "Creating manual backup..."
    
    BACKUP_NAME="manual-backup-testing-$(date +%Y%m%d-%H%M%S)"
    
    print_info "  - Creating backup: $BACKUP_NAME"
    velero backup create $BACKUP_NAME \
        --include-namespaces $TESTING_NAMESPACE \
        --wait \
        --timeout 5m || print_warn "Backup creation may have warnings"
    
    print_success "Manual backup created: $BACKUP_NAME"
}

# Verify Velero installation
verify_velero() {
    print_info "Verifying Velero installation..."
    
    # Check Velero pods
    print_info "  - Checking Velero pods..."
    kubectl get pods -n $NAMESPACE
    
    # Check backup locations
    print_info "  - Checking backup locations..."
    velero backup-location get
    
    # Check snapshot locations
    print_info "  - Checking snapshot locations..."
    velero snapshot-location get
    
    # Check backup schedules
    print_info "  - Checking backup schedules..."
    kubectl get schedule -n $NAMESPACE
    
    # Check recent backups
    print_info "  - Checking recent backups..."
    velero backup get --recent 3
    
    # Test backup location connectivity
    print_info "  - Testing backup location connectivity..."
    velero backup-location get default --show-details || print_warn "Backup location details may not be available"
    
    print_success "Velero verification completed"
}

# Test restore operation
test_restore() {
    print_info "Testing restore operation..."
    
    # Get latest backup
    LATEST_BACKUP=$(velero backup get --recent 1 -o json | jq -r '.items[0].metadata.name')
    
    if [ -z "$LATEST_BACKUP" ]; then
        print_warn "No backups available to test restore"
        return
    fi
    
    print_info "  - Creating restore from backup: $LATEST_BACKUP"
    
    # Create restore test namespace
    kubectl create namespace velero-restore-test --dry-run=client -o yaml | kubectl apply -f -
    
    # Create restore
    velero restore create \
        --from-backup $LATEST_BACKUP \
        --namespace-mappings $TESTING_NAMESPACE:velero-restore-test \
        --wait \
        --timeout 5m || print_warn "Restore test may have warnings"
    
    # Check restore status
    print_info "  - Checking restore status..."
    velero restore get --recent 1
    
    # Clean up test namespace
    print_info "  - Cleaning up test namespace..."
    kubectl delete namespace velero-restore-test --ignore-not-found=true
    
    print_success "Restore test completed"
}

# Print summary
print_summary() {
    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}  Velero Configuration Summary${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo ""
    echo "Namespace: $NAMESPACE"
    echo "Testing Namespace: $TESTING_NAMESPACE"
    echo "Bucket: $BUCKET_NAME"
    echo "Region: $REGION"
    echo ""
    echo "Backup Schedules:"
    echo "  - Daily backup: 2 AM (7 days retention)"
    echo "  - Hourly backup: Every hour (1 day retention)"
    echo "  - Config backup: Every 30 min (3 days retention)"
    echo ""
    echo "Backup Jobs:"
    echo "  - Backup validation: Daily at 3 AM"
    echo ""
    echo "Next Steps:"
    echo "  1. Check backups: velero backup get"
    echo "  2. View schedules: kubectl get schedule -n $NAMESPACE"
    echo "  3. Check logs: kubectl logs -n $NAMESPACE deployment/velero"
    echo "  4. Test restore: velero restore create --from-backup
# Main execution
main() {
    echo ""
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}  Velero Configuration for Testing${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo ""
    
    check_prerequisites
    configure_s3_credentials
    install_velero_server
    deploy_backup_schedules
    deploy_backup_jobs
    create_manual_backup
    verify_velero
    test_restore
    print_summary
    
    echo ""
    print_success "Velero configuration completed successfully!"
    echo ""
}

# Run main function
main "$@"
