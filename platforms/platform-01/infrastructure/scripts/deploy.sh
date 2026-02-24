#!/bin/bash
# =============================================================================
# Eco-Base Infrastructure Deployment Script
# =============================================================================
# 此腳本自動化部署完整的雲原生基礎設施:
# 1. GKE 集群 (使用 Terraform)
# 2. ArgoCD GitOps
# 3. 監控堆疊 (Prometheus/Grafana)
# 4. 應用程式部署
# =============================================================================

set -euo pipefail

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置變數
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-staging}"
GCP_PROJECT_ID="${GCP_PROJECT_ID:-}"
GCP_REGION="${GCP_REGION:-us-central1}"

# =============================================================================
# 輔助函數
# =============================================================================

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

check_prerequisites() {
    log_info "檢查必要工具..."
    
    local tools=("gcloud" "terraform" "kubectl" "helm" "jq")
    local missing=()
    
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing+=("$tool")
        fi
    done
    
    if [ ${#missing[@]} -ne 0 ]; then
        log_error "缺少以下工具: ${missing[*]}"
        log_info "請安裝缺失的工具後重試"
        exit 1
    fi
    
    log_success "所有必要工具已安裝"
}

verify_gcp_auth() {
    log_info "驗證 GCP 認證..."
    
    if [ -z "$GCP_PROJECT_ID" ]; then
        log_error "請設置 GCP_PROJECT_ID 環境變數"
        exit 1
    fi
    
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
        log_error "未檢測到活躍的 GCP 認證"
        log_info "請執行: gcloud auth application-default login"
        exit 1
    fi
    
    gcloud config set project "$GCP_PROJECT_ID" > /dev/null 2>&1
    log_success "GCP 認證驗證通過 (Project: $GCP_PROJECT_ID)"
}

# =============================================================================
# 部署步驟
# =============================================================================

deploy_gke_cluster() {
    log_info "步驟 1/5: 部署 GKE 集群..."
    
    cd "$INFRA_DIR/terraform/environments/$ENVIRONMENT"
    
    # 檢查 terraform.tfvars 是否存在
    if [ ! -f "terraform.tfvars" ]; then
        log_warning "terraform.tfvars 不存在，使用示例文件創建..."
        cp terraform.tfvars.example terraform.tfvars
        log_error "請編輯 terraform.tfvars 填入您的配置後重試"
        exit 1
    fi
    
    # 初始化 Terraform
    log_info "初始化 Terraform..."
    terraform init
    
    # 規劃變更
    log_info "規劃 Terraform 變更..."
    terraform plan -out=tfplan
    
    # 應用變更
    log_info "應用 Terraform 變更..."
    terraform apply tfplan
    
    # 獲取集群憑證
    log_info "配置 kubectl..."
    eval "$(terraform output -raw get_credentials_command)"
    
    # 等待集群準備就緒
    log_info "等待集群節點準備就緒..."
    kubectl wait --for=condition=Ready nodes --all --timeout=300s
    
    log_success "GKE 集群部署完成"
}

setup_workload_identity() {
    log_info "步驟 2/5: 配置 Workload Identity..."
    
    # 為 ArgoCD 配置 Workload Identity
    gcloud iam service-accounts add-iam-policy-binding \
        "argocd-app-controller@$GCP_PROJECT_ID.iam.gserviceaccount.com" \
        --role="roles/iam.workloadIdentityUser" \
        --member="serviceAccount:$GCP_PROJECT_ID.svc.id.goog[argocd/argocd-application-controller]" \
        2>/dev/null || log_warning "Workload Identity 綁定可能已存在"
    
    log_success "Workload Identity 配置完成"
}

deploy_argocd() {
    log_info "步驟 3/5: 部署 ArgoCD..."
    
    cd "$INFRA_DIR/argocd"
    
    # 執行安裝腳本
    chmod +x install-argocd.sh
    ./install-argocd.sh "$ENVIRONMENT"
    
    # 等待 ArgoCD 準備就緒
    kubectl wait --for=condition=available --timeout=300s deployment/argocd-server -n argocd
    
    # 應用 AppProject
    kubectl apply -f projects/
    
    # 應用 ApplicationSet
    kubectl apply -f apps/
    
    log_success "ArgoCD 部署完成"
}

deploy_monitoring() {
    log_info "步驟 4/5: 部署監控堆疊..."
    
    # 添加 Helm 倉庫
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    # 創建監控命名空間
    kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
    
    # 創建 Grafana 管理員密碼 Secret
    kubectl create secret generic grafana-admin-credentials \
        --from-literal=admin-user=admin \
        --from-literal=admin-password="$(openssl rand -base64 20)" \
        -n monitoring --dry-run=client -o yaml | kubectl apply -f -
    
    # 安裝 kube-prometheus-stack
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --values "$INFRA_DIR/monitoring/prometheus-values.yaml" \
        --wait \
        --timeout 600s
    
    log_success "監控堆疊部署完成"
}

deploy_cert_manager() {
    log_info "步驟 5/5: 部署 Cert-Manager..."
    
    # 添加 Jetstack Helm 倉庫
    helm repo add jetstack https://charts.jetstack.io
    helm repo update
    
    # 創命名空間
    kubectl create namespace cert-manager --dry-run=client -o yaml | kubectl apply -f -
    
    # 安裝 CRDs
    kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.3/cert-manager.crds.yaml
    
    # 安裝 Cert-Manager
    helm upgrade --install cert-manager jetstack/cert-manager \
        --namespace cert-manager \
        --version v1.13.3 \
        --set installCRDs=false \
        --wait \
        --timeout 300s
    
    # 創建 ClusterIssuer (Let's Encrypt)
    cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-staging
spec:
  acme:
    server: https://acme-staging-v02.api.letsencrypt.org/directory
    email: admin@your-domain.com
    privateKeySecretRef:
      name: letsencrypt-staging
    solvers:
    - http01:
        ingress:
          class: nginx
---
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@your-domain.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
    
    log_success "Cert-Manager 部署完成"
}

verify_deployment() {
    log_info "驗證部署..."
    
    # 檢查節點
    log_info "集群節點:"
    kubectl get nodes
    
    # 檢查命名空間
    log_info "命名空間:"
    kubectl get namespaces
    
    # 檢查 ArgoCD 應用程式
    log_info "ArgoCD 應用程式:"
    kubectl get applications -n argocd
    
    # 檢查監控組件
    log_info "監控組件:"
    kubectl get pods -n monitoring
    
    log_success "部署驗證完成"
}

print_next_steps() {
    echo ""
    echo "=========================================="
    echo "部署完成！下一步操作:"
    echo "=========================================="
    echo ""
    echo "1. 訪問 ArgoCD UI:"
    echo "   kubectl port-forward svc/argocd-server -n argocd 8080:443"
    echo "   https://localhost:8080"
    echo ""
    echo "2. 獲取 ArgoCD 密碼:"
    echo "   kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath='{.data.password}' | base64 -d"
    echo ""
    echo "3. 訪問 Grafana:"
    echo "   kubectl port-forward svc/prometheus-grafana -n monitoring 3000:80"
    echo "   http://localhost:3000"
    echo ""
    echo "4. 獲取 Grafana 密碼:"
    echo "   kubectl -n monitoring get secret grafana-admin-credentials -o jsonpath='{.data.admin-password}' | base64 -d"
    echo ""
    echo "5. 查看應用程式狀態:"
    echo "   kubectl get applications -n argocd"
    echo ""
    echo "6. 配置 DNS:"
    echo "   將您的域名指向 Ingress IP:"
    echo "   kubectl get svc -n ingress-nginx ingress-nginx-controller -o jsonpath='{.status.loadBalancer.ingress[0].ip}'"
    echo ""
}

# =============================================================================
# 主函數
# =============================================================================

main() {
    echo "=========================================="
    echo "Eco-Base Infrastructure Deployment"
    echo "Environment: $ENVIRONMENT"
    echo "=========================================="
    echo ""
    
    # 檢查前提條件
    check_prerequisites
    verify_gcp_auth
    
    # 確認部署
    read -p "確定要部署到 $ENVIRONMENT 環境嗎? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        log_info "部署已取消"
        exit 0
    fi
    
    # 執行部署步驟
    deploy_gke_cluster
    setup_workload_identity
    deploy_argocd
    deploy_monitoring
    deploy_cert_manager
    
    # 驗證部署
    verify_deployment
    
    # 打印後續步驟
    print_next_steps
    
    log_success "所有部署步驟完成！"
}

# 處理命令行參數
case "${1:-}" in
    --help|-h)
        echo "使用方法: $0 [environment]"
        echo ""
        echo "參數:"
        echo "  environment    部署環境 (staging 或 production)"
        echo ""
        echo "環境變數:"
        echo "  GCP_PROJECT_ID    GCP 專案 ID (必需)"
        echo "  GCP_REGION        GCP 區域 (默認: us-central1)"
        echo ""
        echo "示例:"
        echo "  GCP_PROJECT_ID=my-project $0 staging"
        exit 0
        ;;
    *)
        main
        ;;
esac
