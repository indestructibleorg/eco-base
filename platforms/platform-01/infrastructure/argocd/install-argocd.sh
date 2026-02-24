#!/bin/bash
# =============================================================================
# ArgoCD Installation Script
# =============================================================================
# 此腳本在 GKE 集群上安裝 ArgoCD 並配置 GitOps 工作流程
# 
# 使用方法:
#   ./install-argocd.sh <environment>
#   例如: ./install-argocd.sh production
# =============================================================================

set -euo pipefail

ENVIRONMENT="${1:-staging}"
NAMESPACE="argocd"
ARGOCD_VERSION="v2.9.3"

echo "=========================================="
echo "Installing ArgoCD for environment: $ENVIRONMENT"
echo "=========================================="

# 創建命名空間
echo "Creating namespace..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# 安裝 ArgoCD
echo "Installing ArgoCD $ARGOCD_VERSION..."
kubectl apply -n $NAMESPACE -f https://raw.githubusercontent.com/argoproj/argo-cd/$ARGOCD_VERSION/manifests/install.yaml

# 等待 ArgoCD 準備就緒
echo "Waiting for ArgoCD to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/argocd-server -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/argocd-repo-server -n $NAMESPACE
kubectl wait --for=condition=available --timeout=300s deployment/argocd-application-controller -n $NAMESPACE

# 配置 Workload Identity (如果使用 GCP)
echo "Configuring Workload Identity..."
kubectl annotate serviceaccount argocd-application-controller \
  iam.gke.io/gcp-service-account=argocd-app-controller@$GCP_PROJECT_ID.iam.gserviceaccount.com \
  -n $NAMESPACE --overwrite 2>/dev/null || true

# 配置 ConfigMap
echo "Applying ArgoCD configuration..."
kubectl apply -f argocd-cm.yaml -n $NAMESPACE
kubectl apply -f argocd-rbac-cm.yaml -n $NAMESPACE

# 重啟 ArgoCD 組件以應用配置
echo "Restarting ArgoCD components..."
kubectl rollout restart deployment/argocd-server -n $NAMESPACE
kubectl rollout restart deployment/argocd-repo-server -n $NAMESPACE
kubectl rollout restart statefulset/argocd-application-controller -n $NAMESPACE

# 等待重啟完成
echo "Waiting for ArgoCD to restart..."
sleep 10
kubectl wait --for=condition=available --timeout=300s deployment/argocd-server -n $NAMESPACE

# 配置 Ingress (可選)
if [ "$ENVIRONMENT" == "production" ]; then
  echo "Configuring Ingress for production..."
  kubectl apply -f argocd-ingress.yaml -n $NAMESPACE
fi

# 獲取初始密碼
echo ""
echo "=========================================="
echo "ArgoCD Installation Complete!"
echo "=========================================="
echo ""
echo "Initial admin password:"
kubectl -n $NAMESPACE get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
echo ""
echo ""
echo "Port-forward command:"
echo "  kubectl port-forward svc/argocd-server -n $NAMESPACE 8080:443"
echo ""
echo "Login URL: https://localhost:8080"
echo "Username: admin"
echo ""
echo "Next steps:"
echo "  1. Change the admin password"
echo "  2. Apply the AppProject configuration: kubectl apply -f projects/"
echo "  3. Apply the ApplicationSet: kubectl apply -f apps/"
echo ""
