# Eco-Base 企業級雲原生全自動化架構

這是一個完整的企業級雲原生基礎設施即代碼 (IaC) 方案，實現了**閉環式全自動化部署**。

## 架構概覽

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GitHub Repository                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │
│  │  Source Code │  │  Helm Charts │  │  Terraform  │  │  GitHub Actions │    │
│  │   (App)     │  │  (K8s Manifest)│  │   (GKE)    │  │    (CI/CD)      │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘    │
└─────────┼────────────────┼────────────────┼──────────────────┼─────────────┘
          │                │                │                  │
          ▼                ▼                ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ArgoCD (GitOps)                                 │
│                    ┌─────────────────────────────┐                          │
│                    │   ApplicationSet Controller  │                          │
│                    │  (Auto-sync Staging/Manual Prod)│                        │
│                    └─────────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│  GKE Staging    │         │  GKE Production │         │  Artifact       │
│  Cluster        │         │  Cluster        │         │  Registry       │
│                 │         │                 │         │                 │
│ ┌─────────────┐ │         │ ┌─────────────┐ │         │ ┌─────────────┐ │
│ │  API (2)    │ │         │ │  API (3+)   │ │         │ │  API Image  │ │
│ │  Web (2)    │ │         │ │  Web (3+)   │ │         │ │  Web Image  │ │
│ │  AI (1)     │ │         │ │  AI (2+)    │ │         │ │  AI Image   │ │
│ └─────────────┘ │         │ └─────────────┘ │         │ └─────────────┘ │
└─────────────────┘         └─────────────────┘         └─────────────────┘
          │                           │
          ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Monitoring Stack                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │
│  │ Prometheus  │  │   Grafana   │  │ Alertmanager│  │  Cloud Logging  │    │
│  │  (Metrics)  │  │(Dashboards) │  │  (Alerts)   │  │    (Logs)       │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 目錄結構

```
infrastructure/
├── terraform/                    # Terraform GKE 集群配置
│   ├── environments/
│   │   ├── production/          # 生產環境
│   │   │   ├── main.tf
│   │   │   ├── variables.tf
│   │   │   └── terraform.tfvars.example
│   │   └── staging/             # 測試環境
│   │       ├── main.tf
│   │       └── variables.tf
│   └── modules/
│       └── gke/                 # GKE 模塊
├── argocd/                      # ArgoCD GitOps 配置
│   ├── install-argocd.sh
│   ├── argocd-cm.yaml
│   ├── argocd-rbac-cm.yaml
│   ├── projects/                # AppProject 定義
│   │   └── eco-base.yaml
│   └── apps/                    # ApplicationSet 定義
│       └── eco-base-appset.yaml
├── .github/workflows/           # GitHub Actions CI/CD
│   ├── ci-build.yaml           # 建置與安全掃描
│   └── deploy.yaml             # 部署流程
├── monitoring/                  # 監控配置
│   ├── prometheus-values.yaml
│   └── grafana/
│       └── dashboards/
│           └── eco-base-api.json
├── helm/eco-base/              # Helm Chart
│   ├── Chart.yaml
│   ├── values.yaml             # 默認值
│   ├── values-staging.yaml     # Staging 配置
│   ├── values-production.yaml  # Production 配置
│   └── templates/              # K8s 模板
├── security/                   # 安全策略
│   └── network-policies/       # NetworkPolicy
│       ├── default-deny.yaml
│       ├── allow-dns.yaml
│       └── api-service.yaml
└── scripts/
    └── deploy.sh               # 一鍵部署腳本
```

## 快速開始

### 1. 前置需求

安裝以下工具:
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
- [Terraform](https://developer.hashicorp.com/terraform/downloads) (>= 1.5.0)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- [Helm](https://helm.sh/docs/intro/install/)
- [jq](https://stedolan.github.io/jq/download/)

### 2. 配置 GCP 認證

```bash
# 登入 GCP
gcloud auth login
gcloud auth application-default login

# 設置專案
export GCP_PROJECT_ID="your-project-id"
gcloud config set project $GCP_PROJECT_ID

# 啟用必要 API
gcloud services enable compute.googleapis.com container.googleapis.com \
  monitoring.googleapis.com logging.googleapis.com \
  artifactregistry.googleapis.com cloudbuild.googleapis.com
```

### 3. 創建 Terraform State Bucket

```bash
# 創建 GCS bucket 存儲 Terraform state
gsutil mb -l us-central1 gs://${GCP_PROJECT_ID}-terraform-state

# 啟用版本控制
gsutil versioning set on gs://${GCP_PROJECT_ID}-terraform-state
```

### 4. 配置變數

```bash
cd terraform/environments/production

# 複製示例配置
cp terraform.tfvars.example terraform.tfvars

# 編輯配置
nano terraform.tfvars
```

編輯 `terraform.tfvars`:
```hcl
gcp_project_id = "your-project-id"
gcp_region     = "us-central1"

# 授權網路 (您的辦公室/VPN IP)
authorized_networks = [
  {
    name = "office"
    cidr = "203.0.113.0/24"
  }
]
```

### 5. 一鍵部署

```bash
# 執行部署腳本
export GCP_PROJECT_ID="your-project-id"
./scripts/deploy.sh production
```

或手動執行各步驟:

```bash
# 1. 部署 GKE 集群
cd terraform/environments/production
terraform init
terraform plan
terraform apply

# 2. 配置 kubectl
eval "$(terraform output -raw get_credentials_command)"

# 3. 部署 ArgoCD
cd ../../../argocd
./install-argocd.sh production

# 4. 部署監控
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --values ../../monitoring/prometheus-values.yaml
```

## GitHub Actions 配置

### 1. 配置 GitHub Secrets

在 GitHub Repository → Settings → Secrets and variables → Actions 中添加:

**Secrets:**
```
# GCP 認證 (使用 Workload Identity，無需長期密鑰)
# 參考: https://github.com/google-github-actions/auth#workload-identity-federation

# 或傳統服務帳戶密鑰 (不推薦)
GCP_SA_KEY=<base64-encoded-service-account-key>
```

**Variables:**
```
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
GKE_CLUSTER_STAGING=eco-base-staging
GKE_CLUSTER_PRODUCTION=eco-base-production
WORKLOAD_IDENTITY_PROVIDER=projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/POOL_NAME/providers/PROVIDER_NAME
GCP_SERVICE_ACCOUNT=github-actions@your-project.iam.gserviceaccount.com
GCP_DEPLOY_SERVICE_ACCOUNT=deployer@your-project.iam.gserviceaccount.com
```

### 2. 配置 Workload Identity (推薦)

```bash
# 創建 Workload Identity Pool
gcloud iam workload-identity-pools create "github-actions" \
  --project="${GCP_PROJECT_ID}" \
  --location="global" \
  --display-name="GitHub Actions"

# 創建 Workload Identity Provider
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --project="${GCP_PROJECT_ID}" \
  --location="global" \
  --workload-identity-pool="github-actions" \
  --display-name="GitHub Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
  --issuer-uri="https://token.actions.githubusercontent.com"

# 綁定服務帳戶
gcloud iam service-accounts add-iam-policy-binding \
  "github-actions@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-actions/attribute.repository/YOUR_ORG/eco-base"
```

## 部署流程

### 自動部署到 Staging

```
開發者推送代碼到 main 分支
           │
           ▼
┌─────────────────────┐
│   GitHub Actions    │
│   CI Build Pipeline │
│  (Build + Security) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Push Images to    │
│   Artifact Registry │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Update Helm       │
│   values-staging    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   ArgoCD Auto-Sync  │
│   to Staging        │
└─────────────────────┘
```

### 手動部署到 Production

```
GitHub Actions → Deploy Workflow (workflow_dispatch)
           │
           ▼
┌─────────────────────┐
│   Manual Approval   │
│   Required          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Update Helm       │
│   values-production │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   ArgoCD Manual     │
│   Sync to Prod      │
└─────────────────────┘
```

## 監控與告警

### 訪問 Grafana

```bash
# 端口轉發
kubectl port-forward svc/prometheus-grafana -n monitoring 3000:80

# 獲取密碼
kubectl -n monitoring get secret grafana-admin-credentials \
  -o jsonpath='{.data.admin-password}' | base64 -d

# 訪問 http://localhost:3000
```

### 預設儀表板

- **集群概覽**: Kubernetes 集群整體健康狀況
- **節點詳情**: CPU、記憶體、磁碟使用情況
- **Pod 監控**: 應用程式資源使用
- **Eco-Base API**: 自定義業務指標

### 告警規則

預設配置以下告警:
- HighErrorRate: 錯誤率 > 5%
- HighLatency: P95 延遲 > 500ms
- PodCrashLooping: Pod 反覆重啟
- NodeDiskPressure: 節點磁碟壓力
- CertificateExpiry: SSL 證書即將過期

## 安全特性

### 1. 網路安全

- **Private GKE Cluster**: Master 節點使用私有端點
- **Network Policies**: 零信任網路，默認拒絕所有流量
- **Authorized Networks**: 限制 GKE API 訪問來源

### 2. 身份與訪問

- **Workload Identity**: Pod 使用 GCP 服務帳戶，無需密鑰
- **RBAC**: 最小權限原則
- **Pod Security**: 非 root 用戶、只讀根文件系統

### 3. 供應鏈安全

- **Container Scanning**: Trivy 掃描映像漏洞
- **SBOM Generation**: 生成軟體物料清單
- **Pin Dependencies**: GitHub Actions 使用固定 SHA

## 故障排除

### 查看 ArgoCD 應用狀態

```bash
# 端口轉發
kubectl port-forward svc/argocd-server -n argocd 8080:443

# 登入
argocd login localhost:8080 --insecure

# 查看應用
argocd app list

# 查看詳情
argocd app get eco-base-production

# 同步應用
argocd app sync eco-base-production
```

### 查看 Pod 日誌

```bash
# API 服務日誌
kubectl logs -l app.kubernetes.io/name=api -n production --tail=100 -f

# AI 推理服務日誌
kubectl logs -l app.kubernetes.io/name=ai-inference -n production --tail=100 -f
```

### 常見問題

**Q: Pod 處於 Pending 狀態**
```bash
# 查看事件
kubectl describe pod <pod-name> -n production

# 常見原因:
# - 資源不足: 檢查節點資源
# - GPU 節點未就緒: 檢查 GPU 驅動
# - PVC 未綁定: 檢查 StorageClass
```

**Q: ArgoCD 同步失敗**
```bash
# 查看同步狀態
argocd app get eco-base-production

# 強制同步
argocd app sync eco-base-production --force

# 查看資源詳情
kubectl get applications -n argocd -o yaml
```

## 成本優化

### 1. 節點自動擴展

```yaml
# 已配置在 Terraform 中
autoscaling:
  min_node_count: 1
  max_node_count: 10
```

### 2. GPU 節點調度

```yaml
# AI 推理服務只在需要時調度到 GPU 節點
nodeSelector:
  node-type: gpu
tolerations:
  - key: nvidia.com/gpu
    operator: Equal
    value: "true"
    effect: NoSchedule
```

### 3. 預留實例

```bash
# 為長期運行的工作負載購買 Committed Use Discounts
gcloud compute commitments create ...
```

## 擴展指南

### 添加新服務

1. 在 `helm/eco-base/templates/` 創建新的 Deployment/Service
2. 在 `values.yaml` 添加配置
3. 更新 `values-staging.yaml` 和 `values-production.yaml`
4. 在 `security/network-policies/` 添加網路策略
5. 提交 PR，ArgoCD 自動同步

### 添加新環境

1. 複製 `terraform/environments/staging` 為新目錄
2. 修改變數 (集群名稱、網路 CIDR 等)
3. 創建對應的 `values-<env>.yaml`
4. 更新 ArgoCD ApplicationSet

## 貢獻指南

1. Fork 本倉庫
2. 創建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 創建 Pull Request

## 許可證

Apache License 2.0

## 支援

如有問題，請:
1. 查看 [FAQ](#常見問題)
2. 提交 GitHub Issue
3. 聯繫團隊: team@your-domain.com
