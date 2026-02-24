# Eco-Base 快速入門指南

## 5 分鐘內完成首次部署

### 1. 前置準備 (2 分鐘)

```bash
# 安裝工具 (如果尚未安裝)
# macOS
brew install google-cloud-sdk terraform kubectl helm jq

# 登入 GCP
gcloud auth login
gcloud auth application-default login

# 設置專案
export GCP_PROJECT_ID="your-project-id"
gcloud config set project $GCP_PROJECT_ID
```

### 2. 創建 State Bucket (1 分鐘)

```bash
gsutil mb -l us-central1 gs://${GCP_PROJECT_ID}-terraform-state
gsutil versioning set on gs://${GCP_PROJECT_ID}-terraform-state
```

### 3. 配置變數 (1 分鐘)

```bash
cd terraform/environments/staging
cp terraform.tfvars.example terraform.tfvars

# 編輯變數
nano terraform.tfvars
# 填入:
# gcp_project_id = "your-project-id"
# authorized_networks = [{ name = "my-ip", cidr = "YOUR_IP/32" }]
```

### 4. 部署 (1 分鐘)

```bash
# 返回根目錄
cd ../../..

# 一鍵部署
export GCP_PROJECT_ID="your-project-id"
./scripts/deploy.sh staging
```

## 常用命令速查

### Terraform

```bash
make init ENV=staging      # 初始化
make plan ENV=staging      # 規劃
make apply ENV=staging     # 應用
make destroy ENV=staging   # 銷毀
```

### Kubernetes

```bash
# 配置 kubectl
eval "$(terraform output -raw get_credentials_command)"

# 查看資源
kubectl get nodes
kubectl get pods -n staging
kubectl get services -n staging

# 查看日誌
make logs-api ENV=staging
make logs-ai ENV=staging
```

### ArgoCD

```bash
# 端口轉發
make argocd-port-forward

# 獲取密碼
make argocd-password

# 登入 CLI
make argocd-login

# 查看狀態
make argocd-status
```

### 監控

```bash
# Grafana
make grafana-port-forward
make grafana-password

# Prometheus
make prometheus-port-forward
```

## 目錄結構速覽

```
infrastructure/
├── terraform/          # GKE 集群
│   └── environments/
│       ├── staging/
│       └── production/
├── argocd/            # GitOps
│   ├── install-argocd.sh
│   ├── projects/
│   └── apps/
├── helm/eco-base/     # 應用部署
│   ├── values.yaml
│   ├── values-staging.yaml
│   └── values-production.yaml
├── monitoring/        # 監控
│   └── prometheus-values.yaml
├── security/          # 網路安全
│   └── network-policies/
└── scripts/
    └── deploy.sh      # 一鍵部署
```

## 下一步

1. **配置 GitHub Actions**: 參考 [GITHUB_SETUP.md](GITHUB_SETUP.md)
2. **自定義應用**: 修改 `helm/eco-base/values-*.yaml`
3. **添加監控告警**: 編輯 `monitoring/prometheus-values.yaml`
4. **配置域名**: 更新 Ingress 和 DNS

## 獲取幫助

```bash
# 查看所有可用命令
make help

# 查看環境資訊
make info ENV=staging

# 查看事件
make events ENV=staging
```

## 成本估算

| 組件 | 規格 | 每月估算 |
|-----|------|---------|
| GKE (e2-standard-4) | 3 節點 | ~$200 |
| GPU 節點 (T4) | 按需 | ~$300 |
| Load Balancer | 1 個 | ~$20 |
| Persistent Disk | 200GB SSD | ~$30 |
| **總計** | | **~$550/月** |

> 使用預留實例可節省 30-50% 成本
