# GitHub Actions 配置指南

本文檔說明如何配置 GitHub Actions 以實現全自動化 CI/CD。

## 概述

我們使用 **Workload Identity Federation** (工作負載身份聯合) 來安全地認證 GitHub Actions 到 GCP，無需管理長期服務帳戶密鑰。

## 配置步驟

### 第一步: 創建 GCP 服務帳戶

```bash
# 設置變數
export GCP_PROJECT_ID="your-project-id"
export PROJECT_NUMBER=$(gcloud projects describe $GCP_PROJECT_ID --format="value(projectNumber)")

# 1. 創建 GitHub Actions 服務帳戶
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions" \
  --description="Service account for GitHub Actions CI/CD"

# 2. 創建部署服務帳戶
gcloud iam service-accounts create deployer \
  --display-name="Deployer" \
  --description="Service account for deployment"
```

### 第二步: 授予 IAM 角色

```bash
# GitHub Actions 服務帳戶 (建置映像)
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:github-actions@$GCP_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:github-actions@$GCP_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/cloudbuild.builds.editor"

# 部署服務帳戶 (部署到 GKE)
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:deployer@$GCP_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/container.developer"

gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:deployer@$GCP_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.reader"
```

### 第三步: 配置 Workload Identity Federation

```bash
# 1. 創建 Workload Identity Pool
gcloud iam workload-identity-pools create "github-actions" \
  --project="${GCP_PROJECT_ID}" \
  --location="global" \
  --display-name="GitHub Actions Pool"

# 2. 創建 Workload Identity Provider
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --project="${GCP_PROJECT_ID}" \
  --location="global" \
  --workload-identity-pool="github-actions" \
  --display-name="GitHub Actions Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
  --attribute-condition="assertion.repository_owner == 'YOUR_GITHUB_ORG'" \
  --issuer-uri="https://token.actions.githubusercontent.com"

# 3. 獲取 Provider ID
export PROVIDER_ID="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-actions/providers/github-provider"

# 4. 允許 GitHub Actions 模擬服務帳戶
# 替換 YOUR_GITHUB_ORG 和 YOUR_REPO
gcloud iam service-accounts add-iam-policy-binding \
  "github-actions@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-actions/attribute.repository/YOUR_GITHUB_ORG/YOUR_REPO"

gcloud iam service-accounts add-iam-policy-binding \
  "deployer@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-actions/attribute.repository/YOUR_GITHUB_ORG/YOUR_REPO"
```

### 第四步: 配置 GitHub Secrets 和 Variables

#### Secrets (加密)

前往 GitHub Repository → Settings → Secrets and variables → Actions → New repository secret

| Secret Name | Value | 說明 |
|------------|-------|------|
| `SLACK_WEBHOOK_URL` | `https://hooks.slack.com/...` | 告警通知 (可選) |
| `PAGERDUTY_SERVICE_KEY` | `your-pagerduty-key` | PagerDuty 集成 (可選) |

#### Variables (非加密)

前往 GitHub Repository → Settings → Secrets and variables → Actions → Variables → New repository variable

| Variable Name | Value | 說明 |
|--------------|-------|------|
| `GCP_PROJECT_ID` | `your-project-id` | GCP 專案 ID |
| `GCP_REGION` | `us-central1` | GCP 區域 |
| `GKE_CLUSTER_STAGING` | `eco-base-staging` | Staging 集群名稱 |
| `GKE_CLUSTER_PRODUCTION` | `eco-base-production` | Production 集群名稱 |
| `K8S_NAMESPACE_STAGING` | `staging` | Staging 命名空間 |
| `K8S_NAMESPACE_PRODUCTION` | `production` | Production 命名空間 |
| `REGISTRY` | `us-central1-docker.pkg.dev/your-project/eco-base` | 映像倉庫 |
| `WORKLOAD_IDENTITY_PROVIDER` | `projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-actions/providers/github-provider` | Workload Identity Provider |
| `GCP_SERVICE_ACCOUNT` | `github-actions@your-project.iam.gserviceaccount.com` | 建置服務帳戶 |
| `GCP_DEPLOY_SERVICE_ACCOUNT` | `deployer@your-project.iam.gserviceaccount.com` | 部署服務帳戶 |

### 第五步: 配置 GitHub Environments

為 Production 部署配置環境保護規則:

1. 前往 GitHub Repository → Settings → Environments
2. 點擊 "New environment"
3. 創建 `production` 環境
4. 配置保護規則:
   - **Required reviewers**: 添加需要批准部署的用戶
   - **Wait timer**: 可選，設置等待時間
   - **Deployment branches**: 限制為 `main` 分支

### 第六步: 驗證配置

```bash
# 在本地測試 Workload Identity (可選)
# 安裝 gcloud 憑證輔助工具
gcloud auth application-default login

# 測試訪問
curl -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  "https://cloudresourcemanager.googleapis.com/v1/projects/$GCP_PROJECT_ID"
```

## 故障排除

### 錯誤: `Unable to find provider`

```
Error: Unable to find provider: projects/.../locations/global/workloadIdentityPools/.../providers/...
```

**解決方案:**
```bash
# 確認 Provider 存在
gcloud iam workload-identity-pools providers list \
  --workload-identity-pool="github-actions" \
  --location="global" \
  --project="$GCP_PROJECT_ID"

# 確認變數正確設置
echo $WORKLOAD_IDENTITY_PROVIDER
```

### 錯誤: `Permission denied`

```
Error: googleapi: Error 403: Permission denied
```

**解決方案:**
```bash
# 確認 IAM 綁定正確
gcloud iam service-accounts get-iam-policy \
  github-actions@$GCP_PROJECT_ID.iam.gserviceaccount.com

# 確認 repository 名稱匹配
# 注意區分大小寫: YOUR_ORG/eco-base ≠ your-org/eco-base
```

### 錯誤: `Attribute condition does not match`

```
Error: Attribute condition does not match
```

**解決方案:**
```bash
# 檢查 attribute-condition
gcloud iam workload-identity-pools providers describe github-provider \
  --workload-identity-pool="github-actions" \
  --location="global" \
  --project="$GCP_PROJECT_ID" \
  --format="value(attributeCondition)"

# 更新 condition
gcloud iam workload-identity-pools providers update-oidc github-provider \
  --workload-identity-pool="github-actions" \
  --location="global" \
  --project="$GCP_PROJECT_ID" \
  --attribute-condition="assertion.repository_owner == 'YOUR_ACTUAL_ORG'"
```

## 安全最佳實踐

1. **使用最小權限**: 只授予必要的 IAM 角色
2. **限制 Repository 訪問**: 在 attribute-condition 中指定允許的倉庫
3. **定期輪換**: 雖然 Workload Identity 無需密鑰，但仍應定期審核 IAM 綁定
4. **啟用審計日誌**: 在 GCP 中啟用 IAM 審計日誌

```bash
# 啟用審計日誌
gcloud projects get-iam-policy $GCP_PROJECT_ID > policy.yaml

# 編輯 policy.yaml 添加 auditConfigs
# auditConfigs:
# - auditLogConfigs:
#   - logType: DATA_READ
#   - logType: DATA_WRITE
#   - logType: ADMIN_READ
#   service: allServices

gcloud projects set-iam-policy $GCP_PROJECT_ID policy.yaml
```

## 參考資料

- [GitHub Actions GCP Auth](https://github.com/google-github-actions/auth)
- [Workload Identity Federation](https://cloud.google.com/iam/docs/workload-identity-federation)
- [GitHub Environments](https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment)
