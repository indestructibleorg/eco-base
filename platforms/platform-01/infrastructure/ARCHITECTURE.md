# Eco-Base 架構設計文檔

## 設計原則

本架構遵循以下企業級設計原則:

1. **GitOps**: 所有基礎設施和應用配置存儲在 Git 中，ArgoCD 自動同步
2. **安全默認**: 零信任網路、最小權限、Workload Identity
3. **可觀測性**: 完整的監控、日誌、告警體系
4. **高可用**: 多區域、自動擴展、Pod 中斷預算
5. **成本優化**: 自動擴縮、預留實例、資源請求/限制

## 架構組件

### 1. 計算層 (GKE)

```
┌─────────────────────────────────────────────────────────┐
│                     GKE Cluster                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ System Pool │  │ General Pool│  │    GPU Pool     │ │
│  │ (2 nodes)   │  │ (2-10 nodes)│  │  (0-4 nodes)    │ │
│  │             │  │             │  │                 │ │
│  │ • kube-dns  │  │ • API       │  │ • AI Inference  │ │
│  │ • calico    │  │ • Web       │  │ • Model Serving │ │
│  │ • monitoring│  │ • Workers   │  │                 │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**節點池配置:**
- **System Pool**: e2-standard-4, 固定 2 節點, 運行系統組件
- **General Pool**: e2-standard-8, 自動擴展 2-10 節點, 運行應用
- **GPU Pool**: n1-standard-8 + T4, 自動擴展 0-4 節點, AI 推理

### 2. 網路層

```
┌─────────────────────────────────────────────────────────┐
│                      VPC Network                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │   Subnet    │  │  Pods CIDR  │  │ Services CIDR   │ │
│  │ 10.0.0.0/20 │  │ 10.4.0.0/14 │  │  10.8.0.0/20    │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
│                                                          │
│  • Private Cluster (Master 端點私有)                     │
│  • Cloud NAT (Private 節點訪問外部)                      │
│  • Network Policies (零信任網路)                        │
└─────────────────────────────────────────────────────────┘
```

### 3. 存儲層

| 服務 | 類型 | 用途 |
|-----|------|------|
| Supabase | 託管 PostgreSQL | 主要數據庫 |
| Cloud Storage | 對象存儲 | 文件、模型 |
| Persistent Disk | SSD | 模型緩存 |
| Redis | 內存 KV | 緩存、會話 |

### 4. GitOps 層 (ArgoCD)

```
┌─────────────────────────────────────────────────────────┐
│                      ArgoCD                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │              ApplicationSet                      │   │
│  │  ┌───────────────┐  ┌─────────────────────────┐ │   │
│  │  │   Staging     │  │       Production        │ │   │
│  │  │ • Auto-sync   │  │ • Manual sync           │ │   │
│  │  │ • Prune       │  │ • Approval required     │ │   │
│  │  │ • Self-heal   │  │ • Prune disabled        │ │   │
│  │  └───────────────┘  └─────────────────────────┘ │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 5. CI/CD 層 (GitHub Actions)

```
┌─────────────────────────────────────────────────────────┐
│                   CI/CD Pipeline                         │
│                                                          │
│  Push to main                                            │
│      │                                                   │
│      ▼                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌────────────┐  │
│  │    Build    │───▶│    Scan     │───▶│   Push     │  │
│  │  • Lint     │    │  • Trivy    │    │  • AR      │  │
│  │  • Test     │    │  • Bandit   │    │  • SBOM    │  │
│  │  • Compile  │    │  • Secrets  │    │            │  │
│  └─────────────┘    └─────────────┘    └─────┬──────┘  │
│                                               │          │
│                                               ▼          │
│  ┌────────────────────────────────────────────────────┐ │
│  │                  Deploy                            │ │
│  │  Staging: Auto-sync    Production: Manual approval │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### 6. 監控層

```
┌─────────────────────────────────────────────────────────┐
│                   Observability Stack                    │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ Prometheus  │  │   Grafana   │  │  Alertmanager   │ │
│  │ (Metrics)   │  │(Dashboards) │  │   (Alerts)      │ │
│  └──────┬──────┘  └─────────────┘  └─────────────────┘ │
│         │                                                │
│         ▼                                                │
│  ┌─────────────────────────────────────────────────────┐│
│  │  Data Sources:                                       ││
│  │  • kube-state-metrics  • node-exporter              ││
│  │  • cadvisor            • application metrics        ││
│  │  • custom exporters                                  ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

## 安全架構

### 身份與訪問

```
┌─────────────────────────────────────────────────────────┐
│                  Identity & Access                       │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │   GitHub    │  │   GCP IAM   │  │  K8s RBAC       │ │
│  │   Actions   │──▶│  Workload   │──▶│  ServiceAccount │ │
│  │             │  │  Identity   │  │                 │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
│                                                          │
│  無長期密鑰 → 短期令牌 → Pod 身份                        │
└─────────────────────────────────────────────────────────┘
```

### 網路安全

| 層級 | 機制 | 配置 |
|-----|------|------|
| 集群 | Private Cluster | Master 端點不公開 |
| 節點 | Authorized Networks | 限制 API 訪問來源 |
| Pod | Network Policies | 默認拒絕，明確允許 |
| 應用 | mTLS | 服務間加密 (可選) |

## 數據流

### 用戶請求流程

```
User
  │
  ▼
Cloudflare (CDN + WAF)
  │
  ▼
GCE Load Balancer
  │
  ▼
Ingress-Nginx (SSL termination)
  │
  ├──▶ Web Frontend (React)
  │
  └──▶ API Service (FastAPI)
           │
           ├──▶ Supabase (PostgreSQL)
           │
           ├──▶ Redis (Cache)
           │
           └──▶ AI Inference (GPU)
```

### 部署流程

```
Developer Push
  │
  ▼
GitHub Actions
  │
  ├──▶ Build Image
  │
  ├──▶ Security Scan
  │
  └──▶ Push to AR
           │
           ▼
      Update Git (Helm values)
           │
           ▼
      ArgoCD Sync
           │
           ├──▶ Staging (Auto)
           │
           └──▶ Production (Manual)
```

## 擴展策略

### 水平擴展

```yaml
# HPA 配置
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 60
```

### 垂直擴展

```yaml
# VPA 配置 (可選)
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  updatePolicy:
    updateMode: "Auto"
```

### 集群自動擴展

```yaml
# Node Auto-provisioning
clusterAutoscaling:
  enabled: true
  resourceLimits:
    maximumCores: 100
    maximumMemoryGb: 400
  autoProvisioningDefaults:
    serviceAccount: gke-nodes@project.iam.gserviceaccount.com
    oauthScopes:
      - https://www.googleapis.com/auth/cloud-platform
```

## 災難恢復

### 備份策略

| 組件 | 備份方式 | 頻率 |
|-----|---------|------|
| Supabase | 自動備份 | 每日 |
| Persistent Disks | 快照 | 每日 |
| Git 倉庫 | GitHub | 實時 |
| Terraform State | GCS 版本控制 | 實時 |

### 恢復流程

```bash
# 1. 從備份恢復 Supabase
# 使用 Supabase Dashboard 或 API

# 2. 從快照恢復 PD
kubectl apply -f restore-pvc.yaml

# 3. 重新部署應用
argocd app sync eco-base-production
```

## 性能基準

### 預期性能

| 指標 | 目標 | 測試方法 |
|-----|------|---------|
| API P99 延遲 | < 200ms | k6 load test |
| 吞吐量 | > 1000 RPS | k6 load test |
| 可用性 | 99.9% | Uptime monitoring |
| 錯誤率 | < 0.1% | Prometheus |

### 負載測試

```bash
# 安裝 k6
brew install k6

# 運行負載測試
k6 run --vus 100 --duration 5m load-test.js
```

## 成本優化

### 預留實例

```bash
# 購買 Committed Use Discounts
gcloud compute commitments create general-commitment \
  --region=us-central1 \
  --resources=vcpu=20,memory=80 \
  --plan=12-month \
  --type=GENERAL_PURPOSE
```

### 自動縮減

```yaml
# 非工作時間縮減 Staging
apiVersion: autoscaling.x-k8s.io/v1alpha1
kind: ScheduledScaler
spec:
  scaleTargetRef:
    name: api
  schedules:
    - start: "0 18 * * 1-5"  # 週一至週五 18:00
      end: "0 9 * * 1-5"     # 週一至週五 09:00
      replicas: 1
```

## 參考資料

- [GKE Best Practices](https://cloud.google.com/kubernetes-engine/docs/best-practices)
- [ArgoCD Documentation](https://argo-cd.readthedocs.io/)
- [GitHub Actions Security](https://docs.github.com/en/actions/security-guides)
- [Prometheus Operator](https://prometheus-operator.dev/)
