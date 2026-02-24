# =============================================================================
# Eco-Base Infrastructure Makefile
# =============================================================================
# 常用命令快捷方式
# =============================================================================

.PHONY: help init plan apply destroy deploy-all
.PHONY: argocd-login argocd-sync argocd-status
.PHONY: grafana-port-forward logs

# 默認環境
ENV ?= staging
GCP_PROJECT_ID ?= $(shell gcloud config get-value project 2>/dev/null)

# =============================================================================
# 幫助
# =============================================================================

help: ## 顯示幫助信息
	@echo "Eco-Base Infrastructure Management"
	@echo "==================================="
	@echo ""
	@echo "可用命令:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Terraform 操作
# =============================================================================

init: ## 初始化 Terraform
	cd terraform/environments/$(ENV) && terraform init

plan: ## 規劃 Terraform 變更
	cd terraform/environments/$(ENV) && terraform plan

apply: ## 應用 Terraform 變更
	cd terraform/environments/$(ENV) && terraform apply

destroy: ## 銷毀 Terraform 資源 (警告: 會刪除所有資源!)
	@echo "警告: 這將刪除所有 $(ENV) 環境的資源!"
	@read -p "確定要繼續嗎? (yes/no): " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		cd terraform/environments/$(ENV) && terraform destroy; \
	else \
		echo "已取消"; \
	fi

output: ## 顯示 Terraform 輸出
	cd terraform/environments/$(ENV) && terraform output

# =============================================================================
# 完整部署
# =============================================================================

deploy-all: ## 執行完整部署腳本
	./scripts/deploy.sh $(ENV)

# =============================================================================
# ArgoCD 操作
# =============================================================================

argocd-port-forward: ## 端口轉發 ArgoCD UI
	@echo "ArgoCD UI: https://localhost:8080"
	@kubectl port-forward svc/argocd-server -n argocd 8080:443

argocd-password: ## 獲取 ArgoCD 管理員密碼
	@kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
	@echo ""

argocd-login: ## 登入 ArgoCD CLI
	@argocd_password=$$(kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d); \
	kubectl port-forward svc/argocd-server -n argocd 8080:443 & \
	sleep 3; \
	argocd login localhost:8080 --username admin --password "$$argocd_password" --insecure

argocd-sync: ## 同步所有 ArgoCD 應用
	@argocd app sync -l app.kubernetes.io/managed-by=argocd

argocd-status: ## 查看 ArgoCD 應用狀態
	@argocd app list

# =============================================================================
# 監控
# =============================================================================

grafana-port-forward: ## 端口轉發 Grafana
	@echo "Grafana: http://localhost:3000"
	@kubectl port-forward svc/prometheus-grafana -n monitoring 3000:80

grafana-password: ## 獲取 Grafana 管理員密碼
	@kubectl -n monitoring get secret grafana-admin-credentials -o jsonpath="{.data.admin-password}" | base64 -d
	@echo ""

prometheus-port-forward: ## 端口轉發 Prometheus
	@echo "Prometheus: http://localhost:9090"
	@kubectl port-forward svc/prometheus-kube-prometheus-prometheus -n monitoring 9090:9090

# =============================================================================
# 日誌和調試
# =============================================================================

logs-api: ## 查看 API 服務日誌
	@kubectl logs -l app.kubernetes.io/name=api -n $(ENV) --tail=100 -f

logs-ai: ## 查看 AI 推理服務日誌
	@kubectl logs -l app.kubernetes.io/name=ai-inference -n $(ENV) --tail=100 -f

logs-web: ## 查看 Web 前端日誌
	@kubectl logs -l app.kubernetes.io/name=web -n $(ENV) --tail=100 -f

pods: ## 查看所有 Pod
	@kubectl get pods -n $(ENV)

pods-all: ## 查看所有命名空間的 Pod
	@kubectl get pods --all-namespaces

events: ## 查看事件
	@kubectl get events -n $(ENV) --sort-by='.lastTimestamp'

describe-pod: ## 描述 Pod (需要設置 POD 變數)
	@kubectl describe pod $(POD) -n $(ENV)

# =============================================================================
# 網路和 Ingress
# =============================================================================

ingress-ip: ## 獲取 Ingress IP
	@kubectl get svc -n ingress-nginx ingress-nginx-controller -o jsonpath="{.status.loadBalancer.ingress[0].ip}"
	@echo ""

ingress-logs: ## 查看 Ingress 日誌
	@kubectl logs -l app.kubernetes.io/name=ingress-nginx -n ingress-nginx --tail=100 -f

# =============================================================================
# 證書管理
# =============================================================================

certificates: ## 查看證書狀態
	@kubectl get certificates -A

certificate-describe: ## 描述證書 (需要設置 CERT 和 NS 變數)
	@kubectl describe certificate $(CERT) -n $(NS)

# =============================================================================
# 清理
# =============================================================================

clean-tf: ## 清理 Terraform 緩存
	@find terraform -type d -name ".terraform" -exec rm -rf {} + 2>/dev/null || true
	@find terraform -type f -name "*.tfstate*" -delete 2>/dev/null || true
	@find terraform -type f -name ".terraform.lock.hcl" -delete 2>/dev/null || true

clean-helm: ## 清理 Helm 緩存
	@helm repo update

# =============================================================================
# 開發工具
# =============================================================================

fmt: ## 格式化 Terraform 代碼
	@terraform fmt -recursive terraform/

validate: ## 驗證 Terraform 配置
	@cd terraform/environments/$(ENV) && terraform validate

lint: ## 運行所有檢查
	@echo "格式化 Terraform..."
	@$(MAKE) fmt
	@echo "驗證 Terraform..."
	@$(MAKE) validate
	@echo "檢查完成!"

# =============================================================================
# 資訊
# =============================================================================

info: ## 顯示環境資訊
	@echo "環境: $(ENV)"
	@echo "GCP 專案: $(GCP_PROJECT_ID)"
	@echo ""
	@echo "GKE 集群:"
	@gcloud container clusters list --filter="name:$(ENV)" --format="table(name, location, status)"
	@echo ""
	@echo "節點池:"
	@kubectl get nodes -o custom-columns="NAME:.metadata.name,STATUS:.status.conditions[-1].type,ROLE:.metadata.labels.node-type"
