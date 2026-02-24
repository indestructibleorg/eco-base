# =============================================================================
# Eco-Backend Makefile
# =============================================================================

.PHONY: help install dev test lint format migrate run docker-build docker-up docker-down k8s-deploy k8s-delete clean

# 默認目標
help:
	@echo "Eco-Backend 可用命令:"
	@echo ""
	@echo "  make install       - 安裝依賴"
	@echo "  make dev           - 啟動開發服務器"
	@echo "  make test          - 運行測試"
	@echo "  make lint          - 運行代碼檢查"
	@echo "  make format        - 格式化代碼"
	@echo "  make migrate       - 運行數據庫遷移"
	@echo "  make migrate-create - 創建新的遷移"
	@echo "  make run           - 啟動生產服務器"
	@echo "  make docker-build  - 構建 Docker 鏡像"
	@echo "  make docker-up     - 啟動 Docker Compose"
	@echo "  make docker-down   - 停止 Docker Compose"
	@echo "  make k8s-deploy    - 部署到 Kubernetes"
	@echo "  make k8s-delete    - 從 Kubernetes 刪除"
	@echo "  make clean         - 清理臨時文件"

# 安裝依賴
install:
	pip install -r requirements.txt

# 開發服務器
dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 運行測試
test:
	pytest -v

# 運行測試並生成覆蓋率報告
test-cov:
	pytest --cov=app --cov-report=html --cov-report=term

# 代碼檢查
lint:
	flake8 app tests --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 app tests --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# 格式化代碼
format:
	black app tests
	isort app tests

# 數據庫遷移
migrate:
	alembic upgrade head

# 創建新的遷移
migrate-create:
	@read -p "Enter migration message: " message; \
	alembic revision --autogenerate -m "$$message"

# 生產服務器
run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Docker 構建
docker-build:
	docker build -t eco-backend:latest -f deployments/docker/Dockerfile .

# Docker Compose 啟動
docker-up:
	cd deployments/docker && docker-compose up -d

# Docker Compose 停止
docker-down:
	cd deployments/docker && docker-compose down

# Docker Compose 日誌
docker-logs:
	cd deployments/docker && docker-compose logs -f

# Kubernetes 部署
k8s-deploy:
	kubectl apply -f deployments/k8s/namespace.yaml
	kubectl apply -f deployments/k8s/configmap.yaml
	kubectl apply -f deployments/k8s/secrets.yaml
	kubectl apply -f deployments/k8s/rbac.yaml
	kubectl apply -f deployments/k8s/deployment.yaml
	kubectl apply -f deployments/k8s/service.yaml

# Kubernetes 刪除
k8s-delete:
	kubectl delete -f deployments/k8s/

# 清理
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
