#!/usr/bin/env bash
# SuperAI Platform - Deployment Script
set -euo pipefail

NAMESPACE="${SUPERAI_NAMESPACE:-superai}"
RELEASE_NAME="${SUPERAI_RELEASE:-superai}"
CHART_DIR="./helm"
VALUES_FILE="./helm/values.yaml"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║         SuperAI Platform - Deployment Script            ║"
echo "╚══════════════════════════════════════════════════════════╝"

# ── Pre-flight checks ────────────────────────────────────────
command -v kubectl >/dev/null 2>&1 || { echo "kubectl required"; exit 1; }
command -v helm >/dev/null 2>&1 || { echo "helm required"; exit 1; }

echo "[1/6] Creating namespace..."
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

echo "[2/6] Applying base resources..."
kubectl apply -f k8s/base/namespace.yaml
kubectl apply -f k8s/base/configmap.yaml

echo "[3/6] Deploying infrastructure (Redis, PostgreSQL)..."
kubectl apply -f k8s/base/redis.yaml
kubectl apply -f k8s/base/postgres.yaml

echo "  Waiting for Redis..."
kubectl rollout status statefulset/redis -n "$NAMESPACE" --timeout=120s
echo "  Waiting for PostgreSQL..."
kubectl rollout status statefulset/postgres -n "$NAMESPACE" --timeout=120s

echo "[4/6] Deploying inference engines..."
kubectl apply -f k8s/base/vllm-engine.yaml
kubectl apply -f k8s/base/tgi-engine.yaml
kubectl apply -f k8s/base/sglang-engine.yaml
kubectl apply -f k8s/base/ollama-engine.yaml

echo "[5/6] Deploying API Gateway..."
kubectl apply -f k8s/base/api-gateway.yaml

echo "  Waiting for API Gateway..."
kubectl rollout status deployment/superai-api -n "$NAMESPACE" --timeout=300s

echo "[6/6] Deploying monitoring & ingress..."
kubectl apply -f k8s/monitoring/prometheus.yaml
kubectl apply -f k8s/monitoring/grafana.yaml
kubectl apply -f k8s/ingress/ingress.yaml

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              Deployment Complete                        ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  API:        kubectl port-forward svc/superai-api-svc 8000:8000 -n $NAMESPACE"
echo "║  Prometheus: kubectl port-forward svc/prometheus-svc 9090:9090 -n $NAMESPACE"
echo "║  Grafana:    kubectl port-forward svc/grafana-svc 3000:3000 -n $NAMESPACE"
echo "╚══════════════════════════════════════════════════════════╝"

kubectl get pods -n "$NAMESPACE" -o wide