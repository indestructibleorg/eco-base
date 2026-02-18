#!/usr/bin/env bash
# SuperAI Platform - Build Script
set -euo pipefail

VERSION="${1:-1.0.0}"
REGISTRY="${DOCKER_REGISTRY:-ghcr.io/superai-platform}"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║           SuperAI Platform - Build v${VERSION}              ║"
echo "╚══════════════════════════════════════════════════════════╝"

echo "[1/4] Building API Gateway image..."
docker build \
  -t "${REGISTRY}/api:${VERSION}" \
  -t "${REGISTRY}/api:latest" \
  -f docker/Dockerfile \
  .

echo "[2/4] Building vLLM engine image..."
docker build \
  -t "${REGISTRY}/vllm:${VERSION}" \
  -f docker/Dockerfile.gpu \
  --target vllm \
  .

echo "[3/4] Building SGLang engine image..."
docker build \
  -t "${REGISTRY}/sglang:${VERSION}" \
  -f docker/Dockerfile.gpu \
  --target sglang \
  .

echo "[4/4] Pushing images..."
docker push "${REGISTRY}/api:${VERSION}"
docker push "${REGISTRY}/api:latest"
docker push "${REGISTRY}/vllm:${VERSION}"
docker push "${REGISTRY}/sglang:${VERSION}"

echo ""
echo "Build complete. Images:"
echo "  ${REGISTRY}/api:${VERSION}"
echo "  ${REGISTRY}/vllm:${VERSION}"
echo "  ${REGISTRY}/sglang:${VERSION}"