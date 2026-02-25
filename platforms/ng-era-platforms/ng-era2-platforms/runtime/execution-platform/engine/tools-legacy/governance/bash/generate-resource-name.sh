#!/usr/bin/env bash
# 资源名称生成工具
# 用途: 根据命名规范自动生成标准化的资源名称
# 使用: ./generate_resource_name.sh --environment prod --app payment --resource-type deploy --version v1.0.0

set -euo pipefail

# 默认值
ENVIRONMENT=""
APP=""
RESOURCE_TYPE=""
VERSION=""
NAMESPACE=""
REGION=""
OUTPUT_FORMAT="text"  # text | json | yaml

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 帮助信息
show_help() {
    cat << EOF
资源名称生成工具

用途:
  根据组织命名规范自动生成标准化的资源名称

使用:
  $0 [OPTIONS]

必需选项:
  -e, --environment ENV     环境 (dev|staging|prod)
  -a, --app APP             应用名称 (小写、数字、连字符)
  -r, --resource-type TYPE  资源类型 (deploy|svc|ing|cm|secret等)
  -v, --version VERSION     版本号 (格式: vX.Y.Z)

可选选项:
  -n, --namespace NS        命名空间
  --region REGION           地理区域
  -f, --format FORMAT       输出格式 (text|json|yaml) [默认: text]
  -h, --help                显示此帮助信息

示例:
  # 基本使用
  $0 -e prod -a payment -r deploy -v v1.0.0

  # 包含命名空间
  $0 -e prod -a user-api -r svc -v v2.3.1 -n production

  # JSON 输出
  $0 -e staging -a order -r deploy -v v1.0.0-beta1 -f json

支持的资源类型:
  deploy      - Deployment
  svc         - Service
  ing         - Ingress
  cm          - ConfigMap
  secret      - Secret
  pvc         - PersistentVolumeClaim
  sa          - ServiceAccount
  job         - Job
  cronjob     - CronJob
  hpa         - HorizontalPodAutoscaler

EOF
}

# 验证环境
validate_environment() {
    local env="$1"
    if [[ ! "$env" =~ ^(dev|staging|prod)$ ]]; then
        echo -e "${RED}错误: 环境必须是 dev, staging, 或 prod${NC}" >&2
        exit 1
    fi
}

# 验证应用名称
validate_app() {
    local app="$1"
    if [[ ! "$app" =~ ^[a-z0-9-]{3,30}$ ]]; then
        echo -e "${RED}错误: 应用名称必须是 3-30 个字符的小写字母、数字和连字符${NC}" >&2
        exit 1
    fi
}

# 验证资源类型
validate_resource_type() {
    local type="$1"
    local valid_types="deploy svc ing cm secret pvc sa job cronjob hpa"
    if [[ ! " $valid_types " =~ " $type " ]]; then
        echo -e "${RED}错误: 无效的资源类型 '$type'${NC}" >&2
        echo "支持的类型: $valid_types" >&2
        exit 1
    fi
}

# 验证版本号
validate_version() {
    local version="$1"
    if [[ ! "$version" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?$ ]]; then
        echo -e "${RED}错误: 版本号必须符合语义化版本规范 (vX.Y.Z[-PRERELEASE])${NC}" >&2
        exit 1
    fi
}

# 生成资源名称
generate_name() {
    local env="$1"
    local app="$2"
    local type="$3"
    local version="$4"

    # 基本命名格式: {environment}-{app}-{resource-type}-{version}
    echo "${env}-${app}-${type}-${version}"
}

# 输出结果
output_result() {
    local name="$1"
    local format="$2"

    case "$format" in
        text)
            echo -e "${GREEN}生成的资源名称:${NC}"
            echo "$name"
            ;;
        json)
            cat << EOF
{
  "resourceName": "$name",
  "components": {
    "environment": "$ENVIRONMENT",
    "app": "$APP",
    "resourceType": "$RESOURCE_TYPE",
    "version": "$VERSION"
$(if [ -n "$NAMESPACE" ]; then echo "    ,\"namespace\": \"$NAMESPACE\""; fi)
$(if [ -n "$REGION" ]; then echo "    ,\"region\": \"$REGION\""; fi)
  },
  "compliant": true
}
EOF
            ;;
        yaml)
            cat << EOF
resourceName: $name
components:
  environment: $ENVIRONMENT
  app: $APP
  resourceType: $RESOURCE_TYPE
  version: $VERSION
$(if [ -n "$NAMESPACE" ]; then echo "  namespace: $NAMESPACE"; fi)
$(if [ -n "$REGION" ]; then echo "  region: $REGION"; fi)
compliant: true
EOF
            ;;
        *)
            echo -e "${RED}错误: 不支持的输出格式 '$format'${NC}" >&2
            exit 1
            ;;
    esac
}

# 解析参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -a|--app)
                APP="$2"
                shift 2
                ;;
            -r|--resource-type)
                RESOURCE_TYPE="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --region)
                REGION="$2"
                shift 2
                ;;
            -f|--format)
                OUTPUT_FORMAT="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo -e "${RED}错误: 未知选项 '$1'${NC}" >&2
                show_help
                exit 1
                ;;
        esac
    done
}

# 主函数
main() {
    # 解析参数
    parse_args "$@"

    # 检查必需参数
    if [ -z "$ENVIRONMENT" ] || [ -z "$APP" ] || [ -z "$RESOURCE_TYPE" ] || [ -z "$VERSION" ]; then
        echo -e "${RED}错误: 缺少必需参数${NC}" >&2
        echo ""
        show_help
        exit 1
    fi

    # 验证参数
    validate_environment "$ENVIRONMENT"
    validate_app "$APP"
    validate_resource_type "$RESOURCE_TYPE"
    validate_version "$VERSION"

    # 生成名称
    RESOURCE_NAME=$(generate_name "$ENVIRONMENT" "$APP" "$RESOURCE_TYPE" "$VERSION")

    # 输出结果
    output_result "$RESOURCE_NAME" "$OUTPUT_FORMAT"

    # 提示信息
    if [ "$OUTPUT_FORMAT" = "text" ]; then
        echo ""
        echo -e "${YELLOW}使用提示:${NC}"
        echo "- 确保名称符合 DNS-1123 规范"
        echo "- 总长度不超过 63 字符"
        echo "- 可以在 Kubernetes manifests 中使用此名称"
    fi
}

# 运行主函数
main "$@"
