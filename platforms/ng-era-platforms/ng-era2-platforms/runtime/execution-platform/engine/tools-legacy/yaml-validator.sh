#!/bin/bash
# GL YAML 驗證工具
# 獨立的 YAML 驗證腳本，可手動執行或集成到 CI/CD

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GL_DIR="$REPO_ROOT/gl"

# 統計變量
TOTAL_FILES=0
VALID_FILES=0
INVALID_FILES=0
WARNING_FILES=0

echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     GL YAML Validation Tool v1.0.0        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
echo

# 檢查依賴
check_dependencies() {
    echo -e "${YELLOW}Checking dependencies...${NC}"
    
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ ERROR: python3 not found${NC}"
        exit 1
    fi
    
    if ! python3 -c "import yaml" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  Warning: PyYAML not installed${NC}"
        echo -e "    Install with: pip install pyyaml"
        exit 1
    fi
    
    if ! python3 -c "import json" 2>/dev/null; then
        echo -e "${RED}❌ ERROR: json module not available${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ All dependencies satisfied${NC}"
    echo
}

# 驗證單個 YAML 文件
validate_yaml_file() {
    local file="$1"
    TOTAL_FILES=$((TOTAL_FILES + 1))
    
    echo -e "${BLUE}Validating:${NC} $file"
    
    # 語法驗證
    if ! python3 - "$file" <<'EOF' 2>/dev/null; then
import yaml
import sys

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    yaml.safe_load(f)
EOF
        echo -e "  ${RED}❌ ERROR: Invalid YAML syntax${NC}"
        INVALID_FILES=$((INVALID_FILES + 1))
        return 1
    fi
    
    # 結構驗證
    local validation_result=$(python3 - "$file" <<'EOF'
import yaml
import sys
import json

try:
    file_path = sys.argv[1]
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    issues = []
    warnings = []
    
    # 基本結構檢查
    if not isinstance(data, dict):
        issues.append("Root must be a dictionary")
    
    # 檢查必需字段
    if 'version' in data and data['version'] is not None:
        version = str(data['version'])
        if not version.replace('.', '').isdigit():
            warnings.append("Version should be in semver format")
    
    # GL 特定檢查
    if file_path.endswith('gl/00-strategic/DEFINITION.yaml') or file_path.endswith('gl/00-strategic/DEFINITION.yml'):
        required_fields = ['version', 'description', 'scope']
        for field in required_fields:
            if field not in data:
                issues.append(f"Missing required field: {field}")
    
    # 檢查描述字段
    if 'description' in data and data['description'] is not None:
        if not isinstance(data['description'], str):
            issues.append("description must be a string")
    
    # 輸出結果
    result = {
        'valid': len(issues) == 0,
        'has_warnings': len(warnings) > 0,
        'issues': issues,
        'warnings': warnings
    }
    
    print(json.dumps(result))
except Exception as e:
    import traceback
    error_msg = f"{str(e)}\n{''.join(traceback.format_stack()[-3:])}"
    print(json.dumps({
        'valid': False,
        'has_warnings': False,
        'issues': [error_msg],
        'warnings': []
    }))
EOF
)
    
    local valid=$(echo "$validation_result" | python3 -c "import sys, json; print(json.load(sys.stdin)['valid'])")
    local has_warnings=$(echo "$validation_result" | python3 -c "import sys, json; print(json.load(sys.stdin)['has_warnings'])")
    local issues=$(echo "$validation_result" | python3 -c "import sys, json; print('\\n'.join(json.load(sys.stdin)['issues']))")
    local warnings=$(echo "$validation_result" | python3 -c "import sys, json; print('\\n'.join(json.load(sys.stdin)['warnings']))")
    
    if [ "$valid" == "True" ]; then
        if [ "$has_warnings" == "True" ]; then
            echo -e "  ${YELLOW}⚠️  Warnings:${NC}"
            echo "$warnings" | sed 's/^/    /'
            WARNING_FILES=$((WARNING_FILES + 1))
        else
            echo -e "  ${GREEN}✅ Valid${NC}"
        fi
        VALID_FILES=$((VALID_FILES + 1))
    else
        echo -e "  ${RED}❌ Issues:${NC}"
        echo "$issues" | sed 's/^/    /'
        INVALID_FILES=$((INVALID_FILES + 1))
    fi
    
    echo
}

# 驗證所有 GL YAML 文件
validate_all_yaml() {
    echo -e "${YELLOW}Scanning for YAML files in $GL_DIR...${NC}"
    echo
    
    if [ ! -d "$GL_DIR" ]; then
        echo -e "${RED}❌ ERROR: GL directory not found: $GL_DIR${NC}"
        exit 1
    fi
    
    # 查找所有 YAML 文件
    local yaml_files=$(find "$GL_DIR" -type f -name "*.yaml" -o -name "*.yml" | sort)
    
    if [ -z "$yaml_files" ]; then
        echo -e "${YELLOW}⚠️  No YAML files found${NC}"
        exit 0
    fi
    
    echo -e "Found $(echo "$yaml_files" | wc -l) YAML file(s)"
    echo
    echo -e "${YELLOW}Starting validation...${NC}"
    echo "----------------------------------------"
    echo
    
    # 驗證每個文件
    while IFS= read -r file; do
        validate_yaml_file "$file"
    done <<< "$yaml_files"
}

# 生成報告
generate_report() {
    echo "========================================"
    echo -e "${BLUE}Validation Summary${NC}"
    echo "========================================"
    echo
    echo -e "Total files scanned: ${BLUE}$TOTAL_FILES${NC}"
    echo -e "Valid files:        ${GREEN}$VALID_FILES${NC}"
    echo -e "Files with warnings: ${YELLOW}$WARNING_FILES${NC}"
    echo -e "Invalid files:      ${RED}$INVALID_FILES${NC}"
    echo
    
    if [ $INVALID_FILES -gt 0 ]; then
        echo -e "${RED}❌ VALIDATION FAILED${NC}"
        echo "Please fix the errors above."
        return 1
    elif [ $WARNING_FILES -gt 0 ]; then
        echo -e "${YELLOW}⚠️  VALIDATION COMPLETED WITH WARNINGS${NC}"
        return 0
    else
        echo -e "${GREEN}✅ ALL FILES VALID${NC}"
        return 0
    fi
}

# 主函數
main() {
    cd "$REPO_ROOT"
    
    # 檢查依賴
    check_dependencies
    
    # 驗證所有 YAML 文件
    validate_all_yaml
    
    # 生成報告
    generate_report
}

# 執行主函數
main "$@"