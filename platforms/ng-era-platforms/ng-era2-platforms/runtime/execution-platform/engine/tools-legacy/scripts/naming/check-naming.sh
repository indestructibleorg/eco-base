#!/bin/bash
# 命名規範檢查腳本
# 掃描 Kubernetes manifests 並驗證命名規範

set -euo pipefail

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置
POLICY_DIR="${POLICY_DIR:-governance/policies/conftest}"
MANIFEST_DIR="${MANIFEST_DIR:-deploy}"
REPORT_FILE="${REPORT_FILE:-artifacts/reports/naming/naming-violations.json}"

echo -e "${GREEN}🔍 開始命名規範檢查...${NC}"

# 檢查 conftest 是否安裝
if ! command -v conftest &> /dev/null; then
    echo -e "${RED}❌ 錯誤：conftest 未安裝${NC}"
    echo "請執行: wget https://github.com/open-policy-agent/conftest/releases/download/v0.54.0/conftest_0.54.0_Linux_x86_64.tar.gz && tar xzf conftest_0.54.0_Linux_x86_64.tar.gz && sudo mv conftest /usr/local/bin/"
    exit 1
fi

# 建立報告目錄
mkdir -p "$(dirname "$REPORT_FILE")"

# 初始化計數器
total_files=0
passed_files=0
failed_files=0
warnings=0

# 臨時檔案
violations_json=$(mktemp)
echo "[]" > "$violations_json"

# 掃描所有 YAML 檔案
echo -e "${GREEN}📁 掃描目錄：${MANIFEST_DIR}${NC}"

# 使用 process substitution 避免 subshell 問題
while read -r manifest; do
    total_files=$((total_files + 1))
    echo -e "\n${YELLOW}檢查：${manifest}${NC}"
    
    # 執行 conftest
    if conftest test "$manifest" --policy "$POLICY_DIR" --output json > /tmp/conftest_output.json 2>&1; then
        echo -e "${GREEN}✅ 通過${NC}"
        passed_files=$((passed_files + 1))
    else
        echo -e "${RED}❌ 發現違規${NC}"
        failed_files=$((failed_files + 1))
        
        # 解析違規資訊
        if [ -f /tmp/conftest_output.json ]; then
            cat /tmp/conftest_output.json
        fi
    fi
done < <(find . -type f \( -name "*.yaml" -o -name "*.yml" \) -path "*/${MANIFEST_DIR}/*")

# 生成摘要報告
echo -e "\n${GREEN}📊 檢查摘要${NC}"
echo "總檔案數: $total_files"
echo -e "${GREEN}通過: $passed_files${NC}"
echo -e "${RED}失敗: $failed_files${NC}"
echo -e "${YELLOW}警告: $warnings${NC}"

# 生成 JSON 報告
cat > "$REPORT_FILE" <<EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "summary": {
    "total_files": $total_files,
    "passed": $passed_files,
    "failed": $failed_files,
    "warnings": $warnings
  },
  "policy_dir": "$POLICY_DIR",
  "manifest_dir": "$MANIFEST_DIR"
}
EOF

echo -e "\n${GREEN}📄 報告已儲存：${REPORT_FILE}${NC}"

# 清理
rm -f "$violations_json" /tmp/conftest_output.json

# 退出碼
if [ $failed_files -gt 0 ]; then
    echo -e "${RED}❌ 檢查失敗：發現 $failed_files 個違規檔案${NC}"
    exit 1
else
    echo -e "${GREEN}✅ 所有檢查通過${NC}"
    exit 0
fi
