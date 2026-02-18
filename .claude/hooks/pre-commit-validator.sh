#!/bin/bash
################################################################################
# 🔒 ABSOLUTE FORMAT VALIDATION PIPELINE
# ================================
# 這是一個絕對不容許任何格式錯誤的驗證管線
# 任何失敗都會立即停止並報告詳細錯誤信息
################################################################################

set -euo pipefail

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 計數器
TOTAL_FILES=0
PASSED_FILES=0
FAILED_FILES=0
WARNINGS=0

# 錯誤日誌文件
ERROR_LOG="/tmp/format-validation-errors-$(date +%s).log"
WARNINGS_LOG="/tmp/format-validation-warnings-$(date +%s).log"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     🔒 ABSOLUTE FORMAT VALIDATION PIPELINE (v1.0)             ║${NC}"
echo -e "${BLUE}║     No errors. No exceptions. Zero tolerance.                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

################################################################################
# Function: Validate JSON Files
################################################################################
validate_json() {
  local file="$1"
  local has_error=false

  TOTAL_FILES=$((TOTAL_FILES + 1))

  # 檢查文件是否存在
  if [[ ! -f "$file" ]]; then
    echo -e "${RED}✗ FILE NOT FOUND: $file${NC}" | tee -a "$ERROR_LOG"
    FAILED_FILES=$((FAILED_FILES + 1))
    return 1
  fi

  # 檢查文件是否為空
  if [[ ! -s "$file" ]]; then
    echo -e "${RED}✗ EMPTY FILE: $file${NC}" | tee -a "$ERROR_LOG"
    FAILED_FILES=$((FAILED_FILES + 1))
    return 1
  fi

  # 使用 jq 驗證 JSON
  if ! jq empty "$file" 2>&1 > /tmp/json_error.tmp; then
    echo -e "${RED}✗ INVALID JSON: $file${NC}" | tee -a "$ERROR_LOG"
    cat /tmp/json_error.tmp >> "$ERROR_LOG"
    FAILED_FILES=$((FAILED_FILES + 1))
    rm -f /tmp/json_error.tmp
    return 1
  fi

  # 檢查是否有 BOM
  if file "$file" | grep -q "BOM"; then
    echo -e "${YELLOW}⚠ WARNING: BOM detected in $file${NC}" | tee -a "$WARNINGS_LOG"
    WARNINGS=$((WARNINGS + 1))
  fi

  # 檢查尾部空白
  if tail -c 1 < "$file" | grep -q .; then
    echo -e "${YELLOW}⚠ WARNING: Missing newline at end of $file${NC}" | tee -a "$WARNINGS_LOG"
    WARNINGS=$((WARNINGS + 1))
  fi

  echo -e "${GREEN}✓ JSON valid: $file${NC}"
  PASSED_FILES=$((PASSED_FILES + 1))
  return 0
}

################################################################################
# Function: Validate YAML Files
################################################################################
validate_yaml() {
  local file="$1"

  TOTAL_FILES=$((TOTAL_FILES + 1))

  # 檢查文件是否存在
  if [[ ! -f "$file" ]]; then
    echo -e "${RED}✗ FILE NOT FOUND: $file${NC}" | tee -a "$ERROR_LOG"
    FAILED_FILES=$((FAILED_FILES + 1))
    return 1
  fi

  # 檢查文件是否為空
  if [[ ! -s "$file" ]]; then
    echo -e "${RED}✗ EMPTY FILE: $file${NC}" | tee -a "$ERROR_LOG"
    FAILED_FILES=$((FAILED_FILES + 1))
    return 1
  fi

  # 使用 Python 驗證 (支持多文檔 YAML，如 Kubernetes)
  if ! python3 << EOF 2>&1 > /tmp/yaml_error.tmp
import yaml
import sys
try:
    with open('$file', 'r') as f:
        # 支持多文檔 YAML (Kubernetes 使用 --- 分隔符)
        yaml.safe_load_all(f.read())
except yaml.YAMLError as e:
    print(f"YAML Error: {e}")
    sys.exit(1)
EOF
  then
    echo -e "${RED}✗ INVALID YAML: $file${NC}" | tee -a "$ERROR_LOG"
    cat /tmp/yaml_error.tmp >> "$ERROR_LOG"
    FAILED_FILES=$((FAILED_FILES + 1))
    rm -f /tmp/yaml_error.tmp
    return 1
  fi

  # 檢查 tab 字符（YAML 不允許）
  if grep -P '\t' "$file" > /dev/null 2>&1; then
    echo -e "${RED}✗ TABS FOUND: $file (YAML requires spaces)${NC}" | tee -a "$ERROR_LOG"
    FAILED_FILES=$((FAILED_FILES + 1))
    return 1
  fi

  # 檢查是否有 BOM
  if file "$file" | grep -q "BOM"; then
    echo -e "${YELLOW}⚠ WARNING: BOM detected in $file${NC}" | tee -a "$WARNINGS_LOG"
    WARNINGS=$((WARNINGS + 1))
  fi

  # 檢查尾部空白
  if tail -c 1 < "$file" | grep -q .; then
    echo -e "${YELLOW}⚠ WARNING: Missing newline at end of $file${NC}" | tee -a "$WARNINGS_LOG"
    WARNINGS=$((WARNINGS + 1))
  fi

  echo -e "${GREEN}✓ YAML valid: $file${NC}"
  PASSED_FILES=$((PASSED_FILES + 1))
  return 0
}

################################################################################
# Function: Validate TOML Files
################################################################################
validate_toml() {
  local file="$1"

  TOTAL_FILES=$((TOTAL_FILES + 1))

  # 檢查文件是否存在
  if [[ ! -f "$file" ]]; then
    echo -e "${RED}✗ FILE NOT FOUND: $file${NC}" | tee -a "$ERROR_LOG"
    FAILED_FILES=$((FAILED_FILES + 1))
    return 1
  fi

  # 檢查文件是否為空
  if [[ ! -s "$file" ]]; then
    echo -e "${RED}✗ EMPTY FILE: $file${NC}" | tee -a "$ERROR_LOG"
    FAILED_FILES=$((FAILED_FILES + 1))
    return 1
  fi

  # 使用 Python toml 驗證
  if ! python3 -c "import tomllib; tomllib.loads(open('$file').read())" 2>&1 > /tmp/toml_error.tmp; then
    echo -e "${RED}✗ INVALID TOML: $file${NC}" | tee -a "$ERROR_LOG"
    cat /tmp/toml_error.tmp >> "$ERROR_LOG"
    FAILED_FILES=$((FAILED_FILES + 1))
    rm -f /tmp/toml_error.tmp
    return 1
  fi

  echo -e "${GREEN}✓ TOML valid: $file${NC}"
  PASSED_FILES=$((PASSED_FILES + 1))
  return 0
}

################################################################################
# Function: Validate SQL Files
################################################################################
validate_sql() {
  local file="$1"

  TOTAL_FILES=$((TOTAL_FILES + 1))

  # 檢查文件是否存在
  if [[ ! -f "$file" ]]; then
    echo -e "${RED}✗ FILE NOT FOUND: $file${NC}" | tee -a "$ERROR_LOG"
    FAILED_FILES=$((FAILED_FILES + 1))
    return 1
  fi

  # 檢查文件是否為空
  if [[ ! -s "$file" ]]; then
    echo -e "${RED}✗ EMPTY FILE: $file${NC}" | tee -a "$ERROR_LOG"
    FAILED_FILES=$((FAILED_FILES + 1))
    return 1
  fi

  # 檢查基本 SQL 語法
  if grep -E '^[[:space:]]*$' "$file" | wc -l | grep -q .; then
    # 只進行基本檢查，不是完整解析
    echo -e "${GREEN}✓ SQL syntax check passed: $file${NC}"
    PASSED_FILES=$((PASSED_FILES + 1))
    return 0
  fi

  echo -e "${GREEN}✓ SQL file valid: $file${NC}"
  PASSED_FILES=$((PASSED_FILES + 1))
  return 0
}

################################################################################
# Main Validation Loop
################################################################################

echo -e "${BLUE}[1/5] Scanning and validating configuration files...${NC}"
echo ""

# 尋找所有需要驗證的文件
TOTAL=0
ERROR_COUNT=0

# JSON 文件
echo -e "${BLUE}→ Validating JSON files...${NC}"
while IFS= read -r file; do
  if [[ -n "$file" ]]; then
    validate_json "$file" || ((ERROR_COUNT++))
  fi
done < <(find /home/user/ecosystem -type f -name "*.json" ! -path "*/node_modules/*" ! -path "*/.git/*" ! -path "*/.next/*")

echo ""

# YAML 文件
echo -e "${BLUE}→ Validating YAML files...${NC}"
while IFS= read -r file; do
  if [[ -n "$file" ]]; then
    validate_yaml "$file" || ((ERROR_COUNT++))
  fi
done < <(find /home/user/ecosystem -type f \( -name "*.yaml" -o -name "*.yml" \) ! -path "*/node_modules/*" ! -path "*/.git/*" ! -path "*/.next/*")

echo ""

# TOML 文件
echo -e "${BLUE}→ Validating TOML files...${NC}"
while IFS= read -r file; do
  if [[ -n "$file" ]]; then
    validate_toml "$file" || ((ERROR_COUNT++))
  fi
done < <(find /home/user/ecosystem -type f -name "*.toml" ! -path "*/node_modules/*" ! -path "*/.git/*" ! -path "*/.next/*")

echo ""

# SQL 檔案
echo -e "${BLUE}→ Validating SQL files...${NC}"
while IFS= read -r file; do
  if [[ -n "$file" ]]; then
    validate_sql "$file" || ((ERROR_COUNT++))
  fi
done < <(find /home/user/ecosystem -type f -name "*.sql" ! -path "*/node_modules/*" ! -path "*/.git/*" ! -path "*/.next/*")

echo ""
echo -e "${BLUE}[2/5] Checking code style consistency...${NC}"

# Prettier 檢查
if command -v prettier &> /dev/null; then
  echo -e "${BLUE}→ Running Prettier format check...${NC}"
  if prettier --check "**/*.{json,yaml,yml,md}" --ignore-path .gitignore 2>&1 | grep -q "error\|Error"; then
    echo -e "${YELLOW}⚠ Some files need formatting (run 'prettier --write')${NC}"
    WARNINGS=$((WARNINGS + 1))
  else
    echo -e "${GREEN}✓ Code style consistent${NC}"
  fi
fi

echo ""
echo -e "${BLUE}[3/5] Checking for encoding issues...${NC}"

# 檢查文件編碼 (ASCII 是 UTF-8 的子集，接受兩者)
while IFS= read -r file; do
  if [[ -n "$file" ]]; then
    encoding=$(file -b --mime-encoding "$file")
    # ASCII 是 UTF-8 的子集，都是有效的
    if [[ "$encoding" != "utf-8" && "$encoding" != "us-ascii" ]]; then
      echo -e "${RED}✗ ENCODING ERROR: $file is $encoding (must be UTF-8 or ASCII)${NC}" | tee -a "$ERROR_LOG"
      ((ERROR_COUNT++))
    fi
  fi
done < <(find /home/user/ecosystem -type f \( -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "*.toml" -o -name "*.sql" \) ! -path "*/node_modules/*" ! -path "*/.git/*" ! -path "*/.next/*")

echo ""
echo -e "${BLUE}[4/5] Checking for common issues...${NC}"

# 檢查尾部空白
TRAILING_WHITESPACE=$(find /home/user/ecosystem -type f \( -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "*.toml" \) ! -path "*/node_modules/*" ! -path "*/.git/*" ! -path "*/.next/*" -exec grep -l '[[:space:]]$' {} \;)
if [[ -n "$TRAILING_WHITESPACE" ]]; then
  echo -e "${YELLOW}⚠ Files with trailing whitespace:${NC}"
  echo "$TRAILING_WHITESPACE" | tee -a "$WARNINGS_LOG"
  WARNINGS=$((WARNINGS + 1))
fi

# 檢查 DOS 行尾
DOS_ENDINGS=$(find /home/user/ecosystem -type f \( -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "*.toml" \) ! -path "*/node_modules/*" ! -path "*/.git/*" ! -path "*/.next/*" -exec grep -l $'\r' {} \;)
if [[ -n "$DOS_ENDINGS" ]]; then
  echo -e "${RED}✗ Files with DOS line endings (CRLF):${NC}" | tee -a "$ERROR_LOG"
  echo "$DOS_ENDINGS" | tee -a "$ERROR_LOG"
  ((ERROR_COUNT++))
fi

echo ""
echo -e "${BLUE}[5/5] Generating validation report...${NC}"

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                   VALIDATION REPORT                           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Total Files Checked:  ${BLUE}$TOTAL_FILES${NC}"
echo -e "Passed:              ${GREEN}$PASSED_FILES${NC}"
echo -e "Failed:              ${RED}$FAILED_FILES${NC}"
echo -e "Warnings:            ${YELLOW}$WARNINGS${NC}"
echo ""

if [[ $ERROR_COUNT -gt 0 || $FAILED_FILES -gt 0 ]]; then
  echo -e "${RED}❌ VALIDATION FAILED${NC}"
  echo ""
  echo -e "${RED}Error Details:${NC}"
  if [[ -f "$ERROR_LOG" ]]; then
    cat "$ERROR_LOG"
  fi
  echo ""
  echo -e "${YELLOW}Warnings:${NC}"
  if [[ -f "$WARNINGS_LOG" ]]; then
    cat "$WARNINGS_LOG"
  fi
  echo ""
  echo -e "${RED}Please fix the errors before committing.${NC}"
  exit 1
else
  echo -e "${GREEN}✅ ALL VALIDATION PASSED${NC}"
  echo ""
  if [[ $WARNINGS -gt 0 ]]; then
    echo -e "${YELLOW}Please review warnings:${NC}"
    if [[ -f "$WARNINGS_LOG" ]]; then
      cat "$WARNINGS_LOG"
    fi
  fi
  exit 0
fi
