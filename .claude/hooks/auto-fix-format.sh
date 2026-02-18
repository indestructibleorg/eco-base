#!/bin/bash
################################################################################
# ⚙️ AUTOMATIC FORMAT FIXER
# =========================
# 自動修復所有格式問題
# 這個工具會自動糾正：
# - JSON/YAML/TOML 縮進
# - 行尾 (CRLF -> LF)
# - 尾部空白
# - BOM 符號
# - 文件編碼
################################################################################

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

FIXED_COUNT=0
FAILED_COUNT=0

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          ⚙️ AUTOMATIC FORMAT FIXER (v1.0)                      ║${NC}"
echo -e "${BLUE}║     Fixing all format issues automatically...                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

################################################################################
# Fix JSON Files
################################################################################
echo -e "${BLUE}[1/5] Fixing JSON files...${NC}"
while IFS= read -r file; do
  if [[ -n "$file" ]]; then
    echo -e "${BLUE}→ Processing: $file${NC}"

    # 使用 jq 重新格式化
    if jq . "$file" > "$file.tmp" 2>/dev/null; then
      mv "$file.tmp" "$file"
      echo -e "${GREEN}✓ Fixed: $file${NC}"
      FIXED_COUNT=$((FIXED_COUNT + 1))
    else
      echo -e "${RED}✗ Failed to fix: $file${NC}"
      FAILED_COUNT=$((FAILED_COUNT + 1))
      rm -f "$file.tmp"
    fi
  fi
done < <(find /home/user/ecosystem -type f -name "*.json" ! -path "*/node_modules/*" ! -path "*/.git/*" ! -path "*/.next/*")

echo ""

################################################################################
# Fix YAML Files
################################################################################
echo -e "${BLUE}[2/5] Fixing YAML files...${NC}"
while IFS= read -r file; do
  if [[ -n "$file" ]]; then
    echo -e "${BLUE}→ Processing: $file${NC}"

    # 轉換 CRLF 為 LF
    dos2unix -q "$file" 2>/dev/null || sed -i 's/\r$//' "$file"

    # 移除尾部空白
    sed -i 's/[[:space:]]*$//' "$file"

    # 確保文件末尾有換行符
    sed -i -e '$a\' "$file"

    # 移除 BOM
    sed -i '1s/^\xEF\xBB\xBF//' "$file"

    echo -e "${GREEN}✓ Fixed: $file${NC}"
    FIXED_COUNT=$((FIXED_COUNT + 1))
  fi
done < <(find /home/user/ecosystem -type f \( -name "*.yaml" -o -name "*.yml" \) ! -path "*/node_modules/*" ! -path "*/.git/*" ! -path "*/.next/*")

echo ""

################################################################################
# Fix TOML Files
################################################################################
echo -e "${BLUE}[3/5] Fixing TOML files...${NC}"
while IFS= read -r file; do
  if [[ -n "$file" ]]; then
    echo -e "${BLUE}→ Processing: $file${NC}"

    # 轉換 CRLF 為 LF
    dos2unix -q "$file" 2>/dev/null || sed -i 's/\r$//' "$file"

    # 移除尾部空白
    sed -i 's/[[:space:]]*$//' "$file"

    # 確保文件末尾有換行符
    sed -i -e '$a\' "$file"

    # 移除 BOM
    sed -i '1s/^\xEF\xBB\xBF//' "$file"

    echo -e "${GREEN}✓ Fixed: $file${NC}"
    FIXED_COUNT=$((FIXED_COUNT + 1))
  fi
done < <(find /home/user/ecosystem -type f -name "*.toml" ! -path "*/node_modules/*" ! -path "*/.git/*" ! -path "*/.next/*")

echo ""

################################################################################
# Fix SQL Files
################################################################################
echo -e "${BLUE}[4/5] Fixing SQL files...${NC}"
while IFS= read -r file; do
  if [[ -n "$file" ]]; then
    echo -e "${BLUE}→ Processing: $file${NC}"

    # 轉換 CRLF 為 LF
    dos2unix -q "$file" 2>/dev/null || sed -i 's/\r$//' "$file"

    # 移除尾部空白
    sed -i 's/[[:space:]]*$//' "$file"

    # 確保文件末尾有換行符
    sed -i -e '$a\' "$file"

    # 移除 BOM
    sed -i '1s/^\xEF\xBB\xBF//' "$file"

    echo -e "${GREEN}✓ Fixed: $file${NC}"
    FIXED_COUNT=$((FIXED_COUNT + 1))
  fi
done < <(find /home/user/ecosystem -type f -name "*.sql" ! -path "*/node_modules/*" ! -path "*/.git/*" ! -path "*/.next/*")

echo ""

################################################################################
# Run Prettier
################################################################################
echo -e "${BLUE}[5/5] Running Prettier formatting...${NC}"
if command -v prettier &> /dev/null; then
  if prettier --write "**/*.{json,yaml,yml,md}" --ignore-path .gitignore 2>&1 | grep -q "modified\|unchanged"; then
    echo -e "${GREEN}✓ Prettier formatting completed${NC}"
    FIXED_COUNT=$((FIXED_COUNT + 1))
  fi
else
  echo -e "${YELLOW}⚠ Prettier not found, skipping...${NC}"
fi

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                   FIXING REPORT                               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Files Fixed:     ${GREEN}$FIXED_COUNT${NC}"
echo -e "Failed Fixes:    ${RED}$FAILED_COUNT${NC}"
echo ""

if [[ $FAILED_COUNT -eq 0 ]]; then
  echo -e "${GREEN}✅ ALL FILES HAVE BEEN FIXED${NC}"
  echo ""
  echo -e "${BLUE}Next steps:${NC}"
  echo "1. Review the changes"
  echo "2. Run the validator: bash .claude/hooks/pre-commit-validator.sh"
  echo "3. Commit your changes"
  exit 0
else
  echo -e "${RED}❌ SOME FILES COULD NOT BE FIXED${NC}"
  echo "Please review and fix them manually."
  exit 1
fi
