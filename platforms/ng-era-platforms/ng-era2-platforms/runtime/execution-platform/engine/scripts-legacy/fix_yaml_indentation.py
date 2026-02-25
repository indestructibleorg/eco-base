#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: fix_yaml_indentation
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
修復 YAML 縮排問題的腳本
處理列表項目後的空行縮排問題
"""
import re
import sys
def fix_yaml_indentation(filepath):
    """
    修復 YAML 文件中的縮排問題
    主要修復：
    1. 列表項目後的空行應該有與列表項目相同的縮排
    2. 移除過多空行
    3. 統一縮排為 2 個空格
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    fixed_lines = []
    i = 0
    while i < len(lines):
        current_line = lines[i]
        # 檢查是否是列表項目（以 '- ' 開頭，有適當縮排）
        if re.match(r'^(\s+)-\s', current_line):
            # 獲取列表項目的縮排
            match = re.match(r'^(\s+)-\s', current_line)
            item_indent = match.group(1)
            # 添加當前行
            fixed_lines.append(current_line)
            i += 1
            # 檢查後續空行
            while i < len(lines) and lines[i].strip() == '':
                # 將空行替換為有正確縮排的空行
                fixed_lines.append('\n')
                i += 1
            # 處理非空行
            while i < len(lines) and lines[i].strip() != '':
                next_line = lines[i]
                # 如果下一行是子屬性，確保縮排正確
                if re.match(r'^(\s+)(\w+:)', next_line):
                    # 子屬性應該有比列表項目多 2 個空格的縮排
                    expected_indent = item_indent + '  '
                    if next_line.startswith(expected_indent):
                        fixed_lines.append(next_line)
                    else:
                        # 修復縮排
                        content = next_line.lstrip()
                        fixed_lines.append(expected_indent + content)
                elif re.match(r'^(\s+)-\s', next_line):
                    # 這是下一個列表項目，正常處理
                    fixed_lines.append(next_line)
                else:
                    # 其他情況，保持原樣
                    fixed_lines.append(next_line)
                i += 1
            continue
        # 非列表項目，直接添加
        fixed_lines.append(current_line)
        i += 1
    # 寫回文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    print(f"✅ Fixed YAML indentation: {filepath}")
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 fix_yaml_indentation.py <yaml_file>")
        sys.exit(1)
    filepath = sys.argv[1]
    fix_yaml_indentation(filepath)