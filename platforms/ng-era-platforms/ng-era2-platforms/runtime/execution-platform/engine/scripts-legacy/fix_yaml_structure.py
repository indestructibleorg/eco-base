#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: fix_yaml_structure
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
修復 YAML 結構問題
處理嵌套列表和映射的縮排問題
"""
import re
import sys
def fix_yaml_structure(filepath):
    """
    修復 YAML 文件中的結構問題
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # 跳過 Markdown 內容
        if line.strip().startswith('```') or line.strip().startswith('#') or line.strip().startswith('**'):
            i += 1
            continue
        # 處理列表項目
        match = re.match(r'^(\s+)-\s*(.+)', line)
        if match:
            indent = match.group(1)
            fixed_lines.append(line)
            i += 1
            # 檢查後續行是否需要縮排
            while i < len(lines):
                next_line = lines[i].rstrip()
                # 空行
                if not next_line.strip():
                    fixed_lines.append('\n')
                    i += 1
                    continue
                # 下一個列表項目
                next_match = re.match(r'^(\s+)-\s', next_line)
                if next_match:
                    break
                # 子屬性或子列表
                if next_line.startswith(indent + '  ') or next_line.startswith(indent + '\t'):
                    fixed_lines.append(next_line + '\n')
                else:
                    # 需要修復縮排
                    leading_spaces = len(next_line) - len(next_line.lstrip())
                    if leading_spaces < len(indent) + 2:
                        # 添加正確縮排
                        fixed_lines.append(' ' * (len(indent) + 2) + next_line.lstrip() + '\n')
                    else:
                        fixed_lines.append(next_line + '\n')
                i += 1
            continue
        # 非列表行，直接添加
        if not line.strip().startswith('```') and not line.strip().startswith('#'):
            fixed_lines.append(line)
        i += 1
    # 移除連續空行
    result = []
    prev_empty = False
    for line in fixed_lines:
        if not line.strip():
            if not prev_empty:
                result.append(line)
            prev_empty = True
        else:
            result.append(line)
            prev_empty = False
    # 寫回文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(result)
    print(f"✅ Fixed YAML structure: {filepath}")
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 fix_yaml_structure.py <yaml_file>")
        sys.exit(1)
    filepath = sys.argv[1]
    fix_yaml_structure(filepath)