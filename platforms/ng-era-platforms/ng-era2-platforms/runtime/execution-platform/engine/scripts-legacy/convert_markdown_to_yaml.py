#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: convert_markdown_to_yaml
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
將包含 Markdown 標記的 YAML 文件轉換為純 YAML
"""
import re
import sys
def convert_markdown_to_yaml(filepath):
    """
    轉換 Markdown 格式的 YAML 文件為純 YAML
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    # 移除所有 Markdown 標記
    # 1. 移除標題行 (# 開頭的行)
    content = re.sub(r'^#.*\n', '', content, flags=re.MULTILINE)
    # 2. 移除加粗標記 (**...**)
    content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
    # 3. 移除代碼塊標記 (```yaml ... ```)
    content = re.sub(r'```yaml\n', '', content)
    content = re.sub(r'\n```', '\n', content)
    # 4. 移除元數據行（包含 key: value 格式但不在 YAML 結構中的）
    # 通常這些行在文件開頭或結尾
    lines = content.split('\n')
    yaml_lines = []
    in_yaml_block = False
    for line in lines:
        # 檢查是否是 YAML 開始標記
        if 'apiVersion:' in line or 'kind:' in line or line.strip().startswith('apiVersion'):
            in_yaml_block = True
        # 如果在 YAML 塊中，保留該行
        if in_yaml_block:
            # 檢查是否是結束標記（--- 後面是非 YAML 內容）
            if line.strip() == '---' and in_yaml_block and yaml_lines:
                # 檢查下一行是否是元數據
                next_idx = lines.index(line) + 1
                if next_idx < len(lines):
                    next_line = lines[next_idx]
                    if re.match(r'^\*\*[A-Za-z]+:', next_line) or next_line.strip().startswith('**'):
                        break
            yaml_lines.append(line + '\n')
    # 清理內容
    yaml_content = ''.join(yaml_lines)
    # 移除連續空行
    yaml_content = re.sub(r'\n{3,}', '\n\n', yaml_content)
    # 移除行尾空格
    yaml_content = '\n'.join(line.rstrip() for line in yaml_content.split('\n'))
    # 寫回文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"✅ Converted Markdown to YAML: {filepath}")
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 convert_markdown_to_yaml.py <yaml_file>")
        sys.exit(1)
    filepath = sys.argv[1]
    convert_markdown_to_yaml(filepath)