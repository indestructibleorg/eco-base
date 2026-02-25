#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: fix_yaml_markdown
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
修復包含 Markdown 的 YAML 文件
移除 Markdown 代碼塊標記，提取純 YAML 內容
"""
import re
import sys
def extract_yaml_from_markdown(filepath):
    """
    從包含 Markdown 的文件中提取 YAML 內容
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    # 移除 Markdown 代碼塊標記
    # 模式：```yaml ... ```
    yaml_content = re.sub(r'```yaml\n', '', content)
    yaml_content = re.sub(r'\n```\n', '\n', yaml_content)
    yaml_content = re.sub(r'\n```$', '', yaml_content)
    # 移除文件開頭的標題和元數據（在第一個 ```yaml 之前的內容）
    parts = yaml_content.split('apiVersion:')
    if len(parts) > 1:
        yaml_content = 'apiVersion:' + parts[1]
    # 移除結尾的 markdown 元數據（在最後一個 --- 之後的內容）
    yaml_parts = yaml_content.rsplit('---', 1)
    if len(yaml_parts) > 1:
        # 檢查最後一部分是否包含有效的 YAML
        last_part = yaml_parts[-1].strip()
        if not last_part.startswith('  ') and not last_part.startswith('-'):
            yaml_content = yaml_parts[0]
    # 清理空行
    yaml_content = re.sub(r'\n{3,}', '\n\n', yaml_content)
    # 寫回文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"✅ Extracted pure YAML from: {filepath}")
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 fix_yaml_markdown.py <yaml_file>")
        sys.exit(1)
    filepath = sys.argv[1]
    extract_yaml_from_markdown(filepath)