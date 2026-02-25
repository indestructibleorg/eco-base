#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: clean_yaml_files
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
批量清理 YAML 文件
移除所有 Markdown 標記和多餘的文檔分隔符
"""
import re
import glob
def clean_yaml_file(filepath):
    """清理單個 YAML 文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    # 移除所有 Markdown 標記
    content = re.sub(r'^#.*\n', '', content, flags=re.MULTILINE)
    content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
    content = re.sub(r'```yaml\n', '', content)
    content = re.sub(r'\n```', '\n', content)
    # 分割文檔
    parts = re.split(r'\n---\n', content)
    if len(parts) > 1:
        # 保留第一個有效的 YAML 文檔
        yaml_content = parts[0]
        # 寫回文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        return True
    return False
def main():
    """主函數"""
    yaml_files = glob.glob('gl/**/*.yaml', recursive=True)
    fixed_count = 0
    for yaml_file in yaml_files:
        if clean_yaml_file(yaml_file):
            fixed_count += 1
            print(f"✅ Cleaned: {yaml_file}")
    print(f"\nTotal files cleaned: {fixed_count}")
if __name__ == "__main__":
    main()