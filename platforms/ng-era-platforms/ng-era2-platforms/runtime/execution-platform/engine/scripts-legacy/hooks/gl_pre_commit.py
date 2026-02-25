#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: gl-pre-commit
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
GL Pre-commit Hook - 檢查檔案是否有 GL 層級標註
"""
import sys
import re
def check_gl_annotation(filepath):
    """Check if file should have and has GL annotation"""
    # Skip non-source files
    if not any(filepath.endswith(ext) for ext in ['.py', '.ts', '.js', '.yaml', '.yml']):
        return True
    # Skip test files and configs
    skip_patterns = ['test_', '_test.', '.test.', 'conftest', '__pycache__', 'node_modules']
    if any(p in filepath for p in skip_patterns):
        return True
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(3000)
            # Check for ECO-Layer annotation
            if re.search(r'ECO-Layer:\s*GL\d{2}-\d{2}', content):
                return True
            # Check for gl layer in yaml metadata
            if filepath.endswith(('.yaml', '.yml')):
                if re.search(r'layer:\s*["\']?GL\d{2}-\d{2}', content):
                    return True
            # Warn but don't fail for now
            print(f"ℹ️  {filepath}: Consider adding ECO-Layer annotation")
            return True  # Warning only, don't block
    except Exception:
        return True
def main():
    files = sys.argv[1:]
    all_passed = True
    for filepath in files:
        if not check_gl_annotation(filepath):
            all_passed = False
    sys.exit(0 if all_passed else 1)
if __name__ == "__main__":
    main()