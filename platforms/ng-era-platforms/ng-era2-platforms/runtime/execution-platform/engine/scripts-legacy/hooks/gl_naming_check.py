#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: gl-naming-check
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
GL Naming Convention Check - 確保 GL 檔案使用 kebab-case
"""
import sys
import re
import os
def check_naming(filepath):
    """Check if GL file follows naming convention"""
    filename = os.path.basename(filepath)
    # Check for old naming patterns
    if re.match(r'^GL_', filename):
        print(f"❌ {filepath}: Use kebab-case (gl-*) instead of GL_*")
        return False
    if re.match(r'^ECO-[A-Z]', filename):
        print(f"❌ {filepath}: Use lowercase (gl-*) instead of ECO-*")
        return False
    # Check for underscores in gl files
    if filepath.startswith('gl/') and '_' in filename and not filename.startswith('__'):
        print(f"⚠️  {filepath}: Consider using kebab-case instead of underscores")
    return True
def main():
    files = sys.argv[1:]
    all_passed = True
    for filepath in files:
        if not check_naming(filepath):
            all_passed = False
    if all_passed:
        print("✅ GL naming convention check passed")
    sys.exit(0 if all_passed else 1)
if __name__ == "__main__":
    main()