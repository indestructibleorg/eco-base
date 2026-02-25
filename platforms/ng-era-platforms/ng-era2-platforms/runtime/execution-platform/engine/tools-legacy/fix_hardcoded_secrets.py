# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: fix_hardcoded_secrets
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Fix Hardcoded Secrets Script
This script identifies and helps fix hardcoded secrets and passwords
in Python source files.
"""
import re
from pathlib import Path
from typing import Dict, List
class SecretFixer:
    """Fix hardcoded secrets in Python files."""
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.excluded_dirs = {"__pycache__", ".git", "venv", ".venv", "node_modules"}
    def find_hardcoded_secrets(self, file_path: Path) -> List[Dict]:
        """Find hardcoded secrets in a file."""
        findings = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")
            # Pattern to find hardcoded secrets
            secret_patterns = [
                (r'(password|passwd|pwd)\s*=\s*["\']([^"\']{6,})["\']', "password"),
                (r'(api_key|apikey|api-key)\s*=\s*["\']([^"\']{10,})["\']', "api_key"),
                (
                    r'(secret|token|auth_key|authkey)\s*=\s*["\']([^"\']{10,})["\']',
                    "secret",
                ),
            ]
            for pattern, secret_type in secret_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    line_num = content[: match.start()].count("\n") + 1
                    line_content = lines[line_num - 1]
                    # Skip if it's an example, test, or default value
                    if any(
                        keyword in line_content.lower()
                        for keyword in [
                            "example",
                            "test",
                            "demo",
                            "placeholder",
                            "default",
                            "your_",
                        ]
                    ):
                        continue
                    findings.append(
                        {
                            "line": line_num,
                            "type": secret_type,
                            "content": line_content.strip(),
                            "file": str(file_path.relative_to(self.project_root)),
                        }
                    )
        except Exception:
            pass
        return findings
    def generate_env_template(self, findings: List[Dict]) -> str:
        """Generate .env.example template from findings.
        
        Security: Only generates placeholder template, never includes actual secret values.
        """
        env_vars = set()
        for finding in findings:
            secret_type = finding["type"]
            env_var_name = secret_type.upper()
            env_vars.add(env_var_name)
        # Security: Template only contains placeholders, no actual secrets
        template = "# Security Configuration\n"
        template += "# Copy this file to .env and fill in your actual values\n"
        template += "# WARNING: Never commit .env file with actual secrets\n\n"
        for var in sorted(env_vars):
            template += f"{var}=PLACEHOLDER_VALUE_CHANGE_THIS\n"
        return template
    def fix_file(self, file_path: Path, dry_run: bool = True) -> Dict:
        """Fix hardcoded secrets in a file."""
        findings = self.find_hardcoded_secrets(file_path)
        if not findings:
            return {"file": str(file_path), "findings": 0, "modified": False}
        if not dry_run:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            lines = content.split("\n")
            for finding in findings:
                line_num = finding["line"] - 1
                secret_type = finding["type"]
                # Replace with environment variable
                env_var = f"os.getenv('{secret_type.upper()}')"
                old_line = lines[line_num]
                # Preserve indentation
                indent = len(old_line) - len(old_line.lstrip())
                new_line = " " * indent + f"{secret_type} = {env_var}"
                # Add comment about environment variable
                new_line += "  # Set via environment variable"
                lines[line_num] = new_line
            # Add import at top if needed
            if "import os" not in content:
                # Find first import and add os before it
                import_idx = -1
                for i, line in enumerate(lines):
                    if line.strip().startswith("import ") or line.strip().startswith(
                        "from "
                    ):
                        import_idx = i
                        break
                if import_idx >= 0:
                    lines.insert(import_idx, "import os")
            # Write back
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        # Security: Don't return actual secret values to avoid logging sensitive data
        return {
            "file": str(file_path),
            "findings": len(findings),
            "modified": not dry_run,
            # "secrets" field removed to prevent sensitive data leakage
        }
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fix hardcoded secrets")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show what would be changed without modifying files",
    )
    parser.add_argument(
        "--generate-env", action="store_true", help="Generate .env.example template"
    )
    args = parser.parse_args()
    project_root = Path.cwd()
    fixer = SecretFixer(project_root)
    print("🔍 Scanning for hardcoded secrets...")
    print("=" * 60)
    all_findings = []
    python_files = list(project_root.rglob("*.py"))
    for file_path in python_files:
        if any(excluded in str(file_path) for excluded in fixer.excluded_dirs):
            continue
        findings = fixer.find_hardcoded_secrets(file_path)
        if findings:
            all_findings.extend(findings)
            result = fixer.fix_file(file_path, dry_run=args.dry_run)
            # Security: Only log non-sensitive metadata
            print(f"\n📄 {result['file']}")
            print(f"   Findings: {result['findings']} (details suppressed for security)")
            if result["modified"]:
                print("   ✅ Modified")
            else:
                print("   ℹ️  Would modify (dry-run)")
            for idx, _ in enumerate(findings, start=1):
                print(f"   - Finding {idx} (details suppressed to avoid logging sensitive data)")
    print("\n" + "=" * 60)
    print(f"Total findings: {len(all_findings)}")
    if args.generate_env:
        # Security: Template generation only creates placeholders, no actual secrets
        template = fixer.generate_env_template(all_findings)
        env_file = project_root / ".env.example"
        # Security: Writing only placeholder template to file
        with open(env_file, "w") as f:
            f.write(template)
        print(f"\n📝 Generated {env_file} (placeholder template only)")
        print(f"   Variables: {len(set(f['type'] for f in all_findings))}")
    if args.dry_run:
        print("\n💡 This was a dry-run. No files were modified.")
        print("   Run again without --dry-run to apply fixes.")
    else:
        print("\n✅ Files have been modified.")
        print("   Please review changes and update your .env file.")
if __name__ == "__main__":
    main()
