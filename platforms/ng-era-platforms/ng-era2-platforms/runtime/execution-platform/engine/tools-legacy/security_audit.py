# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: security_audit
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Security Audit Tool
This script performs a comprehensive security audit of the codebase,
analyzing MD5 usage, ast.literal_eval() usage, and other security concerns.
"""
# MNGA-002: Import organization needs review
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import ast  # Added for ast.literal_eval()
@dataclass
class SecurityFinding:
    """Represents a security finding."""
    file_path: str
    line_number: int
    severity: str  # 'critical', 'high', 'medium', 'low', 'info'
    category: str
    issue: str
    code_snippet: str
    recommendation: str
    context: str = ""
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "file": self.file_path,
            "line": self.line_number,
            "severity": self.severity,
            "category": self.category,
            "issue": self.issue,
            "code": self.code_snippet,
            "recommendation": self.recommendation,
            "context": self.context,
        }
class SecurityAuditor:
    """Audits code for security issues."""
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.findings: List[SecurityFinding] = []
        self.excluded_dirs = {
            "__pycache__",
            ".git",
            "venv",
            ".venv",
            "node_modules",
            ".tox",
            "dist",
            "build",
            ".eggs",
            "env",
        }
        self.excluded_files = {
            "setup.py",
            "__init__.py",
            "test_",
            "_test.py",
            ".pyc",
            ".md",
            ".txt",
            ".json",
            ".yaml",
            ".yml",
        }
    def should_analyze(self, file_path: Path) -> bool:
        """Check if file should be analyzed."""
        # Skip excluded directories
        if any(excluded in str(file_path) for excluded in self.excluded_dirs):
            return False
        # Skip excluded files
        if any(
            file_path.name.startswith(excluded) or file_path.name.endswith(excluded)
            for excluded in self.excluded_files
        ):
            return False
        return True
    def check_md5_usage(self, file_path: Path) -> List[SecurityFinding]:
        """Check for MD5 hash usage."""
        findings = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                lines = content.split("\n")
            # Check for MD5 imports
            md5_patterns = [
                r"hashlib\.md5\s*\(",
                r"import\s+.*md5",
                r"from\s+.*md5",
            ]
            for pattern in md5_patterns:
                for match in re.finditer(pattern, content):
                    line_num = content[: match.start()].count("\n") + 1
                    code_snippet = lines[line_num - 1].strip()
                    # Analyze context to determine severity
                    context = self._get_context(lines, line_num)
                    severity = "medium"
                    # Upgrade to high if used for passwords or secrets
                    if any(
                        keyword in context.lower()
                        for keyword in [
                            "password",
                            "secret",
                            "auth",
                            "token",
                            "credential",
                        ]
                    ):
                        severity = "high"
                    findings.append(
                        SecurityFinding(
                            file_path=str(
                                file_path.relative_to(
                                    self.project_root)),
                            line_number=line_num,
                            severity=severity,
                            category="Cryptographic",
                            issue="MD5 hash usage detected",
                            code_snippet=code_snippet,
                            recommendation="Replace with SHA256 for security-sensitive operations. "
                            "MD5 is considered cryptographically broken.",
                            context=context,
                        ))
        except Exception:
            pass
        return findings
    def check_eval_usage(self, file_path: Path) -> List[SecurityFinding]:
        """Check for eval() usage."""
        findings = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                lines = content.split("\n")
            # Check for eval() calls
            eval_pattern = r"\beval\s*\("
            for match in re.finditer(eval_pattern, content):
                line_num = content[: match.start()].count("\n") + 1
                code_snippet = lines[line_num - 1].strip()
                # Analyze context
                context = self._get_context(lines, line_num)
                severity = "high"
                # Check if it's in a comment or docstring
                if "#" in code_snippet and code_snippet.index("#") < code_snippet.find(
                    "eval"
                ):
                    continue
                # Check for potential safe usage patterns
                safe_patterns = [
                    r'eval\(\s*["\'].*["\']\s*\)',  # Literal string
                    r"# TODO.*eval",  # Documented intention
                    r"#.*unsafe",  # Acknowledged unsafe
                ]
                if any(re.search(pattern, context) for pattern in safe_patterns):
                    severity = "medium"
                findings.append(
                    SecurityFinding(
                        file_path=str(file_path.relative_to(self.project_root)),
                        line_number=line_num,
                        severity=severity,
                        category="Code Injection",
                        issue="ast.literal_eval() function usage detected",
                        code_snippet=code_snippet,
                        recommendation="Avoid ast.literal_eval() as it can execute arbitrary code. "
                        "Consider using ast.literal_ast.literal_eval() for parsing literals, "
                        "or implement a proper parser for your use case.",
                        context=context,
                    )
                )
        except Exception:
            pass
        return findings
    def check_hardcoded_secrets(self, file_path: Path) -> List[SecurityFinding]:
        """Check for hardcoded secrets and credentials."""
        findings = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                lines = content.split("\n")
            # Patterns for hardcoded secrets
            secret_patterns = [
                (
                    r'(password|passwd|pwd)\s*=\s*["\'][^"\']{8,}["\']',
                    "Hardcoded password",
                ),
                (
                    r'(api_key|apikey|api-key)\s*=\s*["\'][^"\']{8,}["\']',
                    "Hardcoded API key",
                ),
                (
                    r'(secret|token|auth_key|authkey)\s*=\s*["\'][^"\']{8,}["\']',
                    "Hardcoded secret/token",
                ),
                (
                    r'(private_key|privatekey|privkey)\s*=\s*["\'][^"\']{20,}["\']',
                    "Hardcoded private key",
                ),
                (r"ghp_[a-zA-Z0-9]{36}", "GitHub personal access token"),
                (r"sk-[a-zA-Z0-9]{48}", "Stripe API key"),
                (r"AIza[A-Za-z0-9\\-]{35}", "Google API key"),
            ]
            for pattern, issue_desc in secret_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    line_num = content[: match.start()].count("\n") + 1
                    code_snippet = lines[line_num - 1].strip()
                    # Check if it's an example or default value
                    context = self._get_context(lines, line_num)
                    if any(
                        keyword in context.lower()
                        for keyword in [
                            "example",
                            "test",
                            "demo",
                            "placeholder",
                            "default",
                        ]
                    ):
                        continue
                    findings.append(
                        SecurityFinding(
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=line_num,
                            severity="critical",
                            category="Secrets Management",
                            issue=issue_desc,
                            code_snippet=(
                                code_snippet[:100] + "..."
                                if len(code_snippet) > 100
                                else code_snippet
                            ),
                            recommendation="Move to environment variables or secret management system. "
                            "Never commit secrets to version control.",
                            context=context,
                        )
                    )
        except Exception:
            pass
        return findings
    def check_sql_injection(self, file_path: Path) -> List[SecurityFinding]:
        """Check for potential SQL injection vulnerabilities."""
        findings = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                lines = content.split("\n")
            # Check for unsafe SQL patterns
            unsafe_patterns = [
                (
                    r"\bexecute\s*\(\s*[&quot;\'][^&quot;\']*\%s[&quot;\']",
                    "Potential SQL injection with %s formatting",
                ),
                (
                    r"\bexecute\s*\(\s*[&quot;\'][^&quot;\']*\{[^\}]+\}[&quot;\']",
                    "Potential SQL injection with .format()",
                ),
                (
                    r"\bexecute\s*\(\s*f[&quot;\'][^&quot;\']*\{[^\}]+\}[&quot;\']",
                    "Potential SQL injection with f-string",
                ),
            ]
            for pattern, issue_desc in unsafe_patterns:
                for match in re.finditer(pattern, content):
                    line_num = content[: match.start()].count("\n") + 1
                    code_snippet = lines[line_num - 1].strip()
                    # Check if parameterized queries are used nearby
                    context = self._get_context(lines, line_num)
                    if "cursor.execute" in context or "conn.execute" in context:
                        findings.append(
                            SecurityFinding(
                                file_path=str(
                                    file_path.relative_to(
                                        self.project_root)),
                                line_number=line_num,
                                severity="critical",
                                category="SQL Injection",
                                issue=issue_desc,
                                code_snippet=code_snippet,
                                recommendation='Use parameterized queries: cursor.execute("SELECT * FROM table WHERE col = ?", (value,))',
                                context=context,
                            ))
        except Exception:
            pass
        return findings
    def check_file_operations(self, file_path: Path) -> List[SecurityFinding]:
        """Check for unsafe file operations."""
        findings = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                lines = content.split("\n")
            # Check for pickle usage (can execute arbitrary code)
            if "pickle" in content:
                for match in re.finditer(r"pickle\.(load|loads)\s*\(", content):
                    line_num = content[: match.start()].count("\n") + 1
                    code_snippet = lines[line_num - 1].strip()
                    findings.append(
                        SecurityFinding(
                            file_path=str(
                                file_path.relative_to(
                                    self.project_root)),
                            line_number=line_num,
                            severity="high",
                            category="Deserialization",
                            issue="Unsafe pickle usage",
                            code_snippet=code_snippet,
                            recommendation="Pickle can execute arbitrary code. Use JSON or safer serialization formats.",
                            context=self._get_context(
                                lines,
                                line_num),
                        ))
        except Exception:
            pass
        return findings
    def _get_context(
        self, lines: List[str], line_num: int, context_lines: int = 3
    ) -> str:
        """Get context around a line."""
        start = max(0, line_num - context_lines - 1)
        end = min(len(lines), line_num + context_lines)
        return "\n".join(lines[start:end])
    def analyze_file(self, file_path: Path) -> List[SecurityFinding]:
        """Analyze a single file for security issues."""
        if not self.should_analyze(file_path):
            return []
        findings = []
        # Run all security checks
        findings.extend(self.check_md5_usage(file_path))
        findings.extend(self.check_eval_usage(file_path))
        findings.extend(self.check_hardcoded_secrets(file_path))
        findings.extend(self.check_sql_injection(file_path))
        findings.extend(self.check_file_operations(file_path))
        return findings
    def audit_project(self) -> Dict:
        """Run full security audit."""
        python_files = list(self.project_root.rglob("*.py"))
        print(f"🔍 Found {len(python_files)} Python files to analyze")
        print("=" * 60)
        for i, file_path in enumerate(python_files, 1):
            if not self.should_analyze(file_path):
                continue
            findings = self.analyze_file(file_path)
            if findings:
                self.findings.extend(findings)
                print(
                    f"\n[{len(self.findings)}] {file_path.relative_to(self.project_root)}"
                )
                for finding in findings:
                    print(
                        f"  {finding.severity.upper()}: {finding.category} - {finding.issue}"
                    )
        return self._generate_report()
    def _generate_report(self) -> Dict:
        """Generate comprehensive audit report."""
        # Categorize findings
        by_severity = {"critical": [], "high": [], "medium": [], "low": [], "info": []}
        by_category = {}
        for finding in self.findings:
            # By severity
            by_severity[finding.severity].append(finding.to_dict())
            # By category
            if finding.category not in by_category:
                by_category[finding.category] = []
            by_category[finding.category].append(finding.to_dict())
        return {
            "audit_timestamp": datetime.now().isoformat(),
            "total_findings": len(self.findings),
            "summary": {
                "critical": len(by_severity["critical"]),
                "high": len(by_severity["high"]),
                "medium": len(by_severity["medium"]),
                "low": len(by_severity["low"]),
                "info": len(by_severity["info"]),
            },
            "by_severity": by_severity,
            "by_category": by_category,
            "findings": [f.to_dict() for f in self.findings],
        }
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Security audit tool")
    parser.add_argument("--output", "-o", help="Output file for report (JSON)")
    parser.add_argument("--limit", type=int, help="Limit analysis to N files (testing)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    # Initialize auditor
    project_root = Path.cwd()
    auditor = SecurityAuditor(project_root)
    # Run audit
    print("🔐 Starting Security Audit")
    print("=" * 60)
    report = auditor.audit_project()
    # Print summary
    print("\n" + "=" * 60)
    print("SECURITY AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total findings: {report['total_findings']}")
    print("\nBy Severity:")
    for severity, count in report["summary"].items():
        if count > 0:
            icon = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🟢",
                "info": "🔵",
            }
            print(f"  {icon[severity]} {severity.upper()}: {count}")
    print("\nBy Category:")
    for category, findings in report["by_category"].items():
        print(f"  {category}: {len(findings)}")
    # Save report if requested
    if args.output:
        with open(args.output, "w", encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Report saved to: {args.output}")
    # Verbose output
    if args.verbose:
        print("\n" + "=" * 60)
        print("DETAILED FINDINGS")
        print("=" * 60)
        for finding in report["findings"]:
            print(f"\n🔍 {finding['severity'].upper()}: {finding['category']}")
            print(f"   File: {finding['file']}:{finding['line']}")
            print(f"   Issue: {finding['issue']}")
            print(f"   Code: {finding['code']}")
            print(f"   Recommendation: {finding['recommendation']}")
if __name__ == "__main__":
    main()
