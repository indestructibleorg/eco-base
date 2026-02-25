# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: language-gl-platform.governance-analyzer
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Language Governance Analyzer
分析代碼庫中的語言使用情況，並根據語言策略進行驗證
Usage:
    python language-gl-platform.governance-analyzer.py --config config/language-policy.yaml --repo-root . --output-format json
"""
# MNGA-002: Import organization needs review
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
try:
    import yaml
except ImportError:
    print(
        "Error: PyYAML is required. Install it with: pip install pyyaml",
        file=sys.stderr,
    )
    sys.exit(1)
# Language extensions mapping
LANGUAGE_EXTENSIONS = {
    "TypeScript": [".ts", ".tsx"],
    "JavaScript": [".js", ".jsx"],
    "Python": [".py"],
    "C++": [".cpp", ".hpp", ".cc", ".cxx", ".h"],
    "C": [".c", ".h"],
    "Go": [".go"],
    "Rust": [".rs"],
    "Rego": [".rego"],
    "Swift": [".swift"],
    "Kotlin": [".kt", ".kts"],
    "PHP": [".php"],
    "Ruby": [".rb"],
    "Lua": [".lua"],
    "Perl": [".pl", ".pm"],
    "Bash": [".sh", ".bash"],
    "YAML": [".yaml", ".yml"],
    "JSON": [".json"],
    "Markdown": [".md"],
    "TOML": [".toml"],
    "HCL": [".tf", ".hcl"],
    "XML": [".xml"],
}
# Reverse mapping for faster lookup
EXTENSION_TO_LANGUAGE = {}
for lang, exts in LANGUAGE_EXTENSIONS.items():
    for ext in exts:
        EXTENSION_TO_LANGUAGE[ext] = lang
def load_policy(config_path: str) -> Dict:
    """Load language policy from YAML file"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading policy file {config_path}: {e}", file=sys.stderr)
        sys.exit(1)
def scan_directory(
    repo_root: str, exclude_dirs: List[str] = None
) -> Dict[str, Dict[str, int]]:
    """Scan repository and count files by language in each directory"""
    if exclude_dirs is None:
        exclude_dirs = [
            "node_modules",
            ".git",
            "dist",
            "build",
            "__pycache__",
            ".pytest_cache",
            "venv",
            ".venv",
            "vendor",
            "target",
        ]
    stats = defaultdict(lambda: defaultdict(int))
    repo_path = Path(repo_root)
    for file_path in repo_path.rglob("*"):
        if not file_path.is_file():
            continue
        # Skip excluded directories
        if any(excluded in file_path.parts for excluded in exclude_dirs):
            continue
        # Determine language from extension
        ext = file_path.suffix.lower()
        if ext not in EXTENSION_TO_LANGUAGE:
            continue
        language = EXTENSION_TO_LANGUAGE[ext]
        # Get relative directory - use at least 2 levels for better matching
        try:
            rel_path = file_path.relative_to(repo_path)
            # Use first 2 parts if available (e.g., automation/autonomous/)
            if len(rel_path.parts) >= 2:
                directory = "/".join(rel_path.parts[:2]) + "/"
            elif len(rel_path.parts) >= 1:
                directory = str(rel_path.parts[0]) + "/"
            else:
                directory = ""
        except ValueError:
            continue
        stats[directory][language] += 1
    return dict(stats)
def check_violations(
    stats: Dict[str, Dict[str, int]], policy: Dict
) -> List[Dict[str, Any]]:
    """Check for language policy violations"""
    violations = []
    directory_rules = policy.get("directory_rules", {})
    global_policy = policy.get("global_policy", {})
    forbidden_languages = global_policy.get("forbidden_languages", [])
    for directory, language_counts in stats.items():
        # Check directory-specific rules - find the most specific match
        dir_rule = None
        best_match = ""
        for rule_pattern, rule in directory_rules.items():
            if directory.startswith(rule_pattern):
                # Prefer longer (more specific) matches
                if len(rule_pattern) > len(best_match):
                    best_match = rule_pattern
                    dir_rule = rule
        for language, count in language_counts.items():
            # Check forbidden languages globally
            if language in forbidden_languages:
                violations.append(
                    {
                        "severity": "CRITICAL",
                        "directory": directory,
                        "language": language,
                        "count": count,
                        "message": f"使用了全局禁止的語言 {language}",
                        "rule": "global_policy.forbidden_languages",
                    }
                )
                continue
            # Check directory-specific rules
            if dir_rule:
                allowed = dir_rule.get("allowed_languages", [])
                dir_rule.get("forbidden_patterns", [])
                # Configuration and data formats are generally allowed
                # everywhere
                config_formats = ["YAML", "JSON", "Markdown", "TOML", "XML"]
                # Check if language is allowed
                if (
                    allowed
                    and language not in allowed
                    and language not in config_formats
                ):
                    violations.append(
                        {
                            "severity": "ERROR",
                            "directory": directory,
                            "language": language,
                            "count": count,
                            "message": f"在 {directory} 中使用了未授權的語言 {language}",
                            "rule": f'directory_rules.{directory.rstrip("/")}.allowed_languages',
                            "allowed": allowed,
                        })
    return violations
def generate_report(
    stats: Dict, violations: List[Dict], output_format: str, output_file: str = None
):
    """Generate report in specified format"""
    total_files = sum(sum(langs.values()) for langs in stats.values())
    report = {
        "total_files": total_files,
        "violations": violations,
        "stats": stats,
        "summary": {
            "total_violations": len(violations),
            "critical": sum(1 for v in violations if v["severity"] == "CRITICAL"),
            "error": sum(1 for v in violations if v["severity"] == "ERROR"),
            "warning": sum(1 for v in violations if v["severity"] == "WARNING"),
        },
    }
    if output_format == "json":
        output = json.dumps(report, indent=2, ensure_ascii=False)
    elif output_format == "markdown":
        output = generate_markdown_report(report)
    else:
        output = generate_text_report(report)
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output)
    else:
        print(output)
def generate_markdown_report(report: Dict) -> str:
    """Generate markdown format report"""
    lines = []
    lines.append("# Language Governance Report\n")
    lines.append(f"**Total Files Scanned:** {report['total_files']}\n")
    lines.append(f"**Total Violations:** {report['summary']['total_violations']}\n")
    if report["violations"]:
        lines.append("\n## Violations\n")
        for v in report["violations"]:
            icon = (
                "🔴"
                if v["severity"] == "CRITICAL"
                else "⚠️" if v["severity"] == "ERROR" else "💡"
            )
            lines.append(
                f"- {icon} **{v['severity']}** in `{v['directory']}`: {v['message']}"
            )
            if "count" in v:
                lines.append(f" ({v['count']} files)\n")
            else:
                lines.append("\n")
    else:
        lines.append("\n✅ No violations found!\n")
    lines.append("\n## Language Distribution\n")
    for directory, languages in sorted(report["stats"].items()):
        if languages:
            lines.append(f"\n### {directory}\n")
            for lang, count in sorted(
                languages.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"- {lang}: {count} files\n")
    return "".join(lines)
def generate_text_report(report: Dict) -> str:
    """Generate plain text report"""
    lines = []
    lines.append("=" * 60)
    lines.append("\nLanguage Governance Report")
    lines.append("\n" + "=" * 60)
    lines.append(f"\nTotal Files: {report['total_files']}")
    lines.append(f"Total Violations: {report['summary']['total_violations']}")
    if report["violations"]:
        lines.append("\n\nViolations:")
        lines.append("\n" + "-" * 60)
        for v in report["violations"]:
            lines.append(f"\n[{v['severity']}] {v['directory']}")
            lines.append(f"  {v['message']}")
            if "count" in v:
                lines.append(f"  Files: {v['count']}")
    return "\n".join(lines)
def main():
    parser = argparse.ArgumentParser(description="Language Governance Analyzer")
    parser.add_argument(
        "--config", required=True, help="Path to language policy YAML file"
    )
    parser.add_argument("--repo-root", required=True, help="Repository root directory")
    parser.add_argument(
        "--output-format",
        choices=["json", "markdown", "text"],
        default="text",
        help="Output format",
    )
    parser.add_argument("--output-file", help="Output file path")
    args = parser.parse_args()
    # Load policy
    policy = load_policy(args.config)
    # Scan repository
    print(f"Scanning repository: {args.repo_root}", file=sys.stderr)
    stats = scan_directory(args.repo_root)
    # Check violations
    print("Checking for violations...", file=sys.stderr)
    violations = check_violations(stats, policy)
    # Generate report
    generate_report(stats, violations, args.output_format, args.output_file)
    # Exit with error if violations found
    if violations:
        sys.exit(1)
    return 0
if __name__ == "__main__":
    sys.exit(main())
