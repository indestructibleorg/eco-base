# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: governance
# @ECO-semantic: generate-consolidated-report
# @ECO-audit-trail: ../../engine/governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Generate Consolidated Security Report
整合語言治理、CodeQL 和 Semgrep 報告
Usage:
    python generate-consolidated-report.py --results-dir analysis-results/ --output consolidated-report.md
"""
# MNGA-002: Import organization needs review
import argparse
import json
import os
import sys
from typing import Dict, List
def load_json_safe(file_path: str) -> Dict:
    """Safely load JSON file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {file_path}: {e}", file=sys.stderr)
        return {}
def find_files_recursive(directory: str, pattern: str) -> List[str]:
    """Find files matching pattern recursively"""
    results = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(pattern):
                results.append(os.path.join(root, file))
    return results
def parse_governance_report(results_dir: str) -> Dict:
    """Parse language governance report"""
    governance_file = os.path.join(
        results_dir, "language-governance-report", "governance-report.json"
    )
    if os.path.exists(governance_file):
        return load_json_safe(governance_file)
    return {}
def parse_sarif_results(results_dir: str, name: str) -> Dict:
    """Parse SARIF format security scan results"""
    sarif_files = find_files_recursive(results_dir, ".sarif")
    results = {
        "total_alerts": 0,
        "by_severity": {"error": 0, "warning": 0, "note": 0},
        "alerts": [],
    }
    for sarif_file in sarif_files:
        if name.lower() not in sarif_file.lower():
            continue
        data = load_json_safe(sarif_file)
        if not data:
            continue
        for run in data.get("runs", []):
            for result in run.get("results", []):
                level = result.get("level", "warning")
                results["by_severity"][level] = results["by_severity"].get(level, 0) + 1
                results["total_alerts"] += 1
                results["alerts"].append(
                    {
                        "message": result.get("message", {}).get("text", "Unknown"),
                        "level": level,
                        "rule_id": result.get("ruleId", "unknown"),
                    }
                )
    return results
def generate_consolidated_report(results_dir: str, output_file: str):
    """Generate consolidated markdown report"""
    lines = []
    lines.append("# 🔒 Consolidated Security & Governance Report\n\n")
    lines.append(
        "This report consolidates findings from multiple security and governance tools.\n\n"
    )
    lines.append("---\n\n")
    # Language Governance
    lines.append("## 1️⃣ Language Governance\n\n")
    governance = parse_governance_report(results_dir)
    if governance:
        summary = governance.get("summary", {})
        lines.append(f"- **Total Files Scanned:** {governance.get('total_files', 0)}\n")
        lines.append(f"- **Total Violations:** {summary.get('total_violations', 0)}\n")
        if summary.get("critical", 0) > 0:
            lines.append(f"- 🔴 **Critical:** {summary['critical']}\n")
        if summary.get("error", 0) > 0:
            lines.append(f"- ⚠️ **Errors:** {summary['error']}\n")
        if summary.get("warning", 0) > 0:
            lines.append(f"- 💡 **Warnings:** {summary['warning']}\n")
        violations = governance.get("violations", [])
        if violations:
            lines.append("\n### Top Violations\n\n")
            for v in violations[:5]:  # Show top 5
                icon = "🔴" if v["severity"] == "CRITICAL" else "⚠️"
                lines.append(
                    f"- {icon} `{v.get('directory', 'unknown')}`: {v.get('message', 'Unknown violation')}\n"
                )
    else:
        lines.append(
            "✅ No language governance data available or no violations found.\n"
        )
    lines.append("\n---\n\n")
    # CodeQL Results
    lines.append("## 2️⃣ CodeQL Security Analysis\n\n")
    codeql_results = parse_sarif_results(results_dir, "codeql")
    if codeql_results["total_alerts"] > 0:
        lines.append(f"- **Total Alerts:** {codeql_results['total_alerts']}\n")
        for severity, count in codeql_results["by_severity"].items():
            if count > 0:
                lines.append(f"- **{severity.capitalize()}:** {count}\n")
        if codeql_results["alerts"]:
            lines.append("\n### Top Alerts\n\n")
            for alert in codeql_results["alerts"][:5]:
                lines.append(
                    f"- [{alert['level']}] {alert['message']} (Rule: {alert['rule_id']})\n"
                )
    else:
        lines.append("✅ No CodeQL alerts found or results not available.\n")
    lines.append("\n---\n\n")
    # Semgrep Results
    lines.append("## 3️⃣ Semgrep Security Scan\n\n")
    semgrep_results = parse_sarif_results(results_dir, "semgrep")
    if semgrep_results["total_alerts"] > 0:
        lines.append(f"- **Total Alerts:** {semgrep_results['total_alerts']}\n")
        for severity, count in semgrep_results["by_severity"].items():
            if count > 0:
                lines.append(f"- **{severity.capitalize()}:** {count}\n")
        if semgrep_results["alerts"]:
            lines.append("\n### Top Alerts\n\n")
            for alert in semgrep_results["alerts"][:5]:
                lines.append(
                    f"- [{alert['level']}] {alert['message']} (Rule: {alert['rule_id']})\n"
                )
    else:
        lines.append("✅ No Semgrep alerts found or results not available.\n")
    lines.append("\n---\n\n")
    # Summary
    lines.append("## 📊 Overall Summary\n\n")
    total_issues = (
        governance.get("summary", {}).get("total_violations", 0)
        + codeql_results["total_alerts"]
        + semgrep_results["total_alerts"]
    )
    if total_issues == 0:
        lines.append("✅ **No critical issues found across all scans!**\n\n")
        lines.append("All security and governance checks passed successfully.\n")
    else:
        lines.append(f"⚠️ **Total Issues Found:** {total_issues}\n\n")
        lines.append(
            "Please review the detailed findings above and address critical and high-severity issues.\n"
        )
    lines.append("\n---\n")
    lines.append("\n*Report generated by SynergyMesh Security Pipeline*\n")
    # Write report
    report_content = "".join(lines)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Consolidated report written to: {output_file}", file=sys.stderr)
    print(f"Total issues: {total_issues}", file=sys.stderr)
def main():
    parser = argparse.ArgumentParser(
        description="Generate Consolidated Security Report"
    )
    parser.add_argument(
        "--results-dir", required=True, help="Directory containing analysis results"
    )
    parser.add_argument("--output", required=True, help="Output markdown file")
    args = parser.parse_args()
    if not os.path.exists(args.results_dir):
        print(
            f"Error: Results directory not found: {args.results_dir}", file=sys.stderr
        )
        sys.exit(1)
    generate_consolidated_report(args.results_dir, args.output)
    return 0
if __name__ == "__main__":
    sys.exit(main())
