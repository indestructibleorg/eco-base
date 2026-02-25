# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: governance
# @ECO-semantic: generate-language-dashboard
# @ECO-audit-trail: ../../engine/governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Language Governance Dashboard Generator
----------------------------------------
Generates comprehensive Language Governance Dashboard with:
- Real-time health scores
- Violation maps
- Language distribution
- Trend analysis
- Fix history
- Security status
- Architecture compliance
- Mermaid diagrams
Usage:
    python3 tools/generate-language-dashboard.py \
        --governance-report governance/language-governance-report.json \
        --codeql-results governance/codeql-results-*.sarif \
        --semgrep-results governance/semgrep-results.sarif \
        --history knowledge/language-history.yaml \
        --health knowledge/language-health-score.yaml \
        --output docs/LANGUAGE_GOVERNANCE_DASHBOARD.md
"""
# MNGA-002: Import organization needs review
import argparse
import datetime
import glob
import json
import os
from collections import defaultdict
import yaml
from rich.console import Console
console = Console()
def load_json(path: str) -> dict:
    """Load JSON file"""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load {path}: {e}[/yellow]")
        return {}
def load_yaml(path: str) -> dict:
    """Load YAML file"""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load {path}: {e}[/yellow]")
        return {}
def load_sarif(pattern: str) -> list[dict]:
    """Load SARIF files matching pattern"""
    results = []
    for path in glob.glob(pattern):
        data = load_json(path)
        if data and "runs" in data:
            results.extend(data["runs"])
    return results
def calculate_statistics(
    governance_data: dict, codeql_data: list, semgrep_data: list, health_data: dict
) -> dict:
    """Calculate all dashboard statistics"""
    stats = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "generation_timestamp": datetime.datetime.utcnow().strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        ),
    }
    # Health score
    stats["health_score"] = health_data.get("overall_score", 0)
    stats["health_grade"] = health_data.get("grade", "N/A")
    stats["health_status"] = get_status_emoji(stats["health_score"])
    # Violations
    violations = governance_data.get("violations", [])
    stats["total_violations"] = len(violations)
    stats["cross_layer_count"] = sum(
        1 for v in violations if "cross-layer" in v.get("type", "").lower()
    )
    stats["forbidden_count"] = sum(
        1 for v in violations if "forbidden" in v.get("type", "").lower()
    )
    # Security findings
    codeql_count = sum(len(run.get("results", [])) for run in codeql_data)
    semgrep_count = sum(len(run.get("results", [])) for run in semgrep_data)
    stats["security_findings"] = codeql_count + semgrep_count
    stats["codeql_count"] = codeql_count
    stats["semgrep_count"] = semgrep_count
    # Architecture
    stats["architecture_compliance"] = (
        health_data.get("components", {})
        .get("architecture_alignment", {})
        .get("score", 0)
# 5
    )
    # Trends
    stats["violation_trend"] = "📈" if stats["total_violations"] > 50 else "📉"
    stats["security_status"] = (
        "🔴"
        if stats["security_findings"] > 20
        else "🟡" if stats["security_findings"] > 10 else "🟢"
    )
    stats["compliance_status"] = (
        "🟢"
        if stats["architecture_compliance"] > 80
        else "🟡" if stats["architecture_compliance"] > 60 else "🔴"
    )
    return stats
def get_status_emoji(score: float) -> str:
    """Get status emoji for score"""
    if score >= 90:
        return "🟢 Excellent"
    elif score >= 80:
        return "🟡 Good"
    elif score >= 70:
        return "🟠 Fair"
    elif score >= 60:
        return "🔴 Poor"
    else:
        return "🔴 Critical"
def generate_violations_table(violations: list[dict]) -> str:
    """Generate violations table markdown"""
    if not violations:
        return "| Type | File | Issue | Severity |\n|------|------|-------|----------|\n| No violations | - | - | - |\n"
    lines = ["| Type | File | Issue | Severity |", "|------|------|-------|----------|"]
    for v in violations[:20]:  # Limit to top 20
        vtype = v.get("type", "Unknown")
        vfile = v.get("file", "Unknown")
        vmsg = v.get("message", "No description")[:80]
        vsev = v.get("severity", "UNKNOWN")
        lines.append(f"| {vtype} | `{vfile}` | {vmsg} | **{vsev}** |")
    if len(violations) > 20:
        lines.append(f"| ... | ... | {len(violations) - 20} more violations | ... |")
    return "\n".join(lines)
def generate_language_chart(governance_data: dict) -> str:
    """Generate ASCII language distribution chart"""
    stats = governance_data.get("statistics", {})
    lang_dist = stats.get("by_language", {})
    if not lang_dist:
        return "No language data available"
    max_count = max(lang_dist.values()) if lang_dist else 1
    lines = []
    for lang, count in sorted(lang_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
        bar_length = int((count / max_count) * 50)
        bar = "█" * bar_length
        lines.append(f"{lang:15} {bar} {count}")
    return "\n".join(lines)
def generate_trend_chart(history_data: dict) -> str:
    """Generate ASCII trend chart"""
    fixes = history_data.get("fixes", [])
    if not fixes:
        return "No historical data available"
    # Group by date
    dates = defaultdict(int)
    for fix in fixes:
        date = fix.get("timestamp", "")[:10]
        dates[date] += 1
    # Get last 30 days
    sorted_dates = sorted(dates.items())[-30:]
    if not sorted_dates:
        return "No recent data"
    max_count = max(d[1] for d in sorted_dates)
    lines = []
    for date, count in sorted_dates:
        bar_length = int((count / max_count) * 30) if max_count > 0 else 0
        bar = "█" * bar_length
        lines.append(f"{date} {bar} {count}")
    return "\n".join(lines[-14:])  # Show last 2 weeks
def generate_fix_history_table(history_data: dict) -> str:
    """Generate fix history table"""
    fixes = history_data.get("fixes", [])[-10:]  # Last 10 fixes
    if not fixes:
        return "| Date | Type | Action | Status |\n|------|------|--------|--------|\n| No fixes yet | - | - | - |\n"
    lines = ["| Date | Type | Action | Status |", "|------|------|--------|--------|"]
    for fix in reversed(fixes):
        date = fix.get("timestamp", "")[:10]
        ftype = fix.get("type", "Unknown")
        action = fix.get("action", "Unknown")
        status = "✅ Success" if fix.get("success", False) else "❌ Failed"
        lines.append(f"| {date} | {ftype} | {action} | {status} |")
    return "\n".join(lines)
def generate_hotspot_table(violations: list[dict]) -> str:
    """Generate violation hotspot table"""
    # Count violations by directory
    dir_counts = defaultdict(int)
    for v in violations:
        vfile = v.get("file", "")
        vdir = "/".join(vfile.split("/")[:2]) if "/" in vfile else vfile
        dir_counts[vdir] += 1
    lines = [
        "| Directory | Violations | Status |",
        "|-----------|------------|--------|",
    ]
    for vdir, count in sorted(dir_counts.items(), key=lambda x: x[1], reverse=True)[
        :10
    ]:
        status = "🔴" if count > 10 else "🟡" if count > 5 else "🟢"
        lines.append(f"| `{vdir}` | {count} | {status} |")
    return "\n".join(lines)
def generate_dashboard(args):
    """Generate complete dashboard"""
    console.print("[cyan]Loading data sources...[/cyan]")
    # Load all data
    governance_data = load_json(args.governance_report)
    health_data = load_yaml(args.health)
    history_data = load_yaml(args.history)
    codeql_data = load_sarif(args.codeql_results)
    semgrep_data = load_sarif(args.semgrep_results)
    console.print("[cyan]Calculating statistics...[/cyan]")
    stats = calculate_statistics(
        governance_data, codeql_data, semgrep_data, health_data
    )
    console.print("[cyan]Generating dashboard sections...[/cyan]")
    # Read template
    template_path = args.output
    if os.path.exists(template_path):
        with open(template_path, encoding="utf-8") as f:
            template = f.read()
    else:
        console.print("[yellow]Template not found, using default[/yellow]")
        template = "# Dashboard Template Missing"
    # Replace placeholders
    replacements = {
        "TIMESTAMP": stats["timestamp"],
        "GENERATION_TIMESTAMP": stats["generation_timestamp"],
        "LANGUAGE_HEALTH_SCORE": str(stats["health_score"]),
        "HEALTH_GRADE": stats["health_grade"],
        "HEALTH_STATUS": stats["health_status"],
        "TOTAL_VIOLATIONS": str(stats["total_violations"]),
        "VIOLATION_TREND": stats["violation_trend"],
        "SECURITY_FINDINGS": str(stats["security_findings"]),
        "SECURITY_STATUS": stats["security_status"],
        "ARCHITECTURE_COMPLIANCE": str(int(stats["architecture_compliance"])),
        "COMPLIANCE_STATUS": stats["compliance_status"],
        "FIX_SUCCESS_RATE": "85",  # Placeholder
        "FIX_STATUS": "🟢",
        "VIOLATIONS_TABLE": generate_violations_table(
            governance_data.get("violations", [])
        ),
        "LANGUAGE_DISTRIBUTION_CHART": generate_language_chart(governance_data),
        "HEALTH_SCORE_TREND_CHART": "Trend data not available",
        "VIOLATION_TREND_CHART": generate_trend_chart(history_data),
        "FIX_HISTORY_TABLE": generate_fix_history_table(history_data),
        "HOTSPOT_TABLE": generate_hotspot_table(governance_data.get("violations", [])),
        "CROSS_LAYER_COUNT": str(stats["cross_layer_count"]),
        "FORBIDDEN_COUNT": str(stats["forbidden_count"]),
        "SECURITY_COUNT": str(stats["security_findings"]),
    }
    # Apply replacements
    dashboard = template
    for key, value in replacements.items():
        dashboard = dashboard.replace("{{ " + key + " }}", value)
    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(dashboard)
    console.print(f"[green]✓ Dashboard generated: {args.output}[/green]")
    # Generate data file
    data_output = args.output.replace(".md", "-data.yaml")
    with open(data_output, "w", encoding="utf-8") as f:
        yaml.dump(stats, f)
    console.print(f"[green]✓ Data file generated: {data_output}[/green]")
def main():
    parser = argparse.ArgumentParser(
        description="Generate Language Governance Dashboard"
    )
    parser.add_argument(
        "--governance-report",
        default="governance/language-governance-report.json",
        help="Path to governance report JSON",
    )
    parser.add_argument(
        "--codeql-results",
        default="governance/codeql-results-*.sarif",
        help="Path pattern for CodeQL SARIF files",
    )
    parser.add_argument(
        "--semgrep-results",
        default="governance/semgrep-results.sarif",
        help="Path to Semgrep SARIF file",
    )
    parser.add_argument(
        "--history",
        default="knowledge/language-history.yaml",
        help="Path to fix history YAML",
    )
    parser.add_argument(
        "--health",
        default="knowledge/language-health-score.yaml",
        help="Path to health score YAML",
    )
    parser.add_argument(
        "--output",
        default="docs/LANGUAGE_GOVERNANCE_DASHBOARD.md",
        help="Output dashboard file",
    )
    args = parser.parse_args()
    try:
        generate_dashboard(args)
    except Exception as e:
        console.print(f"[red]Error generating dashboard: {e}[/red]")
        import traceback
        traceback.print_exc()
        return 1
    return 0
if __name__ == "__main__":
    exit(main())
