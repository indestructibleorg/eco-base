# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: ci-cost-dashboard
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
CI Cost Dashboard Generator
Analyzes GitHub Actions workflow runs and generates cost/usage reports.
"""
import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any
import requests
GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.environ.get('ECO_GITHUB_TOKEN')
REPO_OWNER = (
    os.environ.get("GITHUB_REPOSITORY", "").split("/")[0]
    if os.environ.get('ECO_GITHUB_REPOSITORY')
    else None
)
REPO_NAME = (
    os.environ.get("GITHUB_REPOSITORY", "").split("/")[1]
    if os.environ.get('ECO_GITHUB_REPOSITORY')
    else None
)
# Cost estimation (GitHub Actions pricing)
COST_PER_MINUTE = {
    "ubuntu-latest": 0.008,  # $0.008/minute
    "ubuntu-22.04": 0.008,
    "ubuntu-20.04": 0.008,
    "macos-latest": 0.08,  # $0.08/minute
    "windows-latest": 0.016,  # $0.016/minute
}
ANOMALY_THRESHOLDS = {
    "max_runs_per_workflow": 50,  # Per week
    "max_duration_minutes": 30,
    "max_total_minutes_per_workflow": 500,  # Per week
}
def get_headers() -> dict[str, str]:
    """Get headers for GitHub API requests."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers
def fetch_workflow_runs(days: int = 7) -> list[dict[str, Any]]:
    """Fetch workflow runs from the past N days."""
    if not REPO_OWNER or not REPO_NAME:
        print("Error: GITHUB_REPOSITORY environment variable not set", file=sys.stderr)
        sys.exit(1)
    since = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"
    url = f"{GITHUB_API}/repos/{REPO_OWNER}/{REPO_NAME}/actions/runs"
    params = {
        "created": f">={since}",
        "per_page": 100,
        "page": 1,
    }
    all_runs = []
    while True:
        response = requests.get(url, headers=get_headers(), params=params)
        response.raise_for_status()
        data = response.json()
        runs = data.get("workflow_runs", [])
        all_runs.extend(runs)
        # Check if there are more pages
        if len(runs) < params["per_page"]:
            break
        params["page"] += 1
        # Safety limit
        if params["page"] > 20:
            break
    return all_runs
def analyze_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze workflow runs and calculate statistics."""
    workflow_stats = defaultdict(
        lambda: {
            "runs": 0,
            "successful": 0,
            "failed": 0,
            "cancelled": 0,
            "total_duration_minutes": 0,
            "total_cost_usd": 0.0,
            "avg_duration_minutes": 0.0,
            "triggers": defaultdict(int),
            "branches": defaultdict(int),
        }
    )
    total_stats = {
        "total_runs": len(runs),
        "total_minutes": 0,
        "total_cost_usd": 0.0,
        "period_days": 0,
    }
    for run in runs:
        workflow_name = run.get("name", "Unknown")
        status = run.get("conclusion", "unknown")
        # Calculate duration
        created_at = datetime.fromisoformat(run["created_at"].rstrip("Z"))
        updated_at = datetime.fromisoformat(run["updated_at"].rstrip("Z"))
        duration_minutes = (updated_at - created_at).total_seconds() / 60
        # Estimate cost (assuming ubuntu-latest)
        runner_os = "ubuntu-latest"
        cost = duration_minutes * COST_PER_MINUTE.get(runner_os, 0.008)
        # Update workflow stats
        stats = workflow_stats[workflow_name]
        stats["runs"] += 1
        stats["total_duration_minutes"] += duration_minutes
        stats["total_cost_usd"] += cost
        if status == "success":
            stats["successful"] += 1
        elif status == "failure":
            stats["failed"] += 1
        elif status == "cancelled":
            stats["cancelled"] += 1
        stats["triggers"][run.get("event", "unknown")] += 1
        stats["branches"][run.get("head_branch", "unknown")] += 1
        # Update total stats
        total_stats["total_minutes"] += duration_minutes
        total_stats["total_cost_usd"] += cost
    # Calculate averages
    for stats in workflow_stats.values():
        if stats["runs"] > 0:
            stats["avg_duration_minutes"] = (
                stats["total_duration_minutes"] / stats["runs"]
            )
    return {"workflows": dict(workflow_stats), "totals": total_stats}
def detect_anomalies(analysis: dict[str, Any]) -> list[str]:
    """Detect anomalous workflow behavior."""
    anomalies = []
    for workflow_name, stats in analysis["workflows"].items():
        # Check for excessive runs
        if stats["runs"] > ANOMALY_THRESHOLDS["max_runs_per_workflow"]:
            anomalies.append(
                f"⚠️ **{workflow_name}**: Excessive runs ({stats['runs']} runs, threshold: {ANOMALY_THRESHOLDS['max_runs_per_workflow']})"
            )
        # Check for long-running workflows
        if stats["avg_duration_minutes"] > ANOMALY_THRESHOLDS["max_duration_minutes"]:
            anomalies.append(
                f"⚠️ **{workflow_name}**: Long average duration ({stats['avg_duration_minutes']:.1f} min, threshold: {ANOMALY_THRESHOLDS['max_duration_minutes']} min)"
            )
        # Check for excessive total minutes
        if (
            stats["total_duration_minutes"]
            > ANOMALY_THRESHOLDS["max_total_minutes_per_workflow"]
        ):
            anomalies.append(
                f"⚠️ **{workflow_name}**: Excessive total minutes ({stats['total_duration_minutes']:.0f} min, threshold: {ANOMALY_THRESHOLDS['max_total_minutes_per_workflow']} min)"
            )
        # Check for high failure rate
        if stats["runs"] > 10:
            failure_rate = stats["failed"] / stats["runs"]
            if failure_rate > 0.3:
                anomalies.append(
                    f"⚠️ **{workflow_name}**: High failure rate ({failure_rate*100:.0f}%, {stats['failed']}/{stats['runs']} runs)"
                )
    return anomalies
def generate_markdown_report(
    analysis: dict[str, Any], days: int, anomalies: list[str]
) -> str:
    """Generate markdown report."""
    now = datetime.utcnow()
    report_period = f"{(now - timedelta(days=days)).strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}"
    # Sort workflows by cost
    sorted_workflows = sorted(
        analysis["workflows"].items(),
        key=lambda x: x[1]["total_cost_usd"],
        reverse=True,
    )
    report = f"""# CI Cost Dashboard 📊
#*Generated**: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC
#*Period**: {report_period} ({days} days)
#*Repository**: {REPO_OWNER}/{REPO_NAME}
---
## 📊 Summary
| Metric | Value |
|--------|-------|
| **Total Workflow Runs** | {analysis['totals']['total_runs']} |
| **Total Minutes Used** | {analysis['totals']['total_minutes']:.0f} min |
| **Estimated Cost** | ${analysis['totals']['total_cost_usd']:.2f} |
| **Average Cost per Run** | ${(analysis['totals']['total_cost_usd'] / analysis['totals']['total_runs'] if analysis['totals']['total_runs'] > 0 else 0):.3f} |
| **Estimated Monthly Cost** | ${(analysis['totals']['total_cost_usd'] / days * 30):.2f} |
"""
    # Anomalies section
    if anomalies:
        report += "\n## 🚨 Anomalies Detected\n\n"
        for anomaly in anomalies:
            report += f"- {anomaly}\n"
        report += "\n"
    else:
        report += "\n## ✅ No Anomalies Detected\n\nAll workflows are operating within expected parameters.\n\n"
    # Top 10 most expensive workflows
    report += "\n## 💰 Top 10 Most Expensive Workflows\n\n"
    report += "| Rank | Workflow | Runs | Minutes | Est. Cost | Avg Duration | Success Rate |\n"
    report += "|------|----------|------|---------|-----------|--------------|-------------|\n"
    for idx, (workflow_name, stats) in enumerate(sorted_workflows[:10], 1):
        success_rate = (
            (stats["successful"] / stats["runs"] * 100) if stats["runs"] > 0 else 0
        )
        report += (
            f"| {idx} | {workflow_name} | {stats['runs']} | "
            f"{stats['total_duration_minutes']:.0f} | ${stats['total_cost_usd']:.2f} | "
            f"{stats['avg_duration_minutes']:.1f} min | {success_rate:.0f}% |\n"
        )
    # All workflows detailed stats
    report += "\n## 📈 All Workflows Detailed Statistics\n\n"
    for workflow_name, stats in sorted_workflows:
        success_rate = (
            (stats["successful"] / stats["runs"] * 100) if stats["runs"] > 0 else 0
        )
        failure_rate = (
            (stats["failed"] / stats["runs"] * 100) if stats["runs"] > 0 else 0
        )
        report += f"\n### {workflow_name}\n\n"
        report += "| Metric | Value |\n"
        report += "|--------|-------|\n"
        report += f"| Total Runs | {stats['runs']} |\n"
        report += f"| Successful | {stats['successful']} ({success_rate:.0f}%) |\n"
        report += f"| Failed | {stats['failed']} ({failure_rate:.0f}%) |\n"
        report += f"| Cancelled | {stats['cancelled']} |\n"
        report += f"| Total Minutes | {stats['total_duration_minutes']:.0f} min |\n"
        report += f"| Average Duration | {stats['avg_duration_minutes']:.1f} min |\n"
        report += f"| Estimated Cost | ${stats['total_cost_usd']:.2f} |\n"
        # Triggers breakdown
        if stats["triggers"]:
            report += "\n**Triggers**:\n"
            for trigger, count in sorted(
                stats["triggers"].items(), key=lambda x: x[1], reverse=True
            ):
                report += f"- `{trigger}`: {count} runs\n"
        # Top branches
        if stats["branches"]:
            top_branches = sorted(
                stats["branches"].items(), key=lambda x: x[1], reverse=True
            )[:5]
            report += "\n**Top Branches**:\n"
            for branch, count in top_branches:
                report += f"- `{branch}`: {count} runs\n"
    # Cost optimization recommendations
    report += "\n---\n\n## 💡 Cost Optimization Recommendations\n\n"
    high_frequency_workflows = [
        name for name, stats in analysis["workflows"].items() if stats["runs"] > 30
    ]
    if high_frequency_workflows:
        report += "\n### High-Frequency Workflows\n\n"
        report += "Consider optimizing these workflows that run frequently:\n\n"
        for name in high_frequency_workflows[:5]:
            stats = analysis["workflows"][name]
            report += f"- **{name}**: {stats['runs']} runs in {days} days\n"
            report += "  - Consider: Reducing trigger frequency, using path filters, or consolidating jobs\n"
    long_running_workflows = [
        name
        for name, stats in analysis["workflows"].items()
        if stats["avg_duration_minutes"] > 15
    ]
    if long_running_workflows:
        report += "\n### Long-Running Workflows\n\n"
        report += "Consider optimizing these workflows with long durations:\n\n"
        for name in long_running_workflows[:5]:
            stats = analysis["workflows"][name]
            report += f"- **{name}**: {stats['avg_duration_minutes']:.1f} min average\n"
            report += "  - Consider: Caching dependencies, parallelizing jobs, or optimizing build steps\n"
    report += "\n---\n\n"
    report += "*This dashboard is automatically generated by the CI Cost Dashboard workflow.*\n"
    return report
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate CI cost dashboard")
    parser.add_argument("--days", type=int, default=7, help="Number of days to analyze")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument(
        "--check-anomalies", action="store_true", help="Check for anomalies only"
    )
    args = parser.parse_args()
    if not GITHUB_TOKEN:
        print("Error: GITHUB_TOKEN environment variable required", file=sys.stderr)
        sys.exit(1)
    try:
        print(f"Fetching workflow runs for the past {args.days} days...")
        runs = fetch_workflow_runs(args.days)
        print(f"Fetched {len(runs)} workflow runs")
        print("Analyzing workflow runs...")
        analysis = analyze_runs(runs)
        print("Detecting anomalies...")
        anomalies = detect_anomalies(analysis)
        if args.check_anomalies:
            if anomalies:
                for anomaly in anomalies:
                    print(anomaly)
                sys.exit(0)
            else:
                print("No anomalies detected")
                sys.exit(0)
        print("Generating report...")
        report = generate_markdown_report(analysis, args.days, anomalies)
        if args.output:
            with open(args.output, "w", encoding='utf-8') as f:
                f.write(report)
            print(f"Report written to {args.output}")
        else:
            print(report)
        print("\n✅ Dashboard generated successfully!")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
if __name__ == "__main__":
    main()
