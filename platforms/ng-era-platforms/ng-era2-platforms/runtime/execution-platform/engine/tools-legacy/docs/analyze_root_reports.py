# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: analyze_root_reports
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Analyze root-level reports in the SynergyMesh repository.
This script scans and analyzes all root-level report files, extracting key metrics,
findings, and status information to generate a consolidated analysis report.
Usage:
  python tools/docs/analyze_root_reports.py \
    --repo-root . \
    --output docs/reports-analysis.md \
    --json-output docs/reports-analysis.json
  python tools/docs/analyze_root_reports.py --help
"""
# MNGA-002: Import organization needs review
from __future__ import annotations
import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
# Try to import yaml, provide helpful error if not available
try:
    import yaml
except ImportError:
    print("Warning: PyYAML not available. YAML output will be disabled.")
    yaml = None
# Regex patterns for extracting information from markdown reports
SECTION_PATTERN = re.compile(r"^#{1,3}\s+(.+?)\s*$", re.MULTILINE)
STATUS_PATTERN = re.compile(r"[✅✓]|完成|success|passed|clean", re.IGNORECASE)
WARNING_PATTERN = re.compile(r"[⚠️]|警告|warning|caution", re.IGNORECASE)
ERROR_PATTERN = re.compile(r"[❌✗]|錯誤|error|failed|failure", re.IGNORECASE)
METRIC_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:%|errors?|warnings?|tests?|lines?|issues?)", re.IGNORECASE
)
BULLET_PATTERN = re.compile(r"^\s*[-*+]\s+(.+)", re.MULTILINE)
@dataclass
class ReportMetrics:
    """Metrics extracted from a report."""
    total_items: int = 0
    success_count: int = 0
    warning_count: int = 0
    error_count: int = 0
    percentages: list[float] = field(default_factory=list)
    numeric_values: list[int] = field(default_factory=list)
@dataclass
class ReportSection:
    """A section within a report."""
    title: str
    level: int
    content: str
    line_number: int
@dataclass
class ReportAnalysis:
    """Analysis of a single report file."""
    file_path: str
    file_name: str
    category: str
    line_count: int
    word_count: int
    sections: list[ReportSection]
    metrics: ReportMetrics
    key_findings: list[str]
    recommendations: list[str]
    status_summary: str
    has_errors: bool
    has_warnings: bool
    has_successes: bool
    created_date: str | None = None
@dataclass
class ConsolidatedAnalysis:
    """Consolidated analysis of all reports."""
    generated_at: str
    repo_root: str
    total_reports: int
    reports_by_category: dict[str, int]
    overall_status: str
    total_lines: int
    total_words: int
    reports: list[ReportAnalysis]
    summary: dict[str, Any]
def _extract_sections(content: str) -> list[ReportSection]:
    """Extract all sections from markdown content."""
    sections: list[ReportSection] = []
    lines = content.split("\n")
    for i, line in enumerate(lines, start=1):
        match = re.match(r"^(#{1,3})\s+(.+?)\s*$", line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            # Find content until next section of same or higher level
            content_lines: list[str] = []
            j = i
            while j < len(lines):
                next_line = lines[j]
                next_match = re.match(r"^(#{1,3})\s+", next_line)
                if next_match and len(next_match.group(1)) <= level:
                    break
                content_lines.append(next_line)
                j += 1
            section = ReportSection(
                title=title,
                level=level,
                content="\n".join(content_lines).strip(),
                line_number=i,
            )
            sections.append(section)
    return sections
def _extract_metrics(content: str) -> ReportMetrics:
    """Extract numerical metrics from report content."""
    metrics = ReportMetrics()
    # Count status indicators
    metrics.success_count = len(STATUS_PATTERN.findall(content))
    metrics.warning_count = len(WARNING_PATTERN.findall(content))
    metrics.error_count = len(ERROR_PATTERN.findall(content))
    # Extract numeric values and percentages
    for match in METRIC_PATTERN.finditer(content):
        value = float(match.group(1))
        if "%" in match.group(0):
            metrics.percentages.append(value)
        else:
            metrics.numeric_values.append(int(value))
    # Count bullet points as items
    metrics.total_items = len(BULLET_PATTERN.findall(content))
    return metrics
def _extract_key_findings(sections: list[ReportSection]) -> list[str]:
    """Extract key findings from report sections."""
    findings: list[str] = []
    # Keywords that indicate findings sections
    finding_keywords = [
        "finding",
        "result",
        "outcome",
        "achievement",
        "issue",
        "problem",
        "challenge",
        "成就",
        "發現",
        "結果",
        "問題",
    ]
    for section in sections:
        title_lower = section.title.lower()
        if any(keyword in title_lower for keyword in finding_keywords):
            # Extract bullet points
            bullets = BULLET_PATTERN.findall(section.content)
            findings.extend([b.strip() for b in bullets[:5]])  # Limit to top 5
    return findings
def _extract_recommendations(sections: list[ReportSection]) -> list[str]:
    """Extract recommendations from report sections."""
    recommendations: list[str] = []
    # Keywords that indicate recommendation sections
    rec_keywords = [
        "recommendation",
        "action",
        "next step",
        "improvement",
        "suggestion",
        "proposal",
        "建議",
        "行動",
        "改進",
    ]
    for section in sections:
        title_lower = section.title.lower()
        if any(keyword in title_lower for keyword in rec_keywords):
            bullets = BULLET_PATTERN.findall(section.content)
            recommendations.extend([b.strip() for b in bullets[:5]])
    return recommendations
def _determine_status(
    metrics: ReportMetrics, content: str
) -> tuple[str, bool, bool, bool]:
    """Determine overall status from metrics and content."""
    has_errors = metrics.error_count > 0 or ERROR_PATTERN.search(content) is not None
    has_warnings = (
        metrics.warning_count > 0 or WARNING_PATTERN.search(content) is not None
    )
    has_successes = (
        metrics.success_count > 0 or STATUS_PATTERN.search(content) is not None
    )
    if has_errors:
        status = "❌ Issues Detected"
    elif has_warnings:
        status = "⚠️  Warnings Present"
    elif has_successes:
        status = "✅ Healthy"
    else:
        status = "ℹ️  Informational"
    return status, has_errors, has_warnings, has_successes
def _categorize_report(file_path: Path) -> str:
    """Determine the category of a report based on its path and name."""
    path_parts = file_path.parts
    name_lower = file_path.stem.lower()
    # Check path-based categories
    if "self-awareness" in name_lower:
        return "Self-Awareness"
    elif "phase" in name_lower:
        return "Phase Implementation"
    elif "pr" in name_lower or "pull" in name_lower:
        return "Pull Request Analysis"
    elif "comprehensive" in name_lower:
        return "Comprehensive Report"
    elif "validation" in name_lower:
        return "Validation Report"
    elif "integration" in name_lower:
        return "Integration Analysis"
    elif "gl-platform.governance" in name_lower or "ci" in name_lower:
        return "Governance & CI"
    elif "reports" in path_parts:
        return "Detailed Report"
    else:
        return "General Report"
def _extract_date(content: str, file_path: Path) -> str | None:
    """Extract creation/completion date from report content or file metadata."""
    # Try to find date in content
    date_patterns = [
        r"完成日期[：:]\s*(\d{4}-\d{2}-\d{2})",
        r"[Dd]ate[：:]\s*(\d{4}-\d{2}-\d{2})",
        r"[Gg]enerated[：:]\s*(\d{4}-\d{2}-\d{2})",
        r"(\d{4}-\d{2}-\d{2})",
    ]
    for pattern in date_patterns:
        match = re.search(pattern, content)
        if match:
            return match.group(1)
    # Fallback to file modification time
    try:
        mtime = file_path.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=UTC).strftime("%Y-%m-%d")
    except Exception:
        return None
def analyze_report(file_path: Path, repo_root: Path) -> ReportAnalysis:
    """Analyze a single report file."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Try with latin-1 encoding as fallback
        content = file_path.read_text(encoding="latin-1")
    lines = content.split("\n")
    words = content.split()
    sections = _extract_sections(content)
    metrics = _extract_metrics(content)
    key_findings = _extract_key_findings(sections)
    recommendations = _extract_recommendations(sections)
    status, has_errors, has_warnings, has_successes = _determine_status(
        metrics, content
    )
    category = _categorize_report(file_path)
    created_date = _extract_date(content, file_path)
    relative_path = str(file_path.relative_to(repo_root))
    return ReportAnalysis(
        file_path=relative_path,
        file_name=file_path.name,
        category=category,
        line_count=len(lines),
        word_count=len(words),
        sections=sections,
        metrics=metrics,
        key_findings=key_findings,
        recommendations=recommendations,
        status_summary=status,
        has_errors=has_errors,
        has_warnings=has_warnings,
        has_successes=has_successes,
        created_date=created_date,
    )
def scan_reports(repo_root: Path) -> list[ReportAnalysis]:
    """Scan and analyze all report files in the repository."""
    report_dirs = [
        repo_root / "reports",
        repo_root / "docs" / "reports",
    ]
    analyses: list[ReportAnalysis] = []
    for report_dir in report_dirs:
        if not report_dir.exists():
            continue
        # Find all markdown files
        for md_file in report_dir.glob("*.md"):
            if md_file.is_file():
                analysis = analyze_report(md_file, repo_root)
                analyses.append(analysis)
    # Sort by category and then by file name
    analyses.sort(key=lambda a: (a.category, a.file_name))
    return analyses
def generate_consolidated_analysis(
    analyses: list[ReportAnalysis], repo_root: Path
) -> ConsolidatedAnalysis:
    """Generate consolidated analysis from individual report analyses."""
    reports_by_category: dict[str, int] = {}
    total_lines = 0
    total_words = 0
    for analysis in analyses:
        category = analysis.category
        reports_by_category[category] = reports_by_category.get(category, 0) + 1
        total_lines += analysis.line_count
        total_words += analysis.word_count
    # Determine overall status
    error_count = sum(1 for a in analyses if a.has_errors)
    warning_count = sum(1 for a in analyses if a.has_warnings)
    success_count = sum(1 for a in analyses if a.has_successes)
    if error_count > 0:
        overall_status = f"❌ Issues Found ({error_count} reports with errors)"
    elif warning_count > 0:
        overall_status = f"⚠️  Warnings Present ({warning_count} reports with warnings)"
    elif success_count > 0:
        overall_status = f"✅ All Reports Healthy ({success_count} successful)"
    else:
        overall_status = "ℹ️  Reports Informational"
    # Generate summary statistics
    summary = {
        "total_reports": len(analyses),
        "reports_with_errors": error_count,
        "reports_with_warnings": warning_count,
        "reports_with_successes": success_count,
        "total_lines": total_lines,
        "total_words": total_words,
        "average_lines_per_report": total_lines // len(analyses) if analyses else 0,
        "average_words_per_report": total_words // len(analyses) if analyses else 0,
        "categories": list(reports_by_category.keys()),
    }
    generated_at = (
        datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    return ConsolidatedAnalysis(
        generated_at=generated_at,
        repo_root=str(repo_root.absolute()),
        total_reports=len(analyses),
        reports_by_category=reports_by_category,
        overall_status=overall_status,
        total_lines=total_lines,
        total_words=total_words,
        reports=analyses,
        summary=summary,
    )
def render_markdown_report(consolidated: ConsolidatedAnalysis) -> str:
    """Render the consolidated analysis as a Markdown report."""
    lines: list[str] = [
        "# Root-Level Reports Analysis",
        "",
        f"**Generated**: {consolidated.generated_at}",
        f"**Repository**: `{consolidated.repo_root}`",
        "",
        "---",
        "",
        "## 📊 Executive Summary",
        "",
        f"**Total Reports**: {consolidated.total_reports}",
        f"**Overall Status**: {consolidated.overall_status}",
        f"**Total Content**: {consolidated.total_lines:,} lines, {consolidated.total_words:,} words",
        "",
        "### Reports by Category",
        "",
    ]
    for category, count in sorted(consolidated.reports_by_category.items()):
        lines.append(f"- **{category}**: {count} report(s)")
    lines.extend(["",
                  "### Summary Statistics",
                  "",
                  f"- Reports with Errors: {consolidated.summary['reports_with_errors']}",
                  f"- Reports with Warnings: {consolidated.summary['reports_with_warnings']}",
                  f"- Healthy Reports: {consolidated.summary['reports_with_successes']}",
                  f"- Average Lines per Report: {consolidated.summary['average_lines_per_report']:,}",
                  f"- Average Words per Report: {consolidated.summary['average_words_per_report']:,}",
                  "",
                  "---",
                  "",
                  "## 📁 Report Inventory",
                  "",
                  ])
    # Group reports by category
    current_category: str | None = None
    for report in consolidated.reports:
        if report.category != current_category:
            current_category = report.category
            lines.extend(
                [
                    "",
                    f"### {current_category}",
                    "",
                ]
            )
        lines.extend(
            [
                f"#### {report.file_name}",
                "",
                f"- **Path**: `{report.file_path}`",
                f"- **Status**: {report.status_summary}",
                f"- **Size**: {report.line_count:,} lines, {report.word_count:,} words",
            ]
        )
        if report.created_date:
            lines.append(f"- **Date**: {report.created_date}")
        lines.extend(
            [
                f"- **Sections**: {len(report.sections)}",
                f"- **Metrics**: {report.metrics.success_count} ✅ / {report.metrics.warning_count} ⚠️ / {report.metrics.error_count} ❌",
                "",
            ])
        if report.key_findings:
            lines.append("**Key Findings**:")
            for finding in report.key_findings[:3]:  # Show top 3
                lines.append(f"- {finding}")
            lines.append("")
        if report.recommendations:
            lines.append("**Recommendations**:")
            for rec in report.recommendations[:3]:  # Show top 3
                lines.append(f"- {rec}")
            lines.append("")
    lines.extend(
        [
            "---",
            "",
            "## 🔍 Detailed Findings",
            "",
        ]
    )
    # Aggregate all findings
    all_findings: list[tuple[str, str]] = []
    for report in consolidated.reports:
        for finding in report.key_findings:
            all_findings.append((report.file_name, finding))
    if all_findings:
        lines.append("### All Key Findings")
        lines.append("")
        current_file: str | None = None
        for file_name, finding in all_findings[:20]:  # Limit to top 20
            if file_name != current_file:
                current_file = file_name
                lines.append(f"#### From {file_name}")
                lines.append("")
            lines.append(f"- {finding}")
        lines.append("")
    # Aggregate all recommendations
    all_recommendations: list[tuple[str, str]] = []
    for report in consolidated.reports:
        for rec in report.recommendations:
            all_recommendations.append((report.file_name, rec))
    if all_recommendations:
        lines.append("### All Recommendations")
        lines.append("")
        current_file = None
        for file_name, rec in all_recommendations[:20]:  # Limit to top 20
            if file_name != current_file:
                current_file = file_name
                lines.append(f"#### From {file_name}")
                lines.append("")
            lines.append(f"- {rec}")
        lines.append("")
    lines.extend(
        [
            "---",
            "",
            "## 🎯 Action Items",
            "",
            "Based on the consolidated analysis, here are the recommended actions:",
            "",
        ]
    )
    # Generate action items based on status
    if consolidated.summary["reports_with_errors"] > 0:
        lines.extend(
            [
                "### 🔴 High Priority",
                "",
                f"- Review and address issues in {consolidated.summary['reports_with_errors']} report(s) with errors",
                "",
            ])
    if consolidated.summary["reports_with_warnings"] > 0:
        lines.extend(
            [
                "### 🟡 Medium Priority",
                "",
                f"- Investigate warnings in {consolidated.summary['reports_with_warnings']} report(s)",
                "",
            ])
    lines.extend(
        [
            "### 🟢 Maintenance",
            "",
            "- Keep reports up-to-date with latest developments",
            "- Archive or consolidate outdated reports",
            "- Ensure consistent formatting across all reports",
            "",
            "---",
            "",
            "## 📈 Health Indicators",
            "",
            f"- **Report Coverage**: {consolidated.total_reports} reports across {len(consolidated.reports_by_category)} categories",
            f"- **Documentation Density**: {consolidated.total_words:,} words of documentation",
            f"- **Status Health**: {consolidated.summary['reports_with_successes']} / {consolidated.total_reports} reports healthy",
            "",
            "---",
            "",
            f"*Report generated by `tools/docs/analyze_root_reports.py` at {consolidated.generated_at}*",
            "",
        ])
    return "\n".join(lines)
def render_json_output(consolidated: ConsolidatedAnalysis) -> str:
    """Render the consolidated analysis as JSON."""
    # Convert to dict, handling nested dataclasses
    def to_dict(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return {k: to_dict(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, list):
            return [to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        return obj
    data = to_dict(consolidated)
    return json.dumps(data, indent=2, ensure_ascii=False)
def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Root directory of the repository (default: current directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for Markdown report (default: stdout)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Output path for JSON report (optional)",
    )
    parser.add_argument(
        "--yaml-output",
        type=Path,
        default=None,
        help="Output path for YAML report (optional, requires PyYAML)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    args = parser.parse_args()
    # Validate repo root
    if not args.repo_root.exists():
        print(f"Error: Repository root not found: {args.repo_root}", file=sys.stderr)
        sys.exit(1)
    if args.verbose:
        print(f"Scanning reports in: {args.repo_root}")
    # Scan and analyze reports
    analyses = scan_reports(args.repo_root)
    if not analyses:
        print("Warning: No report files found", file=sys.stderr)
        sys.exit(0)
    if args.verbose:
        print(f"Found {len(analyses)} report(s)")
    # Generate consolidated analysis
    consolidated = generate_consolidated_analysis(analyses, args.repo_root)
    # Generate Markdown report
    markdown_report = render_markdown_report(consolidated)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown_report, encoding="utf-8")
        if args.verbose:
            print(f"Markdown report written to: {args.output}")
    else:
        print(markdown_report)
    # Generate JSON output if requested
    if args.json_output:
        json_output = render_json_output(consolidated)
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json_output, encoding="utf-8")
        if args.verbose:
            print(f"JSON report written to: {args.json_output}")
    # Generate YAML output if requested
    if args.yaml_output:
        if yaml is None:
            print(
                "Warning: PyYAML not available, skipping YAML output", file=sys.stderr
            )
        else:
            # Convert to dict for YAML serialization
            data = json.loads(render_json_output(consolidated))
            yaml_output = yaml.dump(data, allow_unicode=True, sort_keys=False)
            args.yaml_output.parent.mkdir(parents=True, exist_ok=True)
            args.yaml_output.write_text(yaml_output, encoding="utf-8")
            if args.verbose:
                print(f"YAML report written to: {args.yaml_output}")
    if args.verbose:
        print("\n✅ Analysis complete!")
        print(f"   Total reports: {consolidated.total_reports}")
        print(f"   Overall status: {consolidated.overall_status}")
if __name__ == "__main__":
    main()
