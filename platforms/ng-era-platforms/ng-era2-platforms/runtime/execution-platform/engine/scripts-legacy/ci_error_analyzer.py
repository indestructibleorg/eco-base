#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: ci-error-analyzer
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
CI Error Analyzer GitHub Action Script
This script analyzes CI/CD workflow failures using the CI Error Analyzer
and posts the results as PR comments or creates issues.
Usage:
    python scripts/ci-error-analyzer.py --workflow-run-id <id> --mode <comment|issue>
"""
# MNGA-002: Import organization needs review
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict
# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent.parent / "workspace" / "src"))
try:
    from core.ci_error_handler import (
        CIError,
        CIErrorAnalyzer,
        ErrorCategory,
        ErrorSeverity,
    )
except ImportError:
    print("⚠️  CI Error Handler module not found. Using standalone mode.")
    CIErrorAnalyzer = None
def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analyze CI/CD workflow failures")
    parser.add_argument(
        "--workflow-run-id", type=str, help="GitHub Actions workflow run ID"
    )
    parser.add_argument("--job-name", type=str, help="Specific job name to analyze")
    parser.add_argument("--log-file", type=str, help="Path to log file to analyze")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "comment",
            "issue",
            "report"],
        default="report",
        help="Output mode: comment (PR comment), issue (create issue), report (JSON report)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ci-error-analysis.json",
        help="Output file path for JSON report",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()
def analyze_log_content(
    log_content: str, source: str = "github_actions"
) -> Dict[str, Any]:
    """
    Analyze log content and return structured error data
    Args:
        log_content: Raw log content
        source: Source of the log
    Returns:
        Dictionary with analysis results
    """
    if CIErrorAnalyzer is None:
        # Fallback: Simple error detection
        return fallback_analyze(log_content)
    analyzer = CIErrorAnalyzer()
    errors = analyzer.analyze_log(log_content, source=source)
    summary = analyzer.summarize_errors(errors)
    auto_fixable = analyzer.get_auto_fixable_errors(errors)
    return {
        "errors": [error.to_dict() for error in errors],
        "summary": summary,
        "auto_fixable_errors": [error.to_dict() for error in auto_fixable],
        "analysis_metadata": {
            "source": source,
            "total_errors": len(errors),
            "auto_fixable_count": len(auto_fixable),
        },
    }
def fallback_analyze(log_content: str) -> Dict[str, Any]:
    """
    Fallback analysis when CI Error Analyzer is not available
    Args:
        log_content: Raw log content
    Returns:
        Basic analysis results
    """
    import re
    # Simple pattern matching
    errors = []
    error_patterns = [
        (r"error:", "error"),
        (r"failed", "failure"),
        (r"exception", "exception"),
        (r"exit code [1-9]", "exit_code"),
    ]
    for pattern, error_type in error_patterns:
        matches = re.finditer(pattern, log_content, re.IGNORECASE)
        for match in matches:
            errors.append(
                {
                    "type": error_type,
                    "message": match.group(0),
                    "position": match.start(),
                }
            )
    return {
        "errors": errors,
        "summary": {
            "total": len(errors),
            "by_type": {},
        },
        "auto_fixable_errors": [],
        "analysis_metadata": {
            "source": "fallback",
            "total_errors": len(errors),
            "auto_fixable_count": 0,
        },
    }
def read_log_file(file_path: str) -> str:
    """Read log file content"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading log file: {e}")
        sys.exit(1)
def format_markdown_report(analysis: Dict[str, Any]) -> str:
    """
    Format analysis results as Markdown
    Args:
        analysis: Analysis results
    Returns:
        Formatted Markdown report
    """
    report = ["# 🔍 CI Error Analysis Report\n"]
    # Summary
    summary = analysis.get("summary", {})
    report.append("## 📊 Summary\n")
    report.append(f"- **Total Errors**: {summary.get('total', 0)}")
    report.append(f"- **Auto-Fixable**: {summary.get('auto_fixable_count', 0)}")
    # Errors by category
    by_category = summary.get("by_category", {})
    if by_category:
        report.append("\n### Errors by Category\n")
        for category, count in by_category.items():
            report.append(f"- **{category}**: {count}")
    # Errors by severity
    by_severity = summary.get("by_severity", {})
    if by_severity:
        report.append("\n### Errors by Severity\n")
        for severity, count in by_severity.items():
            emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(
                severity, "⚪"
            )
            report.append(f"- {emoji} **{severity}**: {count}")
    # Detailed errors
    errors = analysis.get("errors", [])
    if errors:
        report.append("\n## 🐛 Detected Errors\n")
        for idx, error in enumerate(errors[:10], 1):  # Limit to first 10
            report.append(f"### {idx}. {error.get('title', 'Unknown Error')}\n")
            report.append(f"- **Category**: {error.get('category', 'unknown')}")
            report.append(f"- **Severity**: {error.get('severity', 'unknown')}")
            if error.get("file_path"):
                location = error["file_path"]
                if error.get("line_number"):
                    location += f":{error['line_number']}"
                report.append(f"- **Location**: `{location}`")
            if error.get("message"):
                report.append("\n**Message**:")
                report.append(f"```\n{error['message'][:500]}\n```")
            if error.get("fix_suggestion"):
                report.append(f"\n💡 **Fix Suggestion**: {error['fix_suggestion']}")
            report.append("")
    # Auto-fixable errors
    auto_fixable = analysis.get("auto_fixable_errors", [])
    if auto_fixable:
        report.append("\n## ✅ Auto-Fixable Errors\n")
        report.append(
            f"Found {len(auto_fixable)} errors that can be automatically fixed:\n"
        )
        for error in auto_fixable[:5]:  # Limit to first 5
            report.append(
                f"- {error.get('title', 'Unknown')} - {error.get('fix_suggestion', '')}"
            )
    return "\n".join(report)
def save_json_report(analysis: Dict[str, Any], output_path: str) -> None:
    """Save analysis results as JSON"""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)
        print(f"✅ Report saved to: {output_path}")
    except Exception as e:
        print(f"❌ Error saving report: {e}")
def main():
    """Main execution"""
    args = parse_arguments()
    if args.verbose:
        print("🔍 Starting CI Error Analysis...")
    # Read log content
    log_content = ""
    if args.log_file:
        if args.verbose:
            print(f"📄 Reading log file: {args.log_file}")
        log_content = read_log_file(args.log_file)
    elif os.getenv("GITHUB_STEP_SUMMARY"):
        # Running in GitHub Actions - try to read from environment
        if args.verbose:
            print("🔄 Running in GitHub Actions context")
        # For now, use stdin or a predefined log location
        log_content = sys.stdin.read() if not sys.stdin.isatty() else ""
    if not log_content:
        print(
            "⚠️  No log content to analyze. Provide --log-file or pipe log content to stdin."
        )
        sys.exit(0)
    # Analyze
    if args.verbose:
        print(f"🔬 Analyzing {len(log_content)} characters of log data...")
    analysis = analyze_log_content(log_content)
    # Output based on mode
    if args.mode == "report":
        save_json_report(analysis, args.output)
        # Also print summary
        summary = analysis.get("summary", {})
        print("\n📊 Analysis Complete:")
        print(f"   Total Errors: {summary.get('total', 0)}")
        print(f"   Auto-Fixable: {summary.get('auto_fixable_count', 0)}")
    elif args.mode == "comment":
        # Format as markdown for PR comment
        markdown = format_markdown_report(analysis)
        # Save to GitHub Actions output if available
        if os.getenv("GITHUB_OUTPUT"):
            with open(os.getenv("GITHUB_OUTPUT"), "a") as f:
                f.write(f"analysis_report<<EOF\n{markdown}\nEOF\n")
        print(markdown)
    elif args.mode == "issue":
        # Prepare issue data
        errors = analysis.get("errors", [])
        if errors:
            issue_data = {
                "title": f"CI Failure: {errors[0].get('title', 'Unknown Error')}",
                "body": format_markdown_report(analysis),
                "labels": ["ci-failure", "automated"],
            }
            # Save issue data for GitHub Actions
            if os.getenv("GITHUB_OUTPUT"):
                with open(os.getenv("GITHUB_OUTPUT"), "a") as f:
                    f.write(f"issue_title={issue_data['title']}\n")
                    f.write(f"issue_body<<EOF\n{issue_data['body']}\nEOF\n")
            print(json.dumps(issue_data, indent=2))
    # Exit with error code if critical errors found
    summary = analysis.get("summary", {})
    by_severity = summary.get("by_severity", {})
    if by_severity.get("critical", 0) > 0:
        sys.exit(1)
if __name__ == "__main__":
    main()
