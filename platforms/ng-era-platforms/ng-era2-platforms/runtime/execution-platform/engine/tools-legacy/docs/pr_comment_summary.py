# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: pr_comment_summary
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
PR Comment Summary Generator
PR 評論摘要生成器
Generates formatted summary comments for Pull Requests based on
gl-platform.governance pipeline results.
Usage:
    python tools/docs/pr_comment_summary.py --run-id 12345
    python tools/docs/pr_comment_summary.py --input results.json
"""
# MNGA-002: Import organization needs review
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)
def generate_stage_summary(results: dict[str, Any]) -> str:
    """Generate a markdown table summarizing stage results."""
    stages = [
        ("1", "Lint", "lint"),
        ("2", "Format", "format"),
        ("3", "Schema", "schema"),
        ("4", "Vector Test", "vector"),
        ("5", "Policy Gate", "policy"),
        ("6", "SBOM", "sbom"),
        ("7", "Provenance", "provenance"),
        ("8", "Audit", "audit"),
    ]
    lines = [
        "| Stage | Name | Status | Duration |",
        "|-------|------|--------|----------|",
    ]
    for num, name, key in stages:
        stage_result = results.get(key, {})
        status = stage_result.get("status", "unknown")
        duration = stage_result.get("duration_ms", 0)
        if status == "success":
            status_icon = "✅"
        elif status == "failure":
            status_icon = "❌"
        elif status == "skipped":
            status_icon = "⏭️"
        else:
            status_icon = "⚪"
        duration_str = f"{duration}ms" if duration else "-"
        lines.append(f"| {num} | {name} | {status_icon} | {duration_str} |")
    return "\n".join(lines)
def generate_validation_details(results: dict[str, Any]) -> str:
    """Generate details about validation results."""
    details = []
    schema_result = results.get("schema", {})
    if schema_result:
        items = schema_result.get("items_validated", 0)
        errors = schema_result.get("errors", [])
        details.append("### Schema Validation")
        details.append(f"- Items validated: {items}")
        if errors:
            details.append(f"- Errors found: {len(errors)}")
            details.append("```")
            for error in errors[:5]:  # Show max 5 errors
                details.append(f"  • {error}")
            if len(errors) > 5:
                details.append(f"  ... and {len(errors) - 5} more")
            details.append("```")
    policy_result = results.get("policy", {})
    if policy_result:
        violations = policy_result.get("violations", [])
        details.append("### Policy Gate")
        if violations:
            details.append(f"- Violations: {len(violations)}")
            for v in violations[:3]:
                details.append(f"  - {v}")
        else:
            details.append("- No policy violations")
    return "\n".join(details) if details else ""
def generate_supply_chain_status(results: dict[str, Any]) -> str:
    """Generate supply chain status summary."""
    lines = ["### Supply Chain Status"]
    sbom = results.get("sbom", {})
    if sbom.get("generated"):
        lines.append(f"- 📦 SBOM: Generated ({sbom.get('packages', 0)} packages)")
    elif sbom.get("exists"):
        lines.append(f"- 📦 SBOM: Exists ({sbom.get('packages', 0)} packages)")
    else:
        lines.append("- 📦 SBOM: Not available")
    provenance = results.get("provenance", {})
    if provenance.get("generated"):
        lines.append("- 🔏 Provenance: Generated")
        if provenance.get("digest"):
            lines.append(f"  - Digest: `{provenance['digest'][:16]}...`")
    elif provenance.get("exists"):
        lines.append("- 🔏 Provenance: Exists")
    else:
        lines.append("- 🔏 Provenance: Not available")
    return "\n".join(lines)
def generate_pr_comment(results: dict[str, Any], context: dict[str, str] = None) -> str:
    """Generate the full PR comment markdown."""
    if context is None:
        context = {}
    # Determine overall status
    all_passed = all(
        results.get(stage, {}).get("status") in ("success", "skipped")
        for stage in [
            "lint",
            "format",
            "schema",
            "vector",
            "policy",
            "sbom",
            "provenance",
            "audit",
        ]
    )
    header_emoji = "✅" if all_passed else "⚠️"
    header_text = (
        "Governance Pipeline Passed"
        if all_passed
        else "Governance Pipeline Issues Found"
    )
    sections = [
        f"## {header_emoji} {header_text}",
        "",
        generate_stage_summary(results),
        "",
    ]
    # Add details if there are issues
    details = generate_validation_details(results)
    if details:
        sections.append(details)
        sections.append("")
    # Add supply chain status
    sections.append(generate_supply_chain_status(results))
    sections.append("")
    # Add context info
    if context:
        sections.append("---")
        sections.append("<details>")
        sections.append("<summary>Pipeline Details</summary>")
        sections.append("")
        if context.get("commit"):
            sections.append(f"- **Commit:** `{context['commit']}`")
        if context.get("actor"):
            sections.append(f"- **Actor:** @{context['actor']}")
        if context.get("run_id"):
            sections.append(
                f"- **Run ID:** [{context['run_id']}](https://github.com/{context.get('repository', '')}/actions/runs/{context['run_id']})"
            )
        if context.get("timestamp"):
            sections.append(f"- **Timestamp:** {context['timestamp']}")
        sections.append("")
        sections.append("</details>")
    # Add footer
    sections.append("")
    if all_passed:
        sections.append("*All gl-platform.governance checks completed successfully.* 🎉")
    else:
        sections.append(
            "*Please review the issues above and address them before merging.*"
        )
    return "\n".join(sections)
def load_results_from_file(file_path: Path) -> dict[str, Any]:
    """Load results from a JSON or YAML file."""
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.suffix in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        else:
            return json.load(f)
def main():
    parser = argparse.ArgumentParser(description="Generate PR comment summary")
    parser.add_argument("--input", "-i", help="Input results file (JSON/YAML)")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format",
    )
    args = parser.parse_args()
    # Get context from environment
    context = {
        "commit": os.environ.get("GITHUB_SHA", ""),
        "actor": os.environ.get("GITHUB_ACTOR", ""),
        "repository": os.environ.get("GITHUB_REPOSITORY", ""),
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    # Load or generate results
    if args.input:
        results = load_results_from_file(Path(args.input))
    else:
        # Generate sample results for demo
        results = {
            "lint": {"status": "success", "duration_ms": 500},
            "format": {"status": "success", "duration_ms": 200},
            "schema": {
                "status": "success",
                "duration_ms": 1500,
                "items_validated": 30,
                "errors": [],
            },
            "vector": {"status": "success", "duration_ms": 800},
            "policy": {"status": "success", "duration_ms": 600, "violations": []},
            "sbom": {
                "status": "success",
                "duration_ms": 1000,
                "exists": True,
                "packages": 35,
            },
            "provenance": {
                "status": "success",
                "duration_ms": 500,
                "exists": True,
                "digest": "abc123def456789",
            },
            "audit": {"status": "success", "duration_ms": 300},
        }
    # Generate output
    if args.format == "json":
        output = json.dumps(
            {
                "comment": generate_pr_comment(results, context),
                "results": results,
                "context": context,
            },
            indent=2,
        )
    else:
        output = generate_pr_comment(results, context)
    # Write output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"✅ Output written to: {args.output}")
    else:
        print(output)
    return 0
if __name__ == "__main__":
    sys.exit(main())
