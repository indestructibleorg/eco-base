# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: generate-sankey-data
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Generate Sankey Diagram Data for Language Governance Dashboard
This script analyzes the language gl-platform.governance report and generates
Sankey diagram data showing language flow violations.
Usage:
    python tools/generate-sankey-data.py
Output:
    gl-platform.governance/sankey-data.json
"""
# MNGA-002: Import organization needs review
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
def get_project_root() -> Path:
    """Get the project root directory"""
    current = Path(__file__).resolve().parent
    return current.parent
def parse_governance_report(report_path: Path) -> list[dict[str, Any]]:
    """Parse violations from gl-platform.governance report"""
    violations = []
    if not report_path.exists():
        return violations
    with open(report_path, encoding="utf-8") as f:
        content = f.read()
    # Parse violations section
    lines = content.split("\n")
    for line in lines:
        if "—" in line and "**" in line:
            # Extract file path and reason
            match = re.match(
                r"-\s*\*\*(.*?)\*\*\s*—\s*(.*?)(?:\(Layer:\s*(.*?)\))?$", line
            )
            if match:
                file_path = match.group(1).strip()
                reason = match.group(2).strip()
                match.group(3).strip() if match.group(3) else "Unknown"
                # Determine source layer from file path
                source_layer = determine_layer(file_path)
                # Extract language and violation type
                language, violation_type = extract_language_and_violation(
                    file_path, reason
                )
                # Determine fix target
                fix_target = determine_fix_target(
                    language, violation_type, source_layer
                )
                violations.append(
                    {
                        "sourceLayer": source_layer,
                        "language": language,
                        "violationType": violation_type,
                        "fixTarget": fix_target,
                        "file": file_path,
                        "reason": reason,
                        "count": 1,
                    }
                )
    return violations
def determine_layer(file_path: str) -> str:
    """Determine layer from file path"""
    path_lower = file_path.lower()
    if "apps/web" in path_lower or "frontend" in path_lower:
        return "L5: Applications"
    elif "services" in path_lower:
        return "L4: Services"
    elif "automation" in path_lower or "ai" in path_lower:
        return "L3: AI/Automation"
    elif "gl-platform.governance" in path_lower:
        return "L2: Governance"
    elif "core" in path_lower:
        return "L1: Core Engine"
    elif "autonomous" in path_lower or "ros" in path_lower:
        return "L0: OS/Hardware"
    else:
        return "Unknown Layer"
def extract_language_and_violation(file_path: str, reason: str) -> tuple:
    """Extract language and violation type from file and reason"""
    # Determine language from file extension
    ext = Path(file_path).suffix.lower()
    language_map = {
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".tsx": "TypeScript",
        ".py": "Python",
        ".go": "Go",
        ".cpp": "C++",
        ".cc": "C++",
        ".c": "C",
        ".rs": "Rust",
        ".lua": "Lua",
        ".java": "Java",
        ".rb": "Ruby",
        ".php": "PHP",
    }
    language = language_map.get(ext, "Unknown")
    # Determine violation type from reason
    reason_lower = reason.lower()
    if "forbidden" in reason_lower or "not allowed" in reason_lower:
        violation_type = "Forbidden Language"
    elif "cross-layer" in reason_lower or "wrong layer" in reason_lower:
        violation_type = "Layer Violation"
    elif "type hint" in reason_lower or "type safety" in reason_lower:
        violation_type = "Type Safety"
    elif "security" in reason_lower or "vulnerability" in reason_lower:
        violation_type = "Security Issue"
    else:
        violation_type = "Policy Violation"
    return language, violation_type
def determine_fix_target(language: str, violation_type: str, source_layer: str) -> str:
    """Determine fix target based on violation"""
    if violation_type == "Forbidden Language":
        if language in ["JavaScript", "Lua", "Ruby", "PHP"]:
            return "Remove or Rewrite"
        else:
            return "Rewrite to Allowed Language"
    elif violation_type == "Layer Violation":
        return "Move to Correct Layer"
    elif violation_type == "Type Safety":
        return "Add Type Annotations"
    elif violation_type == "Security Issue":
        return "Fix Security Vulnerability"
    else:
        return "Apply Policy Fix"
def aggregate_flows(violations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate duplicate flows"""
    flow_map = {}
    for v in violations:
        key = (
            f"{v['sourceLayer']}|{v['language']}|{v['violationType']}|{v['fixTarget']}"
        )
        if key in flow_map:
            flow_map[key]["count"] += 1
        else:
            flow_map[key] = v.copy()
    return list(flow_map.values())
def generate_sankey_data():
    """Generate Sankey diagram data"""
    project_root = get_project_root()
    # Parse gl-platform.governance report
    report_path = project_root / "gl-platform.governance" / "language-gl-platform.governance-report.md"
    violations = parse_gl-platform.governance_report(report_path)
    # Add some example flows if no violations found
    if not violations:
        violations = [
            {
                "sourceLayer": "L5: Applications",
                "language": "JavaScript",
                "violationType": "Forbidden Language",
                "fixTarget": "Rewrite to TypeScript",
                "file": "apps/web/src/legacy-code.js",
                "reason": "JavaScript file in TypeScript project",
                "count": 1,
            },
            {
                "sourceLayer": "L1: Core Engine",
                "language": "Python",
                "violationType": "Type Safety",
                "fixTarget": "Add Type Hints",
                "file": "core/engine/utils.py",
                "reason": "Python file needs type hints",
                "count": 1,
            },
            {
                "sourceLayer": "L4: Services",
                "language": "C++",
                "violationType": "Layer Violation",
                "fixTarget": "Move to L0: Hardware",
                "file": "services/legacy/driver.cpp",
                "reason": "C++ code should be in hardware layer",
                "count": 1,
            },
            {
                "sourceLayer": "L3: AI/Automation",
                "language": "Lua",
                "violationType": "Forbidden Language",
                "fixTarget": "Remove or Rewrite",
                "file": "automation/scripts/config.lua",
                "reason": "Lua not allowed in automation layer",
                "count": 1,
            },
        ]
    # Aggregate flows
    aggregated = aggregate_flows(violations)
    # Generate output
    output = {
        "generatedAt": datetime.utcnow().isoformat() + "Z",
        "totalFlows": len(aggregated),
        "totalViolations": sum(v["count"] for v in aggregated),
        "flows": aggregated,
    }
    # Save to file
    output_path = project_root / "gl-platform.governance" / "sankey-data.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"✅ Generated Sankey data: {output_path}")
    print(f"   Total flows: {output['totalFlows']}")
    print(f"   Total violations: {output['totalViolations']}")
    return output
if __name__ == "__main__":
    generate_sankey_data()
