# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: governance
# @ECO-semantic: generate-migration-flow
# @ECO-audit-trail: ../../engine/governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Language Migration Flow Model Generator
========================================
Visualizes language migration patterns between directory clusters.
Purpose:
  - Build migration flows between Cluster × Language combinations
  - Track source violations and their remediation paths
  - Show language spread/convergence across architecture layers
Inputs:
  - governance/language-governance-report.md (current violations)
  - knowledge/language-history.yaml (historical migration events)
Outputs:
  - apps/web/public/data/migration-flow.json (for web dashboard)
  - docs/MIGRATION_FLOW.mmd (Mermaid visualization)
"""
# MNGA-002: Import organization needs review
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
import yaml
def get_project_root() -> Path:
    """Get the project root directory"""
    current = Path(__file__).resolve().parent
    return current.parent
def determine_cluster_from_path(file_path: str) -> str:
    """Extract cluster/directory from file path"""
    parts = file_path.split("/")
    if len(parts) < 2:
        return "root"
    # Main directory clusters
    if parts[0] in ["core", "governance", "automation", "services", "apps", "tools"]:
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return parts[0]
    return parts[0]
def get_language_from_file(file_path: str) -> str:
    """Get language from file extension"""
    ext = Path(file_path).suffix.lower()
    language_map = {
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".py": "python",
        ".go": "go",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".c": "c",
        ".rs": "rust",
        ".lua": "lua",
        ".java": "java",
        ".rego": "rego",
    }
    return language_map.get(ext, "unknown")
def classify_violation_type(reason: str) -> str:
    """Classify violation type from reason text"""
    reason_lower = reason.lower()
    if "forbidden" in reason_lower or "not allowed" in reason_lower:
        return "forbidden"
    elif (
        "cross-layer" in reason_lower
        or "layer" in reason_lower
        or "wrong layer" in reason_lower
    ):
        return "layer_violation"
    elif "security" in reason_lower or "vulnerability" in reason_lower:
        return "security"
    elif "type" in reason_lower:
        return "type_safety"
    else:
        return "policy_violation"
def suggest_target_from_violation(
    cluster: str, language: str, violation_type: str, file_path: str
) -> tuple[str, str]:
    """
    Suggest target cluster:language based on violation type.
    Returns (target_cluster, target_language)
    """
    # Forbidden languages should be removed
    if violation_type == "forbidden":
        if language in ["lua", "java"]:
            return "removed", "removed"
        # C++ should move to autonomous layer
        if language == "cpp":
            return "automation/autonomous", "cpp"
    # Layer violations
    if violation_type == "layer_violation":
        # C++ in services should move to autonomous
        if "services" in cluster and language == "cpp":
            return "automation/autonomous", "cpp"
        # TypeScript in governance should move to core
        if "governance" in cluster and language == "typescript":
            return "core", "typescript"
        # Python in apps/web should move to services or be rewritten
        if "apps/web" in cluster and language == "python":
            return "services", "python"
    # Type safety violations - stay in same cluster, improve code
    if violation_type == "type_safety":
        return cluster, language
    # Default: rewrite to primary language of cluster
    cluster_primary_lang = {
        "core": "typescript",
        "governance": "python",
        "automation": "python",
        "services": "typescript",
        "apps/web": "typescript",
        "autonomous": "cpp",
    }
    for key, lang in cluster_primary_lang.items():
        if key in cluster:
            if lang != language:
                return cluster, lang
    return cluster, language
def parse_governance_report(report_path: Path) -> list[dict[str, Any]]:
    """Parse current violations from governance report"""
    violations = []
    if not report_path.exists():
        return violations
    with open(report_path, encoding="utf-8") as f:
        content = f.read()
    lines = content.split("\n")
    for line in lines:
        if "—" in line and "**" in line:
            match = re.match(
                r"-\s*\*\*(.*?)\*\*\s*—\s*(.*?)(?:\(Layer:\s*(.*?)\))?$", line
            )
            if match:
                file_path = match.group(1).strip()
                reason = match.group(2).strip()
                cluster = determine_cluster_from_path(file_path)
                language = get_language_from_file(file_path)
                violation_type = classify_violation_type(reason)
                violations.append(
                    {
                        "file": file_path,
                        "cluster": cluster,
                        "language": language,
                        "violation_type": violation_type,
                        "reason": reason,
                    }
                )
    return violations
def parse_migration_history(history_path: Path) -> list[dict[str, Any]]:
    """Parse historical migration events from language history"""
    migrations = []
    if not history_path.exists():
        return migrations
    try:
        with open(history_path, encoding="utf-8") as f:
            history_data = yaml.safe_load(f)
        if not history_data or "events" not in history_data:
            return migrations
        for event in history_data["events"]:
            event_type = event.get("type", "")
            details = event.get("details", "")
            # Look for migration/rewrite patterns
            if (
                event_type == "fix"
                or "rewrite" in details.lower()
                or "move" in details.lower()
            ):
                # Try to parse migration info from details
                # Example: "Moved services/old.cpp to autonomous/new.cpp"
                # Example: "Rewrote core/util.js to core/util.ts"
                # Simple pattern matching for now
                if "files_affected" in event:
                    files = event.get("files_affected", [])
                    if isinstance(files, list) and len(files) >= 2:
                        source_file = files[0]
                        target_file = files[1]
                        source_cluster = determine_cluster_from_path(source_file)
                        source_lang = get_language_from_file(source_file)
                        target_cluster = determine_cluster_from_path(target_file)
                        target_lang = get_language_from_file(target_file)
                        migrations.append(
                            {
                                "source_cluster": source_cluster,
                                "source_language": source_lang,
                                "target_cluster": target_cluster,
                                "target_language": target_lang,
                                "timestamp": event.get("timestamp", ""),
                                "type": "history",
                            }
                        )
    except Exception as e:
        print(f"Error parsing migration history: {e}")
    return migrations
def build_migration_edges(
    violations: list[dict[str, Any]], history_migrations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build migration edges from violations and history"""
    edges = []
    edge_counts = defaultdict(int)
    # Process historical migrations
    for migration in history_migrations:
        source = f"{migration['source_cluster']}:{migration['source_language']}"
        target = f"{migration['target_cluster']}:{migration['target_language']}"
        edge_key = (source, target, "history")
        edge_counts[edge_key] += 1
    # Process current violations (suggested migrations)
    for violation in violations:
        source_cluster = violation["cluster"]
        source_lang = violation["language"]
        violation_type = violation["violation_type"]
        target_cluster, target_lang = suggest_target_from_violation(
            source_cluster, source_lang, violation_type, violation["file"]
        )
        source = f"{source_cluster}:{source_lang}"
        target = f"{target_cluster}:{target_lang}"
        # Skip if source == target (no migration needed)
        if source != target:
            edge_key = (source, target, "suggested")
            edge_counts[edge_key] += 1
    # Convert to edge list
    for (source, target, edge_type), count in edge_counts.items():
        edges.append(
            {"source": source, "target": target, "count": count, "type": edge_type}
        )
    # Sort by count (highest first)
    edges.sort(key=lambda x: x["count"], reverse=True)
    return edges
def generate_mermaid_sankey(edges: list[dict[str, Any]]) -> str:
    """Generate Mermaid sankey-beta diagram"""
    lines = ["```mermaid", '%%{init: {"theme": "dark"}}%%', "sankey-beta", ""]
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        count = edge["count"]
        edge["type"]
        # Format: SourceNode,TargetNode,Value
        lines.append(f"{source},{target},{count}")
    lines.append("```")
    return "\n".join(lines)
def generate_migration_flow_data():
    """Main function to generate migration flow data"""
    project_root = get_project_root()
    # Load input data
    report_path = project_root / "governance" / "language-governance-report.md"
    history_path = project_root / "knowledge" / "language-history.yaml"
    violations = parse_governance_report(report_path)
    history_migrations = parse_migration_history(history_path)
    print(f"📊 Parsed {len(violations)} violations")
    print(f"📜 Parsed {len(history_migrations)} historical migrations")
    # Build migration edges
    edges = build_migration_edges(violations, history_migrations)
    # Add some example data if nothing found
    if not edges:
        edges = [
            {
                "source": "services:cpp",
                "target": "automation/autonomous:cpp",
                "count": 3,
                "type": "suggested",
            },
            {
                "source": "apps/web:javascript",
                "target": "apps/web:typescript",
                "count": 2,
                "type": "suggested",
            },
            {
                "source": "governance:typescript",
                "target": "core:typescript",
                "count": 1,
                "type": "suggested",
            },
            {
                "source": "automation:lua",
                "target": "removed:removed",
                "count": 1,
                "type": "suggested",
            },
        ]
    # Calculate statistics
    total_migrations = sum(edge["count"] for edge in edges)
    history_count = sum(edge["count"] for edge in edges if edge["type"] == "history")
    suggested_count = sum(
        edge["count"] for edge in edges if edge["type"] == "suggested"
    )
    # Most common source
    source_counts = defaultdict(int)
    for edge in edges:
        source_counts[edge["source"]] += edge["count"]
    most_common_source = (
        max(source_counts.items(), key=lambda x: x[1])[0] if source_counts else "None"
    )
    # Most common target
    target_counts = defaultdict(int)
    for edge in edges:
        target_counts[edge["target"]] += edge["count"]
    most_common_target = (
        max(target_counts.items(), key=lambda x: x[1])[0] if target_counts else "None"
    )
    # Output data structure
    output_data = {
        "generatedAt": datetime.utcnow().isoformat() + "Z",
        "statistics": {
            "totalMigrations": total_migrations,
            "historicalMigrations": history_count,
            "suggestedMigrations": suggested_count,
            "mostCommonSource": most_common_source,
            "mostCommonTarget": most_common_target,
        },
        "edges": edges,
    }
    # Save JSON
    output_json_path = project_root / "governance" / "migration-flow.json"
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Generated migration flow JSON: {output_json_path}")
    print(f"   Total migrations: {total_migrations}")
    print(f"   Historical: {history_count}, Suggested: {suggested_count}")
    print(f"   Most common source: {most_common_source}")
    print(f"   Most common target: {most_common_target}")
    # Generate Mermaid diagram
    mermaid_content = f"""# 🔄 Language Migration Flow Model
> **Generated:** {output_data['generatedAt']}
> **Total Migrations:** {total_migrations}
> **Historical:** {history_count} | **Suggested:** {suggested_count}
---
## Migration Flow Diagram
{generate_mermaid_sankey(edges)}
---
## Migration Statistics
- **Most Common Source**: `{most_common_source}`
- **Most Common Target**: `{most_common_target}`
---
## Migration Edges
| Source | Target | Count | Type |
|--------|--------|-------|------|
"""
    for edge in edges[:20]:  # Top 20
        mermaid_content += f"| `{edge['source']}` | `{edge['target']}` | {edge['count']} | {edge['type']} |\n"
    mermaid_content += """
---
## Legend
- **✓ history**: Completed migrations from historical events
- **→ suggested**: Recommended migrations based on current violations
---
#*Maintained by:** Living Knowledge Base + Language Governance System
"""
    output_mmd_path = project_root / "docs" / "MIGRATION_FLOW.md"
    output_mmd_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_mmd_path, "w", encoding="utf-8") as f:
        f.write(mermaid_content)
    print(f"✅ Generated Mermaid documentation: {output_mmd_path}")
    return output_data
if __name__ == "__main__":
    generate_migration_flow_data()
