# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: load-playbook
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Load Refactor Playbook and Extract Auto-Fix Context
載入重構劇本並提取 Auto-Fix 上下文
Loads a refactor playbook for a specific cluster and extracts the auto-fix
context for use by Auto-Fix Bot and other automation tools.
"""
# MNGA-002: Import organization needs review
import argparse
import json
import re
from pathlib import Path
import yaml
def load_playbook(cluster_id: str, index_path: Path, repo_root: Path) -> dict:
    """
    Load refactor playbook for a cluster and extract key sections.
    Returns a dictionary with:
    - cluster_id
    - playbook_path
    - auto_fix_allowed: list of items that can be auto-fixed
    - manual_review_required: list of items requiring manual review
    - p0_actions: P0 priority actions
    - p1_actions: P1 priority actions
    - p2_actions: P2 priority actions
    """
    # Load index
    with open(index_path) as f:
        index = yaml.safe_load(f)
    # Find cluster in index
    cluster = None
    for c in index.get("clusters", []):
        if c["cluster_id"] == cluster_id:
            cluster = c
            break
    if not cluster:
        raise ValueError(f"Cluster '{cluster_id}' not found in index")
    # Load playbook markdown
    refactor_file = cluster.get("refactor_file")
    if not refactor_file:
        raise ValueError(f"No refactor_file specified for cluster '{cluster_id}'")
    playbook_path = (
        repo_root / "docs" / "refactor_playbooks" / "03_refactor" / refactor_file
    )
    if not playbook_path.exists():
        raise FileNotFoundError(f"Playbook not found: {playbook_path}")
    with open(playbook_path, encoding="utf-8") as f:
        content = f.read()
    # Extract sections
    result = {
        "cluster_id": cluster_id,
        "playbook_path": str(playbook_path.relative_to(repo_root)),
        "status": cluster.get("status", "unknown"),
        "domain": cluster.get("domain", ""),
        "auto_fix_allowed": extract_auto_fix_allowed(content),
        "manual_review_required": extract_manual_review(content),
        "p0_actions": extract_priority_actions(content, "P0"),
        "p1_actions": extract_priority_actions(content, "P1"),
        "p2_actions": extract_priority_actions(content, "P2"),
        "acceptance_criteria": extract_acceptance_criteria(content),
    }
    return result
def extract_section(content: str, heading_pattern: str) -> str:
    """
    Extract content under a specific heading.
    Args:
        content: Full markdown content
        heading_pattern: Regex pattern to match heading (e.g., "P0", "Auto-Fix")
    Returns:
        Content under the heading until next heading of same or higher level
    """
    # Match heading and capture content until next ## or #
    pattern = rf"##+ .*?{heading_pattern}.*?\n(.*?)(?=\n##+ |\Z)"
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""
def extract_list_items(text: str) -> list[str]:
    """Extract list items from markdown text"""
    items = []
    for line in text.split("\n"):
        line = line.strip()
        # Match bullet points: -, *, or numbered lists
        if re.match(r"^[-*]\s+", line) or re.match(r"^\d+\.\s+", line):
            # Remove bullet/number and clean up
            item = re.sub(r"^[-*]\s+", "", line)
            item = re.sub(r"^\d+\.\s+", "", item)
            if item:
                items.append(item)
    return items
def extract_auto_fix_allowed(content: str) -> list[str]:
    """Extract items that can be auto-fixed"""
    section = extract_section(content, r"Auto-Fix Bot.*可以處理")
    # Look for subsection "適合 Auto-Fix"
    allowed_section = re.search(
        r"適合.*Auto-Fix.*?[:：](.*?)(?=必須人工審查|##|\Z)",
        section,
        re.DOTALL | re.IGNORECASE,
    )
    if allowed_section:
        return extract_list_items(allowed_section.group(1))
    # Fallback: extract all items from the section
    return extract_list_items(section)
def extract_manual_review(content: str) -> list[str]:
    """Extract items requiring manual review"""
    section = extract_section(content, r"Auto-Fix Bot.*可以處理")
    # Look for subsection "必須人工審查"
    manual_section = re.search(
        r"必須人工審查.*?[:：](.*?)(?=##|\Z)", section, re.DOTALL | re.IGNORECASE
    )
    if manual_section:
        return extract_list_items(manual_section.group(1))
    return []
def extract_priority_actions(content: str, priority: str) -> dict[str, any]:
    """
    Extract actions for a specific priority level (P0, P1, P2).
    Returns:
        Dictionary with 'objective', 'actions', and 'acceptance_criteria'
    """
    section = extract_section(content, priority)
    if not section:
        return {"objective": "", "actions": [], "acceptance_criteria": []}
    # Extract objective (first paragraph or line after heading)
    objective_match = re.search(
        r"目標[:：](.*?)(?=\n[-*]|\n行動|\Z)", section, re.DOTALL
    )
    objective = objective_match.group(1).strip() if objective_match else ""
    # Extract actions (list items under "行動項目")
    actions_section = re.search(
        r"行動項目.*?[:：](.*?)(?=驗收條件|##|\Z)", section, re.DOTALL
    )
    actions = []
    if actions_section:
        actions = extract_list_items(actions_section.group(1))
    # Extract acceptance criteria
    criteria_section = re.search(r"驗收條件.*?[:：](.*?)(?=##|\Z)", section, re.DOTALL)
    criteria = []
    if criteria_section:
        criteria = extract_list_items(criteria_section.group(1))
    return {"objective": objective, "actions": actions, "acceptance_criteria": criteria}
def extract_acceptance_criteria(content: str) -> dict[str, any]:
    """Extract overall acceptance criteria from the playbook"""
    section = extract_section(content, r"驗收條件與成功指標")
    # Extract metrics tables if present
    language_metrics = []
    security_metrics = []
    architecture_metrics = []
    # Look for table patterns
    table_pattern = r"\|.*?\|.*?\|.*?\|"
    re.findall(table_pattern + r"(?:\n\|.*?\|.*?\|.*?\|)+", section)
    # Also extract bullet points
    metrics_list = extract_list_items(section)
    return {
        "language_metrics": language_metrics,
        "security_metrics": security_metrics,
        "architecture_metrics": architecture_metrics,
        "other_criteria": metrics_list,
    }
def main():
    parser = argparse.ArgumentParser(
        description="Load refactor playbook and extract auto-fix context"
    )
    parser.add_argument(
        "--cluster",
        type=str,
        required=True,
        help="Cluster ID (e.g., core/architecture-stability)",
    )
    parser.add_argument(
        "--index",
        type=str,
        default="docs/refactor_playbooks/03_refactor/index.yaml",
        help="Path to index.yaml file (relative to repo root)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="playbook-context.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--repo-root", type=str, default=".", help="Repository root directory"
    )
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty print JSON output"
    )
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    index_path = repo_root / args.index
    output_path = repo_root / args.output
    try:
        print(f"📂 Loading playbook for cluster: {args.cluster}")
        context = load_playbook(args.cluster, index_path, repo_root)
        print("✅ Playbook loaded successfully")
        print(f"   - Status: {context['status']}")
        print(f"   - Domain: {context['domain']}")
        print(f"   - Auto-fix items: {len(context['auto_fix_allowed'])}")
        print(f"   - Manual review items: {len(context['manual_review_required'])}")
        print(f"   - P0 actions: {len(context['p0_actions']['actions'])}")
        print(f"   - P1 actions: {len(context['p1_actions']['actions'])}")
        print(f"   - P2 actions: {len(context['p2_actions']['actions'])}")
        # Write output
        print(f"💾 Writing output to: {output_path}")
        indent = 2 if args.pretty else None
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(context, f, indent=indent, ensure_ascii=False)
        print(f"✅ Done! Context written to {output_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    return 0
if __name__ == "__main__":
    exit(main())
