# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: map-violations-to-playbooks
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Map Language Governance Violations to Refactor Playbooks
將語言治理違規映射到重構劇本
Maps language gl-platform.governance violations to corresponding refactor playbooks
for display in CI/CD workflows and dashboards.
"""
# MNGA-002: Import organization needs review
import argparse
import json
from collections import defaultdict
from pathlib import Path
import yaml
def load_index(index_path: Path) -> dict:
    """Load index.yaml"""
    with open(index_path) as f:
        return yaml.safe_load(f)
def load_hotspot_data(hotspot_path: Path) -> list:
    """Load hotspot.json data"""
    try:
        with open(hotspot_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
def map_file_to_cluster(file_path: str, clusters: list) -> dict | None:
    """
    Find which cluster a file belongs to based on directory prefix.
    Examples:
        core/legacy_module/old_api.php -> core/architecture-stability
        services/gateway/router.lua -> services/gateway
        automation/autonomous/flight_controller.lua -> automation/autonomous
    """
    for cluster in clusters:
        cluster["cluster_id"]
        domain = cluster["domain"]
        # Match file path to cluster domain
        if file_path.startswith(domain + "/"):
            return cluster
    return None
def generate_summary(hotspot_data: list, index: dict, repo_root: Path) -> str:
    """Generate markdown summary linking violations to playbooks"""
    clusters_data = index.get("clusters", [])
    summary = ["# 語言治理違規與重構劇本對應\n"]
    summary.append("> Language Governance Violations Mapped to Refactor Playbooks\n")
    summary.append("---\n\n")
    # Group violations by cluster
    cluster_violations = defaultdict(lambda: {"cluster": None, "violations": []})
    for violation in hotspot_data:
        file_path = violation.get("file", "")
        cluster = map_file_to_cluster(file_path, clusters_data)
        if cluster:
            cluster_id = cluster["cluster_id"]
            if cluster_violations[cluster_id]["cluster"] is None:
                cluster_violations[cluster_id]["cluster"] = cluster
            cluster_violations[cluster_id]["violations"].append(violation)
    # Generate summary for each cluster
    if not cluster_violations:
        summary.append("✅ **沒有發現語言治理違規** (No violations found)\n")
        return "".join(summary)
    summary.append("## 📊 總覽 (Overview)\n\n")
    summary.append(f"- **Total Clusters Affected**: {len(cluster_violations)}\n")
    summary.append(
        f"- **Total Violations**: {sum(len(data['violations']) for data in cluster_violations.values())}\n"
    )
    summary.append("\n---\n\n")
    for cluster_id, data in sorted(cluster_violations.items()):
        cluster = data["cluster"]
        violations = data["violations"]
        # Severity breakdown
        severity_counts = defaultdict(int)
        for v in violations:
            severity_counts[v.get("severity", "UNKNOWN")] += 1
        summary.append(f"## 📁 {cluster_id}\n\n")
        # Link to refactor playbook
        refactor_file = cluster.get("refactor_file", "")
        if refactor_file:
            playbook_link = f"docs/refactor_playbooks/03_refactor/{refactor_file}"
            summary.append(
                f"**🔗 重構劇本 (Refactor Playbook)**: [{refactor_file}]({playbook_link})\n\n"
            )
        # Severity breakdown
        summary.append("**嚴重性分佈 (Severity Breakdown)**:\n")
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            count = severity_counts.get(severity, 0)
            if count > 0:
                emoji = {
                    "CRITICAL": "🔴",
                    "HIGH": "🟠",
                    "MEDIUM": "🟡",
                    "LOW": "🟢",
                }.get(severity, "⚪")
                summary.append(f"- {emoji} {severity}: {count}\n")
        summary.append("\n")
        # Violation list (show top 5)
        summary.append("**違規清單 (Violations)**:\n\n")
        for i, v in enumerate(violations[:5], 1):
            file_path = v.get("file", "unknown")
            score = v.get("score", 0)
            reason = v.get("reason", "No reason provided")
            severity = v.get("severity", "UNKNOWN")
            emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(
                severity, "⚪"
            )
            summary.append(f"{i}. {emoji} **`{file_path}`** (score: {score})\n")
            summary.append(f"   - {reason}\n")
        if len(violations) > 5:
            summary.append(
                f"\n_...以及 {len(violations) - 5} 個其他違規 (and {len(violations) - 5} more violations)_\n"
            )
        summary.append("\n---\n\n")
    # Add footer with next steps
    summary.append("## 🚀 下一步 (Next Steps)\n\n")
    summary.append("1. 查看對應的重構劇本 (Review corresponding refactor playbooks)\n")
    summary.append("2. 執行 P0 優先級任務 (Execute P0 priority tasks)\n")
    summary.append(
        "3. 使用 Auto-Fix Bot 自動修復適合的項目 (Use Auto-Fix Bot for eligible items)\n"
    )
    summary.append(
        "4. 人工審查需要審查的變更 (Manual review for changes requiring human oversight)\n\n"
    )
    return "".join(summary)
def main():
    parser = argparse.ArgumentParser(
        description="Map language gl-platform.governance violations to refactor playbooks"
    )
    parser.add_argument(
        "--hotspot",
        type=str,
        default="apps/web/public/data/hotspot.json",
        help="Path to hotspot.json file (relative to repo root)",
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
        default="gl-platform.governance-summary.md",
        help="Output markdown file path",
    )
    parser.add_argument(
        "--repo-root", type=str, default=".", help="Repository root directory"
    )
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    hotspot_path = repo_root / args.hotspot
    index_path = repo_root / args.index
    output_path = repo_root / args.output
    # Load data
    print(f"📂 Loading hotspot data from: {hotspot_path}")
    hotspot_data = load_hotspot_data(hotspot_path)
    print(f"   Found {len(hotspot_data)} violations")
    print(f"📂 Loading index from: {index_path}")
    index = load_index(index_path)
    print(f"   Found {len(index.get('clusters', []))} clusters")
    # Generate summary
    print("🔄 Generating summary...")
    summary = generate_summary(hotspot_data, index, repo_root)
    # Write output
    print(f"💾 Writing output to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"✅ Done! Summary written to {output_path}")
if __name__ == "__main__":
    main()
