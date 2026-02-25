# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: lkb-update
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Living Knowledge Base Update Module
------------------------------------
在 AI Auto-Fix 完成後自動更新：
1. docs/KNOWLEDGE_HEALTH.md
2. docs/knowledge-graph.yaml
3. docs/knowledge-health-report.yaml
並觸發 knowledge_cycle.action 階段
使用：
    python tools/lkb-update.py \
        --event "auto-fix" \
        --description "AI Auto-Fix Bot applied repository repairs"
"""
# MNGA-002: Import organization needs review
import argparse
import datetime
import os
from typing import Any
import yaml
from rich.console import Console
console = Console()
def load_yaml(path: str) -> dict[str, Any]:
    """載入 YAML 檔案"""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        try:
            return yaml.safe_load(f) or {}
        except Exception as e:
            console.print(f"[red]Error loading {path}: {e}[/red]")
            return {}
def save_yaml(path: str, data: dict[str, Any]):
    """儲存 YAML 檔案"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            data, f, default_flow_style=False, allow_unicode=True, sort_keys=False
        )
def update_knowledge_health_report(
    event: str, description: str, metrics: dict[str, Any] = None
):
    """更新知識庫健康報告"""
    report_path = "docs/knowledge-health-report.yaml"
    now = datetime.datetime.utcnow().isoformat() + "Z"
    # 載入現有報告
    report = load_yaml(report_path)
    # 初始化結構
    if "knowledge_events" not in report:
        report["knowledge_events"] = []
    if "last_updated" not in report:
        report["last_updated"] = now
    if "health_metrics" not in report:
        report["health_metrics"] = {}
    # 新增事件
    knowledge_event = {
        "timestamp": now,
        "event": event,
        "description": description,
    }
    if metrics:
        knowledge_event["metrics"] = metrics
    report["knowledge_events"].append(knowledge_event)
    report["last_updated"] = now
    # 保留最近 100 個事件
    if len(report["knowledge_events"]) > 100:
        report["knowledge_events"] = report["knowledge_events"][-100:]
    # 儲存報告
    save_yaml(report_path, report)
    console.print(f"[green]✓ Updated knowledge health report: {report_path}[/green]")
def update_knowledge_graph(event: str, nodes: list[dict[str, Any]] = None):
    """更新知識圖譜"""
    graph_path = "docs/knowledge-graph.yaml"
    now = datetime.datetime.utcnow().isoformat() + "Z"
    # 載入現有圖譜
    graph = load_yaml(graph_path)
    # 初始化結構
    if "metadata" not in graph:
        graph["metadata"] = {}
    if "events" not in graph:
        graph["events"] = []
    if "nodes" not in graph:
        graph["nodes"] = []
    # 更新 metadata
    graph["metadata"]["last_updated"] = now
    graph["metadata"]["last_event"] = event
    # 新增事件
    graph["events"].append(
        {
            "timestamp": now,
            "event": event,
        }
    )
    # 新增或更新節點
    if nodes:
        for new_node in nodes:
            # 檢查節點是否已存在
            existing = None
            for i, node in enumerate(graph["nodes"]):
                if node.get("id") == new_node.get("id"):
                    existing = i
                    break
            if existing is not None:
                # 更新現有節點
                graph["nodes"][existing].update(new_node)
                graph["nodes"][existing]["updated_at"] = now
            else:
                # 新增新節點
                new_node["created_at"] = now
                graph["nodes"].append(new_node)
    # 保留最近 50 個事件
    if len(graph["events"]) > 50:
        graph["events"] = graph["events"][-50:]
    # 儲存圖譜
    save_yaml(graph_path, graph)
    console.print(f"[green]✓ Updated knowledge graph: {graph_path}[/green]")
def update_knowledge_health_md(summary: str, metrics: dict[str, Any] = None):
    """更新知識庫健康 Markdown 文檔"""
    md_path = "docs/KNOWLEDGE_HEALTH.md"
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    content = f"""# Knowledge Health Report
#*Last Updated**: {now}
## Summary
{summary}
"""
    if metrics:
        content += "## Metrics\n\n"
        for key, value in metrics.items():
            content += f"- **{key}**: {value}\n"
        content += "\n"
    content += """## Recent Updates
This document is automatically updated by the Living Knowledge Base system when:
- AI Auto-Fix Bot applies repairs
- Language gl-platform.gl-platform.governance violations are resolved
- Security issues are remediated
- System health changes significantly
## Knowledge Cycle
The Living Knowledge Base operates in continuous cycles:
1. **Detection**: Language gl-platform.gl-platform.governance, CodeQL, and Semgrep scans detect issues
2. **Analysis**: AI analyzes violations and generates fix suggestions
3. **Remediation**: Auto-Fix Bot creates and applies patches
4. **Verification**: Changes are validated through CI/CD
5. **Documentation**: Knowledge base is updated with learnings
6. **Monitoring**: Health scores are recalculated and tracked
For detailed event history, see `docs/knowledge-health-report.yaml`.
For knowledge graph visualization, see `docs/knowledge-graph.yaml`.
"""
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)
    console.print(f"[green]✓ Updated knowledge health markdown: {md_path}[/green]")
def main():
    parser = argparse.ArgumentParser(
        description="Update Living Knowledge Base after Auto-Fix"
    )
    parser.add_argument(
        "--event", required=True, help="Event type (e.g., 'auto-fix', 'manual-fix')"
    )
    parser.add_argument("--description", required=True, help="Event description")
    parser.add_argument(
        "--violations-fixed", type=int, help="Number of violations fixed"
    )
    parser.add_argument(
        "--health-score", type=float, help="Current health score (0-100)"
    )
    parser.add_argument("--files-changed", type=int, help="Number of files changed")
    args = parser.parse_args()
    console.print("[cyan]📚 Updating Living Knowledge Base...[/cyan]")
    # 準備 metrics
    metrics = {}
    if args.violations_fixed is not None:
        metrics["violations_fixed"] = args.violations_fixed
    if args.health_score is not None:
        metrics["health_score"] = args.health_score
    if args.files_changed is not None:
        metrics["files_changed"] = args.files_changed
    # 更新知識庫健康報告
    update_knowledge_health_report(
        args.event, args.description, metrics if metrics else None
    )
    # 準備知識圖譜節點
    nodes = [
        {
            "id": f"event-{args.event}-{datetime.datetime.utcnow().timestamp()}",
            "type": "event",
            "event_type": args.event,
            "description": args.description,
            "metrics": metrics if metrics else {},
        }
    ]
    # 更新知識圖譜
    update_knowledge_graph(args.event, nodes)
    # 更新 Markdown 文檔
    summary = f"{args.description}\n\n"
    if metrics:
        summary += "**Impact**: "
        impact_items = []
        if args.violations_fixed:
            impact_items.append(f"{args.violations_fixed} violations fixed")
        if args.files_changed:
            impact_items.append(f"{args.files_changed} files changed")
        if args.health_score:
            impact_items.append(f"Health score: {args.health_score:.1f}/100")
        summary += ", ".join(impact_items)
    update_knowledge_health_md(summary, metrics if metrics else None)
    console.print("[green]✅ Living Knowledge Base updated successfully![/green]")
if __name__ == "__main__":
    main()
