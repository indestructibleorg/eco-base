# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: language-history-writer
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Language History Writer
-----------------------
記錄所有語言治理修復事件到 knowledge/language-history.yaml
使用：
    python tools/language-history-writer.py \
        --violation-type "forbidden-language" \
        --file-path "services/legacy/script.php" \
        --action "removed" \
        --reason "PHP is forbidden in services directory"
"""
# MNGA-002: Import organization needs review
import argparse
import datetime
import os
from typing import Any
import yaml
from rich import print
from rich.console import Console
console = Console()
def load_language_history() -> dict[str, Any]:
    """載入語言歷史記錄"""
    history_path = "knowledge/language-history.yaml"
    if not os.path.exists(history_path):
        return {
            "metadata": {
                "version": "1.0",
                "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                "description": "Language gl-platform.governance fix history tracking",
            },
            "history": [],
        }
    with open(history_path, encoding="utf-8") as f:
        try:
            return yaml.safe_load(f) or {}
        except Exception as e:
            console.print(f"[red]Error loading history: {e}[/red]")
            return {"history": []}
def save_language_history(data: dict[str, Any]):
    """儲存語言歷史記錄"""
    history_path = "knowledge/language-history.yaml"
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        yaml.dump(
            data, f, default_flow_style=False, allow_unicode=True, sort_keys=False
        )
def add_fix_record(
    violation_type: str,
    file_path: str,
    action: str,
    reason: str,
    before_language: str = None,
    after_language: str = None,
    severity: str = "ERROR",
    fixed_by: str = "ai-auto-fix-bot",
):
    """新增修復記錄"""
    history = load_language_history()
    # 確保有 history 欄位
    if "history" not in history:
        history["history"] = []
    # 建立記錄
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "violation_type": violation_type,
        "file_path": file_path,
        "action": action,
        "reason": reason,
        "severity": severity,
        "fixed_by": fixed_by,
    }
    if before_language:
        record["before_language"] = before_language
    if after_language:
        record["after_language"] = after_language
    # 新增記錄
    history["history"].append(record)
    # 更新 metadata
    if "metadata" not in history:
        history["metadata"] = {}
    history["metadata"]["last_updated"] = datetime.datetime.utcnow().isoformat() + "Z"
    history["metadata"]["total_fixes"] = len(history["history"])
    # 計算統計
    if "statistics" not in history:
        history["statistics"] = {}
    stats = history["statistics"]
    # 按類型統計
    if "by_type" not in stats:
        stats["by_type"] = {}
    stats["by_type"][violation_type] = stats["by_type"].get(violation_type, 0) + 1
    # 按動作統計
    if "by_action" not in stats:
        stats["by_action"] = {}
    stats["by_action"][action] = stats["by_action"].get(action, 0) + 1
    # 按嚴重程度統計
    if "by_severity" not in stats:
        stats["by_severity"] = {}
    stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1
    # 保存
    save_language_history(history)
    console.print(f"[green]✓ Added fix record for {file_path}[/green]")
    console.print(f"  Type: {violation_type}, Action: {action}, Severity: {severity}")
def get_recent_fixes(limit: int = 10) -> list[dict[str, Any]]:
    """取得最近的修復記錄"""
    history = load_language_history()
    if "history" not in history:
        return []
    return history["history"][-limit:]
def get_statistics() -> dict[str, Any]:
    """取得統計資訊"""
    history = load_language_history()
    return history.get("statistics", {})
def generate_report() -> str:
    """生成報告"""
    history = load_language_history()
    stats = history.get("statistics", {})
    report = "# Language Governance Fix History Report\n\n"
    report += (
        f"**Total Fixes**: {history.get('metadata', {}).get('total_fixes', 0)}\n\n"
    )
    if "by_type" in stats:
        report += "## By Violation Type\n\n"
        for vtype, count in sorted(
            stats["by_type"].items(), key=lambda x: x[1], reverse=True
        ):
            report += f"- **{vtype}**: {count}\n"
        report += "\n"
    if "by_action" in stats:
        report += "## By Action\n\n"
        for action, count in sorted(
            stats["by_action"].items(), key=lambda x: x[1], reverse=True
        ):
            report += f"- **{action}**: {count}\n"
        report += "\n"
    if "by_severity" in stats:
        report += "## By Severity\n\n"
        for severity, count in sorted(
            stats["by_severity"].items(), key=lambda x: x[1], reverse=True
        ):
            report += f"- **{severity}**: {count}\n"
        report += "\n"
    # 最近修復
    recent = get_recent_fixes(5)
    if recent:
        report += "## Recent Fixes\n\n"
        for fix in recent:
            report += f"- **{fix['timestamp']}**: {fix['file_path']}\n"
            report += f"  - Type: {fix['violation_type']}, Action: {fix['action']}\n"
            report += f"  - Reason: {fix['reason']}\n\n"
    return report
def main():
    parser = argparse.ArgumentParser(
        description="Record language gl-platform.governance fix history"
    )
    parser.add_argument(
        "--violation-type",
        required=True,
        help="Type of violation (e.g., 'forbidden-language', 'wrong-layer', 'security-issue')",
    )
    parser.add_argument(
        "--file-path", required=True, help="Path to the file that was fixed"
    )
    parser.add_argument(
        "--action",
        required=True,
        help="Action taken (e.g., 'removed', 'moved', 'refactored', 'migrated')",
    )
    parser.add_argument("--reason", required=True, help="Reason for the fix")
    parser.add_argument("--before-language", help="Language before fix")
    parser.add_argument("--after-language", help="Language after fix")
    parser.add_argument(
        "--severity",
        default="ERROR",
        choices=["CRITICAL", "ERROR", "WARNING"],
        help="Severity level",
    )
    parser.add_argument(
        "--fixed-by", default="ai-auto-fix-bot", help="Who/what applied the fix"
    )
    parser.add_argument(
        "--report", action="store_true", help="Generate and print report"
    )
    args = parser.parse_args()
    if args.report:
        console.print("[cyan]📊 Generating language history report...[/cyan]")
        report = generate_report()
        print(report)
        # 儲存報告
        report_path = "knowledge/language-history-report.md"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        console.print(f"[green]✓ Report saved to {report_path}[/green]")
    else:
        console.print("[cyan]📝 Recording language fix to history...[/cyan]")
        add_fix_record(
            violation_type=args.violation_type,
            file_path=args.file_path,
            action=args.action,
            reason=args.reason,
            before_language=args.before_language,
            after_language=args.after_language,
            severity=args.severity,
            fixed_by=args.fixed_by,
        )
        console.print("[green]✅ Fix record added successfully![/green]")
if __name__ == "__main__":
    main()
