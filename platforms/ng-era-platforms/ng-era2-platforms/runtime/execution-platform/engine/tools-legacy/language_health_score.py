# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: governance
# @ECO-semantic: language-health-score
# @ECO-audit-trail: ../../engine/governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Language Health Score Calculator
---------------------------------
計算語言層級健康分數（0-100）
分數計算基於：
1. 語言治理違規數量與嚴重程度
2. 安全漏洞數量
3. 架構對齊程度
4. 修復歷史趨勢
使用：
    python tools/language-health-score.py \
        --governance-report governance/language-governance-report.json \
        --history knowledge/language-history.yaml \
        --output knowledge/language-health-score.yaml
"""
# MNGA-002: Import organization needs review
import argparse
import datetime
import json
import os
from typing import Any
import yaml
from rich.console import Console
from rich.table import Table
console = Console()
def load_json(path: str) -> dict[str, Any]:
    """載入 JSON 檔案"""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception as e:
            console.print(f"[red]Error loading {path}: {e}[/red]")
            return {}
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
def calculate_violation_score(governance_report: dict[str, Any]) -> float:
    """計算違規分數 (0-40分)"""
    violations = governance_report.get("violations", [])
    if not violations:
        return 40.0
    # 按嚴重程度計分
    critical_count = sum(1 for v in violations if v.get("severity") == "CRITICAL")
    error_count = sum(1 for v in violations if v.get("severity") == "ERROR")
    warning_count = sum(1 for v in violations if v.get("severity") == "WARNING")
    # 權重：CRITICAL(-5), ERROR(-2), WARNING(-0.5)
    penalty = (critical_count * 5) + (error_count * 2) + (warning_count * 0.5)
    # 最大扣分 40 分
    score = max(0, 40 - penalty)
    return score
def calculate_security_score(
    semgrep_report: dict[str, Any], codeql_report: dict[str, Any]
) -> float:
    """計算安全分數 (0-30分)"""
    # Semgrep 結果
    semgrep_results = semgrep_report.get("results", [])
    semgrep_high = sum(
        1 for r in semgrep_results if r.get("extra", {}).get("severity") == "ERROR"
    )
    semgrep_medium = sum(
        1 for r in semgrep_results if r.get("extra", {}).get("severity") == "WARNING"
    )
    # CodeQL 結果
    codeql_runs = codeql_report.get("runs", [])
    codeql_high = 0
    codeql_medium = 0
    for run in codeql_runs:
        results = run.get("results", [])
        for result in results:
            level = result.get("level", "note")
            if level == "error":
                codeql_high += 1
            elif level == "warning":
                codeql_medium += 1
    # 權重：High(-3), Medium(-1)
    penalty = (semgrep_high + codeql_high) * 3 + (semgrep_medium + codeql_medium) * 1
    # 最大扣分 30 分
    score = max(0, 30 - penalty)
    return score
def calculate_architecture_score(governance_report: dict[str, Any]) -> float:
    """計算架構對齊分數 (0-20分)"""
    violations = governance_report.get("violations", [])
    # 架構違規類型
    arch_violations = [
        v
        for v in violations
        if "layer" in v.get("message", "").lower()
        or "directory" in v.get("message", "").lower()
    ]
    if not arch_violations:
        return 20.0
    # 每個架構違規扣 2 分
    penalty = len(arch_violations) * 2
    # 最大扣分 20 分
    score = max(0, 20 - penalty)
    return score
def calculate_trend_score(history: dict[str, Any]) -> float:
    """計算趨勢分數 (0-10分)"""
    if "history" not in history or not history["history"]:
        return 5.0  # 無歷史記錄給予中等分數
    fixes = history["history"]
    # 看最近 30 天的修復趨勢
    thirty_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=30)
    recent_fixes = []
    for fix in fixes:
        try:
            fix_time = datetime.datetime.fromisoformat(
                fix["timestamp"].replace("Z", "+00:00")
            )
            if fix_time >= thirty_days_ago:
                recent_fixes.append(fix)
        except BaseException:
            continue
    if not recent_fixes:
        return 5.0
    # 修復越多，分數越高
    fix_count = len(recent_fixes)
    if fix_count >= 20:
        return 10.0
    elif fix_count >= 10:
        return 8.0
    elif fix_count >= 5:
        return 6.0
    else:
        return 5.0
def calculate_health_score(
    governance_report: dict[str, Any],
    semgrep_report: dict[str, Any] = None,
    codeql_report: dict[str, Any] = None,
    history: dict[str, Any] = None,
) -> dict[str, Any]:
    """計算整體健康分數"""
    # 各項分數
    violation_score = calculate_violation_score(governance_report)
    security_score = calculate_security_score(semgrep_report or {}, codeql_report or {})
    architecture_score = calculate_architecture_score(governance_report)
    trend_score = calculate_trend_score(history or {})
    # 總分 (0-100)
    total_score = violation_score + security_score + architecture_score + trend_score
    # 等級評定
    if total_score >= 90:
        grade = "A"
        status = "Excellent"
    elif total_score >= 80:
        grade = "B"
        status = "Good"
    elif total_score >= 70:
        grade = "C"
        status = "Fair"
    elif total_score >= 60:
        grade = "D"
        status = "Poor"
    else:
        grade = "F"
        status = "Critical"
    return {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "total_score": round(total_score, 2),
        "grade": grade,
        "status": status,
        "components": {
            "violation_score": round(violation_score, 2),
            "security_score": round(security_score, 2),
            "architecture_score": round(architecture_score, 2),
            "trend_score": round(trend_score, 2),
        },
        "max_scores": {
            "violation_score": 40,
            "security_score": 30,
            "architecture_score": 20,
            "trend_score": 10,
        },
    }
def display_score(score_data: dict[str, Any]):
    """顯示分數表格"""
    table = Table(title="Language Health Score")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="magenta")
    table.add_column("Max", style="white")
    table.add_column("Percentage", style="green")
    components = score_data["components"]
    max_scores = score_data["max_scores"]
    for key in components:
        score = components[key]
        max_score = max_scores[key]
        percentage = (score / max_score * 100) if max_score > 0 else 0
        name = key.replace("_score", "").replace("_", " ").title()
        table.add_row(name, f"{score:.2f}", str(max_score), f"{percentage:.1f}%")
    table.add_row("─" * 20, "─" * 10, "─" * 5, "─" * 10, style="dim")
    table.add_row(
        "Total",
        f"{score_data['total_score']:.2f}",
        "100",
        f"{score_data['total_score']:.1f}%",
        style="bold",
    )
    console.print(table)
    console.print(f"\n[bold]Grade: {score_data['grade']}[/bold]")
    console.print(f"[bold]Status: {score_data['status']}[/bold]")
def main():
    parser = argparse.ArgumentParser(description="Calculate language health score")
    parser.add_argument(
        "--governance-report", required=True, help="Path to governance report JSON"
    )
    parser.add_argument("--semgrep-report", help="Path to Semgrep SARIF report")
    parser.add_argument("--codeql-report", help="Path to CodeQL SARIF report")
    parser.add_argument("--history", help="Path to language history YAML")
    parser.add_argument(
        "--output",
        default="knowledge/language-health-score.yaml",
        help="Output path for score YAML",
    )
    parser.add_argument("--display", action="store_true", help="Display score table")
    args = parser.parse_args()
    console.print("[cyan]🔍 Calculating language health score...[/cyan]")
    # 載入報告
    governance_report = load_json(args.governance_report)
    semgrep_report = load_json(args.semgrep_report) if args.semgrep_report else {}
    codeql_report = load_json(args.codeql_report) if args.codeql_report else {}
    history = load_yaml(args.history) if args.history else {}
    # 計算分數
    score_data = calculate_health_score(
        governance_report, semgrep_report, codeql_report, history
    )
    # 儲存分數
    save_yaml(args.output, score_data)
    console.print(f"[green]✓ Health score saved to {args.output}[/green]")
    # 顯示分數
    if args.display:
        display_score(score_data)
    else:
        console.print(
            f"[bold green]Total Score: {score_data['total_score']:.2f}/100[/bold green]"
        )
        console.print(
            f"[bold]Grade: {score_data['grade']} ({score_data['status']})[/bold]"
        )
    console.print("[green]✅ Health score calculation complete![/green]")
if __name__ == "__main__":
    main()
