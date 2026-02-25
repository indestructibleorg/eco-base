# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: generate_evolution_report
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
generate_evolution_report.py
目的：
  - 讀取 config/system-evolution.yaml
  - 嘗試從治理/掃描輸出中取得指標：
      - language_violations_total
      - semgrep_high_total
      - playbook_coverage_ratio
  - 根據目標與權重算出 0~100 分的演化健康度
  - 輸出：
      - knowledge/evolution-state.yaml   (機器可讀)
      - docs/SYSTEM_EVOLUTION_REPORT.md (人類可讀)
依賴：
  - PyYAML: pip install pyyaml
"""
# MNGA-002: Import organization needs review
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import yaml
ROOT = Path(__file__).resolve().parents[2]  # unmanned-island 根目錄
CONFIG_PATH = ROOT / "config/system-evolution.yaml"
def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
def safe_load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
def load_json(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
def count_language_violations(report_path: Path) -> int:
    """
    簡單假設 language-gl-platform.governance-report.md 每個違規一行，例如：
      - **path/to/file.ts** — reason...
    """
    if not report_path.exists():
        return 0
    pattern = re.compile(r"^- \*\*(.+?)\*\* —")
    count = 0
    with report_path.open("r", encoding="utf-8") as f:
        for line in f:
            if pattern.match(line.strip()):
                count += 1
    return count
def count_semgrep_high(semgrep_path: Path) -> int:
    if not semgrep_path.exists():
        return 0
    data = load_json(semgrep_path)
    results = data.get("results", [])
    high = 0
    for r in results:
        severity = (r.get("extra", {}).get("severity", "") or "").upper()
        if severity == "ERROR" or severity == "HIGH":
            high += 1
    return high
def compute_playbook_coverage(
    cluster_heatmap_path: Path, playbooks_root: Path
) -> float:
    """
    概念：
      - 從 cluster-heatmap.json 抽出所有 clusters 名稱（扁平結構，如 "core/", "services/" 等）
      - 檢查 03_refactor 底下是否有對應 *_refactor.md
      - 回傳 covered / total 比例
    若 cluster_heatmap / playbooks_root 任一不存在 → 回傳 0.0（代表目前還沒有這層治理）
    """
    if (not cluster_heatmap_path.exists()) or (not playbooks_root.exists()):
        return 0.0
    data = load_json(cluster_heatmap_path)
    if not data:
        return 0.0
    # 實際格式是扁平字典：
    # { "core/": {...}, "services/": {...}, ... }
    clusters = list(data.keys())
    if not clusters:
        return 0.0
    # 檢查對應的 playbook 檔是否存在：
    # 約定：cluster 'core/' → playbook 應在
    # docs/refactor_playbooks/03_refactor/core/*_refactor.md
    covered = 0
    for cluster_key in clusters:
        # 移除尾隨斜線以獲得目錄名稱
        domain = cluster_key.rstrip("/")
        playbook_dir = playbooks_root / domain
        # 寬鬆匹配：找底下任何 *_refactor.md 即算有 playbook（你之後可以改得更嚴格）
        if playbook_dir.exists():
            candidates = list(playbook_dir.glob("*_refactor.md"))
            if candidates:
                covered += 1
    return covered / len(clusters)
def score_metric(value: float, target: float, direction: str) -> float:
    """
    根據方向與目標，計算 0~100 分：
      - lower_is_better: value <= target → 100；超過則按比例下降（簡化版本）
      - higher_is_better: value >= target → 100；低於則按比例下降
    """
    if direction == "lower_is_better":
        if value <= target:
            return 100.0
        # 簡單壓縮：當 value 是 target 的 2 倍 → 50 分；3 倍 → ~33 分
        ratio = target / max(value, 1e-6)
        return max(0.0, min(100.0, ratio * 100.0))
    elif direction == "higher_is_better":
        if value >= target:
            return 100.0
        ratio = value / max(target, 1e-6)
        return max(0.0, min(100.0, ratio * 100.0))
    else:
        return 0.0
def main():
    if not CONFIG_PATH.exists():
        raise SystemExit(f"Config not found: {CONFIG_PATH}")
    config = load_yaml(CONFIG_PATH)
    metrics_sources = config.get("metrics_sources", {})
    outputs = config.get("outputs", {})
    lang_report_path = ROOT / metrics_sources.get("language_gl-platform.governance_report", "")
    semgrep_path = ROOT / metrics_sources.get("semgrep_report", "")
    cluster_heatmap_path = ROOT / metrics_sources.get("cluster_heatmap", "")
    playbooks_root = ROOT / metrics_sources.get("refactor_playbooks_root", "")
    # 1) 收集原始指標
    language_violations_total = count_language_violations(lang_report_path)
    semgrep_high_total = count_semgrep_high(semgrep_path)
    playbook_coverage_ratio = compute_playbook_coverage(
        cluster_heatmap_path, playbooks_root
    )
    metrics = {
        "language_violations_total": language_violations_total,
        "semgrep_high_total": semgrep_high_total,
        "playbook_coverage_ratio": playbook_coverage_ratio,
    }
    # 2) 根據 config.objectives 計算各目標分數
    objectives = config.get("objectives", [])
    scored_objectives = []
    total_weight = 0.0
    weighted_sum = 0.0
    for obj in objectives:
        oid = obj.get("id")
        metric_name = obj.get("metric")
        target = obj.get("target", 0.0)
        direction = obj.get("direction", "lower_is_better")
        weight = float(obj.get("weight", 0.0))
        value = float(metrics.get(metric_name, 0.0))
        score = score_metric(value, float(target), direction)
        scored_objectives.append(
            {
                "id": oid,
                "name": obj.get("name", ""),
                "metric": metric_name,
                "value": value,
                "target": target,
                "direction": direction,
                "weight": weight,
                "score": round(score, 2),
            }
        )
        total_weight += weight
        weighted_sum += score * weight
    overall_score = 0.0
    if total_weight > 0:
        overall_score = weighted_sum / total_weight
    # 3) 準備 evolution-state.yaml（機器可讀）
    state = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "metrics": metrics,
        "objectives": scored_objectives,
        "overall_score": round(overall_score, 2),
        "config_version": config.get("version", "0.0.0"),
    }
    state_path = ROOT / outputs.get("state_yaml", "knowledge/evolution-state.yaml")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(state, f, sort_keys=False, allow_unicode=True)
    # 4) 準備 SYSTEM_EVOLUTION_REPORT.md（人類可讀）
    report_path = ROOT / outputs.get(
        "report_markdown", "docs/SYSTEM_EVOLUTION_REPORT.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# 🧬 System Evolution Report\n")
    lines.append(f"- 生成時間：{state['generated_at']}")
    lines.append(f"- Config 版本：{state['config_version']}")
    lines.append(f"- 總體演化健康度：**{state['overall_score']}/100**\n")
    lines.append("## 指標概覽\n")
    for k, v in metrics.items():
        lines.append(f"- `{k}` = `{v}`")
    lines.append("")
    lines.append("## 目標與得分\n")
    for obj in scored_objectives:
        lines.append(f"### {obj['name']} (`{obj['id']}`)")
        lines.append(f"- 指標：`{obj['metric']}`")
        lines.append(
            f"- 目前值：`{obj['value']}`，目標：`{obj['target']}`（{obj['direction']}）"
        )
        lines.append(f"- 權重：{obj['weight']}")
        lines.append(f"- 得分：**{obj['score']}/100**\n")
    lines.append("## 下一步建議（高階）\n")
    lines.append(
        "> 以下是根據分數粗略給出的優先級建議，你可以再交給 AI 做更細的 Refactor Playbook。\n"
    )
    for obj in scored_objectives:
        suggestion = ""
        if obj["id"] == "language-gl-platform.governance":
            suggestion = "優先針對違規最多的 cluster（依 language-gl-platform.governance-report 排序），更新對應的 03_refactor 劇本並排入 P0 任務。"
        elif obj["id"] == "security":
            suggestion = "列出全部 Semgrep HIGH 問題，對應到 services/core 的 cluster，建立或更新這些區域的安全重構 Playbook。"
        elif obj["id"] == "refactor-playbook-coverage":
            suggestion = "找出 cluster-heatmap 中沒有對應 03_refactor Playbook 的 cluster，為其建立最小可用的重構劇本。"
        else:
            suggestion = "根據此目標的當前得分與權重，安排適當的 Refactor 迭代。"
        lines.append(
            f"- `{obj['id']}`（{obj['name']}）：目前得分 {obj['score']}/100 → {suggestion}"
        )
    lines.append("")
    lines.append("---")
    lines.append("本報告由 `tools/evolution/generate_evolution_report.py` 自動生成。")
    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[System Evolution] 狀態已更新：{state_path}")
    print(f"[System Evolution] 報告已生成：{report_path}")
    print(f"[System Evolution] 總體演化健康度：{state['overall_score']}/100")
if __name__ == "__main__":
    main()
