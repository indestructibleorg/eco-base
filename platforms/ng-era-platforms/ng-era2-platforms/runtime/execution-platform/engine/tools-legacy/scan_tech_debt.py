# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: scan_tech_debt
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
技術債務掃描與分類工具
Scans and categorizes technical debt across the repository
目標：識別並優先處理168個技術債務項目，減少至84個
"""
# MNGA-002: Import organization needs review
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
@dataclass
class DebtItem:
    """技術債務項目"""
    file_path: str
    line_number: int
    debt_type: str  # TODO, FIXME, XXX, HACK, DEPRECATED
    severity: str  # HIGH, MEDIUM, LOW
    message: str
    context: str = ""
@dataclass
class DebtReport:
    """債務報告"""
    total_items: int = 0
    by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_severity: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_directory: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    items: List[DebtItem] = field(default_factory=list)
class TechDebtScanner:
    """技術債務掃描器"""
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.report = DebtReport()
        # 要掃描的文件擴展名
        self.extensions = {
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".yaml",
            ".yml",
            ".md",
            ".sh",
        }
        # 要跳過的目錄
        self.skip_dirs = {
            "node_modules",
            ".git",
            "__pycache__",
            ".venv",
            "venv",
            ".pytest_cache",
            "dist",
            "build",
            ".next",
            "coverage",
        }
        # 債務標記模式
        self.debt_patterns = {
            "TODO": re.compile(r"#\s*TODO\s*:?\s*(.+)", re.IGNORECASE),
            "FIXME": re.compile(r"#\s*FIXME\s*:?\s*(.+)", re.IGNORECASE),
            "XXX": re.compile(r"#\s*XXX\s*:?\s*(.+)", re.IGNORECASE),
            "HACK": re.compile(r"#\s*HACK\s*:?\s*(.+)", re.IGNORECASE),
            "DEPRECATED": re.compile(r"@deprecated|#\s*DEPRECATED", re.IGNORECASE),
        }
        # 高複雜度函數模式（Python）
        self.high_complexity_pattern = re.compile(
            r"def\s+\w+\([^)]*\).*?(?=\ndef\s|\nclass\s|\Z)", re.DOTALL
        )
    def scan(self) -> DebtReport:
        """掃描整個儲存庫"""
        print("🔍 掃描技術債務...\n")
        for file_path in self._iter_files():
            self._scan_file(file_path)
        self._calculate_summary()
        return self.report
    def _iter_files(self):
        """遍歷所有要掃描的文件"""
        for root, dirs, files in os.walk(self.repo_root):
            # 過濾跳過的目錄
            dirs[:] = [d for d in dirs if d not in self.skip_dirs]
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in self.extensions:
                    yield file_path
    def _scan_file(self, file_path: Path):
        """掃描單個文件"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            rel_path = str(file_path.relative_to(self.repo_root))
            for line_num, line in enumerate(lines, 1):
                # 檢查債務標記
                for debt_type, pattern in self.debt_patterns.items():
                    match = pattern.search(line)
                    if match:
                        message = match.group(1) if match.lastindex else line.strip()
                        severity = self._determine_severity(debt_type, message)
                        item = DebtItem(
                            file_path=rel_path,
                            line_number=line_num,
                            debt_type=debt_type,
                            severity=severity,
                            message=message.strip(),
                            context=line.strip(),
                        )
                        self.report.items.append(item)
            # 檢查高複雜度函數（僅Python）
            if file_path.suffix == ".py":
                self._check_complexity(file_path, "".join(lines))
        except Exception as e:
            print(f"⚠️  掃描 {file_path} 失敗: {e}")
    def _determine_severity(self, debt_type: str, message: str) -> str:
        """確定嚴重程度"""
        message_lower = message.lower()
        # 高優先級關鍵詞
        high_keywords = [
            "security",
            "critical",
            "urgent",
            "bug",
            "broken",
            "fix immediately",
        ]
        # 中優先級關鍵詞
        medium_keywords = ["important", "should", "refactor", "improve"]
        if any(kw in message_lower for kw in high_keywords):
            return "HIGH"
        elif any(kw in message_lower for kw in medium_keywords):
            return "MEDIUM"
        elif debt_type in ["FIXME", "XXX"]:
            return "MEDIUM"
        else:
            return "LOW"
    def _check_complexity(self, file_path: Path, content: str):
        """檢查函數複雜度"""
        rel_path = str(file_path.relative_to(self.repo_root))
        # 簡化版複雜度檢查：函數行數
        functions = re.findall(r"def\s+(\w+)\([^)]*\):", content)
        for func_name in functions:
            # 查找函數體
            func_pattern = re.compile(
                rf"def\s+{re.escape(func_name)}\([^)]*\):(.+?)(?=\ndef\s|\nclass\s|\Z)",
                re.DOTALL,
            )
            match = func_pattern.search(content)
            if match:
                func_body = match.group(1)
                lines = len(
                    [
                        l
                        for l in func_body.split("\n")
                        if l.strip() and not l.strip().startswith("#")
                    ]
                )
                # 如果函數超過100行，標記為高複雜度
                if lines > 100:
                    item = DebtItem(
                        file_path=rel_path,
                        line_number=content[: match.start()].count("\n") + 1,
                        debt_type="HIGH_COMPLEXITY",
                        severity="MEDIUM",
                        message=f"Function '{func_name}' has {lines} lines (threshold: 100)",
                        context=f"def {func_name}(...)",
                    )
                    self.report.items.append(item)
    def _calculate_summary(self):
        """計算摘要統計"""
        self.report.total_items = len(self.report.items)
        for item in self.report.items:
            self.report.by_type[item.debt_type] += 1
            self.report.by_severity[item.severity] += 1
            # 按目錄分類
            directory = (
                str(Path(item.file_path).parts[0]) if "/" in item.file_path else "root"
            )
            self.report.by_directory[directory] += 1
    def generate_report(self) -> Dict:
        """生成詳細報告"""
        # 按嚴重程度排序
        high_priority = [item for item in self.report.items if item.severity == "HIGH"]
        medium_priority = [
            item for item in self.report.items if item.severity == "MEDIUM"
        ]
        low_priority = [item for item in self.report.items if item.severity == "LOW"]
        return {
            "summary": {
                "total_items": self.report.total_items,
                "target_reduction": self.report.total_items // 2,  # 減少50%
                "by_type": dict(self.report.by_type),
                "by_severity": dict(self.report.by_severity),
                "by_directory": dict(self.report.by_directory),
            },
            "high_priority": [
                {
                    "file": item.file_path,
                    "line": item.line_number,
                    "type": item.debt_type,
                    "message": item.message,
                }
                for item in high_priority[:20]  # 前20個
            ],
            "medium_priority": [
                {
                    "file": item.file_path,
                    "line": item.line_number,
                    "type": item.debt_type,
                    "message": item.message,
                }
                for item in medium_priority[:30]  # 前30個
            ],
            "low_priority_count": len(low_priority),
        }
    def print_summary(self):
        """打印摘要"""
        print("\n" + "=" * 70)
        print("📊 技術債務掃描摘要")
        print("=" * 70)
        print(f"\n總債務項目: {self.report.total_items}")
        print(f"目標減少至: {self.report.total_items // 2} (-50%)")
        print("\n按類型分佈:")
        for debt_type, count in sorted(
            self.report.by_type.items(), key=lambda x: -x[1]
        ):
            print(
                f"  {debt_type:15} {count:4} ({count/self.report.total_items*100:.1f}%)"
            )
        print("\n按嚴重程度分佈:")
        for severity, count in sorted(
            self.report.by_severity.items(), key=lambda x: -x[1]
        ):
            emoji = (
                "🔴" if severity == "HIGH" else "🟡" if severity == "MEDIUM" else "🟢"
            )
            print(
                f"  {emoji} {severity:8} {count:4} ({count/self.report.total_items*100:.1f}%)"
            )
        print("\n按目錄分佈 (Top 5):")
        top_dirs = sorted(self.report.by_directory.items(), key=lambda x: -x[1])[:5]
        for directory, count in top_dirs:
            print(f"  {directory:30} {count:4}")
        print("\n" + "=" * 70)
        # 顯示高優先級項目
        high_priority = [item for item in self.report.items if item.severity == "HIGH"]
        if high_priority:
            print("\n🔴 高優先級項目 (Top 10):")
            for item in high_priority[:10]:
                print(f"  [{item.debt_type}] {item.file_path}:{item.line_number}")
                print(f"      {item.message[:70]}")
        print()
def main():
    """主函數"""
    repo_root = Path(__file__).parent.parent
    scanner = TechDebtScanner(repo_root)
    scanner.scan()
    # 打印摘要
    scanner.print_summary()
    # 生成 JSON 報告
    report = scanner.generate_report()
    report_file = repo_root / "TECH_DEBT_SCAN_REPORT.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"📄 詳細報告已保存: {report_file}\n")
    # 建議行動
    high_count = report["summary"]["by_severity"].get("HIGH", 0)
    print("💡 建議行動:")
    print(f"  1. 優先處理 {high_count} 個高優先級項目")
    print(
        f"  2. 目標：從 {report['summary']['total_items']} 項減少至 {report['summary']['target_reduction']} 項"
    )
    print("  3. 聚焦於高複雜度函數重構")
    print()
if __name__ == "__main__":
    main()
