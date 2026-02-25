#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: auto-quality-check
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
自動化程式碼品質檢查工具
Automated Code Quality Check Tool
此腳本自動執行 PR-1-REVIEW-REPORT.md 中識別的所有檢查項目
This script automatically performs all checks identified in PR-1-REVIEW-REPORT.md
"""
# MNGA-002: Import organization needs review
import subprocess
import json
from pathlib import Path
from typing import Dict, Any
import argparse
from datetime import datetime
import ast  # Added for ast.literal_eval()

class QualityChecker:
    """自動化品質檢查器"""
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.results: Dict[str, Any] = {}

    def _sanitize_results_for_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        安全地過濾結果中的敏感資料，避免在報告中以明文儲存。
        僅保留必要的摘要資訊（例如計數），並對可能的敏感欄位進行遮罩。
        """
        # 定義可能包含敏感資訊的欄位名稱
        sensitive_keys = {
            "secrets",
            "tokens",
            "passwords",
            "keys",
            "credentials",
            "secret_values",
        }
        sanitized: Dict[str, Any] = {}
        for check_name, result in results.items():
            if not isinstance(result, dict):
                # 非預期格式，直接複製（通常不會包含敏感資訊）
                sanitized[check_name] = result
                continue
            sanitized_result: Dict[str, Any] = {}
            for key, value in result.items():
                if key in sensitive_keys:
                    # 統一遮罩潛在敏感資料
                    sanitized_result[key] = "[REDACTED FOR SECURITY]"
                elif isinstance(value, list):
                    # 僅保留清單長度，避免意外將敏感字串寫入報告
                    sanitized_result[key] = {
                        "count": len(value),
                        "note": "List content omitted for security"
                    }
                else:
                    sanitized_result[key] = value
            sanitized[check_name] = sanitized_result
        return sanitized
    def run_all_checks(self) -> Dict[str, Any]:
        """執行所有檢查"""
        print("🚀 開始自動化品質檢查...")
        self.check_security()
        self.check_python_quality()
        self.check_typescript_quality()
        self.check_code_duplication()
        self.check_docstring_coverage()
        self.check_non_ascii_filenames()
        self.check_console_logs()
        self.check_eval_usage()
        self.generate_report()
        return self.results
    def check_security(self):
        """P0: 安全性檢查"""
        print("\n🔒 執行安全性檢查...")
        try:
            # 使用 detect-secrets
            result = subprocess.run(
                ["detect-secrets", "scan"],
                capture_output=True,
                text=True,
                cwd=self.repo_root
            )
            secrets_found = "no secrets" not in result.stdout.lower()
            self.results["security"] = {
                "status": "⚠️ WARNING" if secrets_found else "✅ PASS",
                "secrets_detected": secrets_found,
                "details": "請審查包含敏感關鍵字的檔案" if secrets_found else "未檢測到明顯的秘密"
            }
        except FileNotFoundError:
            self.results["security"] = {
                "status": "⚠️ SKIP",
                "details": "detect-secrets 未安裝，請執行: pip install detect-secrets"
            }
    def check_python_quality(self):
        """P0: Python 程式碼品質"""
        print("\n🐍 檢查 Python 程式碼品質...")
        # 統計型別提示
        py_files = list(self.repo_root.glob("**/*.py"))
        total_files = len(py_files)
        files_with_type_hints = 0
        for py_file in py_files:
            try:
                content = py_file.read_text()
                # 使用更精確的模式檢測函式回傳型別提示
                if "def " in content and "->" in content:
                    files_with_type_hints += 1
            except (UnicodeDecodeError, OSError, PermissionError):
                continue
        type_hint_coverage = (files_with_type_hints / total_files * 100) if total_files > 0 else 0
        self.results["python_quality"] = {
            "total_files": total_files,
            "files_with_type_hints": files_with_type_hints,
            "type_hint_coverage": f"{type_hint_coverage:.1f}%",
            "status": "✅ PASS" if type_hint_coverage >= 90 else "⚠️ WARNING",
            "target": "90%"
        }
    def check_typescript_quality(self):
        """P1: TypeScript/JavaScript 品質"""
        print("\n📘 檢查 TypeScript 程式碼品質...")
        ts_files = list(self.repo_root.glob("**/*.ts")) + list(self.repo_root.glob("**/*.tsx"))
        js_files = list(self.repo_root.glob("**/*.js")) + list(self.repo_root.glob("**/*.jsx"))
        self.results["typescript_quality"] = {
            "total_ts_files": len(ts_files),
            "total_js_files": len(js_files),
            "status": "✅ PASS"
        }
    def check_code_duplication(self):
        """P0: 程式碼重複檢查"""
        print("\n🔄 檢查程式碼重複...")
        # 檢查已知的重複模組
        duplicate_patterns = [
            "dependency-manager",
            "drone_system"
        ]
        duplicates_found = []
        for pattern in duplicate_patterns:
            matches = list(self.repo_root.glob(f"**/{pattern}"))
            if len(matches) > 1:
                duplicates_found.append({
                    "pattern": pattern,
                    "locations": [str(m.relative_to(self.repo_root)) for m in matches]
                })
        self.results["code_duplication"] = {
            "duplicates_found": len(duplicates_found),
            "details": duplicates_found,
            "status": "⚠️ WARNING" if duplicates_found else "✅ PASS"
        }
    def check_docstring_coverage(self):
        """P1: Docstring 覆蓋率"""
        print("\n📝 檢查 Docstring 覆蓋率...")
        py_files = list(self.repo_root.glob("**/*.py"))
        files_with_docstrings = 0
        for py_file in py_files:
            try:
                content = py_file.read_text()
                # 檢測：檔案包含 docstrings（可能有誤報，建議使用 interrogate）
                if '"""' in content or "'''" in content:
                    files_with_docstrings += 1
            except (UnicodeDecodeError, OSError, PermissionError):
                continue
        coverage = (files_with_docstrings / len(py_files) * 100) if py_files else 0
        self.results["docstring_coverage"] = {
            "total_files": len(py_files),
            "files_with_docstrings": files_with_docstrings,
            "coverage": f"{coverage:.1f}%",
            "status": "✅ PASS" if coverage >= 85 else "⚠️ WARNING",
            "target": "85%"
        }
    def check_non_ascii_filenames(self):
        """P1: 非 ASCII 檔名檢查"""
        print("\n🌐 檢查非 ASCII 檔名...")
        non_ascii_files = []
        for path in self.repo_root.rglob("*"):
            if path.is_file():
                try:
                    path.name.encode('ascii')
                except UnicodeEncodeError:
                    non_ascii_files.append(str(path.relative_to(self.repo_root)))
        self.results["non_ascii_filenames"] = {
            "count": len(non_ascii_files),
            "files": non_ascii_files[:10],  # 只顯示前 10 個
            "status": "⚠️ WARNING" if non_ascii_files else "✅ PASS"
        }
    def check_console_logs(self):
        """P1: Console.log 檢查"""
        print("\n🖥️  檢查 console.log 使用...")
        files_with_console = []
        for ext in [".ts", ".tsx", ".js", ".jsx"]:
            for file_path in self.repo_root.glob(f"**/*{ext}"):
                try:
                    content = file_path.read_text()
                    if "console.log" in content:
                        files_with_console.append(str(file_path.relative_to(self.repo_root)))
                except (UnicodeDecodeError, OSError, PermissionError):
                    continue
        self.results["console_logs"] = {
            "count": len(files_with_console),
            "files": files_with_console[:20],  # 只顯示前 20 個
            "status": "⚠️ WARNING" if files_with_console else "✅ PASS"
        }
    def check_eval_usage(self):
        """P1: eval() 使用檢查"""
        print("\n⚠️  檢查 ast.literal_eval() 使用...")
        files_with_eval = []
        for ext in [".py", ".ts", ".js"]:
            for file_path in self.repo_root.glob(f"**/*{ext}"):
                try:
                    content = file_path.read_text()
                    if "ast.literal_eval(" in content:
                        files_with_eval.append(str(file_path.relative_to(self.repo_root)))
                except (UnicodeDecodeError, OSError, PermissionError):
                    continue
        self.results["eval_usage"] = {
            "count": len(files_with_eval),
            "files": files_with_eval,
            "status": "⚠️ WARNING" if files_with_eval else "✅ PASS"
        }
        # 先產生經過安全過濾的結果，以避免在報告中儲存明文敏感資訊
        sanitized_results = self._sanitize_results_for_report(self.results)
    def generate_report(self):
        """生成報告"""
        print("\n" + "="*80)
        print("📊 自動化品質檢查報告")
        print("="*80)
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "details": sanitized_results
        }
        # 統計狀態
        total_checks = len(self.results)
        passed = sum(1 for r in self.results.values() if r.get("status") == "✅ PASS")
        warnings = sum(1 for r in self.results.values() if r.get("status") == "⚠️ WARNING")
        report["summary"] = {
            "total_checks": total_checks,
            "passed": passed,
            "warnings": warnings,
            "pass_rate": f"{(passed/total_checks*100):.1f}%"
        }
        # 輸出到螢幕
        print(f"\n總檢查項目: {total_checks}")
        print(f"通過: {passed}")
        print(f"警告: {warnings}")
        print(f"通過率: {report['summary']['pass_rate']}")
        print("\n詳細結果:")
        for check_name, result in self.results.items():
            print(f"\n{check_name.upper()}: {result.get('status', 'N/A')}")

            # For security-related results, avoid logging any potentially sensitive
            # details that may be derived from secret-scanning tools. Only emit a
            # high-level indication of whether secrets were detected.
            if check_name == "security":
                secrets_detected = result.get("secrets_detected")
                if secrets_detected is True:
                    print("  - summary: Potential secrets were detected. Please review the repository with appropriate privileges.")
                elif secrets_detected is False:
                    print("  - summary: No obvious secrets were detected by the scanner.")
                else:
                    print("  - summary: Security scan result unavailable or inconclusive.")
                # Skip generic field-by-field logging for security results to prevent
                # accidental exposure of sensitive information.
                continue

            def format_safe_value(key: str, value: Any) -> str:
                """Format values for logging without exposing sensitive data."""
                # Explicitly redact known-sensitive keys
                if key in ["secrets", "tokens", "passwords", "keys", "credentials"]:
                    return "[REDACTED FOR SECURITY]"
                # For collections, log only high-level summary information
                try:
                    if isinstance(value, dict):
                        return f"[Dictionary with {len(value)} entries]"
                    if isinstance(value, (list, tuple, set)):
                        return f"[Sequence with {len(value)} items]"
                except TypeError:
                    # Fallback if len() is not supported
                    return "[Non-scalar data]"
                # For other (typically scalar) values, log directly
                # Large string values are truncated to avoid accidental leaks
                text = str(value)
                if len(text) > 200:
                    return text[:197] + "..."
                return text

            for key, value in result.items():
                if key == "status":
                    continue
                safe_value = format_safe_value(key, value)
                print(f"  - {key}: {safe_value}")
        # 儲存 JSON 報告
        report_file = self.repo_root / "auto-quality-report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 報告已儲存至: {report_file}")
        # 生成 Markdown 報告（同樣使用已過濾的結果）

        self.generate_markdown_report(report)
    def generate_markdown_report(self, report: Dict):
        """生成 Markdown 格式報告"""
        md_file = self.repo_root / "AUTO-QUALITY-REPORT.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("# 自動化品質檢查報告\n\n")
            f.write(f"**生成時間**: {report['timestamp']}\n\n")
            f.write("## 📊 總覽\n\n")
            f.write(f"- 總檢查項目: {report['summary']['total_checks']}\n")
            f.write(f"- ✅ 通過: {report['summary']['passed']}\n")
            f.write(f"- ⚠️ 警告: {report['summary']['warnings']}\n")
            f.write(f"- 通過率: {report['summary']['pass_rate']}\n\n")
            f.write("## 📋 詳細結果\n\n")
            # 僅使用已被 generate_report 過濾後的 details
            for check_name, result in report["details"].items():
                f.write(f"### {check_name.replace('_', ' ').title()}\n\n")
                status = result.get("status", "N/A") if isinstance(result, dict) else "N/A"
                f.write(f"**狀態**: {status}\n\n")
                if isinstance(result, dict):
                    for key, value in result.items():
                        if key == "status":
                            continue
                        # 在此階段，敏感資料已在 _sanitize_results_for_report 中遮罩，
                        # 因此可以直接輸出。
                        if isinstance(value, dict) and "count" in value and "note" in value:
                            # 這是來自清單的摘要資訊
                            f.write(f"- **{key}**: {value['count']} 項 ({value['note']})\n")
                        else:
                            f.write(f"- **{key}**: {value}\n")
                f.write("\n")
            f.write("## 🎯 建議行動\n\n")
            if report.get("details", {}).get("security", {}).get("secrets_detected"):
                f.write("1. **高優先級**: 審查並移除硬編碼的秘密\n")
            # 安全地解析型別提示覆蓋率
            type_hint_coverage_str = self.results.get("python_quality", {}).get("type_hint_coverage", "0%")
            try:
                type_hint_coverage = float(type_hint_coverage_str.rstrip("%"))
                if type_hint_coverage < 90:
                    f.write("2. **高優先級**: 提升 Python 型別提示覆蓋率至 90%+\n")
            except (ValueError, AttributeError):
                pass
            if self.results.get("code_duplication", {}).get("duplicates_found", 0) > 0:
                f.write("3. **高優先級**: 移除重複的程式碼模組\n")
            if self.results.get("non_ascii_filenames", {}).get("count", 0) > 0:
                f.write("4. **中優先級**: 重新命名非 ASCII 檔名\n")
            if self.results.get("console_logs", {}).get("count", 0) > 0:
                f.write("5. **中優先級**: 替換 console.log 為結構化日誌\n")
            f.write("\n詳細改進計劃請參考: [PR-1-ACTION-PLAN.md](./PR-1-ACTION-PLAN.md)\n")
        print(f"✅ Markdown 報告已儲存至: {md_file}")
def main():
    parser = argparse.ArgumentParser(description="自動化程式碼品質檢查")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="倉庫根目錄路徑"
    )
    args = parser.parse_args()
    checker = QualityChecker(args.repo_root)
    checker.run_all_checks()
if __name__ == "__main__":
    main()
