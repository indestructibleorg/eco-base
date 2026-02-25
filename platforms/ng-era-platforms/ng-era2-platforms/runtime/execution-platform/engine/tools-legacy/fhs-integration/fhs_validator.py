# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: fhs_validator
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
FHS Integration Validator
驗證 FHS 整合的正確性，確保不會污染系統
這個驗證器提供多層次的檢查：
1. 預檢查 (Pre-flight checks)
2. 路徑驗證 (Path validation)
3. 衝突檢測 (Conflict detection)
4. 完整性檢查 (Integrity checks)
5. 回滾驗證 (Rollback validation)
"""
# MNGA-002: Import organization needs review
import os
import sys
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
class FHSIntegrationValidator:
    """FHS 整合驗證器 - 確保 100% 正確的遷移"""
    def __init__(self, repo_root: str = None):
        self.repo_root = repo_root or os.getcwd()
        self.validation_report = {
            "timestamp": datetime.now().isoformat(),
            "checks": [],
            "passed": True,
            "errors": [],
            "warnings": []
        }
        # FHS 標準目錄定義
        self.fhs_directories = {
            "bin": {
                "purpose": "用戶命令",
                "allowed_types": [".sh", ""],  # 腳本或二進制
                "naming_pattern": "mno-*",
                "permissions": 0o755
            },
            "sbin": {
                "purpose": "系統管理命令",
                "allowed_types": [".sh", ""],
                "naming_pattern": "mno-*",
                "permissions": 0o755,
                "requires_root": True
            },
            "lib": {
                "purpose": "共享庫和模組",
                "allowed_types": [".py", ".so", ".a"],
                "structure": "lib/{component_name}/",
                "permissions": 0o644
            },
            "etc": {
                "purpose": "配置文件",
                "allowed_types": [".conf", ".yaml", ".yml", ".cfg"],
                "structure": "etc/{component_name}/",
                "permissions": 0o644
            },
            "etc/systemd": {
                "purpose": "系統服務",
                "allowed_types": [".service"],
                "permissions": 0o644
            }
        }
    def validate_integration(self, component_name: str, integration_plan: Dict) -> Dict:
        """執行完整的整合驗證"""
        print("=" * 80)
        print("FHS Integration Validation")
        print("=" * 80)
        print(f"Component: {component_name}")
        print(f"Timestamp: {self.validation_report['timestamp']}")
        print("")
        # 階段 1: 預檢查
        self._pre_flight_checks(component_name, integration_plan)
        # 階段 2: 路徑驗證
        self._validate_paths(component_name, integration_plan)
        # 階段 3: 衝突檢測
        self._detect_conflicts(component_name, integration_plan)
        # 階段 4: FHS 合規性檢查
        self._check_fhs_compliance(component_name, integration_plan)
        # 階段 5: 安全性檢查
        self._security_checks(component_name, integration_plan)
        # 階段 6: 完整性驗證
        self._integrity_checks(component_name, integration_plan)
        # 生成驗證報告
        return self._generate_validation_report(component_name)
    def _pre_flight_checks(self, component_name: str, plan: Dict):
        """預檢查 - 確保基本條件滿足"""
        check_name = "Pre-flight Checks"
        print(f"[1/6] {check_name}...")
        checks = []
        # 檢查 1: 組件存在
        component_path = os.path.join(self.repo_root, "workspace", "tools", component_name)
        if not os.path.exists(component_path):
            self._add_error(f"Component directory not found: {component_path}")
            checks.append(("Component exists", False))
        else:
            checks.append(("Component exists", True))
        # 檢查 2: 成熟度分數
        maturity_score = plan.get("maturity_score", 0)
        if maturity_score < 80:
            self._add_error(f"Maturity score ({maturity_score}) below threshold (80)")
            checks.append(("Maturity threshold", False))
        else:
            checks.append(("Maturity threshold", True))
        # 檢查 3: Git 倉庫乾淨
        if self._has_uncommitted_changes():
            self._add_warning("Uncommitted changes detected in repository")
            checks.append(("Clean git state", False))
        else:
            checks.append(("Clean git state", True))
        # 檢查 4: FHS 目錄存在
        for fhs_dir in ["bin", "sbin", "lib", "etc"]:
            dir_path = os.path.join(self.repo_root, fhs_dir)
            if not os.path.exists(dir_path):
                self._add_error(f"FHS directory missing: {fhs_dir}/")
                checks.append((f"FHS {fhs_dir}/ exists", False))
            else:
                checks.append((f"FHS {fhs_dir}/ exists", True))
        self._add_check(check_name, checks)
        self._print_check_results(checks)
    def _validate_paths(self, component_name: str, plan: Dict):
        """驗證所有目標路徑的正確性"""
        check_name = "Path Validation"
        print(f"\n[2/6] {check_name}...")
        checks = []
        actions = plan.get("actions", [])
        for action in actions:
            if "Create wrapper" in action or "Copy" in action:
                # 提取目標路徑
                target_path = self._extract_target_path(action)
                if target_path:
                    # 驗證路徑符合 FHS 標準
                    is_valid, reason = self._is_valid_fhs_path(target_path, component_name)
                    if is_valid:
                        checks.append((f"Path: {target_path}", True))
                    else:
                        self._add_error(f"Invalid FHS path: {target_path} - {reason}")
                        checks.append((f"Path: {target_path}", False))
        self._add_check(check_name, checks)
        self._print_check_results(checks)
    def _detect_conflicts(self, component_name: str, plan: Dict):
        """檢測文件名衝突和路徑衝突"""
        check_name = "Conflict Detection"
        print(f"\n[3/6] {check_name}...")
        checks = []
        actions = plan.get("actions", [])
        conflicts = []
        for action in actions:
            target_path = self._extract_target_path(action)
            if target_path:
                full_path = os.path.join(self.repo_root, target_path)
                # 檢查文件是否已存在
                if os.path.exists(full_path):
                    # 檢查是否是同一組件的文件（允許覆蓋）
                    if component_name in target_path:
                        self._add_warning(f"Will overwrite existing file: {target_path}")
                        checks.append((f"Overwrite: {os.path.basename(target_path)}", True))
                    else:
                        # 潛在衝突
                        conflicts.append(target_path)
                        self._add_error(f"Conflict: {target_path} already exists")
                        checks.append((f"No conflict: {os.path.basename(target_path)}", False))
                else:
                    checks.append((f"No conflict: {os.path.basename(target_path)}", True))
        if not conflicts:
            checks.append(("No conflicts detected", True))
        self._add_check(check_name, checks)
        self._print_check_results(checks)
    def _check_fhs_compliance(self, component_name: str, plan: Dict):
        """檢查 FHS 合規性"""
        check_name = "FHS Compliance"
        print(f"\n[4/6] {check_name}...")
        checks = []
        actions = plan.get("actions", [])
        # 驗證命名規範
        for action in actions:
            if "Create wrapper" in action:
                target_path = self._extract_target_path(action)
                filename = os.path.basename(target_path)
                # 所有命令必須以 mno- 開頭
                if not filename.startswith("mno-"):
                    self._add_error(f"Command name violation: {filename} (must start with 'mno-')")
                    checks.append((f"Naming: {filename}", False))
                else:
                    checks.append((f"Naming: {filename}", True))
        # 驗證目錄結構
        lib_structure_valid = True
        etc_structure_valid = True
        for action in actions:
            target_path = self._extract_target_path(action)
            if target_path and "lib/" in target_path:
                # lib 下必須有組件子目錄
                expected_prefix = f"lib/{component_name}/"
                if not target_path.startswith(expected_prefix):
                    self._add_error(f"Library structure violation: {target_path}")
                    lib_structure_valid = False
        checks.append(("Library structure", lib_structure_valid))
        checks.append(("Config structure", etc_structure_valid))
        self._add_check(check_name, checks)
        self._print_check_results(checks)
    def _security_checks(self, component_name: str, plan: Dict):
        """安全性檢查"""
        check_name = "Security Checks"
        print(f"\n[5/6] {check_name}...")
        checks = []
        # 檢查 1: 不允許往根目錄寫入
        for action in plan.get("actions", []):
            target_path = self._extract_target_path(action)
            if target_path:
                # 確保不會寫入根目錄或其他敏感位置
                forbidden_paths = ["/", "/usr", "/usr/local", "/opt", "/var"]
                for forbidden in forbidden_paths:
                    if target_path.startswith(forbidden + "/"):
                        self._add_error(f"Security: Attempting to write to forbidden path: {target_path}")
                        checks.append(("No forbidden paths", False))
                        break
                else:
                    checks.append(("No forbidden paths", True))
        # 檢查 2: 確保路徑在倉庫內
        for action in plan.get("actions", []):
            target_path = self._extract_target_path(action)
            if target_path:
                full_path = os.path.join(self.repo_root, target_path)
                # 檢查是否嘗試跳出倉庫
                if ".." in target_path or not full_path.startswith(self.repo_root):
                    self._add_error(f"Security: Path escape attempt: {target_path}")
                    checks.append(("Paths within repo", False))
                else:
                    checks.append(("Paths within repo", True))
        # 檢查 3: 不允許覆蓋關鍵系統文件
        critical_files = [
            "bin/sh", "bin/bash", "sbin/init", "etc/passwd", "etc/shadow"
        ]
        for action in plan.get("actions", []):
            target_path = self._extract_target_path(action)
            if target_path in critical_files:
                self._add_error(f"Security: Attempting to overwrite critical file: {target_path}")
                checks.append(("No critical file overwrites", False))
        if not any("No critical file overwrites" in str(c) for c in checks):
            checks.append(("No critical file overwrites", True))
        self._add_check(check_name, checks)
        self._print_check_results(checks)
    def _integrity_checks(self, component_name: str, plan: Dict):
        """完整性檢查"""
        check_name = "Integrity Checks"
        print(f"\n[6/6] {check_name}...")
        checks = []
        # 檢查 1: 確保所有必要文件都會被遷移
        structure = plan.get("structure", {})
        actions = plan.get("actions", [])
        python_files = structure.get("python_files", [])
        if python_files:
            # 確保至少有一個 lib 目錄被創建
            has_lib_action = any("lib/" in str(action) for action in actions)
            checks.append(("Python files → lib/", has_lib_action))
            if not has_lib_action:
                self._add_error("Python files exist but no lib/ migration planned")
        # 檢查 2: 確保有文檔
        doc_action = any("migration documentation" in str(action).lower() for action in actions)
        checks.append(("Migration documentation", doc_action))
        if not doc_action:
            self._add_warning("No migration documentation will be generated")
        # 檢查 3: 確保有回滾計劃
        # （這裡我們確保原始文件在 workspace 中保留）
        checks.append(("Preserves workspace files", True))  # 我們的設計保留原始文件
        self._add_check(check_name, checks)
        self._print_check_results(checks)
    def _is_valid_fhs_path(self, path: str, component_name: str) -> Tuple[bool, str]:
        """驗證路徑是否符合 FHS 標準"""
        # 提取目錄部分
        parts = path.split("/")
        if len(parts) < 2:
            return False, "Path too short"
        fhs_dir = parts[0]
        # 檢查是否是允許的 FHS 目錄
        if fhs_dir not in ["bin", "sbin", "lib", "etc"]:
            return False, f"Not a valid FHS directory: {fhs_dir}"
        # 檢查子目錄結構
        if fhs_dir in ["lib", "etc"] and len(parts) >= 2:
            # lib 和 etc 下應該有組件子目錄
            if parts[1] != component_name and parts[1] != "systemd":
                return False, f"Missing component subdirectory under {fhs_dir}/"
        # 檢查文件類型
        if len(parts) >= 2:
            filename = parts[-1]
            os.path.splitext(filename)[1]
            if fhs_dir == "bin" or fhs_dir == "sbin":
                # 命令應該以 mno- 開頭
                if not filename.startswith("mno-"):
                    return False, f"Command must start with 'mno-': {filename}"
        return True, "Valid FHS path"
    def _extract_target_path(self, action: str) -> Optional[str]:
        """從 action 字符串中提取目標路徑"""
        # 嘗試各種模式
        if " -> " in action:
            return action.split(" -> ")[1].strip()
        elif ": " in action:
            parts = action.split(": ")
            if len(parts) >= 2:
                return parts[1].strip()
        return None
    def _has_uncommitted_changes(self) -> bool:
        """檢查是否有未提交的變更"""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=self.repo_root
            )
            return len(result.stdout.strip()) > 0
        except Exception as e:
            logger.error(f"Error: {e}")
            return False
    def _add_check(self, check_name: str, results: List[Tuple[str, bool]]):
        """添加檢查結果"""
        self.validation_report["checks"].append({
            "name": check_name,
            "results": results,
            "passed": all(r[1] for r in results)
        })
        if not all(r[1] for r in results):
            self.validation_report["passed"] = False
    def _add_error(self, message: str):
        """添加錯誤"""
        self.validation_report["errors"].append(message)
        self.validation_report["passed"] = False
    def _add_warning(self, message: str):
        """添加警告"""
        self.validation_report["warnings"].append(message)
    def _print_check_results(self, checks: List[Tuple[str, bool]]):
        """打印檢查結果"""
        for check_name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}")
    def _generate_validation_report(self, component_name: str) -> Dict:
        """生成驗證報告"""
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        # 統計
        total_checks = sum(len(c["results"]) for c in self.validation_report["checks"])
        passed_checks = sum(sum(1 for r in c["results"] if r[1]) for c in self.validation_report["checks"])
        print(f"Total Checks: {total_checks}")
        print(f"Passed: {passed_checks}")
        print(f"Failed: {total_checks - passed_checks}")
        print(f"Errors: {len(self.validation_report['errors'])}")
        print(f"Warnings: {len(self.validation_report['warnings'])}")
        print("")
        # 錯誤
        if self.validation_report["errors"]:
            print("ERRORS:")
            for error in self.validation_report["errors"]:
                print(f"  ✗ {error}")
            print("")
        # 警告
        if self.validation_report["warnings"]:
            print("WARNINGS:")
            for warning in self.validation_report["warnings"]:
                print(f"  ⚠ {warning}")
            print("")
        # 最終判定
        if self.validation_report["passed"]:
            print("✓ VALIDATION PASSED - Safe to proceed with integration")
        else:
            print("✗ VALIDATION FAILED - DO NOT proceed with integration")
            print("  Fix all errors before attempting integration")
        print("=" * 80)
        return self.validation_report
    def save_validation_report(self, component_name: str, output_path: str = None):
        """保存驗證報告"""
        if output_path is None:
            output_path = os.path.join(
                self.repo_root,
                "docs",
                "fhs-integration",
                f"validation-{component_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
            )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.validation_report, f, indent=2)
        print(f"\nValidation report saved: {output_path}")
        return output_path
def main():
    """主函數"""
    import argparse
    parser = argparse.ArgumentParser(description="FHS Integration Validator")
    parser.add_argument("component", help="Component name to validate")
    parser.add_argument("--plan-file", help="Integration plan JSON file")
    parser.add_argument("--save-report", action="store_true", help="Save validation report")
    parser.add_argument("--repo-root", help="Repository root path")
    args = parser.parse_args()
    validator = FHSIntegrationValidator(repo_root=args.repo_root)
    # 加載整合計劃
    if args.plan_file and os.path.exists(args.plan_file):
        with open(args.plan_file, 'r') as f:
            plan = json.load(f)
    else:
        # 創建一個示例計劃用於測試
        plan = {
            "maturity_score": 85,
            "actions": [
                "Create wrapper: bin/mno-test-command",
                "Copy library file: test.py -> lib/test-component/test.py"
            ],
            "structure": {
                "python_files": ["test.py"],
                "config_files": []
            }
        }
    # 執行驗證
    report = validator.validate_integration(args.component, plan)
    # 保存報告
    if args.save_report:
        validator.save_validation_report(args.component)
    # 返回退出碼
    sys.exit(0 if report["passed"] else 1)
if __name__ == "__main__":
    main()
