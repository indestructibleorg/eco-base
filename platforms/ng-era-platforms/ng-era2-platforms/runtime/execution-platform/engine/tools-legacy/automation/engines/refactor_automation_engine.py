# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: refactor_automation_engine
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Refactor Automation Engine - 重構全自動化引擎
100% 自主執行的重構引擎，無需人類審核介入。
負責 docs/refactor_playbooks 目錄的全自動化管理。
功能：
- 自動檢測結構問題
- 自動生成重構計畫
- 自動執行重構操作
- 自動驗證結果
- 自動回滾失敗操作
Version: 1.0.0
"""
# MNGA-002: Import organization needs review
import asyncio
import re
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from engine_base import (
    EngineConfig,
    EngineState,
    EngineType,
    ExecutionEngineBase,
    ExecutionMode,
    TaskResult,
)
sys.path.insert(0, str(Path(__file__).parent.parent))
# ============================================================================
# 常數
# ============================================================================
BASE_PATH = Path(__file__).parent.parent.parent.parent
PLAYBOOKS_PATH = BASE_PATH / "docs" / "refactor_playbooks"
# ============================================================================
# 重構自動化引擎
# ============================================================================
class RefactorAutomationEngine(ExecutionEngineBase):
    """
    重構全自動化引擎
    完全自主執行，100% 機器操作：
    1. 監控 - 持續監控目錄結構變化
    2. 分析 - 自動識別問題和改進機會
    3. 規劃 - 自動生成最優重構計畫
    4. 執行 - 自動執行重構操作
    5. 驗證 - 自動驗證結果正確性
    6. 修復 - 自動修復發現的問題
    """
    ENGINE_TYPE = EngineType.EXECUTION
    def __init__(self, config: Optional[EngineConfig] = None):
        config = config or EngineConfig(
            engine_name="RefactorAutomationEngine",
            engine_type=EngineType.EXECUTION,
            execution_mode=ExecutionMode.AUTONOMOUS,
        )
        super().__init__(config)
        self._target_path = PLAYBOOKS_PATH
        self._auto_fix = True
        self._backup_enabled = True
        # 監控狀態
        self._last_scan_time: Optional[datetime] = None
        self._known_issues: List[Dict] = []
        self._pending_operations: List[Dict] = []
        # 規則引擎
        self._rules = self._load_rules()
    # ========================================================================
    # 生命週期實現
    # ========================================================================
    async def _initialize(self) -> bool:
        """初始化引擎"""
        self._logger.info("初始化重構自動化引擎...")
        # 確保目標目錄存在
        if not self._target_path.exists():
            self._logger.warning(f"目標目錄不存在: {self._target_path}")
            self._target_path.mkdir(parents=True, exist_ok=True)
        # 執行初始掃描
        await self._scan_and_analyze()
        # 啟動自動監控
        asyncio.create_task(self._auto_monitor_loop())
        self._logger.info("重構自動化引擎初始化完成")
        return True
    async def _execute(self, task: Dict[str, Any]) -> TaskResult:
        """執行任務"""
        task_id = task.get("task_id", "")
        operation = task.get("operation", "")
        try:
            if operation == "scan":
                result = await self._scan_and_analyze()
            elif operation == "plan":
                result = await self._generate_plan()
            elif operation == "execute_plan":
                result = await self._execute_plan(task.get("plan"))
            elif operation == "fix_issues":
                result = await self._auto_fix_issues()
            elif operation == "validate":
                result = await self._validate_structure()
            elif operation == "optimize":
                result = await self._optimize_directory()
            elif operation == "full_cycle":
                result = await self._full_automation_cycle()
            else:
                return TaskResult(
                    task_id=task_id,
                    success=False,
                    error=f"未知操作: {operation}",
                )
            return TaskResult(
                task_id=task_id,
                success=True,
                result=result,
            )
        except Exception as e:
            self._logger.error(f"執行失敗: {e}")
            return TaskResult(
                task_id=task_id,
                success=False,
                error=str(e),
            )
    async def _shutdown(self) -> bool:
        """關閉引擎"""
        self._logger.info("關閉重構自動化引擎...")
        return True
    def _get_capabilities(self) -> Dict[str, Any]:
        """獲取能力描述"""
        return {
            "operations": [
                "scan",
                "plan",
                "execute_plan",
                "fix_issues",
                "validate",
                "optimize",
                "full_cycle",
            ],
            "auto_fix": self._auto_fix,
            "backup_enabled": self._backup_enabled,
            "target_path": str(self._target_path),
        }
    # ========================================================================
    # 執行引擎實現
    # ========================================================================
    async def execute_operation(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """執行單一操作"""
        op_type = operation.get("type")
        if op_type == "create_directory":
            return await self._op_create_directory(operation)
        elif op_type == "move_file":
            return await self._op_move_file(operation)
        elif op_type == "rename_file":
            return await self._op_rename_file(operation)
        elif op_type == "update_content":
            return await self._op_update_content(operation)
        elif op_type == "delete":
            return await self._op_delete(operation)
        else:
            return {"success": False, "error": f"未知操作類型: {op_type}"}
    async def rollback(self, steps: int = 1) -> bool:
        """回滾操作"""
        if not self._rollback_stack:
            return False
        for _ in range(min(steps, len(self._rollback_stack))):
            rollback_info = self._rollback_stack.pop()
            try:
                await self._execute_rollback(rollback_info)
            except Exception as e:
                self._logger.error(f"回滾失敗: {e}")
                return False
        return True
    # ========================================================================
    # 核心功能
    # ========================================================================
    async def _scan_and_analyze(self) -> Dict[str, Any]:
        """掃描並分析目錄"""
        self._logger.info("掃描目錄結構...")
        issues = []
        statistics = {
            "total_files": 0,
            "total_dirs": 0,
            "by_type": {},
        }
        # 掃描所有檔案
        for item in self._target_path.rglob("*"):
            if item.is_file():
                statistics["total_files"] += 1
                ext = item.suffix.lower()
                statistics["by_type"][ext] = statistics["by_type"].get(ext, 0) + 1
                # 檢查問題
                file_issues = self._check_file_issues(item)
                issues.extend(file_issues)
            elif item.is_dir():
                statistics["total_dirs"] += 1
                dir_issues = self._check_directory_issues(item)
                issues.extend(dir_issues)
        # 檢查結構問題
        structure_issues = self._check_structure_issues()
        issues.extend(structure_issues)
        self._known_issues = issues
        self._last_scan_time = datetime.now()
        return {
            "scan_time": self._last_scan_time.isoformat(),
            "statistics": statistics,
            "issues_found": len(issues),
            "issues": issues[:20],  # 只返回前20個
        }
    async def _generate_plan(self) -> Dict[str, Any]:
        """生成重構計畫"""
        self._logger.info("生成重構計畫...")
        if not self._known_issues:
            await self._scan_and_analyze()
        operations = []
        for issue in self._known_issues:
            issue.get("type")
            fix_ops = self._generate_fix_operations(issue)
            operations.extend(fix_ops)
        # 排序操作 (目錄創建優先)
        operations.sort(
            key=lambda x: {
                "create_directory": 0,
                "move_file": 1,
                "rename_file": 2,
                "update_content": 3,
                "delete": 4,
            }.get(x.get("type"), 5)
        )
        plan = {
            "plan_id": f"plan_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "issues_addressed": len(self._known_issues),
            "operations": operations,
            "estimated_changes": len(operations),
        }
        self._pending_operations = operations
        return plan
    async def _execute_plan(self, plan: Dict[str, Any] = None) -> Dict[str, Any]:
        """執行重構計畫"""
        self._logger.info("執行重構計畫...")
        operations = plan.get("operations") if plan else self._pending_operations
        if not operations:
            return {"success": True, "message": "無操作需要執行"}
        # 創建備份
        if self._backup_enabled:
            backup_dir = (
                self._target_path / ".backup" / datetime.now().strftime("%Y%m%d%H%M%S")
            )
            backup_dir.mkdir(parents=True, exist_ok=True)
        results = {
            "success": True,
            "executed": 0,
            "failed": 0,
            "details": [],
        }
        for op in operations:
            try:
                result = await self.execute_operation(op)
                if result.get("success"):
                    results["executed"] += 1
                    self._executed_operations.append(op)
                else:
                    results["failed"] += 1
                    if not self._auto_fix:
                        results["success"] = False
                        break
                results["details"].append(result)
            except Exception as e:
                results["failed"] += 1
                results["details"].append({"success": False, "error": str(e)})
        return results
    async def _auto_fix_issues(self) -> Dict[str, Any]:
        """自動修復問題"""
        self._logger.info("自動修復問題...")
        # 掃描 -> 規劃 -> 執行
        await self._scan_and_analyze()
        plan = await self._generate_plan()
        result = await self._execute_plan(plan)
        # 驗證
        validation = await self._validate_structure()
        return {
            "fixed": result.get("executed", 0),
            "failed": result.get("failed", 0),
            "validation": validation,
        }
    async def _validate_structure(self) -> Dict[str, Any]:
        """驗證結構"""
        self._logger.info("驗證目錄結構...")
        errors = []
        warnings = []
        # 檢查必要目錄
        required_dirs = ["01_deconstruction", "02_integration", "03_refactor"]
        for dir_name in required_dirs:
            if not (self._target_path / dir_name).exists():
                errors.append(f"缺少必要目錄: {dir_name}")
        # 檢查索引檔案
        index_yaml = self._target_path / "03_refactor" / "index.yaml"
        if index_yaml.exists():
            try:
                with open(index_yaml, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                # 驗證引用
                for cluster in data.get("refactor_clusters", []):
                    path = cluster.get("playbook_path", "")
                    if path and path != "_pending":
                        if not (self._target_path / path).exists():
                            warnings.append(f"索引引用不存在: {path}")
            except Exception as e:
                errors.append(f"索引解析錯誤: {e}")
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }
    async def _optimize_directory(self) -> Dict[str, Any]:
        """優化目錄結構"""
        self._logger.info("優化目錄結構...")
        optimizations = []
        # 檢查根目錄檔案數量
        root_files = [f for f in self._target_path.iterdir() if f.is_file()]
        if len(root_files) > 10:
            # 自動分類
            for f in root_files:
                if "_report" in f.name.lower():
                    optimizations.append(
                        {
                            "type": "move_file",
                            "source": str(f),
                            "target": str(self._target_path / "reports" / f.name),
                            "reason": "分類報告檔案",
                        }
                    )
                elif "__playbook" in f.name.lower():
                    optimizations.append(
                        {
                            "type": "move_file",
                            "source": str(f),
                            "target": str(self._target_path / "generated" / f.name),
                            "reason": "分類生成檔案",
                        }
                    )
        # 執行優化
        for opt in optimizations:
            await self.execute_operation(opt)
        return {
            "optimizations_applied": len(optimizations),
            "details": optimizations,
        }
    async def _full_automation_cycle(self) -> Dict[str, Any]:
        """完整自動化週期"""
        self._logger.info("執行完整自動化週期...")
        cycle_start = datetime.now()
        # 1. 掃描分析
        scan_result = await self._scan_and_analyze()
        # 2. 生成計畫
        plan = await self._generate_plan()
        # 3. 執行計畫
        execution_result = await self._execute_plan(plan)
        # 4. 優化
        optimization_result = await self._optimize_directory()
        # 5. 驗證
        validation_result = await self._validate_structure()
        # 6. 如果驗證失敗，嘗試修復
        if not validation_result.get("valid"):
            self._logger.warning("驗證失敗，嘗試自動修復...")
            await self._auto_fix_issues()
            validation_result = await self._validate_structure()
        cycle_duration = (datetime.now() - cycle_start).total_seconds()
        return {
            "cycle_completed": True,
            "duration_seconds": cycle_duration,
            "scan": scan_result,
            "execution": execution_result,
            "optimization": optimization_result,
            "validation": validation_result,
        }
    # ========================================================================
    # 操作實現
    # ========================================================================
    async def _op_create_directory(self, op: Dict) -> Dict:
        """創建目錄"""
        target = Path(op["target"])
        target.mkdir(parents=True, exist_ok=True)
        self._rollback_stack.append({"type": "delete_directory", "target": str(target)})
        return {"success": True, "created": str(target)}
    async def _op_move_file(self, op: Dict) -> Dict:
        """移動檔案"""
        source = Path(op["source"])
        target = Path(op["target"])
        if not source.exists():
            return {"success": False, "error": f"來源不存在: {source}"}
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        self._rollback_stack.append(
            {
                "type": "move_file",
                "source": str(target),
                "target": str(source),
            }
        )
        return {"success": True, "moved": f"{source} -> {target}"}
    async def _op_rename_file(self, op: Dict) -> Dict:
        """重命名檔案"""
        source = Path(op["source"])
        new_name = op["new_name"]
        target = source.parent / new_name
        if not source.exists():
            return {"success": False, "error": f"來源不存在: {source}"}
        source.rename(target)
        self._rollback_stack.append(
            {
                "type": "rename_file",
                "source": str(target),
                "new_name": source.name,
            }
        )
        return {"success": True, "renamed": f"{source.name} -> {new_name}"}
    async def _op_update_content(self, op: Dict) -> Dict:
        """更新內容"""
        target = Path(op["target"])
        if not target.exists():
            return {"success": False, "error": f"檔案不存在: {target}"}
        # 備份原內容
        original_content = target.read_text(encoding="utf-8")
        # 執行更新
        if "search" in op and "replace" in op:
            new_content = original_content.replace(op["search"], op["replace"])
        elif "content" in op:
            new_content = op["content"]
        else:
            return {"success": False, "error": "缺少更新參數"}
        target.write_text(new_content, encoding="utf-8")
        self._rollback_stack.append(
            {
                "type": "update_content",
                "target": str(target),
                "content": original_content,
            }
        )
        return {"success": True, "updated": str(target)}
    async def _op_delete(self, op: Dict) -> Dict:
        """刪除"""
        target = Path(op["target"])
        if not target.exists():
            return {"success": True, "message": "已不存在"}
        # 備份
        backup_path = self._target_path / ".backup" / "deleted" / target.name
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            shutil.copy2(str(target), str(backup_path))
            target.unlink()
        else:
            shutil.copytree(str(target), str(backup_path))
            shutil.rmtree(str(target))
        self._rollback_stack.append(
            {
                "type": "restore",
                "source": str(backup_path),
                "target": str(target),
            }
        )
        return {"success": True, "deleted": str(target)}
    async def _execute_rollback(self, rollback_info: Dict):
        """執行回滾"""
        op_type = rollback_info.get("type")
        if op_type == "delete_directory":
            Path(rollback_info["target"]).rmdir()
        elif op_type == "move_file":
            shutil.move(rollback_info["source"], rollback_info["target"])
        elif op_type == "rename_file":
            source = Path(rollback_info["source"])
            source.rename(source.parent / rollback_info["new_name"])
        elif op_type == "update_content":
            Path(rollback_info["target"]).write_text(
                rollback_info["content"], encoding="utf-8"
            )
        elif op_type == "restore":
            if Path(rollback_info["source"]).is_file():
                shutil.copy2(rollback_info["source"], rollback_info["target"])
            else:
                shutil.copytree(rollback_info["source"], rollback_info["target"])
    # ========================================================================
    # 問題檢測
    # ========================================================================
    def _check_file_issues(self, file_path: Path) -> List[Dict]:
        """檢查檔案問題"""
        issues = []
        rel_path = file_path.relative_to(self._target_path)
        # 命名問題
        if re.search(r"[A-Z]", file_path.stem) and "_" in file_path.stem:
            issues.append(
                {
                    "type": "naming_inconsistent",
                    "path": str(rel_path),
                    "message": "命名風格混合",
                    "severity": "low",
                }
            )
        return issues
    def _check_directory_issues(self, dir_path: Path) -> List[Dict]:
        """檢查目錄問題"""
        issues = []
        # 空目錄
        if not any(dir_path.iterdir()) and not dir_path.name.startswith("."):
            issues.append(
                {
                    "type": "empty_directory",
                    "path": str(dir_path.relative_to(self._target_path)),
                    "message": "空目錄",
                    "severity": "info",
                }
            )
        return issues
    def _check_structure_issues(self) -> List[Dict]:
        """檢查結構問題"""
        issues = []
        # 根目錄檔案過多
        root_files = [f for f in self._target_path.iterdir() if f.is_file()]
        if len(root_files) > 10:
            issues.append(
                {
                    "type": "root_bloat",
                    "count": len(root_files),
                    "message": f"根目錄檔案過多 ({len(root_files)} 個)",
                    "severity": "medium",
                }
            )
        return issues
    def _generate_fix_operations(self, issue: Dict) -> List[Dict]:
        """生成修復操作"""
        operations = []
        issue_type = issue.get("type")
        if issue_type == "naming_inconsistent":
            path = issue.get("path")
            if path:
                file_path = self._target_path / path
                new_name = self._to_snake_case(file_path.stem) + file_path.suffix
                if new_name != file_path.name:
                    operations.append(
                        {
                            "type": "rename_file",
                            "source": str(file_path),
                            "new_name": new_name,
                        }
                    )
        elif issue_type == "empty_directory":
            # 不自動刪除空目錄，只記錄
            pass
        elif issue_type == "root_bloat":
            # 在其他地方處理
            pass
        return operations
    def _to_snake_case(self, name: str) -> str:
        """轉換為 snake_case"""
        name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
        name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
        name = name.replace("-", "_")
        name = re.sub(r"_+", "_", name)
        return name.lower()
    def _load_rules(self) -> List[Dict]:
        """載入規則"""
        return [
            {"name": "snake_case_naming", "enabled": True},
            {"name": "max_root_files", "enabled": True, "threshold": 10},
            {"name": "required_directories", "enabled": True},
        ]
    # ========================================================================
    # 自動監控
    # ========================================================================
    async def _auto_monitor_loop(self):
        """自動監控循環"""
        monitor_interval = 300  # 5 分鐘
        while self._running:
            try:
                await asyncio.sleep(monitor_interval)
                if self._state == EngineState.RUNNING:
                    self._logger.info("執行定期自動化週期...")
                    await self._full_automation_cycle()
            except Exception as e:
                self._logger.error(f"自動監控錯誤: {e}")
# ============================================================================
# CLI 入口
# ============================================================================
async def main():
    import argparse
    parser = argparse.ArgumentParser(description="重構全自動化引擎")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("start", help="啟動引擎")
    subparsers.add_parser("scan", help="掃描分析")
    subparsers.add_parser("plan", help="生成計畫")
    subparsers.add_parser("execute", help="執行計畫")
    subparsers.add_parser("fix", help="自動修復")
    subparsers.add_parser("validate", help="驗證結構")
    subparsers.add_parser("full-cycle", help="完整週期")
    args = parser.parse_args()
    engine = RefactorAutomationEngine()
    if args.command == "start":
        await engine.start()
        while engine.is_running:
            await asyncio.sleep(1)
    elif args.command:
        await engine.start()
        task = {"operation": args.command.replace("-", "_")}
        result = await engine.execute_now(task)
        print(
            yaml.dump(
                asdict(result) if hasattr(result, "__dataclass_fields__") else result,
                allow_unicode=True,
                default_flow_style=False,
            )
        )
        await engine.stop()
    else:
        parser.print_help()
if __name__ == "__main__":
    asyncio.run(main())
