# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: validation_automation_engine
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Validation Automation Engine - 驗證全自動化引擎
100% 自主執行的驗證引擎，負責：
- 自動驗證結構正確性
- 自動驗證引用有效性
- 自動驗證內容格式
- 自動修復發現的問題
Version: 1.0.0
"""
# MNGA-002: Import organization needs review
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from engine_base import (
    EngineConfig,
    EngineType,
    ExecutionMode,
    TaskResult,
    ValidationEngineBase,
)
sys.path.insert(0, str(Path(__file__).parent.parent))
BASE_PATH = Path(__file__).parent.parent.parent.parent
PLAYBOOKS_PATH = BASE_PATH / "docs" / "refactor_playbooks"
class ValidationAutomationEngine(ValidationEngineBase):
    """驗證全自動化引擎"""
    ENGINE_TYPE = EngineType.VALIDATION
    def __init__(self, config: Optional[EngineConfig] = None):
        config = config or EngineConfig(
            engine_name="ValidationAutomationEngine",
            engine_type=EngineType.VALIDATION,
            execution_mode=ExecutionMode.AUTONOMOUS,
        )
        super().__init__(config)
        self._target_path = PLAYBOOKS_PATH
        self._auto_fix = True
    async def _initialize(self) -> bool:
        self._logger.info("初始化驗證自動化引擎...")
        return True
    async def _execute(self, task: Dict[str, Any]) -> TaskResult:
        task_id = task.get("task_id", "")
        operation = task.get("operation", "")
        try:
            if operation == "validate_all":
                result = await self.validate(self._target_path)
            elif operation == "validate_structure":
                result = await self._validate_structure()
            elif operation == "validate_references":
                result = await self._validate_references()
            elif operation == "validate_content":
                result = await self._validate_content()
            elif operation == "auto_fix":
                result = await self._auto_fix_all()
            else:
                return TaskResult(
                    task_id=task_id, success=False, error=f"未知操作: {operation}"
                )
            return TaskResult(task_id=task_id, success=True, result=result)
        except Exception as e:
            return TaskResult(task_id=task_id, success=False, error=str(e))
    async def _shutdown(self) -> bool:
        return True
    def _get_capabilities(self) -> Dict[str, Any]:
        return {
            "operations": [
                "validate_all",
                "validate_structure",
                "validate_references",
                "validate_content",
                "auto_fix",
            ],
            "auto_fix": self._auto_fix,
        }
    async def validate(self, target: Any, rules: List[str] = None) -> Dict[str, Any]:
        """執行完整驗證"""
        results = {
            "structure": await self._validate_structure(),
            "references": await self._validate_references(),
            "content": await self._validate_content(),
        }
        total_errors = sum(len(r.get("errors", [])) for r in results.values())
        total_warnings = sum(len(r.get("warnings", [])) for r in results.values())
        return {
            "valid": total_errors == 0,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "details": results,
        }
    async def fix(self, issues: List[Dict]) -> Dict[str, Any]:
        """自動修復問題"""
        fixed = 0
        failed = 0
        for issue in issues:
            try:
                if await self._fix_issue(issue):
                    fixed += 1
                else:
                    failed += 1
            except BaseException:
                failed += 1
        return {"fixed": fixed, "failed": failed}
    async def _validate_structure(self) -> Dict:
        """驗證結構"""
        errors = []
        warnings = []
        # 檢查必要目錄
        required = ["01_deconstruction", "02_integration", "03_refactor"]
        for dir_name in required:
            if not (self._target_path / dir_name).exists():
                errors.append(f"缺少必要目錄: {dir_name}")
        # 檢查深度
        for path in self._target_path.rglob("*"):
            depth = len(path.relative_to(self._target_path).parts)
            if depth > 5:
                warnings.append(f"路徑過深: {path.relative_to(self._target_path)}")
                break
        return {"errors": errors, "warnings": warnings}
    async def _validate_references(self) -> Dict:
        """驗證引用"""
        errors = []
        warnings = []
        for md_file in self._target_path.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)
                for text, href in links:
                    if href.startswith(("http://", "https://", "#", "mailto:")):
                        continue
                    resolved = (md_file.parent / href.split("#")[0]).resolve()
                    if not resolved.exists():
                        errors.append(f"{md_file.name}: 斷開的連結 -> {href}")
            except Exception as e:
                warnings.append(f"無法讀取 {md_file.name}: {e}")
        return {"errors": errors[:20], "warnings": warnings}
    async def _validate_content(self) -> Dict:
        """驗證內容"""
        errors = []
        warnings = []
        # 驗證 YAML 檔案
        for yaml_file in self._target_path.rglob("*.yaml"):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    yaml.safe_load(f)
            except yaml.YAMLError as e:
                errors.append(f"YAML 錯誤 {yaml_file.name}: {str(e)[:50]}")
        # 驗證 Markdown 標題
        for md_file in self._target_path.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                if not re.search(r"^#\s+", content, re.MULTILINE):
                    warnings.append(f"{md_file.name}: 缺少標題")
            except BaseException:
                pass
        return {"errors": errors, "warnings": warnings}
    async def _auto_fix_all(self) -> Dict:
        """自動修復所有問題"""
        validation = await self.validate(self._target_path)
        issues = []
        for category, result in validation.get("details", {}).items():
            for error in result.get("errors", []):
                issues.append({"category": category, "error": error})
        fix_result = await self.fix(issues)
        # 重新驗證
        revalidation = await self.validate(self._target_path)
        return {
            "initial_errors": validation["total_errors"],
            "fixed": fix_result["fixed"],
            "remaining_errors": revalidation["total_errors"],
        }
    async def _fix_issue(self, issue: Dict) -> bool:
        """修復單一問題"""
        category = issue.get("category")
        error = issue.get("error", "")
        if category == "structure":
            if "缺少必要目錄" in error:
                dir_name = error.split(": ")[-1]
                (self._target_path / dir_name).mkdir(parents=True, exist_ok=True)
                return True
        return False
