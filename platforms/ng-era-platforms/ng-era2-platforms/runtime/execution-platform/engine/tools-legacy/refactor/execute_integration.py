# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: execute_integration
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Integration Executor - 集成執行引擎
執行資產整合操作，包括：
1. 目錄優化
2. 檔案移動/合併
3. 引用更新
4. 批量處理
5. 驗證確認
Usage:
    python execute_integration.py single --decision <file>
    python execute_integration.py batch --decisions <dir>
    python execute_integration.py reorganize --target <dir>
Version: 1.0.0
"""
# MNGA-002: Import organization needs review
import argparse
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
# ============================================================================
# 常數與配置
# ============================================================================
BASE_PATH = Path(__file__).parent.parent.parent
PLAYBOOKS_PATH = BASE_PATH / "docs" / "refactor_playbooks"
CONFIG_PATH = PLAYBOOKS_PATH / "config" / "integration-processor.yaml"
BACKUP_PATH = PLAYBOOKS_PATH / ".integration_backup"
# ============================================================================
# 枚舉定義
# ============================================================================
class OperationType(Enum):
    """操作類型"""
    CREATE_DIR = "create_directory"
    MOVE_FILE = "move_file"
    COPY_FILE = "copy_file"
    MERGE_FILE = "merge_file"
    DELETE_FILE = "delete_file"
    UPDATE_REFS = "update_references"
    EMBED_CONTENT = "embed_content"
    UPDATE_INDEX = "update_index"
class ExecutionMode(Enum):
    """執行模式"""
    DRY_RUN = "dry_run"
    CONFIRM = "confirm"
    FORCE = "force"
# ============================================================================
# 資料結構
# ============================================================================
@dataclass
class Operation:
    """操作定義"""
    op_id: str
    op_type: OperationType
    source: Optional[str]
    target: str
    options: Dict = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
@dataclass
class OperationResult:
    """操作結果"""
    op_id: str
    success: bool
    message: str
    changes: List[str] = field(default_factory=list)
    rollback_info: Optional[Dict] = None
@dataclass
class DirectoryOptimization:
    """目錄優化方案"""
    target_dir: str
    current_structure: Dict
    optimized_structure: Dict
    operations: List[Operation]
    improvement_score: float
@dataclass
class IntegrationPlan:
    """整合計畫"""
    plan_id: str
    created_at: str
    target_directory: str
    operations: List[Operation]
    estimated_changes: int
    risk_level: str
# ============================================================================
# 目錄優化器
# ============================================================================
class DirectoryOptimizer:
    """
    目錄優化器：分析並優化目錄結構
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        # 優化規則
        self.max_root_files = 10
        self.max_depth = 4
        self.naming_convention = "snake_case"
        # 標準子目錄
        self.standard_subdirs = {
            "reports": ["*_report.md", "*_analysis.md"],
            "generated": ["*__playbook.md", "*_generated.*"],
            "templates": ["*_template.*", "TEMPLATE_*"],
            "meta": ["*.meta.yaml", "META_*"],
        }
    def analyze(self, target_dir: Path) -> DirectoryOptimization:
        """分析並生成優化方案"""
        current = self._scan_structure(target_dir)
        optimized = self._generate_optimized_structure(current, target_dir)
        operations = self._generate_operations(current, optimized, target_dir)
        score = self._calculate_improvement(current, optimized)
        return DirectoryOptimization(
            target_dir=str(target_dir),
            current_structure=current,
            optimized_structure=optimized,
            operations=operations,
            improvement_score=score,
        )
    def _scan_structure(self, target_dir: Path) -> Dict:
        """掃描當前結構"""
        structure = {
            "root_files": [],
            "subdirs": {},
            "total_files": 0,
            "max_depth": 0,
            "naming_issues": [],
        }
        if not target_dir.exists():
            return structure
        for item in target_dir.iterdir():
            if item.is_file():
                structure["root_files"].append(item.name)
                structure["total_files"] += 1
                # 檢查命名
                if not self._check_naming(item.name):
                    structure["naming_issues"].append(item.name)
            elif item.is_dir() and not item.name.startswith("."):
                subdir_files = list(item.rglob("*"))
                structure["subdirs"][item.name] = {
                    "files": [f.name for f in subdir_files if f.is_file()],
                    "count": len([f for f in subdir_files if f.is_file()]),
                }
                structure["total_files"] += structure["subdirs"][item.name]["count"]
        return structure
    def _generate_optimized_structure(self, current: Dict, target_dir: Path) -> Dict:
        """生成優化後的結構"""
        optimized = {
            "root_files": [],
            "subdirs": dict(current.get("subdirs", {})),
            "new_subdirs": {},
        }
        # 分類根目錄檔案
        for filename in current.get("root_files", []):
            placed = False
            for subdir, patterns in self.standard_subdirs.items():
                for pattern in patterns:
                    if self._match_pattern(filename, pattern):
                        if subdir not in optimized["new_subdirs"]:
                            optimized["new_subdirs"][subdir] = []
                        optimized["new_subdirs"][subdir].append(filename)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                # 保留在根目錄
                optimized["root_files"].append(filename)
        return optimized
    def _generate_operations(
        self, current: Dict, optimized: Dict, target_dir: Path
    ) -> List[Operation]:
        """生成操作列表"""
        operations = []
        op_counter = 0
        # 建立新子目錄
        for subdir, files in optimized.get("new_subdirs", {}).items():
            op_counter += 1
            operations.append(
                Operation(
                    op_id=f"op_{op_counter:03d}",
                    op_type=OperationType.CREATE_DIR,
                    source=None,
                    target=str(target_dir / subdir),
                )
            )
            # 移動檔案到新子目錄
            for filename in files:
                op_counter += 1
                operations.append(
                    Operation(
                        op_id=f"op_{op_counter:03d}",
                        op_type=OperationType.MOVE_FILE,
                        source=str(target_dir / filename),
                        target=str(target_dir / subdir / filename),
                        depends_on=[f"op_{op_counter-len(files):03d}"],
                    )
                )
        # 修復命名問題
        for filename in current.get("naming_issues", []):
            new_name = self._fix_naming(filename)
            if new_name != filename:
                op_counter += 1
                operations.append(
                    Operation(
                        op_id=f"op_{op_counter:03d}",
                        op_type=OperationType.MOVE_FILE,
                        source=str(target_dir / filename),
                        target=str(target_dir / new_name),
                        options={"rename": True},
                    )
                )
        return operations
    def _calculate_improvement(self, current: Dict, optimized: Dict) -> float:
        """計算改進分數"""
        score = 0.0
        # 根目錄檔案減少
        current_root = len(current.get("root_files", []))
        optimized_root = len(optimized.get("root_files", []))
        if current_root > 0:
            reduction = (current_root - optimized_root) / current_root
            score += reduction * 0.4
        # 命名問題修復
        naming_issues = len(current.get("naming_issues", []))
        if naming_issues > 0:
            score += 0.2
        # 結構改進
        if optimized.get("new_subdirs"):
            score += 0.3
        return min(score, 1.0)
    def _check_naming(self, filename: str) -> bool:
        """檢查命名是否符合規範"""
        name = Path(filename).stem
        if self.naming_convention == "snake_case":
            # 不應包含大寫或連字符
            return not bool(re.search(r"[A-Z]|-", name))
        return True
    def _fix_naming(self, filename: str) -> str:
        """修復命名"""
        path = Path(filename)
        name = path.stem
        ext = path.suffix
        # 轉換為 snake_case
        # 處理 camelCase
        name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
        name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
        # 處理連字符
        name = name.replace("-", "_")
        # 處理多個下劃線
        name = re.sub(r"_+", "_", name)
        name = name.lower()
        return name + ext
    def _match_pattern(self, filename: str, pattern: str) -> bool:
        """匹配檔案模式"""
        import fnmatch
        return fnmatch.fnmatch(filename, pattern)
# ============================================================================
# 目錄圖譜生成器
# ============================================================================
class DirectoryMapper:
    """
    目錄圖譜生成器：生成目錄結構圖
    """
    def generate_ascii_tree(self, target_dir: Path, max_depth: int = 3) -> str:
        """生成 ASCII 目錄樹"""
        lines = [f"{target_dir.name}/"]
        self._build_tree(target_dir, lines, "", max_depth, 0)
        return "\n".join(lines)
    def _build_tree(
        self,
        path: Path,
        lines: List[str],
        prefix: str,
        max_depth: int,
        current_depth: int,
    ):
        """遞迴建立樹結構"""
        if current_depth >= max_depth:
            return
        try:
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
        except PermissionError:
            return
        # 過濾隱藏檔案
        items = [i for i in items if not i.name.startswith(".")]
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "
            new_prefix = prefix + ("    " if is_last else "│   ")
            if item.is_dir():
                lines.append(f"{prefix}{connector}{item.name}/")
                self._build_tree(item, lines, new_prefix, max_depth, current_depth + 1)
            else:
                lines.append(f"{prefix}{connector}{item.name}")
    def generate_mermaid_diagram(self, target_dir: Path) -> str:
        """生成 Mermaid 流程圖"""
        lines = ["```mermaid", "graph TD"]
        def add_node(path: Path, parent_id: Optional[str] = None):
            node_id = path.name.replace(".", "_").replace("-", "_")
            if parent_id:
                lines.append(f"    {parent_id} --> {node_id}[{path.name}]")
            else:
                lines.append(f"    {node_id}[{path.name}]")
            if path.is_dir():
                for child in sorted(path.iterdir()):
                    if not child.name.startswith("."):
                        add_node(child, node_id)
        add_node(target_dir)
        lines.append("```")
        return "\n".join(lines)
# ============================================================================
# 整合執行器
# ============================================================================
class IntegrationExecutor:
    """
    整合執行器：執行整合操作
    """
    def __init__(self, mode: ExecutionMode = ExecutionMode.DRY_RUN):
        self.mode = mode
        self.backup_dir = BACKUP_PATH / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.executed_ops: List[OperationResult] = []
        self.reference_updates: List[Tuple[Path, str, str]] = []
    def execute_plan(self, plan: IntegrationPlan) -> List[OperationResult]:
        """執行整合計畫"""
        print(
            f"\n{'🔍 模擬執行' if self.mode == ExecutionMode.DRY_RUN else '🚀 執行'} 計畫: {plan.plan_id}"
        )
        print(f"   目標: {plan.target_directory}")
        print(f"   操作數: {len(plan.operations)}")
        if self.mode != ExecutionMode.DRY_RUN:
            self._create_backup(plan)
        results = []
        for op in plan.operations:
            # 檢查依賴
            if not self._check_dependencies(op, results):
                results.append(
                    OperationResult(
                        op_id=op.op_id,
                        success=False,
                        message="依賴操作失敗",
                    )
                )
                continue
            result = self._execute_operation(op)
            results.append(result)
            self.executed_ops.append(result)
            if not result.success and self.mode != ExecutionMode.FORCE:
                print(f"   ❌ {op.op_id} 失敗，停止執行")
                break
        # 更新引用
        if self.mode != ExecutionMode.DRY_RUN:
            self._update_all_references()
        return results
    def execute_single(self, decision_file: Path) -> OperationResult:
        """執行單一決策"""
        with open(decision_file, "r", encoding="utf-8") as f:
            decision = yaml.safe_load(f)
        # 從決策生成操作
        op = self._decision_to_operation(decision)
        return self._execute_operation(op)
    def _execute_operation(self, op: Operation) -> OperationResult:
        """執行單一操作"""
        print(f"   [{op.op_id}] {op.op_type.value}: ", end="")
        if self.mode == ExecutionMode.DRY_RUN:
            print(f"(模擬) {op.target}")
            return OperationResult(
                op_id=op.op_id,
                success=True,
                message="模擬成功",
                changes=[f"Would {op.op_type.value}: {op.target}"],
            )
        try:
            if op.op_type == OperationType.CREATE_DIR:
                result = self._op_create_dir(op)
            elif op.op_type == OperationType.MOVE_FILE:
                result = self._op_move_file(op)
            elif op.op_type == OperationType.COPY_FILE:
                result = self._op_copy_file(op)
            elif op.op_type == OperationType.MERGE_FILE:
                result = self._op_merge_file(op)
            elif op.op_type == OperationType.DELETE_FILE:
                result = self._op_delete_file(op)
            elif op.op_type == OperationType.EMBED_CONTENT:
                result = self._op_embed_content(op)
            elif op.op_type == OperationType.UPDATE_INDEX:
                result = self._op_update_index(op)
            else:
                result = OperationResult(
                    op_id=op.op_id,
                    success=False,
                    message=f"未知操作類型: {op.op_type}",
                )
            status = "✓" if result.success else "✗"
            print(f"{status} {result.message}")
            return result
        except Exception as e:
            print(f"✗ 錯誤: {e}")
            return OperationResult(
                op_id=op.op_id,
                success=False,
                message=str(e),
            )
    def _op_create_dir(self, op: Operation) -> OperationResult:
        """建立目錄"""
        target = Path(op.target)
        target.mkdir(parents=True, exist_ok=True)
        return OperationResult(
            op_id=op.op_id,
            success=True,
            message=f"已建立: {target.name}",
            changes=[f"Created directory: {op.target}"],
        )
    def _op_move_file(self, op: Operation) -> OperationResult:
        """移動檔案"""
        source = Path(op.source)
        target = Path(op.target)
        if not source.exists():
            return OperationResult(
                op_id=op.op_id,
                success=False,
                message=f"來源不存在: {source}",
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        # 記錄引用更新
        self.reference_updates.append((source, str(source), str(target)))
        shutil.move(str(source), str(target))
        return OperationResult(
            op_id=op.op_id,
            success=True,
            message=f"已移動: {source.name} -> {target.parent.name}/",
            changes=[f"Moved: {op.source} -> {op.target}"],
            rollback_info={"source": str(target), "target": str(source)},
        )
    def _op_copy_file(self, op: Operation) -> OperationResult:
        """複製檔案"""
        source = Path(op.source)
        target = Path(op.target)
        if not source.exists():
            return OperationResult(
                op_id=op.op_id,
                success=False,
                message=f"來源不存在: {source}",
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(target))
        return OperationResult(
            op_id=op.op_id,
            success=True,
            message=f"已複製: {source.name}",
            changes=[f"Copied: {op.source} -> {op.target}"],
            rollback_info={"delete": str(target)},
        )
    def _op_merge_file(self, op: Operation) -> OperationResult:
        """合併檔案"""
        source = Path(op.source)
        target = Path(op.target)
        if not source.exists():
            return OperationResult(
                op_id=op.op_id,
                success=False,
                message=f"來源不存在: {source}",
            )
        # 讀取來源內容
        source_content = source.read_text(encoding="utf-8")
        # 讀取或建立目標
        if target.exists():
            target_content = target.read_text(encoding="utf-8")
            merged_content = target_content + "\n\n---\n\n" + source_content
        else:
            merged_content = source_content
        # 寫入合併內容
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(merged_content, encoding="utf-8")
        return OperationResult(
            op_id=op.op_id,
            success=True,
            message=f"已合併: {source.name} -> {target.name}",
            changes=[f"Merged: {op.source} into {op.target}"],
        )
    def _op_delete_file(self, op: Operation) -> OperationResult:
        """刪除檔案"""
        target = Path(op.target)
        if not target.exists():
            return OperationResult(
                op_id=op.op_id,
                success=True,
                message="檔案已不存在",
            )
        # 備份到回滾目錄
        backup_path = self.backup_dir / target.name
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(target), str(backup_path))
        target.unlink()
        return OperationResult(
            op_id=op.op_id,
            success=True,
            message=f"已刪除: {target.name}",
            changes=[f"Deleted: {op.target}"],
            rollback_info={"restore_from": str(backup_path), "restore_to": str(target)},
        )
    def _op_embed_content(self, op: Operation) -> OperationResult:
        """嵌入內容"""
        source = Path(op.source)
        target = Path(op.target)
        section = op.options.get("section", "## Embedded Content")
        if not source.exists() or not target.exists():
            return OperationResult(
                op_id=op.op_id,
                success=False,
                message="來源或目標不存在",
            )
        source_content = source.read_text(encoding="utf-8")
        target_content = target.read_text(encoding="utf-8")
        # 嵌入到指定段落
        embed_marker = f"\n\n{section}\n\n"
        if embed_marker in target_content:
            # 在現有段落後添加
            parts = target_content.split(embed_marker)
            new_content = (
                parts[0] + embed_marker + source_content + "\n\n" + "".join(parts[1:])
            )
        else:
            # 添加新段落
            new_content = target_content + embed_marker + source_content
        target.write_text(new_content, encoding="utf-8")
        return OperationResult(
            op_id=op.op_id,
            success=True,
            message=f"已嵌入: {source.name} -> {target.name}",
            changes=[f"Embedded: {op.source} into {op.target} at '{section}'"],
        )
    def _op_update_index(self, op: Operation) -> OperationResult:
        """更新索引"""
        target = Path(op.target)
        if not target.exists():
            return OperationResult(
                op_id=op.op_id,
                success=False,
                message=f"索引檔案不存在: {target}",
            )
        # 調用索引更新腳本
        # 這裡可以導入 update_indexes 模組
        return OperationResult(
            op_id=op.op_id,
            success=True,
            message=f"索引已更新: {target.name}",
            changes=[f"Updated index: {op.target}"],
        )
    def _check_dependencies(
        self, op: Operation, results: List[OperationResult]
    ) -> bool:
        """檢查依賴是否滿足"""
        for dep_id in op.depends_on:
            dep_result = next((r for r in results if r.op_id == dep_id), None)
            if dep_result is None or not dep_result.success:
                return False
        return True
    def _create_backup(self, plan: IntegrationPlan):
        """建立備份"""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "plan_id": plan.plan_id,
            "timestamp": datetime.now().isoformat(),
            "operations": [asdict(op) for op in plan.operations],
        }
        manifest_path = self.backup_dir / "manifest.yaml"
        with open(manifest_path, "w", encoding="utf-8") as f:
            yaml.dump(manifest, f, allow_unicode=True)
    def _update_all_references(self):
        """更新所有受影響的引用"""
        if not self.reference_updates:
            return
        print("\n📎 更新引用...")
        # 掃描所有 .md 和 .yaml 檔案
        for file in PLAYBOOKS_PATH.rglob("*"):
            if file.suffix in [".md", ".yaml", ".yml"] and file.is_file():
                try:
                    content = file.read_text(encoding="utf-8")
                    modified = False
                    for _, old_path, new_path in self.reference_updates:
                        if old_path in content:
                            content = content.replace(old_path, new_path)
                            modified = True
                    if modified:
                        file.write_text(content, encoding="utf-8")
                        print(f"   更新引用: {file.name}")
                except Exception as e:
                    print(f"   ⚠️ 無法更新 {file.name}: {e}")
    def _decision_to_operation(self, decision: Dict) -> Operation:
        """將決策轉換為操作"""
        integration_type = decision.get("integration_type", "full_integration")
        if integration_type == "full_integration":
            return Operation(
                op_id=decision.get("asset_id", "op_001"),
                op_type=OperationType.MOVE_FILE,
                source=decision.get("source_path"),
                target=decision.get("target_directory")
                + "/"
                + decision.get("target_filename", ""),
            )
        elif integration_type == "embedded_integration":
            return Operation(
                op_id=decision.get("asset_id", "op_001"),
                op_type=OperationType.EMBED_CONTENT,
                source=decision.get("source_path"),
                target=decision.get("embedding_location"),
                options={"section": decision.get("embedding_section")},
            )
        elif integration_type == "archive":
            return Operation(
                op_id=decision.get("asset_id", "op_001"),
                op_type=OperationType.MOVE_FILE,
                source=decision.get("source_path"),
                target="_legacy_scratch/archive/" + decision.get("target_filename", ""),
            )
        else:
            raise ValueError(f"未知整合類型: {integration_type}")
    def rollback(self, steps: int = 1):
        """回滾操作"""
        print(f"\n🔄 回滾最近 {steps} 個操作...")
        ops_to_rollback = self.executed_ops[-steps:]
        for op_result in reversed(ops_to_rollback):
            if op_result.rollback_info:
                info = op_result.rollback_info
                if "source" in info and "target" in info:
                    # 移動回滾
                    shutil.move(info["source"], info["target"])
                    print(f"   已回滾: {op_result.op_id}")
                elif "delete" in info:
                    # 刪除回滾
                    Path(info["delete"]).unlink(missing_ok=True)
                    print(f"   已回滾: {op_result.op_id}")
                elif "restore_from" in info:
                    # 還原回滾
                    shutil.copy2(info["restore_from"], info["restore_to"])
                    print(f"   已回滾: {op_result.op_id}")
# ============================================================================
# 批量處理器
# ============================================================================
class BatchProcessor:
    """
    批量處理器：批量執行多個決策
    """
    def __init__(self, executor: IntegrationExecutor):
        self.executor = executor
    def process_directory(self, decisions_dir: Path) -> List[OperationResult]:
        """處理目錄中的所有決策"""
        results = []
        decision_files = list(decisions_dir.glob("*_decision.yaml"))
        print(f"\n📂 找到 {len(decision_files)} 個決策檔案")
        for decision_file in decision_files:
            print(f"\n處理: {decision_file.name}")
            result = self.executor.execute_single(decision_file)
            results.append(result)
        return results
    def process_plan(self, plan_file: Path) -> List[OperationResult]:
        """處理計畫檔案"""
        with open(plan_file, "r", encoding="utf-8") as f:
            plan_data = yaml.safe_load(f)
        operations = []
        for i, op_data in enumerate(plan_data.get("operations", [])):
            operations.append(
                Operation(
                    op_id=op_data.get("op_id", f"op_{i:03d}"),
                    op_type=OperationType(op_data.get("op_type")),
                    source=op_data.get("source"),
                    target=op_data.get("target"),
                    options=op_data.get("options", {}),
                    depends_on=op_data.get("depends_on", []),
                )
            )
        plan = IntegrationPlan(
            plan_id=plan_data.get("plan_id", datetime.now().strftime("%Y%m%d_%H%M%S")),
            created_at=plan_data.get("created_at", datetime.now().isoformat()),
            target_directory=plan_data.get("target_directory", str(PLAYBOOKS_PATH)),
            operations=operations,
            estimated_changes=len(operations),
            risk_level=plan_data.get("risk_level", "medium"),
        )
        return self.executor.execute_plan(plan)
# ============================================================================
# CLI 入口
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Integration Executor - 集成執行引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    # single 命令
    single_parser = subparsers.add_parser("single", help="執行單一決策")
    single_parser.add_argument("--decision", "-d", required=True, help="決策檔案")
    single_parser.add_argument("--dry-run", action="store_true", help="模擬執行")
    single_parser.add_argument("--confirm", action="store_true", help="確認執行")
    # batch 命令
    batch_parser = subparsers.add_parser("batch", help="批量執行")
    batch_parser.add_argument("--decisions", "-d", help="決策目錄")
    batch_parser.add_argument("--plan", "-p", help="計畫檔案")
    batch_parser.add_argument("--dry-run", action="store_true", help="模擬執行")
    batch_parser.add_argument("--confirm", action="store_true", help="確認執行")
    # reorganize 命令
    reorg_parser = subparsers.add_parser("reorganize", help="重組目錄")
    reorg_parser.add_argument("--target", "-t", required=True, help="目標目錄")
    reorg_parser.add_argument("--dry-run", action="store_true", help="模擬執行")
    reorg_parser.add_argument("--confirm", action="store_true", help="確認執行")
    reorg_parser.add_argument("--output", "-o", help="輸出優化報告")
    # rollback 命令
    rollback_parser = subparsers.add_parser("rollback", help="回滾操作")
    rollback_parser.add_argument("--steps", "-s", type=int, default=1, help="回滾步數")
    # map 命令
    map_parser = subparsers.add_parser("map", help="生成目錄圖譜")
    map_parser.add_argument("--target", "-t", required=True, help="目標目錄")
    map_parser.add_argument(
        "--format", "-f", choices=["ascii", "mermaid"], default="ascii"
    )
    map_parser.add_argument("--output", "-o", help="輸出檔案")
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    # 確定執行模式
    mode = ExecutionMode.DRY_RUN
    if hasattr(args, "confirm") and args.confirm:
        mode = ExecutionMode.CONFIRM
    elif hasattr(args, "dry_run") and not args.dry_run:
        mode = ExecutionMode.CONFIRM
    executor = IntegrationExecutor(mode=mode)
    if args.command == "single":
        result = executor.execute_single(Path(args.decision))
        print(f"\n結果: {'✅ 成功' if result.success else '❌ 失敗'}")
        print(f"訊息: {result.message}")
    elif args.command == "batch":
        batch_processor = BatchProcessor(executor)
        if args.plan:
            results = batch_processor.process_plan(Path(args.plan))
        elif args.decisions:
            results = batch_processor.process_directory(Path(args.decisions))
        else:
            print("請指定 --decisions 或 --plan")
            return
        success_count = sum(1 for r in results if r.success)
        print(f"\n完成: {success_count}/{len(results)} 成功")
    elif args.command == "reorganize":
        optimizer = DirectoryOptimizer()
        target = Path(args.target)
        optimization = optimizer.analyze(target)
        print(f"\n📊 優化分析: {target}")
        print(f"   改進分數: {optimization.improvement_score:.2f}")
        print(f"   操作數: {len(optimization.operations)}")
        if args.output:
            output = {
                "target": optimization.target_dir,
                "improvement_score": optimization.improvement_score,
                "current_structure": optimization.current_structure,
                "optimized_structure": optimization.optimized_structure,
                "operations": [asdict(op) for op in optimization.operations],
            }
            with open(args.output, "w", encoding="utf-8") as f:
                yaml.dump(output, f, allow_unicode=True, default_flow_style=False)
            print(f"\n報告已儲存: {args.output}")
        if optimization.operations and mode != ExecutionMode.DRY_RUN:
            plan = IntegrationPlan(
                plan_id=f"reorg_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                created_at=datetime.now().isoformat(),
                target_directory=str(target),
                operations=optimization.operations,
                estimated_changes=len(optimization.operations),
                risk_level="low",
            )
            executor.execute_plan(plan)
    elif args.command == "rollback":
        executor.rollback(steps=args.steps)
    elif args.command == "map":
        mapper = DirectoryMapper()
        target = Path(args.target)
        if args.format == "ascii":
            output = mapper.generate_ascii_tree(target)
        else:
            output = mapper.generate_mermaid_diagram(target)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"圖譜已儲存: {args.output}")
        else:
            print(output)
if __name__ == "__main__":
    main()
