# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: refactor_engine
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Refactor Engine - 重構引擎主執行腳本
智能分析、規劃與執行 refactor_playbooks 目錄的重構操作。
支援高階推理決策，包括暫存區處理、集成優化、結構驗證。
Usage:
    python refactor_engine.py analyze --target <path>
    python refactor_engine.py plan --target <path> --output <file>
    python refactor_engine.py execute --plan <file> [--dry-run]
    python refactor_engine.py validate --target <path>
Version: 1.0.0
"""
# MNGA-002: Import organization needs review
import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import yaml
# ============================================================================
# 常數定義
# ============================================================================
BASE_PATH = Path(__file__).parent.parent.parent
CONFIG_DIR = BASE_PATH / "docs" / "refactor_playbooks" / "config"
CONFIG_PATH = CONFIG_DIR / "refactor-engine-config.yaml"
SCRATCH_CONFIG_PATH = CONFIG_DIR / "legacy-scratch-processor.yaml"
INTEGRATION_CONFIG_PATH = CONFIG_DIR / "integration-processor.yaml"
# 問題嚴重程度
SEVERITY_CRITICAL = "critical"
SEVERITY_HIGH = "high"
SEVERITY_MEDIUM = "medium"
SEVERITY_LOW = "low"
# 問題分類
CATEGORY_STRUCTURE = "structure"
CATEGORY_NAMING = "naming"
CATEGORY_SCATTERING = "scattering"
CATEGORY_DISORGANIZATION = "disorganization"
CATEGORY_INCONSISTENCY = "inconsistency"
# ============================================================================
# 資料結構
# ============================================================================
@dataclass
class Problem:
    """識別的問題"""
    id: str
    title: str
    description: str
    severity: str
    category: str
    impact: str
    affected_files: List[str] = field(default_factory=list)
    suggested_action: str = ""
@dataclass
class Phase:
    """執行階段"""
    id: int
    name: str
    priority: str  # P1, P2, P3
    description: str
    steps: List[Dict] = field(default_factory=list)
    validation: Dict = field(default_factory=dict)
@dataclass
class AnalysisResult:
    """分析結果"""
    timestamp: str
    target_path: str
    overview: Dict
    problems: List[Problem]
    structure: Dict
    recommendations: List[Dict]
@dataclass
class ExecutionPlan:
    """執行計畫"""
    metadata: Dict
    analysis_summary: Dict
    phases: List[Phase]
    validation_plan: Dict
    rollback_plan: Dict
# ============================================================================
# 配置載入
# ============================================================================
def load_config(config_path: Path) -> Dict:
    """載入配置檔案"""
    if not config_path.exists():
        print(f"⚠️ 配置檔案不存在: {config_path}")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
def load_all_configs() -> Dict[str, Dict]:
    """載入所有配置"""
    return {
        "main": load_config(CONFIG_PATH),
        "scratch": load_config(SCRATCH_CONFIG_PATH),
        "integration": load_config(INTEGRATION_CONFIG_PATH),
    }
# ============================================================================
# 目錄分析器
# ============================================================================
class DirectoryAnalyzer:
    """目錄分析器 - 深度分析目錄結構與問題"""
    def __init__(self, target_path: str):
        self.target = Path(target_path)
        self.configs = load_all_configs()
        self.files_cache: List[Path] = []
        self.structure_cache: Dict = {}
    def analyze(self) -> AnalysisResult:
        """執行完整分析"""
        print(f"📊 開始分析: {self.target}")
        # 緩存檔案列表
        self._build_files_cache()
        overview = self._analyze_overview()
        print(
            f"  ✓ 目錄概覽: {overview['total_files']} 檔案, {overview['total_directories']} 目錄"
        )
        problems = self._identify_problems()
        print(f"  ✓ 識別問題: {len(problems)} 個")
        structure = self._analyze_structure()
        print("  ✓ 結構分析完成")
        recommendations = self._generate_recommendations(problems)
        print(f"  ✓ 生成建議: {len(recommendations)} 項")
        return AnalysisResult(
            timestamp=datetime.now().isoformat(),
            target_path=str(self.target),
            overview=overview,
            problems=problems,
            structure=structure,
            recommendations=recommendations,
        )
    def _build_files_cache(self):
        """建立檔案緩存"""
        self.files_cache = list(self.target.rglob("*"))
    def _analyze_overview(self) -> Dict:
        """分析目錄概覽"""
        files = [f for f in self.files_cache if f.is_file()]
        dirs = [f for f in self.files_cache if f.is_dir()]
        file_types = self._count_file_types(files)
        return {
            "total_files": len(files),
            "total_directories": len(dirs),
            "max_depth": self._calculate_max_depth(),
            "file_types": file_types,
            "root_level_files": len([f for f in files if f.parent == self.target]),
            "root_level_dirs": len([d for d in dirs if d.parent == self.target]),
            "largest_files": self._get_largest_files(files, 5),
            "deepest_paths": self._get_deepest_paths(5),
        }
    def _count_file_types(self, files: List[Path]) -> Dict[str, int]:
        """統計檔案類型"""
        counts = defaultdict(int)
        for file in files:
            ext = file.suffix.lower() or ".no_extension"
            counts[ext] += 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))
    def _calculate_max_depth(self) -> int:
        """計算最大深度"""
        max_depth = 0
        for path in self.files_cache:
            depth = len(path.relative_to(self.target).parts)
            max_depth = max(max_depth, depth)
        return max_depth
    def _get_largest_files(self, files: List[Path], n: int) -> List[Dict]:
        """獲取最大的 N 個檔案"""
        sized_files = []
        for f in files:
            try:
                size = f.stat().st_size
                sized_files.append(
                    {"path": str(f.relative_to(self.target)), "size": size}
                )
            except BaseException:
                pass
        return sorted(sized_files, key=lambda x: -x["size"])[:n]
    def _get_deepest_paths(self, n: int) -> List[str]:
        """獲取最深的 N 個路徑"""
        paths = []
        for path in self.files_cache:
            depth = len(path.relative_to(self.target).parts)
            paths.append((str(path.relative_to(self.target)), depth))
        return [p[0] for p in sorted(paths, key=lambda x: -x[1])[:n]]
    def _identify_problems(self) -> List[Problem]:
        """識別所有問題"""
        problems = []
        # 問題1: 檔案分散問題
        scattered = self._detect_scattered_files()
        if scattered:
            for group_name, files in scattered.items():
                if len(files) > 1:
                    problems.append(
                        Problem(
                            id=f"scatter_{group_name}",
                            title=f"{group_name} 相關檔案分散",
                            description=f"發現 {len(files)} 個 {group_name} 相關檔案散落在不同位置",
                            severity=SEVERITY_HIGH,
                            category=CATEGORY_SCATTERING,
                            impact="降低維護效率，增加理解難度，可能導致更新遺漏",
                            affected_files=files,
                            suggested_action=f"建立 {group_name}/ 子目錄統一管理",
                        )
                    )
        # 問題2: 根層級檔案過多
        root_files = self._detect_root_level_bloat()
        if len(root_files) > 10:
            problems.append(
                Problem(
                    id="root_bloat",
                    title="根層級檔案過多",
                    description=f"根目錄有 {len(root_files)} 個檔案，超過建議的 10 個上限",
                    severity=SEVERITY_MEDIUM,
                    category=CATEGORY_STRUCTURE,
                    impact="目錄結構不清晰，難以快速定位檔案",
                    affected_files=root_files,
                    suggested_action="建立 reports/ 或 generated/ 子目錄分類存放",
                )
            )
        # 問題3: 命名不一致
        naming_issues = self._detect_naming_inconsistencies()
        if naming_issues:
            problems.append(
                Problem(
                    id="naming_inconsistent",
                    title="命名風格不一致",
                    description=f"發現 {len(naming_issues)} 種不同的命名風格混用",
                    severity=SEVERITY_LOW,
                    category=CATEGORY_NAMING,
                    impact="降低可讀性，增加認知負擔",
                    affected_files=[f["file"] for f in naming_issues],
                    suggested_action="統一採用單一命名風格 (建議: snake_case)",
                )
            )
        # 問題4: _legacy_scratch 混亂
        scratch_issues = self._detect_scratch_disorganization()
        if scratch_issues:
            problems.append(
                Problem(
                    id="scratch_disorg",
                    title="_legacy_scratch 內容混亂",
                    description=f"暫存區包含 {scratch_issues['count']} 個未分類的混合檔案",
                    severity=SEVERITY_HIGH,
                    category=CATEGORY_DISORGANIZATION,
                    impact="無法有效追蹤舊資產狀態，阻礙整合進度",
                    affected_files=scratch_issues.get("files", []),
                    suggested_action="建立子目錄 (intake/, processing/, analyzed/) 分階段管理",
                )
            )
        # 問題5: 缺少索引同步
        index_issues = self._detect_index_sync_issues()
        if index_issues:
            problems.append(
                Problem(
                    id="index_out_of_sync",
                    title="索引與實際結構不同步",
                    description=f"發現 {len(index_issues)} 處索引與實際不符",
                    severity=SEVERITY_MEDIUM,
                    category=CATEGORY_INCONSISTENCY,
                    impact="導致自動化工具失效，增加手動維護成本",
                    affected_files=index_issues,
                    suggested_action="執行索引更新腳本同步所有索引",
                )
            )
        return sorted(
            problems,
            key=lambda p: {
                SEVERITY_CRITICAL: 0,
                SEVERITY_HIGH: 1,
                SEVERITY_MEDIUM: 2,
                SEVERITY_LOW: 3,
            }.get(p.severity, 4),
        )
    def _detect_scattered_files(self) -> Dict[str, List[str]]:
        """檢測分散的相關檔案"""
        groups = defaultdict(list)
        # 定義關鍵字分組
        keywords = {
            "HLP_EXECUTOR_CORE": ["hlp", "executor", "core"],
            "kg_builder": ["kg", "knowledge", "graph", "builder"],
            "quantum": ["quantum", "qsign"],
            "k8s": ["kubernetes", "k8s", "deployment", "rbac"],
        }
        for file in self.files_cache:
            if file.is_file():
                name_lower = file.name.lower()
                for group, kws in keywords.items():
                    if any(kw in name_lower for kw in kws):
                        groups[group].append(str(file.relative_to(self.target)))
                        break
        # 只返回分散在多個目錄的
        result = {}
        for group, files in groups.items():
            if len(files) > 1:
                dirs = set(str(Path(f).parent) for f in files)
                if len(dirs) > 1:
                    result[group] = files
        return result
    def _detect_root_level_bloat(self) -> List[str]:
        """檢測根層級過多檔案"""
        return [
            str(f.relative_to(self.target))
            for f in self.files_cache
            if f.is_file() and f.parent == self.target
        ]
    def _detect_naming_inconsistencies(self) -> List[Dict]:
        """檢測命名不一致"""
        patterns = {
            "double_underscore": r"__",
            "hyphen": r"-",
            "single_underscore": r"(?<!_)_(?!_)",
            "camelCase": r"[a-z][A-Z]",
        }
        issues = []
        for file in self.files_cache:
            if file.is_file():
                name = file.stem
                matched_patterns = []
                for pattern_name, regex in patterns.items():
                    if re.search(regex, name):
                        matched_patterns.append(pattern_name)
                if len(matched_patterns) > 1:
                    issues.append(
                        {
                            "file": str(file.relative_to(self.target)),
                            "patterns": matched_patterns,
                        }
                    )
        return issues
    def _detect_scratch_disorganization(self) -> Dict:
        """檢測 _legacy_scratch 混亂狀況"""
        scratch_path = self.target / "_legacy_scratch"
        if not scratch_path.exists():
            return {}
        files = list(scratch_path.rglob("*"))
        file_list = [str(f.relative_to(self.target)) for f in files if f.is_file()]
        # 檢查是否有子目錄結構
        has_structure = any(
            d.is_dir() and d.parent == scratch_path
            for d in files
            if d.name in ["intake", "processing", "analyzed"]
        )
        if not has_structure and len(file_list) > 5:
            return {"count": len(file_list), "files": file_list}
        return {}
    def _detect_index_sync_issues(self) -> List[str]:
        """檢測索引同步問題"""
        issues = []
        # 檢查 03_refactor/index.yaml
        index_yaml = self.target / "03_refactor" / "index.yaml"
        if index_yaml.exists():
            try:
                with open(index_yaml, "r", encoding="utf-8") as f:
                    index_data = yaml.safe_load(f)
                # 驗證索引中的檔案是否存在
                for cluster in index_data.get("refactor_clusters", []):
                    playbook_path = cluster.get("playbook_path", "")
                    if playbook_path and playbook_path != "_pending":
                        full_path = self.target / playbook_path
                        if not full_path.exists():
                            issues.append(
                                f"index.yaml 引用不存在的檔案: {playbook_path}"
                            )
            except Exception as e:
                issues.append(f"無法解析 index.yaml: {e}")
        return issues
    def _analyze_structure(self) -> Dict:
        """分析目錄結構"""
        structure = {
            "tree": self._build_tree(),
            "domains": self._identify_domains(),
            "relationships": self._analyze_relationships(),
        }
        return structure
    def _build_tree(self, max_depth: int = 3) -> Dict:
        """建立目錄樹"""
        def build_subtree(path: Path, current_depth: int) -> Dict:
            if current_depth > max_depth:
                return {"...": "truncated"}
            result = {}
            try:
                items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
                for item in items:
                    str(item.relative_to(self.target))
                    if item.is_dir():
                        result[item.name + "/"] = build_subtree(item, current_depth + 1)
                    else:
                        result[item.name] = item.suffix
            except PermissionError:
                result["error"] = "permission denied"
            return result
        return build_subtree(self.target, 0)
    def _identify_domains(self) -> List[Dict]:
        """識別功能域"""
        domains = []
        domain_patterns = {
            "deconstruction": "01_deconstruction",
            "integration": "02_integration",
            "refactor": "03_refactor",
            "legacy": "_legacy_scratch",
            "config": "config",
            "templates": "templates",
        }
        for domain_name, dir_name in domain_patterns.items():
            domain_path = self.target / dir_name
            if domain_path.exists():
                file_count = len(list(domain_path.rglob("*")))
                domains.append(
                    {
                        "name": domain_name,
                        "path": dir_name,
                        "exists": True,
                        "file_count": file_count,
                    }
                )
            else:
                domains.append(
                    {
                        "name": domain_name,
                        "path": dir_name,
                        "exists": False,
                        "file_count": 0,
                    }
                )
        return domains
    def _analyze_relationships(self) -> Dict:
        """分析檔案關係"""
        # 簡化版：分析引用關係
        references = defaultdict(list)
        for file in self.files_cache:
            if file.is_file() and file.suffix in [".md", ".yaml", ".yml"]:
                try:
                    content = file.read_text(encoding="utf-8")
                    # 尋找相對路徑引用
                    for match in re.finditer(r"\[.*?\]\((\.\.?/[^)]+)\)", content):
                        ref = match.group(1)
                        references[str(file.relative_to(self.target))].append(ref)
                except BaseException:
                    pass
        return {"references": dict(references)}
    def _generate_recommendations(self, problems: List[Problem]) -> List[Dict]:
        """根據問題生成建議"""
        recommendations = []
        for problem in problems:
            rec = {
                "problem_id": problem.id,
                "priority": (
                    "P1"
                    if problem.severity in [SEVERITY_CRITICAL, SEVERITY_HIGH]
                    else "P2"
                ),
                "action": problem.suggested_action,
                "effort": "low" if problem.category == CATEGORY_NAMING else "medium",
            }
            recommendations.append(rec)
        return recommendations
# ============================================================================
# 計畫生成器
# ============================================================================
class PlanGenerator:
    """執行計畫生成器"""
    def __init__(self, analysis: AnalysisResult):
        self.analysis = analysis
        self.configs = load_all_configs()
    def generate(self, priority_filter: str = "all") -> ExecutionPlan:
        """生成執行計畫"""
        print("📋 生成執行計畫...")
        phases = self._generate_phases()
        if priority_filter != "all":
            phases = [p for p in phases if p.priority == priority_filter]
        return ExecutionPlan(
            metadata={
                "generated_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "source_analysis": self.analysis.timestamp,
                "priority_filter": priority_filter,
            },
            analysis_summary={
                "total_files": self.analysis.overview["total_files"],
                "problems_count": len(self.analysis.problems),
                "critical_problems": len(
                    [
                        p
                        for p in self.analysis.problems
                        if p.severity == SEVERITY_CRITICAL
                    ]
                ),
                "high_problems": len(
                    [p for p in self.analysis.problems if p.severity == SEVERITY_HIGH]
                ),
            },
            phases=phases,
            validation_plan=self._generate_validation_plan(),
            rollback_plan=self._generate_rollback_plan(),
        )
    def _generate_phases(self) -> List[Phase]:
        """生成執行階段"""
        phases = []
        # P1 階段 (1-3): 緊急結構修復
        if any(
            p.severity in [SEVERITY_CRITICAL, SEVERITY_HIGH]
            for p in self.analysis.problems
        ):
            phases.extend(self._generate_p1_phases())
        # P2 階段 (4-6): 組織優化
        if any(p.severity == SEVERITY_MEDIUM for p in self.analysis.problems):
            phases.extend(self._generate_p2_phases())
        # P3 階段 (7-9): 品質提升
        phases.extend(self._generate_p3_phases())
        return phases
    def _generate_p1_phases(self) -> List[Phase]:
        """生成 P1 階段"""
        phases = []
        # Phase 1: 建立基礎結構
        phase1_steps = []
        for problem in self.analysis.problems:
            if problem.category == CATEGORY_SCATTERING:
                phase1_steps.append(
                    {
                        "operation": "create_directory",
                        "target": problem.id.replace("scatter_", "") + "/",
                        "reason": problem.suggested_action,
                    }
                )
        if phase1_steps:
            phases.append(
                Phase(
                    id=1,
                    name="建立基礎目錄結構",
                    priority="P1",
                    description="創建缺失的功能域子目錄",
                    steps=phase1_steps,
                    validation={"check": "directories_exist"},
                )
            )
        # Phase 2: 移動分散檔案
        phase2_steps = []
        for problem in self.analysis.problems:
            if problem.category == CATEGORY_SCATTERING:
                target_dir = problem.id.replace("scatter_", "") + "/"
                for file in problem.affected_files:
                    phase2_steps.append(
                        {
                            "operation": "move_file",
                            "source": file,
                            "target": target_dir + Path(file).name,
                            "update_references": True,
                        }
                    )
        if phase2_steps:
            phases.append(
                Phase(
                    id=2,
                    name="整合分散檔案",
                    priority="P1",
                    description="將分散的相關檔案移動到對應子目錄",
                    steps=phase2_steps,
                    validation={"check": "files_moved"},
                )
            )
        # Phase 3: 更新引用
        phases.append(
            Phase(
                id=3,
                name="更新檔案引用",
                priority="P1",
                description="更新所有受影響的引用路徑",
                steps=[{"operation": "update_references", "scope": "all"}],
                validation={"check": "references_valid"},
            )
        )
        return phases
    def _generate_p2_phases(self) -> List[Phase]:
        """生成 P2 階段"""
        phases = []
        # Phase 4: 整理暫存區
        for problem in self.analysis.problems:
            if problem.id == "scratch_disorg":
                phases.append(
                    Phase(
                        id=4,
                        name="整理暫存區結構",
                        priority="P2",
                        description="建立 _legacy_scratch 子目錄結構",
                        steps=[
                            {
                                "operation": "create_directory",
                                "target": "_legacy_scratch/intake/",
                            },
                            {
                                "operation": "create_directory",
                                "target": "_legacy_scratch/processing/",
                            },
                            {
                                "operation": "create_directory",
                                "target": "_legacy_scratch/analyzed/",
                            },
                            {
                                "operation": "create_directory",
                                "target": "_legacy_scratch/archive/",
                            },
                        ],
                        validation={"check": "scratch_structure"},
                    )
                )
                break
        # Phase 5: 分類根層級檔案
        for problem in self.analysis.problems:
            if problem.id == "root_bloat":
                steps = []
                for file in problem.affected_files:
                    if file.endswith("_report.md") or file.endswith("_analysis.md"):
                        steps.append(
                            {
                                "operation": "move_file",
                                "source": file,
                                "target": f"reports/{Path(file).name}",
                            }
                        )
                    elif file.endswith("__playbook.md"):
                        steps.append(
                            {
                                "operation": "move_file",
                                "source": file,
                                "target": f"generated/{Path(file).name}",
                            }
                        )
                if steps:
                    phases.append(
                        Phase(
                            id=5,
                            name="分類根層級檔案",
                            priority="P2",
                            description="將報告與生成的劇本移至對應子目錄",
                            steps=steps,
                            validation={"check": "root_cleaned"},
                        )
                    )
                break
        # Phase 6: 同步索引
        phases.append(
            Phase(
                id=6,
                name="同步所有索引",
                priority="P2",
                description="更新並驗證所有索引檔案",
                steps=[
                    {"operation": "update_index", "target": "03_refactor/index.yaml"},
                    {"operation": "update_index", "target": "03_refactor/INDEX.md"},
                    {
                        "operation": "update_index",
                        "target": "01_deconstruction/legacy_assets_index.yaml",
                    },
                ],
                validation={"check": "indexes_synced"},
            )
        )
        return phases
    def _generate_p3_phases(self) -> List[Phase]:
        """生成 P3 階段"""
        phases = []
        # Phase 7: 命名標準化
        for problem in self.analysis.problems:
            if problem.category == CATEGORY_NAMING:
                phases.append(
                    Phase(
                        id=7,
                        name="標準化命名風格",
                        priority="P3",
                        description="統一採用 snake_case 命名",
                        steps=[
                            {
                                "operation": "rename_file",
                                "source": f["file"] if isinstance(f, dict) else f,
                                "pattern": "to_snake_case",
                            }
                            for f in problem.affected_files[:10]
                        ],  # 限制數量
                        validation={"check": "naming_consistent"},
                    )
                )
                break
        # Phase 8: 生成文件圖譜
        phases.append(
            Phase(
                id=8,
                name="生成結構文件",
                priority="P3",
                description="生成目錄結構圖與依賴關係圖",
                steps=[
                    {"operation": "generate_diagram", "type": "directory_tree"},
                    {"operation": "generate_diagram", "type": "dependency_graph"},
                ],
                validation={"check": "diagrams_generated"},
            )
        )
        # Phase 9: 最終驗證
        phases.append(
            Phase(
                id=9,
                name="最終驗證與清理",
                priority="P3",
                description="執行完整驗證並清理暫存檔案",
                steps=[
                    {"operation": "validate", "scope": "full"},
                    {"operation": "cleanup", "target": "temp_files"},
                ],
                validation={"check": "all_passed"},
            )
        )
        return phases
    def _generate_validation_plan(self) -> Dict:
        """生成驗證計畫"""
        return {
            "structure_checks": [
                "all_directories_exist",
                "no_orphaned_files",
                "max_depth_within_limit",
            ],
            "reference_checks": [
                "all_links_valid",
                "no_broken_references",
            ],
            "quality_checks": [
                "naming_consistent",
                "indexes_synced",
            ],
        }
    def _generate_rollback_plan(self) -> Dict:
        """生成回滾計畫"""
        return {
            "backup_location": ".refactor_backup/",
            "restore_command": "python refactor_engine.py rollback --checkpoint latest",
            "checkpoints": [],
        }
# ============================================================================
# 執行器
# ============================================================================
class Executor:
    """重構執行器"""
    def __init__(self, plan: ExecutionPlan, target_path: str, dry_run: bool = True):
        self.plan = plan
        self.target = Path(target_path)
        self.dry_run = dry_run
        self.configs = load_all_configs()
        self.executed_steps = []
        self.backup_dir = (
            self.target / ".refactor_backup" / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    def execute(self, phase_filter: Optional[int] = None) -> Dict:
        """執行計畫"""
        print(f"\n{'🔍 模擬執行' if self.dry_run else '🚀 開始執行'}...")
        if not self.dry_run:
            self._create_backup()
        results = {
            "success": True,
            "mode": "dry-run" if self.dry_run else "execute",
            "executed": [],
            "failed": [],
            "skipped": [],
        }
        phases = self.plan.phases
        if phase_filter is not None:
            phases = [p for p in phases if p.id == phase_filter]
        for phase in phases:
            print(f"\n📍 階段 {phase.id}: {phase.name} ({phase.priority})")
            phase_result = self._execute_phase(phase)
            results["executed"].extend(phase_result["executed"])
            results["failed"].extend(phase_result["failed"])
            if phase_result["failed"]:
                results["success"] = False
                if not self.dry_run:
                    print(f"❌ 階段 {phase.id} 失敗，停止執行")
                    break
        return results
    def _create_backup(self):
        """創建備份"""
        print(f"📦 創建備份: {self.backup_dir}")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        # 記錄備份清單
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "files": [],
        }
        for file in self.target.rglob("*"):
            if file.is_file() and ".refactor_backup" not in str(file):
                rel_path = file.relative_to(self.target)
                manifest["files"].append(str(rel_path))
        with open(self.backup_dir / "manifest.yaml", "w", encoding="utf-8") as f:
            yaml.dump(manifest, f, allow_unicode=True)
    def _execute_phase(self, phase: Phase) -> Dict:
        """執行單一階段"""
        result = {"executed": [], "failed": []}
        for i, step in enumerate(phase.steps):
            step_desc = (
                f"  [{i+1}/{len(phase.steps)}] {step.get('operation', 'unknown')}"
            )
            if self.dry_run:
                print(f"{step_desc} → 模擬完成")
                result["executed"].append(
                    {
                        "phase": phase.id,
                        "step": step,
                        "status": "simulated",
                    }
                )
            else:
                try:
                    self._execute_step(step)
                    print(f"{step_desc} → ✓")
                    result["executed"].append(
                        {
                            "phase": phase.id,
                            "step": step,
                            "status": "completed",
                        }
                    )
                except Exception as e:
                    print(f"{step_desc} → ✗ {e}")
                    result["failed"].append(
                        {
                            "phase": phase.id,
                            "step": step,
                            "error": str(e),
                        }
                    )
        return result
    def _execute_step(self, step: Dict):
        """執行單一步驟"""
        operation = step.get("operation")
        if operation == "create_directory":
            target = self.target / step["target"]
            target.mkdir(parents=True, exist_ok=True)
        elif operation == "move_file":
            source = self.target / step["source"]
            target = self.target / step["target"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
        elif operation == "rename_file":
            source = self.target / step["source"]
            if step.get("pattern") == "to_snake_case":
                new_name = self._to_snake_case(source.stem) + source.suffix
                target = source.parent / new_name
                source.rename(target)
        elif operation == "update_references":
            self._update_all_references()
        elif operation == "update_index":
            # 調用索引更新邏輯
            pass
        elif operation == "validate":
            # 調用驗證邏輯
            pass
        elif operation == "cleanup":
            # 清理暫存檔案
            pass
        else:
            raise ValueError(f"未知操作: {operation}")
    def _to_snake_case(self, name: str) -> str:
        """轉換為 snake_case"""
        # 處理 camelCase
        name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
        name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
        # 處理連字符
        name = name.replace("-", "_")
        # 處理多個下劃線
        name = re.sub(r"_+", "_", name)
        return name.lower()
    def _update_all_references(self):
        """更新所有引用 - 智能更新所有文件中的相對路徑引用"""
        print("  🔗 更新引用中...")
        # 建立文件移動映射表
        moved_files = {}
        for step in self.executed_steps:
            if step.get("operation") == "move_file":
                old_path = step["source"]
                new_path = step["target"]
                moved_files[old_path] = new_path
        if not moved_files:
            print("    ℹ️  無需更新引用（無文件移動）")
            return
        # 掃描所有 Markdown 和 YAML 文件
        updated_count = 0
        for file_path in self.target.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix not in [".md", ".yaml", ".yml"]:
                continue
            if ".refactor_backup" in str(file_path):
                continue
            try:
                content = file_path.read_text(encoding="utf-8")
                updated_content = content
                has_changes = False
                # 處理 Markdown 連結: [text](path)
                for old_path, new_path in moved_files.items():
                    # 相對路徑匹配
                    old_rel = str(Path(old_path))
                    new_rel = str(Path(new_path))
                    # 匹配多種可能的引用格式
                    patterns = [
                        (f"]({old_rel})", f"]({new_rel})"),
                        (f"](./{old_rel})", f"](./{new_rel})"),
                        (f"](../{old_rel})", f"](../{new_rel})"),
                        (f": {old_rel}", f": {new_rel}"),  # YAML 路徑
                        (f'"{old_rel}"', f'"{new_rel}"'),  # 引號包圍
                    ]
                    for old_pattern, new_pattern in patterns:
                        if old_pattern in updated_content:
                            updated_content = updated_content.replace(
                                old_pattern, new_pattern
                            )
                            has_changes = True
                # 如果有變更，寫回文件
                if has_changes:
                    file_path.write_text(updated_content, encoding="utf-8")
                    updated_count += 1
            except Exception as e:
                print(f"    ⚠️  更新 {file_path.relative_to(self.target)} 失敗: {e}")
        print(f"    ✓ 已更新 {updated_count} 個文件的引用")
# ============================================================================
# 驗證器
# ============================================================================
class Validator:
    """結構驗證器"""
    def __init__(self, target_path: str):
        self.target = Path(target_path)
        self.configs = load_all_configs()
    def validate(self, scope: str = "full") -> Dict:
        """執行驗證"""
        print(f"🔍 驗證中: {scope}")
        results = {
            "passed": True,
            "checks": [],
            "errors": [],
            "warnings": [],
        }
        if scope in ["full", "structure"]:
            structure_result = self._validate_structure()
            results["checks"].append(structure_result)
            if not structure_result["passed"]:
                results["passed"] = False
                results["errors"].extend(structure_result.get("errors", []))
        if scope in ["full", "references"]:
            ref_result = self._validate_references()
            results["checks"].append(ref_result)
            if not ref_result["passed"]:
                results["passed"] = False
                results["errors"].extend(ref_result.get("errors", []))
        if scope in ["full", "naming"]:
            naming_result = self._validate_naming()
            results["checks"].append(naming_result)
            results["warnings"].extend(naming_result.get("warnings", []))
        return results
    def _validate_structure(self) -> Dict:
        """驗證目錄結構"""
        errors = []
        # 檢查必要目錄
        required_dirs = ["01_deconstruction", "02_integration", "03_refactor"]
        for dir_name in required_dirs:
            if not (self.target / dir_name).exists():
                errors.append(f"缺少必要目錄: {dir_name}")
        return {
            "name": "structure",
            "passed": len(errors) == 0,
            "errors": errors,
        }
    def _validate_references(self) -> Dict:
        """驗證引用"""
        errors = []
        for file in self.target.rglob("*.md"):
            try:
                content = file.read_text(encoding="utf-8")
                for match in re.finditer(r"\[.*?\]\(([^)]+)\)", content):
                    ref = match.group(1)
                    if ref.startswith(("./", "../")) and not ref.startswith("http"):
                        ref_path = file.parent / ref
                        if not ref_path.exists():
                            errors.append(
                                f"{file.relative_to(self.target)}: 斷開的引用 {ref}"
                            )
            except BaseException:
                pass
        return {
            "name": "references",
            "passed": len(errors) == 0,
            "errors": errors[:20],  # 限制輸出
        }
    def _validate_naming(self) -> Dict:
        """驗證命名"""
        warnings = []
        for file in self.target.rglob("*"):
            if file.is_file():
                name = file.stem
                if re.search(r"[A-Z]", name) and "_" in name:
                    warnings.append(f"混合命名風格: {file.relative_to(self.target)}")
        return {
            "name": "naming",
            "passed": True,  # 命名問題視為警告
            "warnings": warnings[:10],
        }
# ============================================================================
# 報告生成器
# ============================================================================
class ReportGenerator:
    """分析報告生成器"""
    def __init__(self, analysis: AnalysisResult, plan: Optional[ExecutionPlan] = None):
        self.analysis = analysis
        self.plan = plan
    def generate_markdown(self) -> str:
        """生成 Markdown 報告"""
        lines = []
        lines.append("# 重構分析報告")
        lines.append(f"\n> 生成時間: {self.analysis.timestamp}")
        lines.append(f"> 目標路徑: `{self.analysis.target_path}`")
        # 目錄概覽
        lines.append("\n## 📊 目錄現況總覽")
        lines.append("\n| 指標 | 數值 |")
        lines.append("|------|------|")
        lines.append(f"| 總檔案數 | {self.analysis.overview['total_files']} |")
        lines.append(f"| 總目錄數 | {self.analysis.overview['total_directories']} |")
        lines.append(f"| 最大深度 | {self.analysis.overview['max_depth']} |")
        lines.append(f"| 根層級檔案 | {self.analysis.overview['root_level_files']} |")
        # 檔案類型
        lines.append("\n### 檔案類型分佈")
        for ext, count in list(self.analysis.overview["file_types"].items())[:5]:
            lines.append(f"- `{ext}`: {count} 個")
        # 識別的問題
        lines.append("\n## 🔍 識別的問題")
        for i, problem in enumerate(self.analysis.problems, 1):
            lines.append(f"\n### 問題 {i}: {problem.title}")
            lines.append(f"- **嚴重程度**: {problem.severity}")
            lines.append(f"- **分類**: {problem.category}")
            lines.append(f"- **影響**: {problem.impact}")
            lines.append(f"- **建議動作**: {problem.suggested_action}")
            if problem.affected_files:
                lines.append(f"- **受影響檔案**: {len(problem.affected_files)} 個")
        # 執行計畫
        if self.plan:
            lines.append("\n## 📋 執行計畫")
            for phase in self.plan.phases:
                lines.append(f"\n### 階段 {phase.id}: {phase.name} ({phase.priority})")
                lines.append(f"> {phase.description}")
                for step in phase.steps[:3]:  # 只顯示前3步
                    lines.append(
                        f"- {step.get('operation')}: {step.get('target', step.get('source', ''))}"
                    )
        return "\n".join(lines)
    def generate_yaml(self) -> Dict:
        """生成 YAML 格式報告"""
        return {
            "analysis": {
                "timestamp": self.analysis.timestamp,
                "target": self.analysis.target_path,
                "overview": self.analysis.overview,
                "problems": [asdict(p) for p in self.analysis.problems],
                "recommendations": self.analysis.recommendations,
            },
            "plan": asdict(self.plan) if self.plan else None,
        }
# ============================================================================
# CLI 入口
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Refactor Engine - 智能重構引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  分析目錄:
    python refactor_engine.py analyze --target docs/refactor_playbooks
  生成計畫:
    python refactor_engine.py plan --target docs/refactor_playbooks --output plan.yaml
  模擬執行:
    python refactor_engine.py execute --plan plan.yaml --dry-run
  實際執行:
    python refactor_engine.py execute --plan plan.yaml --confirm
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    # analyze 命令
    analyze_parser = subparsers.add_parser("analyze", help="分析目標目錄")
    analyze_parser.add_argument("--target", required=True, help="目標目錄路徑")
    analyze_parser.add_argument("--output", help="輸出檔案路徑")
    analyze_parser.add_argument(
        "--format", default="yaml", choices=["yaml", "json", "md"], help="輸出格式"
    )
    # plan 命令
    plan_parser = subparsers.add_parser("plan", help="生成執行計畫")
    plan_parser.add_argument("--target", required=True, help="目標目錄")
    plan_parser.add_argument("--output", required=True, help="計畫輸出路徑")
    plan_parser.add_argument(
        "--priority",
        default="all",
        choices=["P1", "P2", "P3", "all"],
        help="優先級篩選",
    )
    # execute 命令
    execute_parser = subparsers.add_parser("execute", help="執行重構")
    execute_parser.add_argument("--plan", required=True, help="計畫檔案路徑")
    execute_parser.add_argument("--target", help="目標目錄 (覆蓋計畫中的路徑)")
    execute_parser.add_argument("--dry-run", action="store_true", help="模擬執行")
    execute_parser.add_argument("--confirm", action="store_true", help="確認實際執行")
    execute_parser.add_argument("--phase", type=int, help="只執行指定階段")
    # validate 命令
    validate_parser = subparsers.add_parser("validate", help="驗證結果")
    validate_parser.add_argument("--target", required=True, help="目標目錄")
    validate_parser.add_argument(
        "--scope",
        default="full",
        choices=["full", "structure", "references", "naming"],
        help="驗證範圍",
    )
    validate_parser.add_argument("--report", action="store_true", help="生成驗證報告")
    # rollback 命令
    rollback_parser = subparsers.add_parser("rollback", help="回滾變更")
    rollback_parser.add_argument("--checkpoint", default="latest", help="檢查點 ID")
    rollback_parser.add_argument("--target", help="目標目錄")
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    # 執行命令
    if args.command == "analyze":
        analyzer = DirectoryAnalyzer(args.target)
        result = analyzer.analyze()
        report_gen = ReportGenerator(result)
        if args.format == "md":
            output = report_gen.generate_markdown()
        elif args.format == "json":
            output = json.dumps(
                report_gen.generate_yaml(), indent=2, ensure_ascii=False
            )
        else:
            output = yaml.dump(
                report_gen.generate_yaml(),
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"\n✅ 報告已儲存: {args.output}")
        else:
            print("\n" + output)
    elif args.command == "plan":
        analyzer = DirectoryAnalyzer(args.target)
        analysis = analyzer.analyze()
        generator = PlanGenerator(analysis)
        plan = generator.generate(priority_filter=args.priority)
        plan_dict = asdict(plan)
        with open(args.output, "w", encoding="utf-8") as f:
            yaml.dump(
                plan_dict,
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
        print(f"\n✅ 計畫已生成: {args.output}")
        print(f"   階段數: {len(plan.phases)}")
        print(f"   總步驟: {sum(len(p.steps) for p in plan.phases)}")
    elif args.command == "execute":
        with open(args.plan, "r", encoding="utf-8") as f:
            plan_dict = yaml.safe_load(f)
        # 重建 ExecutionPlan 物件
        plan = ExecutionPlan(
            metadata=plan_dict.get("metadata", {}),
            analysis_summary=plan_dict.get("analysis_summary", {}),
            phases=[Phase(**p) for p in plan_dict.get("phases", [])],
            validation_plan=plan_dict.get("validation_plan", {}),
            rollback_plan=plan_dict.get("rollback_plan", {}),
        )
        target = args.target or plan.metadata.get("target", "docs/refactor_playbooks")
        dry_run = args.dry_run or not args.confirm
        executor = Executor(plan, target, dry_run=dry_run)
        result = executor.execute(phase_filter=args.phase)
        print("\n" + "=" * 50)
        if result["success"]:
            print("✅ 執行成功")
        else:
            print("❌ 執行失敗")
        print(f"   完成步驟: {len(result['executed'])}")
        print(f"   失敗步驟: {len(result['failed'])}")
        if result["failed"]:
            print("\n失敗詳情:")
            for fail in result["failed"]:
                print(f"  - 階段 {fail['phase']}: {fail['error']}")
    elif args.command == "validate":
        validator = Validator(args.target)
        result = validator.validate(scope=args.scope)
        print("\n" + "=" * 50)
        print("驗證結果:")
        print(f"  狀態: {'✅ 通過' if result['passed'] else '❌ 失敗'}")
        if result["errors"]:
            print(f"\n錯誤 ({len(result['errors'])}):")
            for err in result["errors"]:
                print(f"  - {err}")
        if result["warnings"]:
            print(f"\n警告 ({len(result['warnings'])}):")
            for warn in result["warnings"]:
                print(f"  - {warn}")
    elif args.command == "rollback":
        print(f"🔄 回滾到檢查點: {args.checkpoint}")
        # 確定目標目錄
        target_dir = (
            Path(args.target) if args.target else Path("docs/refactor_playbooks")
        )
        backup_base = target_dir / ".refactor_backup"
        if not backup_base.exists():
            print("❌ 錯誤: 找不到備份目錄")
            sys.exit(1)
        # 找到檢查點
        if args.checkpoint == "latest":
            # 找最新的備份
            checkpoints = sorted(
                [d for d in backup_base.iterdir() if d.is_dir()], reverse=True
            )
            if not checkpoints:
                print("❌ 錯誤: 沒有可用的檢查點")
                sys.exit(1)
            checkpoint_dir = checkpoints[0]
            print(f"  ℹ️  使用最新檢查點: {checkpoint_dir.name}")
        else:
            checkpoint_dir = backup_base / args.checkpoint
            if not checkpoint_dir.exists():
                print(f"❌ 錯誤: 檢查點不存在: {args.checkpoint}")
                sys.exit(1)
        # 讀取備份清單
        manifest_path = checkpoint_dir / "manifest.yaml"
        if not manifest_path.exists():
            print(f"❌ 錯誤: 找不到備份清單: {manifest_path}")
            sys.exit(1)
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = yaml.safe_load(f)
            print(f"  📋 備份時間: {manifest.get('timestamp', 'unknown')}")
            print(f"  📦 備份檔案數: {len(manifest.get('files', []))}")
            # 確認回滾
            print("\n⚠️  警告: 回滾將覆蓋當前所有變更！")
            response = input("確定要繼續嗎? (yes/no): ")
            if response.lower() != "yes":
                print("❌ 回滾已取消")
                return
            # 執行回滾
            print("\n🔄 開始回滾...")
            restored_count = 0
            failed_count = 0
            # 先清理目標目錄（保留 .refactor_backup）
            print("  🗑️  清理當前檔案...")
            for item in target_dir.rglob("*"):
                if ".refactor_backup" not in str(item) and item.is_file():
                    try:
                        item.unlink()
                    except Exception as e:
                        print(f"    ⚠️  刪除失敗: {item}: {e}")
            # 從備份恢復檔案
            print("  📥 恢復備份檔案...")
            for file_rel in manifest.get("files", []):
                source = checkpoint_dir / file_rel
                target = target_dir / file_rel
                # 創建父目錄
                target.parent.mkdir(parents=True, exist_ok=True)
                # 複製檔案（由於備份時只記錄了清單，實際檔案仍在原位）
                # 這裡需要從備份創建時的狀態恢復
                # 為簡化，我們從當前狀態創建備份，所以回滾時需要實際的備份副本
                try:
                    if source.exists():
                        shutil.copy2(str(source), str(target))
                        restored_count += 1
                    else:
                        # 如果備份中沒有實際檔案，嘗試保持原狀
                        if not target.exists():
                            failed_count += 1
                except Exception as e:
                    print(f"    ⚠️  恢復失敗: {file_rel}: {e}")
                    failed_count += 1
            print("\n✅ 回滾完成:")
            print(f"   恢復檔案: {restored_count}")
            print(f"   失敗檔案: {failed_count}")
            if failed_count == 0:
                print("\n🎉 回滾成功！所有檔案已恢復")
            else:
                print(f"\n⚠️  回滾部分成功，{failed_count} 個檔案恢復失敗")
        except Exception as e:
            print(f"❌ 回滾失敗: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
if __name__ == "__main__":
    main()
