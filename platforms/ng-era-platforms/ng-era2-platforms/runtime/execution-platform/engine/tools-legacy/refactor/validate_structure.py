# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: validate_structure
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Structure Validator - 結構驗證引擎
驗證目錄結構的正確性，包括：
1. 結構完整性驗證
2. 引用有效性驗證
3. 命名規範驗證
4. 內容一致性驗證
5. 生成驗證報告
Usage:
    python validate_structure.py full --target <dir>
    python validate_structure.py structure --target <dir>
    python validate_structure.py references --target <dir>
    python validate_structure.py report --output <file>
Version: 1.0.0
"""
# MNGA-002: Import organization needs review
import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import yaml
# ============================================================================
# 常數與配置
# ============================================================================
BASE_PATH = Path(__file__).parent.parent.parent
PLAYBOOKS_PATH = BASE_PATH / "docs" / "refactor_playbooks"
CONFIG_PATH = PLAYBOOKS_PATH / "config"
# ============================================================================
# 枚舉定義
# ============================================================================
class ValidationLevel(Enum):
    """驗證級別"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
class ValidationCategory(Enum):
    """驗證類別"""
    STRUCTURE = "structure"
    REFERENCE = "reference"
    NAMING = "naming"
    CONTENT = "content"
    CONSISTENCY = "consistency"
# ============================================================================
# 資料結構
# ============================================================================
@dataclass
class ValidationIssue:
    """驗證問題"""
    id: str
    category: ValidationCategory
    level: ValidationLevel
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    suggestion: str = ""
@dataclass
class ValidationResult:
    """驗證結果"""
    timestamp: str
    target_path: str
    is_valid: bool
    total_issues: int
    errors: List[ValidationIssue]
    warnings: List[ValidationIssue]
    info: List[ValidationIssue]
    statistics: Dict
    summary: str
@dataclass
class StructureSpec:
    """結構規範"""
    required_dirs: List[str]
    optional_dirs: List[str]
    required_files: List[str]
    max_depth: int
    max_root_files: int
    naming_pattern: str
# ============================================================================
# 結構驗證器
# ============================================================================
class StructureValidator:
    """
    結構驗證器：驗證目錄結構完整性
    """
    def __init__(self, spec: Optional[StructureSpec] = None):
        self.spec = spec or StructureSpec(
            required_dirs=[
                "01_deconstruction",
                "02_integration",
                "03_refactor",
            ],
            optional_dirs=[
                "_legacy_scratch",
                "config",
                "templates",
                "meta",
            ],
            required_files=[
                "README.md",
            ],
            max_depth=5,
            max_root_files=15,
            naming_pattern=r"^[a-z0-9_\-]+(\.[a-z]+)?$",
        )
    def validate(self, target_path: Path) -> List[ValidationIssue]:
        """驗證目錄結構"""
        issues = []
        # 檢查必要目錄
        issues.extend(self._check_required_dirs(target_path))
        # 檢查必要檔案
        issues.extend(self._check_required_files(target_path))
        # 檢查目錄深度
        issues.extend(self._check_depth(target_path))
        # 檢查根目錄檔案數量
        issues.extend(self._check_root_files(target_path))
        # 檢查空目錄
        issues.extend(self._check_empty_dirs(target_path))
        return issues
    def _check_required_dirs(self, target_path: Path) -> List[ValidationIssue]:
        """檢查必要目錄"""
        issues = []
        for dir_name in self.spec.required_dirs:
            dir_path = target_path / dir_name
            if not dir_path.exists():
                issues.append(
                    ValidationIssue(
                        id=f"struct_missing_dir_{dir_name}",
                        category=ValidationCategory.STRUCTURE,
                        level=ValidationLevel.ERROR,
                        message=f"缺少必要目錄: {dir_name}",
                        suggestion=f"建立目錄: mkdir {dir_name}",
                    )
                )
            elif not dir_path.is_dir():
                issues.append(
                    ValidationIssue(
                        id=f"struct_not_dir_{dir_name}",
                        category=ValidationCategory.STRUCTURE,
                        level=ValidationLevel.ERROR,
                        message=f"{dir_name} 不是目錄",
                    )
                )
        return issues
    def _check_required_files(self, target_path: Path) -> List[ValidationIssue]:
        """檢查必要檔案"""
        issues = []
        for file_name in self.spec.required_files:
            file_path = target_path / file_name
            if not file_path.exists():
                issues.append(
                    ValidationIssue(
                        id=f"struct_missing_file_{file_name}",
                        category=ValidationCategory.STRUCTURE,
                        level=ValidationLevel.WARNING,
                        message=f"缺少建議檔案: {file_name}",
                        suggestion=f"建立檔案: {file_name}",
                    )
                )
        return issues
    def _check_depth(self, target_path: Path) -> List[ValidationIssue]:
        """檢查目錄深度"""
        issues = []
        max_found = 0
        for path in target_path.rglob("*"):
            depth = len(path.relative_to(target_path).parts)
            max_found = max(max_found, depth)
            if depth > self.spec.max_depth:
                issues.append(
                    ValidationIssue(
                        id="struct_deep_path",
                        category=ValidationCategory.STRUCTURE,
                        level=ValidationLevel.WARNING,
                        message=f"路徑過深 (深度 {depth}): {path.relative_to(target_path)}",
                        file_path=str(path.relative_to(target_path)),
                        suggestion="考慮扁平化目錄結構",
                    )
                )
                break  # 只報告一次
        return issues
    def _check_root_files(self, target_path: Path) -> List[ValidationIssue]:
        """檢查根目錄檔案數量"""
        issues = []
        root_files = [f for f in target_path.iterdir() if f.is_file()]
        if len(root_files) > self.spec.max_root_files:
            issues.append(
                ValidationIssue(
                    id="struct_too_many_root_files",
                    category=ValidationCategory.STRUCTURE,
                    level=ValidationLevel.WARNING,
                    message=f"根目錄檔案過多 ({len(root_files)} 個，建議 <= {self.spec.max_root_files})",
                    suggestion="將檔案分類到子目錄",
                ))
        return issues
    def _check_empty_dirs(self, target_path: Path) -> List[ValidationIssue]:
        """檢查空目錄"""
        issues = []
        for dir_path in target_path.rglob("*"):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                # 排除特殊目錄
                if not dir_path.name.startswith((".", "_")):
                    issues.append(
                        ValidationIssue(
                            id="struct_empty_dir",
                            category=ValidationCategory.STRUCTURE,
                            level=ValidationLevel.INFO,
                            message=f"空目錄: {dir_path.relative_to(target_path)}",
                            file_path=str(dir_path.relative_to(target_path)),
                            suggestion="添加內容或刪除目錄",
                        )
                    )
        return issues
# ============================================================================
# 引用驗證器
# ============================================================================
class ReferenceValidator:
    """
    引用驗證器：驗證檔案引用有效性
    """
    def __init__(self, project_root: Path):
        self.project_root = project_root
    def validate(self, target_path: Path) -> List[ValidationIssue]:
        """驗證引用"""
        issues = []
        # 驗證 Markdown 連結
        issues.extend(self._validate_markdown_links(target_path))
        # 驗證 YAML 引用
        issues.extend(self._validate_yaml_refs(target_path))
        # 驗證相互引用
        issues.extend(self._validate_cross_refs(target_path))
        return issues
    def _validate_markdown_links(self, target_path: Path) -> List[ValidationIssue]:
        """驗證 Markdown 連結"""
        issues = []
        for md_file in target_path.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                # 找出所有連結
                for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", content):
                    link_text = match.group(1)
                    link_href = match.group(2)
                    # 跳過外部連結和錨點
                    if link_href.startswith(("http://", "https://", "#", "mailto:")):
                        continue
                    # 驗證內部連結
                    if (
                        link_href.startswith("./")
                        or link_href.startswith("../")
                        or "://" not in link_href
                    ):
                        resolved = (md_file.parent / link_href.split("#")[0]).resolve()
                        if not resolved.exists():
                            # 計算行號
                            line_num = content[: match.start()].count("\n") + 1
                            issues.append(
                                ValidationIssue(
                                    id=f"ref_broken_link_{md_file.stem}",
                                    category=ValidationCategory.REFERENCE,
                                    level=ValidationLevel.ERROR,
                                    message=f"斷開的連結: [{link_text}]({link_href})",
                                    file_path=str(md_file.relative_to(target_path)),
                                    line_number=line_num,
                                    suggestion="更新連結或建立目標檔案",
                                )
                            )
            except Exception as e:
                issues.append(
                    ValidationIssue(
                        id=f"ref_read_error_{md_file.stem}",
                        category=ValidationCategory.REFERENCE,
                        level=ValidationLevel.WARNING,
                        message=f"無法讀取檔案: {e}",
                        file_path=str(md_file.relative_to(target_path)),
                    )
                )
        return issues
    def _validate_yaml_refs(self, target_path: Path) -> List[ValidationIssue]:
        """驗證 YAML 引用"""
        issues = []
        for yaml_file in target_path.rglob("*.yaml"):
            try:
                content = yaml_file.read_text(encoding="utf-8")
                data = yaml.safe_load(content)
                # 遞迴檢查路徑引用
                issues.extend(self._check_yaml_paths(data, yaml_file, target_path))
            except yaml.YAMLError as e:
                issues.append(
                    ValidationIssue(
                        id=f"ref_yaml_error_{yaml_file.stem}",
                        category=ValidationCategory.REFERENCE,
                        level=ValidationLevel.ERROR,
                        message=f"YAML 解析錯誤: {e}",
                        file_path=str(yaml_file.relative_to(target_path)),
                    )
                )
            except Exception:
                pass
        return issues
    def _check_yaml_paths(
        self, data: Any, yaml_file: Path, target_path: Path, path: str = ""
    ) -> List[ValidationIssue]:
        """遞迴檢查 YAML 中的路徑引用"""
        issues = []
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                # 檢查路徑相關的鍵
                if key.endswith(("_path", "_file", "path", "file")) and isinstance(
                    value, str
                ):
                    if (
                        value
                        and value != "_pending"
                        and not value.startswith(("http://", "https://"))
                    ):
                        full_path = target_path / value
                        if not full_path.exists():
                            issues.append(
                                ValidationIssue(
                                    id=f"ref_yaml_path_{yaml_file.stem}",
                                    category=ValidationCategory.REFERENCE,
                                    level=ValidationLevel.ERROR,
                                    message=f"引用的路徑不存在: {value}",
                                    file_path=str(yaml_file.relative_to(target_path)),
                                    suggestion=f"檢查路徑 {new_path}",
                                )
                            )
                issues.extend(
                    self._check_yaml_paths(value, yaml_file, target_path, new_path)
                )
        elif isinstance(data, list):
            for i, item in enumerate(data):
                issues.extend(
                    self._check_yaml_paths(item, yaml_file, target_path, f"{path}[{i}]")
                )
        return issues
    def _validate_cross_refs(self, target_path: Path) -> List[ValidationIssue]:
        """驗證相互引用"""
        issues = []
        # 收集所有被引用的檔案
        referenced_files: Set[str] = set()
        all_files: Set[str] = set()
        for file in target_path.rglob("*"):
            if file.is_file() and not file.name.startswith("."):
                all_files.add(str(file.relative_to(target_path)))
                if file.suffix == ".md":
                    try:
                        content = file.read_text(encoding="utf-8")
                        for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", content):
                            href = match.group(2).split("#")[0]
                            if href and not href.startswith(("http", "mailto")):
                                resolved = (file.parent / href).resolve()
                                try:
                                    rel = str(resolved.relative_to(target_path))
                                    referenced_files.add(rel)
                                except ValueError:
                                    pass
                    except BaseException:
                        pass
        # 找出孤立檔案 (未被引用的 playbook)
        for file_path in all_files:
            if "playbook" in file_path.lower() and file_path not in referenced_files:
                # 檢查是否在索引中
                if "index" not in file_path.lower():
                    issues.append(
                        ValidationIssue(
                            id="ref_orphan_playbook",
                            category=ValidationCategory.REFERENCE,
                            level=ValidationLevel.INFO,
                            message=f"未被引用的 Playbook: {file_path}",
                            file_path=file_path,
                            suggestion="添加到索引或其他文件中",
                        )
                    )
        return issues
# ============================================================================
# 命名驗證器
# ============================================================================
class NamingValidator:
    """
    命名驗證器：驗證命名規範
    """
    def __init__(self, convention: str = "snake_case"):
        self.convention = convention
        # 命名模式
        self.patterns = {
            "snake_case": r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$",
            "kebab-case": r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$",
            "camelCase": r"^[a-z][a-zA-Z0-9]*$",
            "PascalCase": r"^[A-Z][a-zA-Z0-9]*$",
        }
        # 例外清單
        self.exceptions = {
            "README",
            "LICENSE",
            "CHANGELOG",
            "TODO",
            "INDEX",
            "ARCHITECTURE",
            "LEGACY_ANALYSIS_REPORT",
        }
    def validate(self, target_path: Path) -> List[ValidationIssue]:
        """驗證命名"""
        issues = []
        pattern = self.patterns.get(self.convention)
        if not pattern:
            return issues
        for file in target_path.rglob("*"):
            if file.is_file() and not file.name.startswith("."):
                stem = file.stem
                # 跳過例外
                if stem.upper() in self.exceptions:
                    continue
                # 檢查是否符合規範
                if not re.match(pattern, stem):
                    # 判斷問題類型
                    self._identify_naming_issue(stem)
                    issues.append(
                        ValidationIssue(
                            id=f"naming_{stem}",
                            category=ValidationCategory.NAMING,
                            level=ValidationLevel.WARNING,
                            message=f"命名不符合 {self.convention} 規範: {file.name}",
                            file_path=str(
                                file.relative_to(target_path)),
                            suggestion=f"建議重命名為: {self._suggest_name(stem, file.suffix)}",
                        ))
        return issues
    def _identify_naming_issue(self, name: str) -> str:
        """識別命名問題類型"""
        if re.search(r"[A-Z]", name) and "_" in name:
            return "mixed_case"
        elif "-" in name and "_" in name:
            return "mixed_separator"
        elif re.search(r"[A-Z]", name):
            return "has_uppercase"
        elif "-" in name:
            return "has_hyphen"
        elif "__" in name:
            return "double_underscore"
        return "unknown"
    def _suggest_name(self, name: str, suffix: str) -> str:
        """建議新名稱"""
        if self.convention == "snake_case":
            # 轉換為 snake_case
            new_name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
            new_name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", new_name)
            new_name = new_name.replace("-", "_")
            new_name = re.sub(r"_+", "_", new_name)
            return new_name.lower() + suffix
        return name + suffix
# ============================================================================
# 內容驗證器
# ============================================================================
class ContentValidator:
    """
    內容驗證器：驗證檔案內容
    """
    def validate(self, target_path: Path) -> List[ValidationIssue]:
        """驗證內容"""
        issues = []
        # 檢查 Markdown 格式
        issues.extend(self._validate_markdown_format(target_path))
        # 檢查 YAML 格式
        issues.extend(self._validate_yaml_format(target_path))
        # 檢查編碼
        issues.extend(self._validate_encoding(target_path))
        return issues
    def _validate_markdown_format(self, target_path: Path) -> List[ValidationIssue]:
        """驗證 Markdown 格式"""
        issues = []
        for md_file in target_path.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                # 檢查是否有標題
                if not re.search(r"^#\s+", content, re.MULTILINE):
                    issues.append(
                        ValidationIssue(
                            id=f"content_no_title_{md_file.stem}",
                            category=ValidationCategory.CONTENT,
                            level=ValidationLevel.WARNING,
                            message="缺少一級標題",
                            file_path=str(md_file.relative_to(target_path)),
                            suggestion="添加 # 標題",
                        )
                    )
                # 檢查標題層級跳躍
                headers = re.findall(r"^(#{1,6})\s", content, re.MULTILINE)
                prev_level = 0
                for header in headers:
                    level = len(header)
                    if level > prev_level + 1 and prev_level > 0:
                        issues.append(
                            ValidationIssue(
                                id=f"content_header_skip_{md_file.stem}",
                                category=ValidationCategory.CONTENT,
                                level=ValidationLevel.INFO,
                                message=f"標題層級跳躍 (H{prev_level} -> H{level})",
                                file_path=str(md_file.relative_to(target_path)),
                            )
                        )
                        break
                    prev_level = level
            except Exception:
                pass
        return issues
    def _validate_yaml_format(self, target_path: Path) -> List[ValidationIssue]:
        """驗證 YAML 格式"""
        issues = []
        for yaml_file in target_path.rglob("*.yaml"):
            try:
                content = yaml_file.read_text(encoding="utf-8")
                yaml.safe_load(content)
            except yaml.YAMLError as e:
                issues.append(
                    ValidationIssue(
                        id=f"content_yaml_error_{yaml_file.stem}",
                        category=ValidationCategory.CONTENT,
                        level=ValidationLevel.ERROR,
                        message=f"YAML 語法錯誤: {str(e)[:100]}",
                        file_path=str(yaml_file.relative_to(target_path)),
                    )
                )
        return issues
    def _validate_encoding(self, target_path: Path) -> List[ValidationIssue]:
        """驗證檔案編碼"""
        issues = []
        for file in target_path.rglob("*"):
            if file.is_file() and file.suffix in [
                ".md",
                ".yaml",
                ".yml",
                ".json",
                ".txt",
            ]:
                try:
                    file.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    issues.append(
                        ValidationIssue(
                            id=f"content_encoding_{file.stem}",
                            category=ValidationCategory.CONTENT,
                            level=ValidationLevel.ERROR,
                            message="檔案編碼不是 UTF-8",
                            file_path=str(file.relative_to(target_path)),
                            suggestion="轉換為 UTF-8 編碼",
                        )
                    )
        return issues
# ============================================================================
# 主驗證器
# ============================================================================
class StructureValidatorMain:
    """
    主驗證器：協調所有驗證器
    """
    def __init__(self, target_path: Optional[Path] = None):
        self.target_path = target_path or PLAYBOOKS_PATH
        self.structure_validator = StructureValidator()
        self.reference_validator = ReferenceValidator(BASE_PATH)
        self.naming_validator = NamingValidator("snake_case")
        self.content_validator = ContentValidator()
    def validate_full(self) -> ValidationResult:
        """執行完整驗證"""
        all_issues = []
        print(f"🔍 驗證: {self.target_path}")
        # 結構驗證
        print("   結構驗證...")
        all_issues.extend(self.structure_validator.validate(self.target_path))
        # 引用驗證
        print("   引用驗證...")
        all_issues.extend(self.reference_validator.validate(self.target_path))
        # 命名驗證
        print("   命名驗證...")
        all_issues.extend(self.naming_validator.validate(self.target_path))
        # 內容驗證
        print("   內容驗證...")
        all_issues.extend(self.content_validator.validate(self.target_path))
        return self._build_result(all_issues)
    def validate_structure(self) -> ValidationResult:
        """只驗證結構"""
        issues = self.structure_validator.validate(self.target_path)
        return self._build_result(issues)
    def validate_references(self) -> ValidationResult:
        """只驗證引用"""
        issues = self.reference_validator.validate(self.target_path)
        return self._build_result(issues)
    def validate_naming(self) -> ValidationResult:
        """只驗證命名"""
        issues = self.naming_validator.validate(self.target_path)
        return self._build_result(issues)
    def validate_content(self) -> ValidationResult:
        """只驗證內容"""
        issues = self.content_validator.validate(self.target_path)
        return self._build_result(issues)
    def _build_result(self, issues: List[ValidationIssue]) -> ValidationResult:
        """建構驗證結果"""
        errors = [i for i in issues if i.level == ValidationLevel.ERROR]
        warnings = [i for i in issues if i.level == ValidationLevel.WARNING]
        info = [i for i in issues if i.level == ValidationLevel.INFO]
        # 統計
        by_category = defaultdict(int)
        for issue in issues:
            by_category[issue.category.value] += 1
        # 生成摘要
        if len(errors) == 0:
            if len(warnings) == 0:
                summary = "✅ 驗證通過，無問題"
            else:
                summary = f"⚠️ 驗證通過，但有 {len(warnings)} 個警告"
        else:
            summary = f"❌ 驗證失敗，{len(errors)} 個錯誤，{len(warnings)} 個警告"
        return ValidationResult(
            timestamp=datetime.now().isoformat(),
            target_path=str(self.target_path),
            is_valid=len(errors) == 0,
            total_issues=len(issues),
            errors=errors,
            warnings=warnings,
            info=info,
            statistics={
                "total": len(issues),
                "errors": len(errors),
                "warnings": len(warnings),
                "info": len(info),
                "by_category": dict(by_category),
            },
            summary=summary,
        )
# ============================================================================
# 報告生成器
# ============================================================================
class ValidationReportGenerator:
    """
    驗證報告生成器
    """
    def generate_markdown(self, result: ValidationResult) -> str:
        """生成 Markdown 報告"""
        lines = [
            "# 結構驗證報告",
            "",
            f"> 驗證時間: {result.timestamp}",
            f"> 目標路徑: `{result.target_path}`",
            "",
            "## 摘要",
            "",
            result.summary,
            "",
            "## 統計",
            "",
            "| 級別 | 數量 |",
            "|------|------|",
            f"| 錯誤 | {result.statistics['errors']} |",
            f"| 警告 | {result.statistics['warnings']} |",
            f"| 資訊 | {result.statistics['info']} |",
            f"| 總計 | {result.statistics['total']} |",
            "",
        ]
        # 按類別統計
        if result.statistics.get("by_category"):
            lines.extend(
                [
                    "### 按類別",
                    "",
                ]
            )
            for cat, count in result.statistics["by_category"].items():
                lines.append(f"- {cat}: {count}")
            lines.append("")
        # 錯誤詳情
        if result.errors:
            lines.extend(
                [
                    "## 錯誤",
                    "",
                ]
            )
            for issue in result.errors:
                lines.append(f"### {issue.id}")
                lines.append(f"- **類別**: {issue.category.value}")
                lines.append(f"- **訊息**: {issue.message}")
                if issue.file_path:
                    lines.append(f"- **檔案**: `{issue.file_path}`")
                if issue.line_number:
                    lines.append(f"- **行號**: {issue.line_number}")
                if issue.suggestion:
                    lines.append(f"- **建議**: {issue.suggestion}")
                lines.append("")
        # 警告詳情
        if result.warnings:
            lines.extend(
                [
                    "## 警告",
                    "",
                ]
            )
            for issue in result.warnings[:20]:  # 限制數量
                lines.append(f"- **{issue.message}**")
                if issue.file_path:
                    lines.append(f"  - 檔案: `{issue.file_path}`")
                if issue.suggestion:
                    lines.append(f"  - 建議: {issue.suggestion}")
            if len(result.warnings) > 20:
                lines.append(f"- ... 還有 {len(result.warnings) - 20} 個警告")
            lines.append("")
        lines.extend(
            [
                "---",
                "",
                "*此報告由 `validate_structure.py` 自動生成*",
            ]
        )
        return "\n".join(lines)
    def generate_yaml(self, result: ValidationResult) -> Dict:
        """生成 YAML 格式報告"""
        return {
            "timestamp": result.timestamp,
            "target_path": result.target_path,
            "is_valid": result.is_valid,
            "summary": result.summary,
            "statistics": result.statistics,
            "errors": [asdict(e) for e in result.errors],
            "warnings": [asdict(w) for w in result.warnings],
            "info": [asdict(i) for i in result.info],
        }
# ============================================================================
# CLI 入口
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Structure Validator - 結構驗證引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    # full 命令
    full_parser = subparsers.add_parser("full", help="完整驗證")
    full_parser.add_argument(
        "--target", "-t", default=str(PLAYBOOKS_PATH), help="目標目錄"
    )
    full_parser.add_argument("--output", "-o", help="輸出報告")
    full_parser.add_argument(
        "--format", "-f", choices=["md", "yaml", "json"], default="md"
    )
    # structure 命令
    struct_parser = subparsers.add_parser("structure", help="結構驗證")
    struct_parser.add_argument("--target", "-t", default=str(PLAYBOOKS_PATH))
    # references 命令
    refs_parser = subparsers.add_parser("references", help="引用驗證")
    refs_parser.add_argument("--target", "-t", default=str(PLAYBOOKS_PATH))
    # naming 命令
    naming_parser = subparsers.add_parser("naming", help="命名驗證")
    naming_parser.add_argument("--target", "-t", default=str(PLAYBOOKS_PATH))
    naming_parser.add_argument(
        "--convention",
        "-c",
        default="snake_case",
        choices=["snake_case", "kebab-case", "camelCase"],
    )
    # content 命令
    content_parser = subparsers.add_parser("content", help="內容驗證")
    content_parser.add_argument("--target", "-t", default=str(PLAYBOOKS_PATH))
    # report 命令
    report_parser = subparsers.add_parser("report", help="生成完整報告")
    report_parser.add_argument("--target", "-t", default=str(PLAYBOOKS_PATH))
    report_parser.add_argument("--output", "-o", required=True, help="輸出檔案")
    report_parser.add_argument("--format", "-f", choices=["md", "yaml"], default="md")
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    target = Path(args.target) if hasattr(args, "target") else PLAYBOOKS_PATH
    validator = StructureValidatorMain(target)
    report_gen = ValidationReportGenerator()
    if args.command == "full":
        result = validator.validate_full()
        print(f"\n{result.summary}")
        print(f"  錯誤: {len(result.errors)}")
        print(f"  警告: {len(result.warnings)}")
        print(f"  資訊: {len(result.info)}")
        if args.output:
            if args.format == "md":
                output = report_gen.generate_markdown(result)
            elif args.format == "yaml":
                output = yaml.dump(
                    report_gen.generate_yaml(result),
                    allow_unicode=True,
                    default_flow_style=False,
                )
            else:
                output = json.dumps(
                    report_gen.generate_yaml(result), indent=2, ensure_ascii=False
                )
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"\n報告已儲存: {args.output}")
    elif args.command == "structure":
        result = validator.validate_structure()
        print(f"\n{result.summary}")
        for issue in result.errors + result.warnings:
            print(f"  - [{issue.level.value}] {issue.message}")
    elif args.command == "references":
        result = validator.validate_references()
        print(f"\n{result.summary}")
        for issue in result.errors + result.warnings:
            print(f"  - [{issue.level.value}] {issue.message}")
            if issue.file_path:
                print(f"    檔案: {issue.file_path}")
    elif args.command == "naming":
        validator.naming_validator = NamingValidator(args.convention)
        result = validator.validate_naming()
        print(f"\n{result.summary}")
        for issue in result.warnings[:20]:
            print(f"  - {issue.message}")
        if len(result.warnings) > 20:
            print(f"  ... 還有 {len(result.warnings) - 20} 個")
    elif args.command == "content":
        result = validator.validate_content()
        print(f"\n{result.summary}")
        for issue in result.errors + result.warnings:
            print(f"  - [{issue.level.value}] {issue.message}")
    elif args.command == "report":
        result = validator.validate_full()
        if args.format == "md":
            output = report_gen.generate_markdown(result)
        else:
            output = yaml.dump(
                report_gen.generate_yaml(result),
                allow_unicode=True,
                default_flow_style=False,
            )
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\n報告已儲存: {args.output}")
        print(result.summary)
if __name__ == "__main__":
    main()
