# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: path_validator
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Path Validator - 路徑驗證器
驗證路徑的有效性，檢測斷開的連結、無效引用等問題。
Features:
1. 路徑安全驗證 (防止注入和遍歷)
2. Markdown 連結驗證
3. YAML 引用驗證
4. 結構完整性驗證
5. 生成詳細驗證報告
Usage:
    python path_validator.py --target <dir>
    python path_validator.py --target <dir> --full
    python path_validator.py --target <dir> --report report.json
"""
# MNGA-002: Import organization needs review
import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional
try:
    import yaml
except ImportError:
    yaml = None
class ValidationLevel(Enum):
    """驗證級別"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
class ValidationCategory(Enum):
    """驗證類別"""
    SECURITY = "security"
    STRUCTURE = "structure"
    REFERENCE = "reference"
    NAMING = "naming"
    CONTENT = "content"
@dataclass
class ValidationIssue:
    """驗證問題"""
    id: str
    category: str
    level: str
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
    errors: int
    warnings: int
    info: int
    issues: List[ValidationIssue]
    summary: str
def validate_path_security(input_path: str, base_dir: Optional[Path] = None) -> str:
    """
    驗證路徑安全性 - 防止路徑注入和遍歷攻擊
    Args:
        input_path: 用戶輸入的路徑
        base_dir: 基礎目錄 (用於限制路徑範圍)
    Returns:
        驗證後的絕對路徑
    Raises:
        ValueError: 路徑無效、包含禁止字符或超出允許範圍
    """
    if base_dir is None:
        base_dir = Path.cwd().resolve()
    else:
        base_dir = base_dir.resolve()
    forbidden_chars = re.compile(r"[\x00-\x1f;&|$`><]")
    if forbidden_chars.search(input_path):
        raise ValueError(f"路徑包含禁止字符: {input_path}")
    if not input_path or input_path.isspace():
        raise ValueError("路徑不能為空")
    try:
        resolved = Path(input_path).expanduser().resolve()
    except Exception as e:
        raise ValueError(f"無效路徑: {input_path}") from e
    if not resolved.exists():
        raise ValueError(f"路徑不存在: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"路徑不是目錄: {resolved}")
    try:
        resolved.relative_to(base_dir)
    except ValueError:
        raise ValueError(f"路徑必須在基礎目錄內: {base_dir}")
    return str(resolved)
class PathValidator:
    """路徑驗證器"""
    def __init__(self, target_path: Path):
        self.target_path = target_path.resolve()
        self.issues: List[ValidationIssue] = []
    def validate_all(self) -> ValidationResult:
        """執行完整驗證"""
        self.issues = []
        self._validate_structure()
        self._validate_references()
        self._validate_naming()
        self._validate_security()
        errors = [i for i in self.issues if i.level == ValidationLevel.ERROR.value]
        warnings = [i for i in self.issues if i.level == ValidationLevel.WARNING.value]
        info = [i for i in self.issues if i.level == ValidationLevel.INFO.value]
        return ValidationResult(
            timestamp=datetime.now().isoformat(),
            target_path=str(self.target_path),
            is_valid=len(errors) == 0,
            total_issues=len(self.issues),
            errors=len(errors),
            warnings=len(warnings),
            info=len(info),
            issues=self.issues,
            summary=self._generate_summary(errors, warnings, info),
        )
    def _validate_structure(self):
        """驗證結構完整性"""
        required_files = ["README.md"]
        for file_name in required_files:
            if not (self.target_path / file_name).exists():
                self.issues.append(
                    ValidationIssue(
                        id=f"struct_missing_{file_name}",
                        category=ValidationCategory.STRUCTURE.value,
                        level=ValidationLevel.WARNING.value,
                        message=f"缺少建議檔案: {file_name}",
                        suggestion=f"建立 {file_name} 檔案",
                    )
                )
        empty_dirs = []
        for dir_path in self.target_path.rglob("*"):
            if dir_path.is_dir():
                if not any(dir_path.iterdir()):
                    if not dir_path.name.startswith((".", "_")):
                        empty_dirs.append(str(dir_path.relative_to(self.target_path)))
        for empty_dir in empty_dirs[:5]:
            self.issues.append(
                ValidationIssue(
                    id="struct_empty_dir",
                    category=ValidationCategory.STRUCTURE.value,
                    level=ValidationLevel.INFO.value,
                    message=f"空目錄: {empty_dir}",
                    file_path=empty_dir,
                    suggestion="添加內容或刪除目錄",
                )
            )
    def _validate_references(self):
        """驗證引用有效性"""
        for md_file in self.target_path.rglob("*.md"):
            self._validate_markdown_links(md_file)
        if yaml:
            for yaml_file in self.target_path.rglob("*.yaml"):
                self._validate_yaml_refs(yaml_file)
            for yml_file in self.target_path.rglob("*.yml"):
                self._validate_yaml_refs(yml_file)
    def _validate_markdown_links(self, md_file: Path):
        """驗證 Markdown 連結"""
        try:
            content = md_file.read_text(encoding="utf-8")
            rel_path = str(md_file.relative_to(self.target_path))
            for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", content):
                link_text = match.group(1)
                link_href = match.group(2).split("#")[0]
                line_num = content[: match.start()].count("\n") + 1
                if link_href.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                if link_href:
                    resolved = (md_file.parent / link_href).resolve()
                    if not resolved.exists():
                        self.issues.append(
                            ValidationIssue(
                                id=f"ref_broken_link_{md_file.stem}",
                                category=ValidationCategory.REFERENCE.value,
                                level=ValidationLevel.ERROR.value,
                                message=f"斷開的連結: [{link_text}]({link_href})",
                                file_path=rel_path,
                                line_number=line_num,
                                suggestion="更新連結或建立目標檔案",
                            )
                        )
        except Exception as e:
            self.issues.append(
                ValidationIssue(
                    id=f"ref_read_error_{md_file.stem}",
                    category=ValidationCategory.REFERENCE.value,
                    level=ValidationLevel.WARNING.value,
                    message=f"無法讀取檔案: {str(e)[:100]}",
                    file_path=str(md_file.relative_to(self.target_path)),
                )
            )
    def _validate_yaml_refs(self, yaml_file: Path):
        """驗證 YAML 引用"""
        try:
            content = yaml_file.read_text(encoding="utf-8")
            data = yaml.safe_load(content)
            self._check_yaml_paths(data, yaml_file)
        except yaml.YAMLError as e:
            self.issues.append(
                ValidationIssue(
                    id=f"ref_yaml_error_{yaml_file.stem}",
                    category=ValidationCategory.CONTENT.value,
                    level=ValidationLevel.ERROR.value,
                    message=f"YAML 語法錯誤: {str(e)[:100]}",
                    file_path=str(yaml_file.relative_to(self.target_path)),
                )
            )
        except Exception:
            pass
    def _check_yaml_paths(self, data: Any, yaml_file: Path, key_path: str = ""):
        """遞迴檢查 YAML 中的路徑"""
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{key_path}.{key}" if key_path else key
                if key.endswith(("_path", "_file", "path", "file")) and isinstance(
                    value, str
                ):
                    if (
                        value
                        and value != "_pending"
                        and not value.startswith(("http://", "https://"))
                    ):
                        full_path = self.target_path / value
                        if not full_path.exists():
                            self.issues.append(
                                ValidationIssue(
                                    id=f"ref_yaml_path_{yaml_file.stem}",
                                    category=ValidationCategory.REFERENCE.value,
                                    level=ValidationLevel.ERROR.value,
                                    message=f"YAML 引用的路徑不存在: {value}",
                                    file_path=str(
                                        yaml_file.relative_to(self.target_path)
                                    ),
                                    suggestion=f"檢查路徑 {new_path}",
                                )
                            )
                self._check_yaml_paths(value, yaml_file, new_path)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                self._check_yaml_paths(item, yaml_file, f"{key_path}[{i}]")
    def _validate_naming(self):
        """驗證命名規範"""
        snake_case_pattern = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
        exceptions = {"README", "LICENSE", "CHANGELOG", "TODO", "INDEX"}
        for file_path in self.target_path.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith("."):
                stem = file_path.stem
                if stem.upper() in exceptions:
                    continue
                if not snake_case_pattern.match(stem):
                    if re.search(r"[A-Z]", stem) or "-" in stem:
                        rel_path = str(file_path.relative_to(self.target_path))
                        suggested = self._suggest_snake_case(stem, file_path.suffix)
                        self.issues.append(
                            ValidationIssue(
                                id=f"naming_{stem}",
                                category=ValidationCategory.NAMING.value,
                                level=ValidationLevel.INFO.value,
                                message=f"命名不符合 snake_case 規範: {file_path.name}",
                                file_path=rel_path,
                                suggestion=f"建議: {suggested}",
                            )
                        )
    def _suggest_snake_case(self, name: str, suffix: str) -> str:
        """建議 snake_case 名稱"""
        new_name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
        new_name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", new_name)
        new_name = new_name.replace("-", "_")
        new_name = re.sub(r"_+", "_", new_name)
        return new_name.lower() + suffix
    def _validate_security(self):
        """驗證安全性問題"""
        for file_path in self.target_path.rglob("*"):
            if file_path.is_symlink():
                try:
                    resolved = file_path.resolve()
                    resolved.relative_to(self.target_path)
                except ValueError:
                    self.issues.append(
                        ValidationIssue(
                            id="security_symlink_escape",
                            category=ValidationCategory.SECURITY.value,
                            level=ValidationLevel.ERROR.value,
                            message=f"符號連結指向外部: {file_path.relative_to(self.target_path)}",
                            file_path=str(
                                file_path.relative_to(
                                    self.target_path)),
                            suggestion="移除或修正符號連結",
                        ))
    def _generate_summary(self, errors: list, warnings: list, info: list) -> str:
        """生成摘要"""
        if not errors and not warnings:
            return "✅ 驗證通過，未發現重大問題"
        elif not errors:
            return f"⚠️ 發現 {len(warnings)} 個警告"
        else:
            return f"❌ 發現 {len(errors)} 個錯誤, {len(warnings)} 個警告"
def main():
    parser = argparse.ArgumentParser(description="路徑驗證器 - 驗證路徑有效性")
    parser.add_argument("--target", "-t", default=".", help="目標目錄")
    parser.add_argument("--full", action="store_true", help="完整驗證")
    parser.add_argument("--report", "-r", help="輸出報告檔案")
    parser.add_argument("--quiet", "-q", action="store_true", help="只顯示錯誤")
    args = parser.parse_args()
    try:
        target = Path(args.target).resolve()
        if not target.exists() or not target.is_dir():
            print(f"錯誤: 目標目錄不存在: {target}")
            sys.exit(1)
    except Exception as e:
        print(f"錯誤: {e}")
        sys.exit(1)
    print(f"🔍 驗證目錄: {target}")
    validator = PathValidator(target)
    result = validator.validate_all()
    if not args.quiet:
        print(f"\n{result.summary}")
        print("\n📊 驗證結果:")
        print(f"   錯誤: {result.errors}")
        print(f"   警告: {result.warnings}")
        print(f"   資訊: {result.info}")
        if result.issues:
            print("\n📋 問題列表:")
            for issue in result.issues:
                level_icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(
                    issue.level, "•"
                )
                print(f"   {level_icon} [{issue.category}] {issue.message}")
                if issue.file_path:
                    print(
                        f"      檔案: {issue.file_path}"
                        + (f":{issue.line_number}" if issue.line_number else "")
                    )
                if issue.suggestion:
                    print(f"      建議: {issue.suggestion}")
    if args.report:
        output_data = {
            "timestamp": result.timestamp,
            "target_path": result.target_path,
            "is_valid": result.is_valid,
            "total_issues": result.total_issues,
            "errors": result.errors,
            "warnings": result.warnings,
            "info": result.info,
            "summary": result.summary,
            "issues": [asdict(i) for i in result.issues],
        }
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 報告已儲存至: {args.report}")
    sys.exit(0 if result.is_valid else 1)
if __name__ == "__main__":
    main()
