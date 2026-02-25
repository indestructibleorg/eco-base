# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: path_fixer
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Path Fixer - 路徑修復器
自動修復常見的路徑問題，如斷開連結、錯誤引用等。
Features:
1. 斷開連結修復
2. 路徑引用更新
3. 檔案重命名建議
4. 批量修復支援
5. 乾運行模式
Usage:
    python path_fixer.py --target <dir> --dry-run
    python path_fixer.py --target <dir> --fix
    python path_fixer.py --target <dir> --fix --backup
"""
# MNGA-002: Import organization needs review
import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
try:
    import yaml
except ImportError:
    yaml = None
@dataclass
class FixAction:
    """修復動作"""
    id: str
    action_type: str
    source_file: str
    description: str
    old_value: str
    new_value: str
    line_number: Optional[int] = None
    applied: bool = False
    error: Optional[str] = None
@dataclass
class FixResult:
    """修復結果"""
    timestamp: str
    target_path: str
    total_actions: int
    applied_actions: int
    failed_actions: int
    skipped_actions: int
    actions: List[FixAction]
    summary: str
class PathFixer:
    """路徑修復器"""
    def __init__(self, target_path: Path, dry_run: bool = True, backup: bool = False):
        self.target_path = target_path.resolve()
        self.dry_run = dry_run
        self.backup = backup
        self.actions: List[FixAction] = []
        self.file_index: Dict[str, Path] = {}
        self._build_file_index()
    def _build_file_index(self):
        """建立檔案索引"""
        for file_path in self.target_path.rglob("*"):
            if file_path.is_file():
                self.file_index[file_path.name] = file_path
                self.file_index[file_path.stem] = file_path
    def analyze_and_fix(self) -> FixResult:
        """分析並修復問題"""
        self.actions = []
        self._find_broken_markdown_links()
        self._find_broken_yaml_refs()
        self._find_naming_issues()
        if not self.dry_run:
            self._apply_fixes()
        applied = len([a for a in self.actions if a.applied])
        failed = len([a for a in self.actions if a.error])
        skipped = len([a for a in self.actions if not a.applied and not a.error])
        return FixResult(
            timestamp=datetime.now().isoformat(),
            target_path=str(self.target_path),
            total_actions=len(self.actions),
            applied_actions=applied,
            failed_actions=failed,
            skipped_actions=skipped,
            actions=self.actions,
            summary=self._generate_summary(applied, failed, skipped),
        )
    def _find_broken_markdown_links(self):
        """找出並建議修復斷開的 Markdown 連結"""
        for md_file in self.target_path.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                rel_path = str(md_file.relative_to(self.target_path))
                for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", content):
                    link_text = match.group(1)
                    link_href = match.group(2).split("#")[0]
                    anchor = (
                        match.group(2)[len(link_href):]
                        if "#" in match.group(2)
                        else ""
                    )
                    line_num = content[: match.start()].count("\n") + 1
                    if link_href.startswith(("http://", "https://", "mailto:", "#")):
                        continue
                    if link_href:
                        resolved = (md_file.parent / link_href).resolve()
                        if not resolved.exists():
                            suggested = self._suggest_fix(link_href, md_file.parent)
                            if suggested:
                                self.actions.append(
                                    FixAction(
                                        id=f"fix_link_{md_file.stem}_{line_num}",
                                        action_type="update_link",
                                        source_file=rel_path,
                                        description=f"更新斷開連結: [{link_text}]",
                                        old_value=link_href,
                                        new_value=suggested + anchor,
                                        line_number=line_num,
                                    )
                                )
            except Exception:
                pass
    def _find_broken_yaml_refs(self):
        """找出並建議修復斷開的 YAML 引用"""
        if yaml is None:
            return
        for yaml_file in self.target_path.rglob("*.yaml"):
            try:
                content = yaml_file.read_text(encoding="utf-8")
                data = yaml.safe_load(content)
                self._scan_yaml_for_fixes(data, yaml_file, content)
            except Exception:
                pass
        for yml_file in self.target_path.rglob("*.yml"):
            try:
                content = yml_file.read_text(encoding="utf-8")
                data = yaml.safe_load(content)
                self._scan_yaml_for_fixes(data, yml_file, content)
            except Exception:
                pass
    def _scan_yaml_for_fixes(
        self, data: Any, yaml_file: Path, content: str, key_path: str = ""
    ):
        """遞迴掃描 YAML 中需要修復的路徑"""
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
                            suggested = self._suggest_fix(value, yaml_file.parent)
                            if suggested:
                                rel_path = str(yaml_file.relative_to(self.target_path))
                                self.actions.append(
                                    FixAction(
                                        id=f"fix_yaml_{yaml_file.stem}_{key}",
                                        action_type="update_yaml_ref",
                                        source_file=rel_path,
                                        description=f"更新 YAML 路徑引用: {new_path}",
                                        old_value=value,
                                        new_value=suggested,
                                    )
                                )
                self._scan_yaml_for_fixes(value, yaml_file, content, new_path)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                self._scan_yaml_for_fixes(item, yaml_file, content, f"{key_path}[{i}]")
    def _find_naming_issues(self):
        """找出命名問題並建議修復"""
        snake_case_pattern = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
        exceptions = {"README", "LICENSE", "CHANGELOG", "TODO", "INDEX"}
        for file_path in self.target_path.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith("."):
                stem = file_path.stem
                if stem.upper() in exceptions:
                    continue
                if not snake_case_pattern.match(stem):
                    if re.search(r"[A-Z]", stem) or "-" in stem:
                        new_name = self._to_snake_case(stem) + file_path.suffix
                        rel_path = str(file_path.relative_to(self.target_path))
                        if new_name != file_path.name:
                            self.actions.append(
                                FixAction(
                                    id=f"rename_{stem}",
                                    action_type="rename_file",
                                    source_file=rel_path,
                                    description="重命名為 snake_case",
                                    old_value=file_path.name,
                                    new_value=new_name,
                                )
                            )
    def _suggest_fix(self, broken_path: str, source_dir: Path) -> Optional[str]:
        """嘗試建議修復路徑"""
        path_parts = Path(broken_path)
        target_name = path_parts.name
        if target_name in self.file_index:
            found = self.file_index[target_name]
            try:
                rel = os.path.relpath(found, source_dir)
                return rel.replace("\\", "/")
            except ValueError:
                return None
        stem = path_parts.stem
        if stem in self.file_index:
            found = self.file_index[stem]
            try:
                rel = os.path.relpath(found, source_dir)
                return rel.replace("\\", "/")
            except ValueError:
                return None
        snake_stem = self._to_snake_case(stem)
        if snake_stem in self.file_index:
            found = self.file_index[snake_stem]
            try:
                rel = os.path.relpath(found, source_dir)
                return rel.replace("\\", "/")
            except ValueError:
                return None
        for name, path in self.file_index.items():
            if name.lower() == target_name.lower():
                try:
                    rel = os.path.relpath(path, source_dir)
                    return rel.replace("\\", "/")
                except ValueError:
                    pass
        return None
    def _to_snake_case(self, name: str) -> str:
        """轉換為 snake_case"""
        new_name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
        new_name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", new_name)
        new_name = new_name.replace("-", "_")
        new_name = re.sub(r"_+", "_", new_name)
        return new_name.lower()
    def _apply_fixes(self):
        """套用修復"""
        for action in self.actions:
            try:
                if action.action_type == "update_link":
                    self._apply_link_fix(action)
                elif action.action_type == "update_yaml_ref":
                    self._apply_yaml_fix(action)
                elif action.action_type == "rename_file":
                    self._apply_rename(action)
            except Exception as e:
                action.error = str(e)
    def _apply_link_fix(self, action: FixAction):
        """套用連結修復"""
        file_path = self.target_path / action.source_file
        if self.backup:
            shutil.copy2(file_path, str(file_path) + ".bak")
        content = file_path.read_text(encoding="utf-8")
        old_pattern = re.escape(f"]({action.old_value})")
        new_value = f"]({action.new_value})"
        new_content = re.sub(old_pattern, new_value, content, count=1)
        if content != new_content:
            file_path.write_text(new_content, encoding="utf-8")
            action.applied = True
    def _apply_yaml_fix(self, action: FixAction):
        """套用 YAML 修復"""
        file_path = self.target_path / action.source_file
        if self.backup:
            shutil.copy2(file_path, str(file_path) + ".bak")
        content = file_path.read_text(encoding="utf-8")
        new_content = content.replace(action.old_value, action.new_value, 1)
        if content != new_content:
            file_path.write_text(new_content, encoding="utf-8")
            action.applied = True
    def _apply_rename(self, action: FixAction):
        """套用重命名"""
        file_path = self.target_path / action.source_file
        new_path = file_path.parent / action.new_value
        if new_path.exists():
            action.error = f"目標檔案已存在: {action.new_value}"
            return
        if self.backup:
            shutil.copy2(file_path, str(file_path) + ".bak")
        file_path.rename(new_path)
        action.applied = True
    def _generate_summary(self, applied: int, failed: int, skipped: int) -> str:
        """生成摘要"""
        if self.dry_run:
            return f"🔍 乾運行模式: 發現 {len(self.actions)} 個可修復問題"
        elif failed > 0:
            return f"⚠️ 套用 {applied} 個修復, {failed} 個失敗"
        elif applied > 0:
            return f"✅ 成功套用 {applied} 個修復"
        else:
            return "ℹ️ 無需修復"
def main():
    parser = argparse.ArgumentParser(description="路徑修復器 - 自動修復路徑問題")
    parser.add_argument("--target", "-t", default=".", help="目標目錄")
    parser.add_argument(
        "--dry-run", "-n", action="store_true", help="乾運行模式 (只分析不修改)"
    )
    parser.add_argument("--fix", action="store_true", help="套用修復")
    parser.add_argument("--backup", "-b", action="store_true", help="修復前備份")
    parser.add_argument("--report", "-r", help="輸出報告檔案")
    parser.add_argument("--quiet", "-q", action="store_true", help="安靜模式")
    args = parser.parse_args()
    dry_run = not args.fix
    try:
        target = Path(args.target).resolve()
        if not target.exists() or not target.is_dir():
            print(f"錯誤: 目標目錄不存在: {target}")
            sys.exit(1)
    except Exception as e:
        print(f"錯誤: {e}")
        sys.exit(1)
    mode = "乾運行" if dry_run else "修復"
    print(f"🔧 路徑修復器 ({mode}模式)")
    print(f"   目標: {target}")
    fixer = PathFixer(target, dry_run=dry_run, backup=args.backup)
    result = fixer.analyze_and_fix()
    if not args.quiet:
        print(f"\n{result.summary}")
        print("\n📊 結果:")
        print(f"   發現問題: {result.total_actions}")
        if not dry_run:
            print(f"   已修復: {result.applied_actions}")
            print(f"   失敗: {result.failed_actions}")
        if result.actions:
            print("\n📋 動作列表:")
            for action in result.actions:
                status = "✅" if action.applied else ("❌" if action.error else "⏸")
                print(f"   {status} [{action.action_type}] {action.description}")
                print(f"      檔案: {action.source_file}")
                print(f"      舊值: {action.old_value}")
                print(f"      新值: {action.new_value}")
                if action.error:
                    print(f"      錯誤: {action.error}")
    if args.report:
        output_data = {
            "timestamp": result.timestamp,
            "target_path": result.target_path,
            "dry_run": dry_run,
            "total_actions": result.total_actions,
            "applied_actions": result.applied_actions,
            "failed_actions": result.failed_actions,
            "skipped_actions": result.skipped_actions,
            "summary": result.summary,
            "actions": [asdict(a) for a in result.actions],
        }
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 報告已儲存至: {args.report}")
    if dry_run and result.total_actions > 0:
        print("\n💡 提示: 使用 --fix 參數套用修復")
    sys.exit(0)
if __name__ == "__main__":
    main()
