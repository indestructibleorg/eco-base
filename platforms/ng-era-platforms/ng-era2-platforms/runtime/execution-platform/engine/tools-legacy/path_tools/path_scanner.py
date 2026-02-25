# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: path_scanner
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Path Scanner - 路徑掃描器
掃描專案目錄結構，識別所有檔案路徑並生成索引。
Features:
1. 遞迴掃描目錄結構
2. 識別 Markdown 連結和 YAML 引用
3. 生成結構化索引報告
4. 支援多種輸出格式 (JSON, YAML)
Usage:
    python path_scanner.py --target <dir>
    python path_scanner.py --target <dir> --output result.json
    python path_scanner.py --target <dir> --include "*.md" "*.yaml"
"""
# MNGA-002: Import organization needs review
import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
try:
    import yaml
except ImportError:
    yaml = None
@dataclass
class FileInfo:
    """檔案資訊"""
    path: str
    name: str
    extension: str
    size: int
    is_directory: bool
    depth: int
    modified_time: str
    hash: Optional[str] = None
@dataclass
class LinkInfo:
    """連結資訊"""
    source_file: str
    link_text: str
    link_href: str
    line_number: int
    link_type: str
    is_valid: bool = True
    resolved_path: Optional[str] = None
@dataclass
class ScanResult:
    """掃描結果"""
    timestamp: str
    target_path: str
    total_files: int
    total_directories: int
    total_links: int
    files: List[FileInfo]
    links: List[LinkInfo]
    statistics: Dict[str, Any]
class PathScanner:
    """路徑掃描器"""
    EXCLUDE_DIRS = {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "build",
        ".next",
        ".cache",
        "coverage",
    }
    EXCLUDE_PATTERNS = [
        r"\.pyc$",
        r"\.pyo$",
        r"\.class$",
        r"\.o$",
        r"\.so$",
        r"\.dll$",
        r"\.exe$",
        r"\.lock$",
        r"\.min\.",
    ]
    def __init__(self, target_path: Path, include_patterns: Optional[List[str]] = None):
        self.target_path = target_path.resolve()
        self.include_patterns = include_patterns or ["*"]
        self.files: List[FileInfo] = []
        self.links: List[LinkInfo] = []
        self.seen_paths: Set[str] = set()
    def scan(self) -> ScanResult:
        """執行掃描"""
        self._scan_directory(self.target_path)
        self._extract_links()
        stats = self._calculate_statistics()
        return ScanResult(
            timestamp=datetime.now().isoformat(),
            target_path=str(self.target_path),
            total_files=len([f for f in self.files if not f.is_directory]),
            total_directories=len([f for f in self.files if f.is_directory]),
            total_links=len(self.links),
            files=self.files,
            links=self.links,
            statistics=stats,
        )
    def _scan_directory(self, dir_path: Path, depth: int = 0):
        """遞迴掃描目錄"""
        try:
            for entry in sorted(dir_path.iterdir()):
                if entry.name.startswith(".") and entry.name not in [".github"]:
                    continue
                if entry.is_dir():
                    if entry.name in self.EXCLUDE_DIRS:
                        continue
                    rel_path = str(entry.relative_to(self.target_path))
                    self.files.append(
                        FileInfo(
                            path=rel_path,
                            name=entry.name,
                            extension="",
                            size=0,
                            is_directory=True,
                            depth=depth,
                            modified_time=datetime.fromtimestamp(
                                entry.stat().st_mtime
                            ).isoformat(),
                        )
                    )
                    self._scan_directory(entry, depth + 1)
                elif entry.is_file():
                    if self._should_exclude(entry):
                        continue
                    if not self._matches_include_pattern(entry):
                        continue
                    rel_path = str(entry.relative_to(self.target_path))
                    if rel_path in self.seen_paths:
                        continue
                    self.seen_paths.add(rel_path)
                    stat = entry.stat()
                    self.files.append(
                        FileInfo(
                            path=rel_path,
                            name=entry.name,
                            extension=entry.suffix,
                            size=stat.st_size,
                            is_directory=False,
                            depth=depth,
                            modified_time=datetime.fromtimestamp(
                                stat.st_mtime
                            ).isoformat(),
                            hash=(
                                self._calculate_hash(entry)
                                if stat.st_size < 1024 * 1024
                                else None
                            ),
                        )
                    )
        except PermissionError:
            pass
    def _should_exclude(self, path: Path) -> bool:
        """檢查是否應排除"""
        name = path.name
        for pattern in self.EXCLUDE_PATTERNS:
            if re.search(pattern, name):
                return True
        return False
    def _matches_include_pattern(self, path: Path) -> bool:
        """檢查是否符合包含模式"""
        if self.include_patterns == ["*"]:
            return True
        for pattern in self.include_patterns:
            if path.match(pattern):
                return True
        return False
    def _calculate_hash(self, file_path: Path) -> str:
        """計算檔案 SHA256 雜湊"""
        sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for block in iter(lambda: f.read(4096), b""):
                    sha256.update(block)
            return sha256.hexdigest()[:16]
        except Exception:
            return ""
    def _extract_links(self):
        """擷取檔案中的連結"""
        for file_info in self.files:
            if file_info.is_directory:
                continue
            file_path = self.target_path / file_info.path
            if file_info.extension == ".md":
                self._extract_markdown_links(file_path, file_info.path)
            elif file_info.extension in [".yaml", ".yml"]:
                self._extract_yaml_refs(file_path, file_info.path)
    def _extract_markdown_links(self, file_path: Path, rel_path: str):
        """擷取 Markdown 連結"""
        try:
            content = file_path.read_text(encoding="utf-8")
            for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", content):
                link_text = match.group(1)
                link_href = match.group(2).split("#")[0]
                line_num = content[: match.start()].count("\n") + 1
                if link_href.startswith(("http://", "https://", "mailto:", "#")):
                    link_type = "external"
                    is_valid = True
                    resolved = None
                else:
                    link_type = "internal"
                    resolved = (file_path.parent / link_href).resolve()
                    try:
                        resolved.relative_to(self.target_path)
                        is_valid = resolved.exists()
                        resolved = (
                            str(resolved.relative_to(self.target_path))
                            if is_valid
                            else None
                        )
                    except ValueError:
                        is_valid = False
                        resolved = None
                self.links.append(
                    LinkInfo(
                        source_file=rel_path,
                        link_text=link_text,
                        link_href=link_href,
                        line_number=line_num,
                        link_type=link_type,
                        is_valid=is_valid,
                        resolved_path=resolved,
                    )
                )
        except Exception:
            pass
    def _extract_yaml_refs(self, file_path: Path, rel_path: str):
        """擷取 YAML 路徑引用"""
        if yaml is None:
            return
        try:
            content = file_path.read_text(encoding="utf-8")
            data = yaml.safe_load(content)
            self._scan_yaml_paths(data, file_path, rel_path)
        except Exception:
            pass
    def _scan_yaml_paths(
        self, data: Any, file_path: Path, rel_path: str, key_path: str = ""
    ):
        """遞迴掃描 YAML 中的路徑"""
        if isinstance(data, dict):
            for key, value in data.items():
                new_key = f"{key_path}.{key}" if key_path else key
                if key.endswith(("_path", "_file", "path", "file")) and isinstance(
                    value, str
                ):
                    if value and not value.startswith(("http://", "https://")):
                        resolved = (file_path.parent / value).resolve()
                        try:
                            resolved.relative_to(self.target_path)
                            is_valid = resolved.exists()
                            resolved_str = (
                                str(resolved.relative_to(self.target_path))
                                if is_valid
                                else None
                            )
                        except ValueError:
                            is_valid = False
                            resolved_str = None
                        self.links.append(
                            LinkInfo(
                                source_file=rel_path,
                                link_text=new_key,
                                link_href=value,
                                line_number=0,
                                link_type="yaml_ref",
                                is_valid=is_valid,
                                resolved_path=resolved_str,
                            )
                        )
                self._scan_yaml_paths(value, file_path, rel_path, new_key)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                self._scan_yaml_paths(item, file_path, rel_path, f"{key_path}[{i}]")
    def _calculate_statistics(self) -> Dict[str, Any]:
        """計算統計資訊"""
        ext_count: Dict[str, int] = {}
        total_size = 0
        for f in self.files:
            if not f.is_directory:
                ext = f.extension or "no_ext"
                ext_count[ext] = ext_count.get(ext, 0) + 1
                total_size += f.size
        broken_links = [l for l in self.links if not l.is_valid]
        return {
            "extensions": ext_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "max_depth": max((f.depth for f in self.files), default=0),
            "broken_links_count": len(broken_links),
            "internal_links_count": len(
                [l for l in self.links if l.link_type == "internal"]
            ),
            "external_links_count": len(
                [l for l in self.links if l.link_type == "external"]
            ),
        }
def main():
    parser = argparse.ArgumentParser(description="路徑掃描器 - 掃描專案目錄結構")
    parser.add_argument("--target", "-t", default=".", help="目標目錄")
    parser.add_argument("--output", "-o", help="輸出檔案路徑")
    parser.add_argument("--include", nargs="+", default=["*"], help="包含模式 (glob)")
    parser.add_argument("--json", action="store_true", help="輸出 JSON 格式")
    parser.add_argument("--summary", action="store_true", help="只顯示摘要")
    args = parser.parse_args()
    target = Path(args.target).resolve()
    if not target.exists() or not target.is_dir():
        print(f"錯誤: 目標目錄不存在: {target}")
        sys.exit(1)
    print(f"🔍 掃描目錄: {target}")
    scanner = PathScanner(target, args.include)
    result = scanner.scan()
    if args.summary:
        print("\n📊 掃描結果摘要:")
        print(f"   檔案數: {result.total_files}")
        print(f"   目錄數: {result.total_directories}")
        print(f"   連結數: {result.total_links}")
        print(f"   斷開連結: {result.statistics['broken_links_count']}")
        print(f"   總大小: {result.statistics['total_size_mb']} MB")
        return 0
    output_data = {
        "timestamp": result.timestamp,
        "target_path": result.target_path,
        "total_files": result.total_files,
        "total_directories": result.total_directories,
        "total_links": result.total_links,
        "statistics": result.statistics,
        "files": [asdict(f) for f in result.files],
        "links": [asdict(l) for l in result.links],
    }
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            if args.json or output_path.suffix == ".json":
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            elif yaml and output_path.suffix in [".yaml", ".yml"]:
                yaml.dump(output_data, f, allow_unicode=True, default_flow_style=False)
            else:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"✅ 結果已儲存至: {output_path}")
    else:
        print(json.dumps(output_data, indent=2, ensure_ascii=False))
    if result.statistics["broken_links_count"] > 0:
        print(f"\n⚠️  發現 {result.statistics['broken_links_count']} 個斷開的連結")
        return 1
    return 0
if __name__ == "__main__":
    sys.exit(main())
