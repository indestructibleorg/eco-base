# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: update_indexes
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Index Updater - 索引更新引擎
更新並同步各種索引檔案，包括：
1. index.yaml (機器可讀)
2. INDEX.md (人類可讀)
3. legacy_assets_index.yaml
4. 驗證索引一致性
Usage:
    python update_indexes.py all
    python update_indexes.py machine
    python update_indexes.py human
    python update_indexes.py verify
Version: 1.0.0
"""
import argparse
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import yaml
# ============================================================================
# 常數與配置
# ============================================================================
BASE_PATH = Path(__file__).parent.parent.parent
PLAYBOOKS_PATH = BASE_PATH / "docs" / "refactor_playbooks"
INDEX_YAML_PATH = PLAYBOOKS_PATH / "03_refactor" / "index.yaml"
INDEX_MD_PATH = PLAYBOOKS_PATH / "03_refactor" / "INDEX.md"
LEGACY_INDEX_PATH = PLAYBOOKS_PATH / "01_deconstruction" / "legacy_assets_index.yaml"
# ============================================================================
# 資料結構
# ============================================================================
@dataclass
class IndexEntry:
    """索引條目"""
    id: str
    name: str
    path: str
    type: str
    status: str
    description: str = ""
    metadata: Dict = field(default_factory=dict)
@dataclass
class ClusterEntry:
    """叢集條目"""
    cluster_id: str
    name: str
    description: str
    playbook_path: str
    status: str
    priority: str
    files: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
@dataclass
class LegacyAssetEntry:
    """遺留資產條目"""
    asset_id: str
    name: str
    source_path: str
    type: str
    status: str
    target_path: Optional[str] = None
    notes: str = ""
@dataclass
class IndexVerificationResult:
    """索引驗證結果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    missing_files: List[str]
    orphan_files: List[str]
    sync_status: Dict[str, bool]
# ============================================================================
# 索引掃描器
# ============================================================================
class IndexScanner:
    """
    索引掃描器：掃描目錄結構生成索引資料
    """
    def __init__(self, base_path: Path):
        self.base_path = base_path
    def scan_refactor_clusters(self) -> List[ClusterEntry]:
        """掃描重構叢集"""
        clusters = []
        refactor_path = self.base_path / "03_refactor"
        if not refactor_path.exists():
            return clusters
        # 掃描 playbook 檔案
        playbook_files = list(refactor_path.rglob("*_playbook.md")) + list(
            refactor_path.rglob("*__playbook.md")
        )
        for pb_file in playbook_files:
            cluster_info = self._parse_playbook(pb_file)
            if cluster_info:
                clusters.append(cluster_info)
        # 掃描子目錄作為叢集
        for subdir in refactor_path.iterdir():
            if subdir.is_dir() and not subdir.name.startswith(("_", ".")):
                # 檢查是否已有 playbook
                existing = next(
                    (c for c in clusters if subdir.name in c.playbook_path), None
                )
                if not existing:
                    clusters.append(
                        ClusterEntry(
                            cluster_id=subdir.name,
                            name=self._format_name(subdir.name),
                            description=f"{subdir.name} 相關檔案",
                            playbook_path="_pending",
                            status="pending",
                            priority="P3",
                            files=self._list_files(subdir),
                            tags=[subdir.name],
                        )
                    )
        return clusters
    def scan_legacy_assets(self) -> List[LegacyAssetEntry]:
        """掃描遺留資產"""
        assets = []
        legacy_path = self.base_path / "_legacy_scratch"
        if not legacy_path.exists():
            return assets
        for file in legacy_path.rglob("*"):
            if file.is_file() and not file.name.startswith("."):
                assets.append(
                    LegacyAssetEntry(
                        asset_id=file.stem[:8],
                        name=file.name,
                        source_path=str(file.relative_to(self.base_path)),
                        type=self._classify_type(file),
                        status="intake",
                    )
                )
        return assets
    def scan_all_documents(self) -> List[IndexEntry]:
        """掃描所有文檔"""
        entries = []
        for file in self.base_path.rglob("*.md"):
            if file.is_file() and not any(p.startswith("_") for p in file.parts):
                entries.append(
                    IndexEntry(
                        id=file.stem,
                        name=self._format_name(file.stem),
                        path=str(file.relative_to(self.base_path)),
                        type="document",
                        status="active",
                        description=self._extract_description(file),
                    )
                )
        return entries
    def _parse_playbook(self, pb_file: Path) -> Optional[ClusterEntry]:
        """解析 playbook 檔案"""
        try:
            content = pb_file.read_text(encoding="utf-8")
            # 提取標題
            title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            name = title_match.group(1) if title_match else pb_file.stem
            # 提取描述
            desc_match = re.search(r"^>\s*(.+)$", content, re.MULTILINE)
            description = desc_match.group(1) if desc_match else ""
            # 提取狀態
            status = "active"
            if "pending" in content.lower() or "TODO" in content:
                status = "pending"
            # 提取優先級
            priority = "P2"
            if "P1" in content:
                priority = "P1"
            elif "P3" in content:
                priority = "P3"
            return ClusterEntry(
                cluster_id=pb_file.stem.replace("__playbook", "").replace(
                    "_playbook", ""
                ),
                name=name,
                description=description,
                playbook_path=str(pb_file.relative_to(self.base_path)),
                status=status,
                priority=priority,
                files=[],
                tags=[],
            )
        except Exception as e:
            print(f"⚠️ 無法解析 {pb_file}: {e}")
            return None
    def _format_name(self, raw_name: str) -> str:
        """格式化名稱"""
        # 移除前綴數字
        name = re.sub(r"^\d+_", "", raw_name)
        # 轉換下劃線為空格
        name = name.replace("_", " ").replace("-", " ")
        # 首字母大寫
        return name.title()
    def _classify_type(self, file: Path) -> str:
        """分類檔案類型"""
        ext = file.suffix.lower()
        type_map = {
            ".md": "documentation",
            ".yaml": "configuration",
            ".yml": "configuration",
            ".json": "data",
            ".py": "code",
            ".txt": "text",
        }
        return type_map.get(ext, "unknown")
    def _extract_description(self, file: Path) -> str:
        """提取檔案描述"""
        try:
            content = file.read_text(encoding="utf-8")
            # 嘗試提取第一段非標題文字
            for line in content.split("\n"):
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith(">"):
                    return line[:200]
        except BaseException:
            pass
        return ""
    def _list_files(self, directory: Path) -> List[str]:
        """列出目錄中的檔案"""
        return [
            str(f.relative_to(self.base_path))
            for f in directory.rglob("*")
            if f.is_file()
        ]
# ============================================================================
# 索引生成器
# ============================================================================
class IndexGenerator:
    """
    索引生成器：生成各種格式的索引
    """
    def __init__(self, scanner: IndexScanner):
        self.scanner = scanner
    def generate_machine_index(self, clusters: List[ClusterEntry]) -> Dict:
        """生成機器可讀索引 (YAML)"""
        return {
            "version": "1.0.0",
            "generated_at": datetime.now().isoformat(),
            "description": "Refactor Playbooks 機器可讀索引",
            "refactor_clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "name": c.name,
                    "description": c.description,
                    "playbook_path": c.playbook_path,
                    "status": c.status,
                    "priority": c.priority,
                    "file_count": len(c.files),
                    "tags": c.tags,
                }
                for c in clusters
            ],
            "statistics": {
                "total_clusters": len(clusters),
                "by_status": self._count_by_field(clusters, "status"),
                "by_priority": self._count_by_field(clusters, "priority"),
            },
        }
    def generate_human_index(self, clusters: List[ClusterEntry]) -> str:
        """生成人類可讀索引 (Markdown)"""
        lines = [
            "# Refactor Playbooks 索引",
            "",
            f"> 自動生成於 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "## 概覽",
            "",
            f"- 總叢集數: {len(clusters)}",
            f"- 待處理: {sum(1 for c in clusters if c.status == 'pending')}",
            f"- 進行中: {sum(1 for c in clusters if c.status == 'active')}",
            f"- 已完成: {sum(1 for c in clusters if c.status == 'completed')}",
            "",
            "## 按優先級分類",
            "",
        ]
        # 按優先級分組
        by_priority = defaultdict(list)
        for c in clusters:
            by_priority[c.priority].append(c)
        for priority in ["P1", "P2", "P3"]:
            if priority in by_priority:
                lines.append(
                    f"### {priority} - {'緊急' if priority == 'P1' else '一般' if priority == 'P2' else '低優先級'}"
                )
                lines.append("")
                for c in by_priority[priority]:
                    status_emoji = {
                        "pending": "⏳",
                        "active": "🔄",
                        "completed": "✅",
                    }.get(c.status, "❓")
                    if c.playbook_path != "_pending":
                        lines.append(f"- [{c.name}]({c.playbook_path}) {status_emoji}")
                    else:
                        lines.append(f"- {c.name} {status_emoji} *(playbook 待建立)*")
                lines.append("")
        # 添加目錄結構
        lines.extend(
            [
                "## 目錄結構",
                "",
                "```",
                "03_refactor/",
            ]
        )
        for c in sorted(clusters, key=lambda x: x.cluster_id):
            lines.append(f"├── {c.cluster_id}/")
        lines.extend(
            [
                "└── index.yaml",
                "```",
                "",
                "---",
                "",
                "*此檔案由 `update_indexes.py` 自動生成*",
            ]
        )
        return "\n".join(lines)
    def generate_legacy_index(self, assets: List[LegacyAssetEntry]) -> Dict:
        """生成遺留資產索引"""
        return {
            "version": "1.0.0",
            "generated_at": datetime.now().isoformat(),
            "description": "遺留資產清單",
            "assets": [
                {
                    "asset_id": a.asset_id,
                    "name": a.name,
                    "source_path": a.source_path,
                    "type": a.type,
                    "status": a.status,
                    "target_path": a.target_path,
                    "notes": a.notes,
                }
                for a in assets
            ],
            "statistics": {
                "total_assets": len(assets),
                "by_type": self._count_by_field(assets, "type"),
                "by_status": self._count_by_field(assets, "status"),
            },
        }
    def _count_by_field(self, items: List, field: str) -> Dict[str, int]:
        """按欄位統計數量"""
        counts = defaultdict(int)
        for item in items:
            value = getattr(item, field, "unknown")
            counts[value] += 1
        return dict(counts)
# ============================================================================
# 索引驗證器
# ============================================================================
class IndexVerifier:
    """
    索引驗證器：驗證索引與實際結構的一致性
    """
    def __init__(self, base_path: Path):
        self.base_path = base_path
    def verify_all(self) -> IndexVerificationResult:
        """執行完整驗證"""
        errors = []
        warnings = []
        missing_files = []
        orphan_files = []
        sync_status = {}
        # 驗證 index.yaml
        yaml_result = self._verify_yaml_index()
        errors.extend(yaml_result.get("errors", []))
        missing_files.extend(yaml_result.get("missing", []))
        sync_status["index.yaml"] = len(yaml_result.get("errors", [])) == 0
        # 驗證 INDEX.md
        md_result = self._verify_md_index()
        warnings.extend(md_result.get("warnings", []))
        sync_status["INDEX.md"] = len(md_result.get("warnings", [])) == 0
        # 驗證 legacy_assets_index.yaml
        legacy_result = self._verify_legacy_index()
        errors.extend(legacy_result.get("errors", []))
        sync_status["legacy_assets_index.yaml"] = (
            len(legacy_result.get("errors", [])) == 0
        )
        # 檢查孤立檔案
        orphan_files = self._find_orphan_files()
        if orphan_files:
            warnings.append(f"發現 {len(orphan_files)} 個未被索引的檔案")
        return IndexVerificationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            missing_files=missing_files,
            orphan_files=orphan_files,
            sync_status=sync_status,
        )
    def _verify_yaml_index(self) -> Dict:
        """驗證 YAML 索引"""
        result = {"errors": [], "missing": []}
        if not INDEX_YAML_PATH.exists():
            result["errors"].append("index.yaml 不存在")
            return result
        try:
            with open(INDEX_YAML_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            # 檢查結構
            if "refactor_clusters" not in data:
                result["errors"].append("缺少 refactor_clusters 欄位")
                return result
            # 驗證每個叢集
            for cluster in data["refactor_clusters"]:
                playbook_path = cluster.get("playbook_path", "")
                if playbook_path and playbook_path != "_pending":
                    full_path = self.base_path / playbook_path
                    if not full_path.exists():
                        result["missing"].append(playbook_path)
                        result["errors"].append(f"Playbook 不存在: {playbook_path}")
        except yaml.YAMLError as e:
            result["errors"].append(f"YAML 解析錯誤: {e}")
        return result
    def _verify_md_index(self) -> Dict:
        """驗證 Markdown 索引"""
        result = {"warnings": []}
        if not INDEX_MD_PATH.exists():
            result["warnings"].append("INDEX.md 不存在")
            return result
        try:
            content = INDEX_MD_PATH.read_text(encoding="utf-8")
            # 檢查連結
            links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)
            for text, href in links:
                if not href.startswith(("http", "#")):
                    full_path = INDEX_MD_PATH.parent / href
                    if not full_path.exists():
                        result["warnings"].append(f"斷開的連結: {href}")
        except Exception as e:
            result["warnings"].append(f"讀取錯誤: {e}")
        return result
    def _verify_legacy_index(self) -> Dict:
        """驗證遺留資產索引"""
        result = {"errors": []}
        if not LEGACY_INDEX_PATH.exists():
            # 遺留索引可選
            return result
        try:
            with open(LEGACY_INDEX_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            # 驗證每個資產
            for asset in data.get("assets", []):
                source_path = asset.get("source_path", "")
                if source_path:
                    full_path = self.base_path / source_path
                    if not full_path.exists():
                        result["errors"].append(f"資產不存在: {source_path}")
        except yaml.YAMLError as e:
            result["errors"].append(f"YAML 解析錯誤: {e}")
        return result
    def _find_orphan_files(self) -> List[str]:
        """找出未被索引的檔案"""
        orphans = []
        # 收集索引中的所有路徑
        indexed_paths = set()
        if INDEX_YAML_PATH.exists():
            try:
                with open(INDEX_YAML_PATH, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                for cluster in data.get("refactor_clusters", []):
                    if cluster.get("playbook_path"):
                        indexed_paths.add(cluster["playbook_path"])
            except BaseException:
                pass
        # 掃描實際檔案
        refactor_path = self.base_path / "03_refactor"
        if refactor_path.exists():
            for file in refactor_path.rglob("*.md"):
                rel_path = str(file.relative_to(self.base_path))
                if rel_path not in indexed_paths and "INDEX" not in file.name:
                    orphans.append(rel_path)
        return orphans
# ============================================================================
# 索引更新器 (主類)
# ============================================================================
class IndexUpdater:
    """
    索引更新器：協調所有索引操作
    """
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or PLAYBOOKS_PATH
        self.scanner = IndexScanner(self.base_path)
        self.generator = IndexGenerator(self.scanner)
        self.verifier = IndexVerifier(self.base_path)
    def update_all(self) -> Dict[str, bool]:
        """更新所有索引"""
        results = {}
        print("📚 更新所有索引...")
        # 更新機器索引
        results["machine"] = self.update_machine_index()
        # 更新人類索引
        results["human"] = self.update_human_index()
        # 更新遺留索引
        results["legacy"] = self.update_legacy_index()
        return results
    def update_machine_index(self) -> bool:
        """更新機器可讀索引"""
        print("   更新 index.yaml...")
        try:
            clusters = self.scanner.scan_refactor_clusters()
            index_data = self.generator.generate_machine_index(clusters)
            INDEX_YAML_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(INDEX_YAML_PATH, "w", encoding="utf-8") as f:
                yaml.dump(
                    index_data,
                    f,
                    allow_unicode=True,
                    default_flow_style=False,
                    sort_keys=False,
                )
            print(f"   ✓ 已更新 ({len(clusters)} 叢集)")
            return True
        except Exception as e:
            print(f"   ✗ 錯誤: {e}")
            return False
    def update_human_index(self) -> bool:
        """更新人類可讀索引"""
        print("   更新 INDEX.md...")
        try:
            clusters = self.scanner.scan_refactor_clusters()
            index_content = self.generator.generate_human_index(clusters)
            INDEX_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(INDEX_MD_PATH, "w", encoding="utf-8") as f:
                f.write(index_content)
            print("   ✓ 已更新")
            return True
        except Exception as e:
            print(f"   ✗ 錯誤: {e}")
            return False
    def update_legacy_index(self) -> bool:
        """更新遺留資產索引"""
        print("   更新 legacy_assets_index.yaml...")
        try:
            assets = self.scanner.scan_legacy_assets()
            index_data = self.generator.generate_legacy_index(assets)
            LEGACY_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(LEGACY_INDEX_PATH, "w", encoding="utf-8") as f:
                yaml.dump(
                    index_data,
                    f,
                    allow_unicode=True,
                    default_flow_style=False,
                    sort_keys=False,
                )
            print(f"   ✓ 已更新 ({len(assets)} 資產)")
            return True
        except Exception as e:
            print(f"   ✗ 錯誤: {e}")
            return False
    def verify(self) -> IndexVerificationResult:
        """驗證索引"""
        print("🔍 驗證索引...")
        return self.verifier.verify_all()
# ============================================================================
# CLI 入口
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Index Updater - 索引更新引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    # all 命令
    subparsers.add_parser("all", help="更新所有索引")
    # machine 命令
    subparsers.add_parser("machine", help="更新 index.yaml")
    # human 命令
    subparsers.add_parser("human", help="更新 INDEX.md")
    # legacy 命令
    subparsers.add_parser(
        "legacy", help="更新 legacy_assets_index.yaml"
    )
    # verify 命令
    verify_parser = subparsers.add_parser("verify", help="驗證索引")
    verify_parser.add_argument("--fix", action="store_true", help="自動修復問題")
    # scan 命令
    scan_parser = subparsers.add_parser("scan", help="掃描目錄結構")
    scan_parser.add_argument("--output", "-o", help="輸出檔案")
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    updater = IndexUpdater()
    if args.command == "all":
        results = updater.update_all()
        print("\n結果:")
        for name, success in results.items():
            status = "✅" if success else "❌"
            print(f"  {name}: {status}")
    elif args.command == "machine":
        success = updater.update_machine_index()
        print(f"\n{'✅ 成功' if success else '❌ 失敗'}")
    elif args.command == "human":
        success = updater.update_human_index()
        print(f"\n{'✅ 成功' if success else '❌ 失敗'}")
    elif args.command == "legacy":
        success = updater.update_legacy_index()
        print(f"\n{'✅ 成功' if success else '❌ 失敗'}")
    elif args.command == "verify":
        result = updater.verify()
        print(f"\n驗證結果: {'✅ 通過' if result.is_valid else '❌ 失敗'}")
        if result.errors:
            print(f"\n錯誤 ({len(result.errors)}):")
            for err in result.errors:
                print(f"  - {err}")
        if result.warnings:
            print(f"\n警告 ({len(result.warnings)}):")
            for warn in result.warnings:
                print(f"  - {warn}")
        if result.missing_files:
            print(f"\n缺失檔案 ({len(result.missing_files)}):")
            for f in result.missing_files:
                print(f"  - {f}")
        if result.orphan_files:
            print(f"\n孤立檔案 ({len(result.orphan_files)}):")
            for f in result.orphan_files[:10]:  # 只顯示前10個
                print(f"  - {f}")
            if len(result.orphan_files) > 10:
                print(f"  ... 還有 {len(result.orphan_files) - 10} 個")
        print("\n同步狀態:")
        for name, synced in result.sync_status.items():
            status = "✅" if synced else "❌"
            print(f"  {name}: {status}")
        if args.fix and not result.is_valid:
            print("\n🔧 嘗試自動修復...")
            updater.update_all()
    elif args.command == "scan":
        scanner = IndexScanner(PLAYBOOKS_PATH)
        output = {
            "clusters": [asdict(c) for c in scanner.scan_refactor_clusters()],
            "legacy_assets": [asdict(a) for a in scanner.scan_legacy_assets()],
            "documents": [asdict(d) for d in scanner.scan_all_documents()],
        }
        output_str = yaml.dump(output, allow_unicode=True, default_flow_style=False)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_str)
            print(f"掃描結果已儲存: {args.output}")
        else:
            print(output_str)
if __name__ == "__main__":
    main()
