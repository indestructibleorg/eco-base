# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: sync_external_repos
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Multi-Repository Auto-Sync Tool | 多倉庫自動同步工具
==================================================
自動從其他倉庫拉取最新代碼到 keystone-ai
Usage:
    python tools/sync_external_repos.py
    python tools/sync_external_repos.py --core-only
    python tools/sync_external_repos.py --repo repo-name
"""
# MNGA-002: Import organization needs review
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
try:
    import yaml
except ImportError:
    print("❌ PyYAML not installed. Installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pyyaml"], check=True)
    import yaml
# Colors
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    END = "\033[0m"
    BOLD = "\033[1m"
def log_info(msg):
    print(f"{Colors.CYAN}ℹ️  {msg}{Colors.END}")
def log_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")
def log_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")
def log_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")
class RepoSyncer:
    """Repository synchronization manager"""
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.repo_root = Path(__file__).parent.parent
        self.external_dir = self.repo_root / "external"
        self.config = self._load_config()
        self.stats = {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    def _load_config(self) -> dict:
        """Load configuration from YAML"""
        if not self.config_path.exists():
            log_error(f"Config file not found: {self.config_path}")
            log_info("Creating example config...")
            self._create_example_config()
            sys.exit(1)
        with open(self.config_path, encoding='utf-8') as f:
            return yaml.safe_load(f)
    def _create_example_config(self):
        """Create example configuration file"""
        example_config = {
            "core_repositories": [
                {
                    "name": "example-core-repo",
                    "url": "https://github.com/your-org/core-repo.git",
                    "branch": "main",
                    "priority": "high",
                    "description": "Core authentication service",
                }
            ],
            "sync_repositories": [
                {
                    "name": "example-sync-repo",
                    "url": "https://github.com/your-org/sync-repo.git",
                    "branch": "main",
                    "priority": "medium",
                    "description": "Legacy system integration",
                }
            ],
            "sync_options": {
                "exclude_patterns": [
                    "*.pyc",
                    "__pycache__",
                    "node_modules",
                    ".git",
                    ".env",
                    "*.log",
                ],
                "include_paths": ["src/", "lib/", "config/"],
                "preserve_permissions": True,
            },
        }
        config_file = self.repo_root / "config" / "external_repos.yaml"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, "w", encoding='utf-8') as f:
            yaml.dump(example_config, f, default_flow_style=False, allow_unicode=True)
        log_success(f"Example config created: {config_file}")
        log_info("Please edit this file and add your repositories")
    def sync_repo(self, repo_config: dict, dry_run: bool = False) -> bool:
        """Sync a single repository"""
        name = repo_config["name"]
        url = repo_config["url"]
        branch = repo_config.get("branch", "main")
        priority = repo_config.get("priority", "medium")
        log_info(f"Syncing: {name} ({priority} priority)")
        # Target directory
        target_dir = self.external_dir / name
        temp_dir = Path(
            f"/tmp/keystone_sync_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        try:
            # Clone repository
            log_info(f"  Cloning from {url}...")
            result = subprocess.run(
                ["git", "clone", "--depth=1", "--branch", branch, url, str(temp_dir)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                log_error(f"  Clone failed: {result.stderr}")
                return False
            # Remove .git directory
            git_dir = temp_dir / ".git"
            if git_dir.exists():
                shutil.rmtree(git_dir)
            # Apply exclusion patterns
            excluded = self._apply_exclusions(temp_dir)
            if excluded:
                log_info(f"  Excluded {excluded} items")
            # Copy to target
            if not dry_run:
                if target_dir.exists():
                    log_warning(f"  Removing existing: {target_dir}")
                    shutil.rmtree(target_dir)
                log_info(f"  Copying to: {target_dir}")
                shutil.copytree(temp_dir, target_dir)
                # Create metadata file
                self._create_metadata(target_dir, repo_config)
                log_success(f"  ✓ {name} synced successfully")
            else:
                log_info(f"  [DRY RUN] Would sync to: {target_dir}")
            # Cleanup
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            return True
        except subprocess.TimeoutExpired:
            log_error(f"  Timeout while cloning {name}")
            return False
        except Exception as e:
            log_error(f"  Failed to sync {name}: {e}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            return False
    def _apply_exclusions(self, repo_dir: Path) -> int:
        """Apply exclusion patterns"""
        exclude_patterns = self.config.get("sync_options", {}).get(
            "exclude_patterns", []
        )
        removed_count = 0
        for pattern in exclude_patterns:
            if "*" in pattern:
                # Glob pattern
                for item in repo_dir.rglob(pattern):
                    if item.exists():
                        if item.is_file():
                            item.unlink()
                        else:
                            shutil.rmtree(item)
                        removed_count += 1
            else:
                # Exact match
                for item in repo_dir.rglob(pattern):
                    if item.name == pattern:
                        if item.is_file():
                            item.unlink()
                        else:
                            shutil.rmtree(item)
                        removed_count += 1
        return removed_count
    def _create_metadata(self, target_dir: Path, repo_config: dict):
        """Create metadata file for synced repository"""
        metadata = {
            "name": repo_config["name"],
            "source_url": repo_config["url"],
            "branch": repo_config.get("branch", "main"),
            "synced_at": datetime.now().isoformat(),
            "priority": repo_config.get("priority", "medium"),
            "description": repo_config.get("description", ""),
        }
        metadata_file = target_dir / ".sync_metadata.json"
        with open(metadata_file, "w", encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    def sync_all(
        self, core_only: bool = False, exclude_core: bool = False, dry_run: bool = False
    ):
        """Sync all repositories"""
        log_info("=" * 60)
        log_info("Multi-Repository Sync Tool")
        log_info("=" * 60)
        # Create external directory
        self.external_dir.mkdir(exist_ok=True)
        # Get repositories to sync
        repos_to_sync = []
        if core_only:
            repos_to_sync = self.config.get("core_repositories", [])
            log_info(f"Syncing {len(repos_to_sync)} CORE repositories only")
        elif exclude_core:
            # Exclude core repositories (for hybrid mode)
            repos_to_sync = self.config.get("sync_repositories", [])
            log_info(
                f"Syncing {len(repos_to_sync)} REGULAR repositories (excluding core)"
            )
        else:
            core_repos = self.config.get("core_repositories", [])
            sync_repos = self.config.get("sync_repositories", [])
            repos_to_sync = core_repos + sync_repos
            log_info(
                f"Syncing {len(repos_to_sync)} repositories ({len(core_repos)} core + {len(sync_repos)} sync)"
            )
        if dry_run:
            log_warning("DRY RUN MODE - No changes will be made")
        # Sync each repository
        self.stats["total"] = len(repos_to_sync)
        for idx, repo in enumerate(repos_to_sync, 1):
            log_info(f"\n[{idx}/{len(repos_to_sync)}] Processing: {repo['name']}")
            if self.sync_repo(repo, dry_run=dry_run):
                self.stats["success"] += 1
            else:
                self.stats["failed"] += 1
        # Print summary
        self._print_summary()
    def sync_single(self, repo_name: str, dry_run: bool = False):
        """Sync a single repository by name"""
        # Find repository in config
        all_repos = self.config.get("core_repositories", []) + self.config.get(
            "sync_repositories", []
        )
        repo_config = next((r for r in all_repos if r["name"] == repo_name), None)
        if not repo_config:
            log_error(f"Repository '{repo_name}' not found in config")
            log_info("Available repositories:")
            for r in all_repos:
                print(f"  - {r['name']}")
            return False
        log_info(f"Syncing single repository: {repo_name}")
        self.stats["total"] = 1
        if self.sync_repo(repo_config, dry_run=dry_run):
            self.stats["success"] += 1
            self._print_summary()
            return True
        else:
            self.stats["failed"] += 1
            self._print_summary()
            return False
    def _print_summary(self):
        """Print sync summary"""
        log_info("\n" + "=" * 60)
        log_info("Sync Summary")
        log_info("=" * 60)
        log_info(f"Total:    {self.stats['total']}")
        log_success(f"Success:  {self.stats['success']}")
        log_error(f"Failed:   {self.stats['failed']}")
        log_info("=" * 60)
        if self.stats["success"] > 0:
            log_success(
                f"\n✨ {self.stats['success']} repositories synced to: {self.external_dir}"
            )
            log_info("Next steps:")
            log_info("  1. Review changes: git status")
            log_info(
                "  2. Commit: git add external/ && git commit -m 'chore: sync external repos'"
            )
            log_info("  3. Push: git push")
def main():
    parser = argparse.ArgumentParser(description="Multi-Repository Auto-Sync Tool")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/external_repos.yaml"),
        help="Configuration file path",
    )
    parser.add_argument(
        "--core-only", action="store_true", help="Sync core repositories only"
    )
    parser.add_argument(
        "--exclude-core",
        action="store_true",
        help="Exclude core repositories (sync regular repos only, for hybrid mode)",
    )
    parser.add_argument("--repo", type=str, help="Sync single repository by name")
    parser.add_argument(
        "--dry-run", action="store_true", help="Dry run mode (no actual changes)"
    )
    args = parser.parse_args()
    # Initialize syncer
    syncer = RepoSyncer(config_path=args.config)
    # Execute
    if args.repo:
        syncer.sync_single(args.repo, dry_run=args.dry_run)
    else:
        syncer.sync_all(
            core_only=args.core_only,
            exclude_core=args.exclude_core,
            dry_run=args.dry_run,
        )
if __name__ == "__main__":
    main()
