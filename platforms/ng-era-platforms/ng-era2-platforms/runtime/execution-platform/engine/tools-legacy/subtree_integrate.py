# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: subtree_integrate
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Git Subtree Integration Helper | Git Subtree 整合輔助工具
========================================================
Python 輔助工具，配合 integrate_repositories.sh 使用
解析 YAML 配置並提供倉庫信息
Usage:
    python tools/subtree_integrate.py --list
    python tools/subtree_integrate.py --core-only --list
    python tools/subtree_integrate.py --repo repo-name --list
    python tools/subtree_integrate.py --update repo-name
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
try:
    import yaml
except ImportError:
    print("❌ PyYAML not installed. Installing...", file=sys.stderr)
    subprocess.run([sys.executable, "-m", "pip", "install", "pyyaml"], check=True)
    import yaml
class SubtreeHelper:
    """Git Subtree integration helper"""
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.repo_root = Path(__file__).parent.parent
        self.config = self._load_config()
    def _load_config(self) -> dict:
        """Load YAML configuration"""
        if not self.config_path.exists():
            print(f"❌ Config not found: {self.config_path}", file=sys.stderr)
            sys.exit(1)
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    def list_repositories(
        self, core_only: bool = False, repo_name: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Get list of repositories
        Returns list of dicts with keys: name, url, branch
        """
        repos = []
        # Get repository lists
        core_repos = self.config.get("core_repositories", [])
        sync_repos = self.config.get("sync_repositories", [])
        # Select repositories based on parameters
        if repo_name:
            # Find specific repository
            all_repos = core_repos + sync_repos
            found = next((r for r in all_repos if r["name"] == repo_name), None)
            if found:
                repos = [found]
            else:
                print(f"❌ Repository '{repo_name}' not found", file=sys.stderr)
                sys.exit(1)
        elif core_only:
            repos = core_repos
        else:
            repos = core_repos + sync_repos
        return repos
    def print_repositories(
        self,
        core_only: bool = False,
        repo_name: Optional[str] = None,
        format: str = "pipe",
    ):
        """
        Print repositories in specified format
        Formats:
            pipe: name|url|branch (for shell script parsing)
            table: Pretty table
            json: JSON output
        """
        repos = self.list_repositories(core_only=core_only, repo_name=repo_name)
        if format == "pipe":
            # Pipe-delimited for shell parsing
            for repo in repos:
                name = repo.get("name", "")
                url = repo.get("url", "")
                branch = repo.get("branch", "main")
                print(f"{name}|{url}|{branch}")
        elif format == "table":
            # Pretty table
            print(f"{'Name':<30} {'URL':<60} {'Branch':<10}")
            print("-" * 100)
            for repo in repos:
                name = repo.get("name", "")
                url = repo.get("url", "")
                branch = repo.get("branch", "main")
                print(f"{name:<30} {url:<60} {branch:<10}")
        elif format == "json":
            import json
            print(json.dumps(repos, indent=2))
    def update_subtree(self, repo_name: str, squash: bool = True):
        """
        Update existing subtree
        Executes: git subtree pull --prefix=external/REPO remote branch
        """
        repos = self.list_repositories(repo_name=repo_name)
        if not repos:
            print(f"❌ Repository '{repo_name}' not found", file=sys.stderr)
            sys.exit(1)
        repo = repos[0]
        name = repo["name"]
        branch = repo.get("branch", "main")
        prefix = f"external/{name}"
        # Check if directory exists
        target_dir = self.repo_root / prefix
        if not target_dir.exists():
            print(f"❌ Directory not found: {target_dir}", file=sys.stderr)
            print(
                f"   Run: ./tools/integrate_repositories.sh --repo {name}",
                file=sys.stderr,
            )
            sys.exit(1)
        # Build command
        cmd = ["git", "subtree", "pull", f"--prefix={prefix}", name, branch]
        if squash:
            cmd.append("--squash")
        print(f"🔄 Updating subtree: {name}")
        print(f"   Prefix: {prefix}")
        print(f"   Remote: {name}")
        print(f"   Branch: {branch}")
        print(f"   Command: {' '.join(cmd)}")
        print()
        # Execute
        try:
            subprocess.run(cmd, check=True, capture_output=False)
            print()
            print(f"✅ Successfully updated: {name}")
            return True
        except subprocess.CalledProcessError as e:
            print()
            print(f"❌ Failed to update: {name}")
            print(f"   Error: {e}")
            return False
    def push_subtree(self, repo_name: str):
        """
        Push changes back to source repository
        Executes: git subtree push --prefix=external/REPO remote branch
        """
        repos = self.list_repositories(repo_name=repo_name)
        if not repos:
            print(f"❌ Repository '{repo_name}' not found", file=sys.stderr)
            sys.exit(1)
        repo = repos[0]
        name = repo["name"]
        branch = repo.get("branch", "main")
        prefix = f"external/{name}"
        # Check if directory exists
        target_dir = self.repo_root / prefix
        if not target_dir.exists():
            print(f"❌ Directory not found: {target_dir}", file=sys.stderr)
            sys.exit(1)
        # Build command
        cmd = ["git", "subtree", "push", f"--prefix={prefix}", name, branch]
        print(f"⬆️  Pushing subtree: {name}")
        print(f"   Prefix: {prefix}")
        print(f"   Remote: {name}")
        print(f"   Branch: {branch}")
        print(f"   Command: {' '.join(cmd)}")
        print()
        print("⚠️  Warning: This will push changes to the source repository!")
        print()
        # Confirm
        response = input("Continue? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted")
            return False
        # Execute
        try:
            subprocess.run(cmd, check=True, capture_output=False)
            print()
            print(f"✅ Successfully pushed: {name}")
            return True
        except subprocess.CalledProcessError as e:
            print()
            print(f"❌ Failed to push: {name}")
            print(f"   Error: {e}")
            return False
    def status(self):
        """Show integration status"""
        all_repos = self.config.get("core_repositories", []) + self.config.get(
            "sync_repositories", []
        )
        print("📊 Repository Integration Status")
        print("=" * 80)
        integrated = 0
        not_integrated = 0
        for repo in all_repos:
            name = repo["name"]
            target_dir = self.repo_root / "external" / name
            status_icon = "✅" if target_dir.exists() else "❌"
            status_text = "Integrated" if target_dir.exists() else "Not integrated"
            print(f"{status_icon} {name:<30} {status_text}")
            if target_dir.exists():
                integrated += 1
            else:
                not_integrated += 1
        print("=" * 80)
        print(
            f"Total: {len(all_repos)} | Integrated: {integrated} | Pending: {not_integrated}"
        )
def main():
    parser = argparse.ArgumentParser(description="Git Subtree Integration Helper")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/external_repos.yaml"),
        help="Configuration file path",
    )
    parser.add_argument("--list", action="store_true", help="List repositories")
    parser.add_argument(
        "--core-only", action="store_true", help="Only core repositories"
    )
    parser.add_argument("--repo", type=str, help="Specific repository name")
    parser.add_argument(
        "--format",
        choices=["pipe", "table", "json"],
        default="pipe",
        help="Output format",
    )
    parser.add_argument(
        "--update", type=str, metavar="REPO_NAME", help="Update existing subtree"
    )
    parser.add_argument(
        "--push",
        type=str,
        metavar="REPO_NAME",
        help="Push changes back to source repository",
    )
    parser.add_argument(
        "--no-squash", action="store_true", help="Don't squash commits when updating"
    )
    parser.add_argument("--status", action="store_true", help="Show integration status")
    args = parser.parse_args()
    # Initialize helper
    helper = SubtreeHelper(config_path=args.config)
    # Execute command
    if args.list:
        helper.print_repositories(
            core_only=args.core_only, repo_name=args.repo, format=args.format
        )
    elif args.update:
        squash = not args.no_squash
        success = helper.update_subtree(args.update, squash=squash)
        sys.exit(0 if success else 1)
    elif args.push:
        success = helper.push_subtree(args.push)
        sys.exit(0 if success else 1)
    elif args.status:
        helper.status()
    else:
        parser.print_help()
        sys.exit(1)
if __name__ == "__main__":
    main()
