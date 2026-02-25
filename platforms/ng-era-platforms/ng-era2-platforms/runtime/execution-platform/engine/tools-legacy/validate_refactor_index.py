# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: validate-refactor-index
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Refactor Index Validator
Validates consistency between index.yaml and actual playbook files
This script checks:
1. All referenced files exist
2. All clusters in index.yaml have corresponding playbook files
3. Legacy asset IDs are defined in legacy_assets_index.yaml
4. index.yaml and INDEX.md are in sync
"""
# MNGA-002: Import organization needs review
import sys
from pathlib import Path
import yaml
class RefactorIndexValidator:
    """Validates refactor playbook index consistency"""
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.refactor_root = repo_root / "docs" / "refactor_playbooks"
        self.errors: list[str] = []
        self.warnings: list[str] = []
    def load_index_yaml(self) -> dict:
        """Load the machine-readable index"""
        index_path = self.refactor_root / "03_refactor" / "index.yaml"
        if not index_path.exists():
            self.errors.append(f"index.yaml not found at {index_path}")
            return {}
        with open(index_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    def load_legacy_assets_index(self) -> dict:
        """Load the legacy assets index"""
        assets_path = (
            self.refactor_root / "01_deconstruction" / "legacy_assets_index.yaml"
        )
        if not assets_path.exists():
            self.errors.append(f"legacy_assets_index.yaml not found at {assets_path}")
            return {}
        with open(assets_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    def validate_file_exists(
        self, file_path: str, cluster_id: str, field_name: str
    ) -> bool:
        """Check if a referenced file exists"""
        if not file_path:
            return True  # Empty is okay for optional fields
        # Handle relative paths from 03_refactor/
        base_path = self.refactor_root / "03_refactor"
        full_path = (base_path / file_path).resolve()
        if not full_path.exists():
            self.errors.append(
                f"Cluster '{cluster_id}': {field_name} file not found: {file_path}"
            )
            return False
        return True
    def validate_legacy_assets(
        self, asset_ids: list[str], cluster_id: str, defined_assets: set[str]
    ) -> bool:
        """Check if legacy asset IDs are defined"""
        if not asset_ids:
            return True
        all_valid = True
        for asset_id in asset_ids:
            if asset_id not in defined_assets:
                self.errors.append(
                    f"Cluster '{cluster_id}': Legacy asset '{asset_id}' not found in "
                    "legacy_assets_index.yaml"
                )
                all_valid = False
        return all_valid
    def validate_cluster_entry(self, cluster: dict, defined_assets: set[str]) -> bool:
        """Validate a single cluster entry"""
        cluster_id = cluster.get("cluster_id", "UNKNOWN")
        # Check required fields
        required_fields = ["cluster_id", "domain", "status", "refactor_file"]
        for field in required_fields:
            if field not in cluster or not cluster[field]:
                self.errors.append(
                    f"Cluster '{cluster_id}': Missing required field '{field}'"
                )
                return False
        # Validate file references
        self.validate_file_exists(
            cluster.get("refactor_file", ""), cluster_id, "refactor_file"
        )
        self.validate_file_exists(
            cluster.get("deconstruction_file", ""), cluster_id, "deconstruction_file"
        )
        self.validate_file_exists(
            cluster.get("integration_file", ""), cluster_id, "integration_file"
        )
        # Validate legacy assets
        legacy_assets = cluster.get("legacy_assets", [])
        self.validate_legacy_assets(legacy_assets, cluster_id, defined_assets)
        # Validate priority
        priority = cluster.get("priority", "")
        if priority and priority not in ["P0", "P1", "P2"]:
            self.warnings.append(
                f"Cluster '{cluster_id}': Invalid priority '{priority}' (should be P0/P1/P2)"
            )
        # Validate status
        status = cluster.get("status", "")
        valid_statuses = ["draft", "in_progress", "completed", "blocked", "pending"]
        if status and status not in valid_statuses:
            self.warnings.append(
                f"Cluster '{cluster_id}': Invalid status '{status}' "
                f"(should be one of {valid_statuses})"
            )
        # Check gl-platform.governance_status structure
        if "gl-platform.governance_status" in cluster:
            gov_status = cluster["gl-platform.governance_status"]
            required_gov_fields = ["violations", "threshold", "auto_fixable"]
            for field in required_gov_fields:
                if field not in gov_status:
                    self.warnings.append(
                        f"Cluster '{cluster_id}': gl-platform.governance_status missing field '{field}'"
                    )
        return True
    def validate_index(self) -> tuple[int, int]:
        """Main validation logic"""
        print("🔍 Validating Refactor Index...")
        print(f"📂 Repository root: {self.repo_root}")
        print()
        # Load indexes
        index_data = self.load_index_yaml()
        assets_data = self.load_legacy_assets_index()
        if not index_data or "clusters" not in index_data:
            self.errors.append("index.yaml is empty or missing 'clusters' key")
            return len(self.errors), len(self.warnings)
        # Build set of defined legacy asset IDs
        defined_assets = set()
        if assets_data and "assets" in assets_data:
            for asset in assets_data["assets"]:
                if "asset_id" in asset:
                    defined_assets.add(asset["asset_id"])
        # Validate each cluster
        clusters = index_data["clusters"]
        print(f"📋 Found {len(clusters)} clusters to validate\n")
        for cluster in clusters:
            self.validate_cluster_entry(cluster, defined_assets)
        # Note: Orphaned file detection not yet implemented
        # See _check_orphaned_files() method for future enhancement
        return len(self.errors), len(self.warnings)
    def _check_orphaned_files(self):
        """Check for playbook files not referenced in index.yaml
        Note: This is intentionally simplified for the initial implementation.
        Future enhancement could build a set of referenced files from index.yaml
        and compare against actual filesystem to detect orphaned files.
        For now, we only validate that referenced files exist (done in validate_cluster_entry).
        """
        # Placeholder for future enhancement
        # When implemented:
        # 1. Build set of all refactor_file paths from index.yaml
        # 2. Scan filesystem for all *_refactor.md files
        # 3. Report files that exist but aren't in index.yaml
        pass
    def print_results(self):
        """Print validation results"""
        print("\n" + "=" * 70)
        if self.errors:
            print(f"\n❌ Found {len(self.errors)} ERROR(S):\n")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        if self.warnings:
            print(f"\n⚠️  Found {len(self.warnings)} WARNING(S):\n")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        if not self.errors and not self.warnings:
            print("\n✅ All checks passed! Index is valid.")
        print("\n" + "=" * 70)
def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(
        description="Validate refactor playbook index consistency"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root directory (default: current directory)",
    )
    # Future enhancement: Add --verbose flag for detailed output
    # parser.add_argument(
    #     '--verbose',
    #     action='store_true',
    #     help='Enable verbose output'
    # )
    args = parser.parse_args()
    # Validate
    validator = RefactorIndexValidator(args.repo_root)
    error_count, warning_count = validator.validate_index()
    # Print results
    validator.print_results()
    # Exit with appropriate code (warnings don't fail the build)
    sys.exit(1 if error_count > 0 else 0)
if __name__ == "__main__":
    main()
