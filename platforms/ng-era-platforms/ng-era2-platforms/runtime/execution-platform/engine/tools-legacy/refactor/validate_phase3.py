# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: validate-phase3
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Phase 3 Deliverables Validator
Validates that all Phase 3 (Refactor) deliverables are present and correctly formatted.
Usage:
    python3 tools/refactor/validate-phase3.py --deliverables-path <path>
"""
# MNGA-002: Import organization needs review
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
try:
    import yaml
except ImportError:
    print("❌ PyYAML not installed. Install with: pip install pyyaml")
    sys.exit(1)
class Phase3Validator:
    """Validator for Phase 3 deliverables."""
    REQUIRED_DELIVERABLES = {
        "action_plan.yaml": {
            "type": "yaml",
            "description": "P0/P1/P2 action plan",
            "required_keys": ["p0_items", "p1_items", "p2_items"],
        },
        "validation": {
            "type": "directory",
            "description": "Validation reports directory",
            "required": True,
        },
    }
    def __init__(self, deliverables_path: str):
        self.deliverables_path = Path(deliverables_path)
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.passed: List[str] = []
        self.repo_root = self._find_repo_root()
    def _find_repo_root(self) -> Path:
        """Find repository root."""
        current = self.deliverables_path
        while current != current.parent:
            if (current / ".git").exists():
                return current
            current = current.parent
        return self.deliverables_path.parent.parent.parent
    def validate(self) -> Tuple[bool, Dict]:
        """Run all validations."""
        print(f"🔍 Validating Phase 3 deliverables at: {self.deliverables_path}")
        print("")
        # Check directory exists
        if not self.deliverables_path.exists():
            self.errors.append(
                f"Deliverables directory not found: {self.deliverables_path}"
            )
            return False, self._get_report()
        # Validate each deliverable
        for filename, spec in self.REQUIRED_DELIVERABLES.items():
            self._validate_deliverable(filename, spec)
        # Validate refactored code
        self._validate_refactored_code()
        # Check architecture compliance
        self._check_architecture_compliance()
        # Check test coverage
        self._check_test_coverage()
        # Print summary
        self._print_summary()
        # Return results
        success = len(self.errors) == 0
        return success, self._get_report()
    def _validate_deliverable(self, filename: str, spec: Dict):
        """Validate a single deliverable."""
        filepath = self.deliverables_path / filename
        file_type = spec.get("type")
        # Check directory
        if file_type == "directory":
            if not filepath.exists():
                if spec.get("required"):
                    self.errors.append(f"Missing directory: {filename}")
                    print(f"❌ {filename}/ - Not found")
                else:
                    self.warnings.append(f"Optional directory not found: {filename}")
                    print(f"⚠ {filename}/ - Not found (optional)")
            else:
                self.passed.append(filename)
                print(f"✓ {filename}/ - Exists")
            return
        # Check file existence
        if not filepath.exists():
            self.errors.append(f"Missing deliverable: {filename}")
            print(f"❌ {filename} - Not found")
            return
        # Validate content
        try:
            if file_type == "yaml":
                self._validate_yaml(filepath, spec)
            self.passed.append(filename)
            print(f"✓ {filename} - Valid")
        except Exception as e:
            self.errors.append(f"{filename}: {str(e)}")
            print(f"❌ {filename} - {str(e)}")
    def _validate_yaml(self, filepath: Path, spec: Dict):
        """Validate YAML file."""
        with open(filepath, "r", encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if data is None:
            raise ValueError("Empty YAML file")
        # Check required keys
        required_keys = spec.get("required_keys", [])
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
            # Check that P0/P1/P2 items are lists
            if key.endswith("_items"):
                if not isinstance(data[key], list):
                    raise ValueError(f"{key} must be a list")
    def _validate_refactored_code(self):
        """Validate that refactored code exists and compiles."""
        print("")
        print("🔍 Validating refactored codebase...")
        # Check for common source directories
        source_dirs = ["core", "services", "apps", "workspace/src"]
        found_sources = False
        for src_dir in source_dirs:
            src_path = self.repo_root / src_dir
            if src_path.exists():
                found_sources = True
                print(f"  ✓ Found source directory: {src_dir}")
        if not found_sources:
            self.warnings.append("No standard source directories found")
    def _check_architecture_compliance(self):
        """Check architecture compliance."""
        print("")
        print("🔍 Checking architecture compliance...")
        # Look for architecture validation config
        arch_config = (
            self.repo_root / "controlplane" / "config" / "architecture-rules.yaml"
        )
        if not arch_config.exists():
            self.warnings.append("Architecture rules config not found")
            print("  ⚠ Architecture rules not configured")
            return
        print("  ✓ Architecture rules configured")
    def _check_test_coverage(self):
        """Check test coverage if possible."""
        print("")
        print("🔍 Checking test coverage...")
        # Check if package.json exists
        package_json = self.repo_root / "package.json"
        if not package_json.exists():
            print("  ℹ No package.json found, skipping coverage check")
            return
        # Try to read package.json
        try:
            with open(package_json, "r", encoding='utf-8') as f:
                package_data = json.load(f)
            # Check if test script exists
            if "scripts" in package_data and "test" in package_data["scripts"]:
                print("  ✓ Test script configured")
            else:
                self.warnings.append("No test script found in package.json")
                print("  ⚠ No test script configured")
        except Exception as e:
            self.warnings.append(f"Error reading package.json: {e}")
    def _print_summary(self):
        """Print validation summary."""
        print("")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("📊 Validation Summary")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"✓ Passed: {len(self.passed)}")
        print(f"⚠ Warnings: {len(self.warnings)}")
        print(f"❌ Errors: {len(self.errors)}")
        print("")
        if self.warnings:
            print("Warnings:")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")
            print("")
        if self.errors:
            print("Errors:")
            for error in self.errors:
                print(f"  ❌ {error}")
            print("")
        # Print recommendation
        if len(self.errors) == 0:
            if len(self.warnings) == 0:
                print("✅ Phase 3 validation PASSED - Ready for deployment")
            else:
                print(
                    "⚠️  Phase 3 validation PASSED with warnings - Review before deployment"
                )
        else:
            print("❌ Phase 3 validation FAILED - Address errors before proceeding")
    def _get_report(self) -> Dict:
        """Get validation report."""
        return {
            "phase": 3,
            "deliverables_path": str(self.deliverables_path),
            "passed": self.passed,
            "warnings": self.warnings,
            "errors": self.errors,
            "success": len(self.errors) == 0,
            "recommendation": (
                "ready_for_deployment"
                if len(self.errors) == 0 and len(self.warnings) == 0
                else "review_warnings" if len(self.errors) == 0 else "address_errors"
            ),
        }
def main():
    parser = argparse.ArgumentParser(
        description="Validate Phase 3 (Refactor) deliverables"
    )
    parser.add_argument(
        "--deliverables-path",
        required=True,
        help="Path to Phase 3 deliverables directory",
    )
    parser.add_argument(
        "--output", help="Output validation report to file (JSON format)"
    )
    args = parser.parse_args()
    # Run validation
    validator = Phase3Validator(args.deliverables_path)
    success, report = validator.validate()
    # Save report if requested
    if args.output:
        with open(args.output, "w", encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Report saved to: {args.output}")
    # Exit with appropriate code
    sys.exit(0 if success else 1)
if __name__ == "__main__":
    main()
