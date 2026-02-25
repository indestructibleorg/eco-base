# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: namespace-validator
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
MachineNativeOps Namespace Validator
Validates that all resources comply with MachineNativeOps namespace standards.
Usage:
    python namespace-validator.py [--verbose] [--strict] <path>
Version: 2.0.0
Author: MachineNativeOps Platform Team
"""
import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List
import yaml
# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
class ValidationResult:
    """Result of a validation check"""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.passed = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
    def add_error(self, message: str):
        """Add an error (causes validation to fail)"""
        self.errors.append(message)
        self.passed = False
    def add_warning(self, message: str):
        """Add a warning (doesn't cause validation to fail)"""
        self.warnings.append(message)
    def add_info(self, message: str):
        """Add informational message"""
        self.info.append(message)
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "file": self.file_path,
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
        }
class NamespaceValidator:
    """Validates namespace compliance"""
    def __init__(self, config_path: str = "mno-namespace.yaml"):
        """Initialize validator with namespace configuration"""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.results: List[ValidationResult] = []
        self.stats = {
            "files_checked": 0,
            "files_passed": 0,
            "files_failed": 0,
            "errors": 0,
            "warnings": 0,
        }
    def _load_config(self) -> dict:
        """Load namespace configuration from YAML"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded namespace config from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    def validate_file(self, file_path: Path) -> ValidationResult:
        """
        Validate a single file
        Args:
            file_path: Path to file to validate
        Returns:
            ValidationResult object
        """
        result = ValidationResult(str(file_path))
        self.stats["files_checked"] += 1
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")
            # Check each validation rule
            for rule_id, rule in self.validation_rules.items():
                # Check forbidden patterns
                if "forbidden_pattern" in rule:
                    self._check_forbidden_pattern(
                        content, lines, file_path, rule_id, rule, result
                    )
                # Check required patterns (only for YAML/JSON files)
                if "pattern" in rule and file_path.suffix in {".yaml", ".yml", ".json"}:
                    self._check_required_pattern(
                        content, file_path, rule_id, rule, result
                    )
                # Check with custom function
                if "check_function" in rule:
                    check_func = getattr(self, rule["check_function"], None)
                    if check_func:
                        check_func(content, lines, file_path, rule_id, rule, result)
        except Exception as e:
            result.add_issue(
                ValidationIssue(
                    file_path=str(file_path),
                    line_number=None,
                    rule_id="SYSTEM",
                    severity=Severity.ERROR,
                    message=f"Error validating file: {e}",
                )
            )
        return result
    def _check_forbidden_pattern(
        self,
        content: str,
        lines: List[str],
        file_path: Path,
        rule_id: str,
        rule: Dict,
        result: ValidationResult,
    ):
        """Check for forbidden patterns."""
        pattern = rule["forbidden_pattern"]
        matches = list(re.finditer(pattern, content, re.MULTILINE))
        if matches:
            for match in matches:
                # Find line number
                line_number = content[: match.start()].count("\n") + 1
                result.add_issue(
                    ValidationIssue(
                        file_path=str(file_path),
                        line_number=line_number,
                        rule_id=rule_id,
                        severity=rule["severity"],
                        message=f"{rule['description']}: Found '{match.group()}'",
                        suggestion=rule.get("suggestion"),
                    )
                )
    def _check_required_pattern(
        self,
        content: str,
        file_path: Path,
        rule_id: str,
        rule: Dict,
        result: ValidationResult,
    ):
        """Check for required patterns in YAML/JSON files."""
        pattern = rule["pattern"]
        # Only check if the file seems to define the relevant resource
        # (e.g., only check namespace if 'namespace:' exists)
        if rule_id == "NS-001" and "namespace:" in content:
            if not re.search(pattern, content, re.MULTILINE):
                result.add_issue(
                    ValidationIssue(
                        file_path=str(file_path),
                        line_number=None,
                        rule_id=rule_id,
                        severity=rule["severity"],
                        message=rule["description"],
                        suggestion=rule.get("suggestion"),
                    )
                )
    def _validate_kebab_case(
        self,
        content: str,
        lines: List[str],
        file_path: Path,
        rule_id: str,
        rule: Dict,
        result: ValidationResult,
    ):
        """Validate that resource names use kebab-case."""
        # Check for 'name:' fields in YAML
        if file_path.suffix in {".yaml", ".yml"}:
            name_pattern = r'^\s*name:\s*(["\']?)([^"\'\s]+)\1\s*$'
            for i, line in enumerate(lines):
                match = re.match(name_pattern, line)
                if match:
                    name = match.group(2)
                    # Skip if it's a variable or environment reference
                    if "{{" in name or "${" in name:
                        continue
                    # Check kebab-case
                    if not re.match(r"^[a-z][a-z0-9-]*[a-z0-9]$", name):
                        result.add_issue(
                            ValidationIssue(
                                file_path=str(file_path),
                                line_number=i + 1,
                                rule_id=rule_id,
                                severity=rule["severity"],
                                message=f"Resource name '{name}' does not use kebab-case",
                                suggestion="Use lowercase letters, numbers, and hyphens only",
                            ))
    def _validate_yaml_keys(
        self,
        content: str,
        lines: List[str],
        file_path: Path,
        rule_id: str,
        rule: Dict,
        result: ValidationResult,
    ):
        """Validate that YAML keys use snake_case."""
        if file_path.suffix not in {".yaml", ".yml"}:
            return
        # Pattern to match YAML keys
        key_pattern = r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:"
        for i, line in enumerate(lines):
            match = re.match(key_pattern, line)
            if match:
                key = match.group(1)
                # Skip special cases (environment variables, metadata fields)
                if key.isupper() or key in {"apiVersion", "kind", "metadata"}:
                    continue
                # Check snake_case
                if not re.match(r"^[a-z][a-z0-9_]*$", key):
                    result.add_issue(
                        ValidationIssue(
                            file_path=str(file_path),
                            line_number=i + 1,
                            rule_id=rule_id,
                            severity=rule["severity"],
                            message=f"YAML key '{key}' does not use snake_case",
                            suggestion="Use lowercase letters, numbers, and underscores only",
                        ))
    def should_process_file(self, file_path: Path) -> bool:
        """Determine if a file should be validated."""
        # Check file extension
        if file_path.suffix not in self.processable_extensions:
            return False
        # Check if in excluded directory
        for parent in file_path.parents:
            if parent.name in self.excluded_dirs:
                return False
        # Check if file is readable
        if not os.access(file_path, os.R_OK):
            return False
        return True
    def validate_directory(self, directory_path: Path) -> List[ValidationResult]:
        """Recursively validate all files in a directory."""
        results = []
        for file_path in directory_path.rglob("*"):
            if file_path.is_file() and self.should_process_file(file_path):
                result = self.validate_file(file_path)
                if not result.passed or result.issues:
                    results.append(result)
        return results
    def validate_path(self, path: Path) -> List[ValidationResult]:
        """Validate a file or directory."""
        if path.is_file():
            return [self.validate_file(path)]
        elif path.is_dir():
            return self.validate_directory(path)
        else:
            print(f"Error: {path} is not a valid file or directory")
            return []
    def generate_report(self) -> str:
        """Generate a detailed validation report."""
        total_files = len(self.results)
        total_issues = sum(len(r.issues) for r in self.results)
        total_errors = sum(
            len([i for i in r.issues if i.severity == Severity.ERROR])
            for r in self.results
        )
        total_warnings = sum(
            len([i for i in r.issues if i.severity == Severity.WARNING])
            for r in self.results
        )
        files_passed = sum(1 for r in self.results if r.passed)
        report = []
        report.append("=" * 80)
        report.append("MachineNativeOps Namespace Validation Report")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Mode: {'Strict' if self.strict else 'Standard'}")
        report.append("")
        report.append("Summary")
        report.append("-" * 80)
        report.append(f"Files validated:     {total_files}")
        report.append(f"Files passed:        {files_passed}")
        report.append(f"Files failed:        {total_files - files_passed}")
        report.append(f"Total issues:        {total_issues}")
        report.append(f"  Errors:            {total_errors}")
        report.append(f"  Warnings:          {total_warnings}")
        report.append("")
        if self.results:
            # Group issues by rule
            issues_by_rule: Dict[str, List[ValidationIssue]] = {}
            for result in self.results:
                for issue in result.issues:
                    if issue.rule_id not in issues_by_rule:
                        issues_by_rule[issue.rule_id] = []
                    issues_by_rule[issue.rule_id].append(issue)
            report.append("Issues by Rule")
            report.append("-" * 80)
            for rule_id in sorted(issues_by_rule.keys()):
                issues = issues_by_rule[rule_id]
                report.append(f"\n{rule_id}: {len(issues)} occurrences")
                rule = self.validation_rules.get(rule_id, {})
                if rule:
                    report.append(f"  Description: {rule.get('description', 'N/A')}")
                # Show first few examples
                for issue in issues[:3]:
                    report.append(f"  - {issue.file_path}:{issue.line_number or '?'}")
                    report.append(f"    {issue.message}")
                    if issue.suggestion:
                        report.append(f"    💡 {issue.suggestion}")
                if len(issues) > 3:
                    report.append(f"  ... and {len(issues) - 3} more")
        report.append("")
        report.append("=" * 80)
        if total_errors == 0 and total_warnings == 0:
            report.append(
                "✓ All files comply with MachineNativeOps namespace standards"
            )
        elif total_errors == 0:
            report.append(f"⚠ Validation completed with {total_warnings} warnings")
        else:
            report.append(f"✗ Validation failed with {total_errors} errors")
        report.append("=" * 80)
        return "\n".join(report)
    def save_report(
        self, report: str, output_path: str = "namespace-validation-report.txt"
    ):
        """Save report to file."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\nReport saved to: {output_path}")
        except Exception as e:
            print(f"Error saving report: {e}")
def main():
    """Main entry point for the namespace validator."""
    parser = argparse.ArgumentParser(
        description="MachineNativeOps Namespace Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate current directory
  python namespace-validator.py .
  # Strict validation with verbose output
  python namespace-validator.py --strict --verbose src/
  # Generate detailed report
  python namespace-validator.py --report .
        """,
    )
    parser.add_argument("path", type=str, help="File or directory path to validate")
    parser.add_argument(
        "--strict", action="store_true", help="Enable strict validation mode"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--report", action="store_true", help="Generate detailed report file"
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="namespace-validation-report.txt",
        help="Path for report file (default: namespace-validation-report.txt)",
    )
    args = parser.parse_args()
    # Validate path exists
    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path {path} does not exist")
        sys.exit(1)
    # Create validator
    validator = NamespaceValidator(strict=args.strict, verbose=args.verbose)
    # Load namespace config if available
    validator.load_namespace_config()
    # Perform validation
    print(f"Validating namespace compliance in: {path}")
    if args.strict:
        print("(STRICT MODE - warnings treated as errors)")
    print()
    validator.results = validator.validate_path(path)
    # Generate and display report
    report = validator.generate_report()
    print("\n" + report)
    # Save report if requested
    if args.report:
        validator.save_report(report, args.report_path)
    # Exit with appropriate code
    total_errors = sum(
        len([i for i in r.issues if i.severity == Severity.ERROR])
        for r in validator.results
    )
    sys.exit(1 if total_errors > 0 else 0)
