# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: namespace-converter
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
MachineNativeOps Namespace Converter
Converts legacy namespace references to MachineNativeOps standard namespace.
Supports comprehensive conversion rules aligned with mno-namespace.yaml.
Usage:
    python namespace-converter.py [--dry-run] [--verbose] <path>
    python namespace-converter.py --validate <path>
Examples:
    python namespace-converter.py --dry-run .
    python namespace-converter.py --verbose src/
    python namespace-converter.py --validate config/
Version: 2.0.0
Author: MachineNativeOps Platform Team
"""
# MNGA-002: Import organization needs review
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import yaml
@dataclass
class ConversionResult:
    """Represents the result of a conversion operation."""
    file_path: str
    conversions: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    patterns_matched: Dict[str, int] = field(default_factory=dict)
class NamespaceConverter:
    """
    Advanced namespace converter aligned with MachineNativeOps standards.
    """
    def __init__(self, dry_run=False, verbose=False, validate_only=False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.validate_only = validate_only
        self.results: List[ConversionResult] = []
        # Comprehensive conversion rules aligned with mno-namespace.yaml
        self.conversion_rules = {
            # API Version conversions
            r"apiVersion:\s*axiom\.io/v(\d+)": r"apiVersion: machinenativeops.io/v\1",
            r"axiom\.io/v(\d+)": r"machinenativeops.io/v\1",
            # Namespace conversions
            r"\baxiom\b(?!\.io)": r"machinenativeops",
            r"namespace:\s*axiom": r"namespace: machinenativeops",
            # Kind conversions
            r"\bMachineNativeOpsGlobalBaseline\b": r"MachineNativeOpsGlobalBaseline",
            r"\bAxiom([A-Z]\w+)": r"MachineNativeOps\1",
            # URN conversions
            r"urn:machinenativeops:": r"urn:machinenativeops:",
            # Domain conversions
            r"axiom\.io/": r"machinenativeops.io/",
            # Resource name conversions (with word boundaries)
            r"\bmachinenativeops-": r"machinenativeops-",
            r'(["\']|^)machinenativeops-': r"\1machinenativeops-",
            # Registry conversions
            r"registry\.axiom\.io": r"registry.machinenativeops.io",
            # Path conversions
            r"/etc/gl-runtime": r"/etc/machinenativeops",
            r"/opt/gl-runtime": r"/opt/machinenativeops",
            r"/var/lib/gl-runtime": r"/var/lib/machinenativeops",
            r"/var/log/gl-runtime": r"/var/log/machinenativeops",
            # Cluster name conversions
            r"\bmachinenativeops-etcd-cluster\b": r"super-agent-etcd-cluster",
            # Label key conversions
            r"axiom\.io/(\w+)": r"machinenativeops.io/\1",
        }
        # Validation patterns (must NOT exist after conversion)
        self.forbidden_patterns = [
            (r"\baxiom\.io/", "Legacy machinenativeops.io domain found"),
            (r"\bAxiom[A-Z]", "Legacy Axiom class name found"),
            (r"urn:machinenativeops:", "Legacy axiom URN found"),
            (r"/etc/machinenativeops/", "Legacy axiom path found"),
        ]
        # File extensions to process
        self.processable_extensions = {
            ".yaml",
            ".yml",
            ".json",
            ".py",
            ".js",
            ".ts",
            ".md",
            ".sh",
            ".txt",
            ".conf",
            ".toml",
        }
        # Excluded directories
        self.excluded_dirs = {
            ".git",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            "dist",
            "build",
            "target",
            "archive",
        }
    def load_namespace_config(self, config_path: str = "mno-namespace.yaml") -> Dict:
        """Load namespace configuration from mno-namespace.yaml."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            if self.verbose:
                print(f"Warning: {config_path} not found, using default rules")
            return {}
        except Exception as e:
            print(f"Error loading namespace config: {e}")
            return {}
    def convert_file(self, file_path: Path) -> ConversionResult:
        """Convert namespace references in a single file."""
        result = ConversionResult(file_path=str(file_path))
        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Apply conversion rules
            for pattern, replacement in self.conversion_rules.items():
                matches = list(re.finditer(pattern, content, re.MULTILINE))
                if matches:
                    match_count = len(matches)
                    result.conversions += match_count
                    result.patterns_matched[pattern] = match_count
                    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
                    if self.verbose:
                        print(f"  Pattern '{pattern}': {match_count} matches")
            # Validate converted content
            if self.validate_only or (result.conversions > 0):
                self._validate_content(content, result)
            # Write converted content
            if result.conversions > 0 and not self.dry_run and not self.validate_only:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                if self.verbose:
                    print(f"✓ Updated {file_path}: {result.conversions} conversions")
            elif result.conversions > 0 and self.dry_run:
                if self.verbose:
                    print(
                        f"⊡ Would update {file_path}: {result.conversions} conversions"
                    )
        except Exception as e:
            error_msg = f"Error processing {file_path}: {e}"
            result.errors.append(error_msg)
            print(error_msg)
        return result
    def _validate_content(self, content: str, result: ConversionResult):
        """Validate that no forbidden patterns exist in content."""
        for pattern, message in self.forbidden_patterns:
            matches = list(re.finditer(pattern, content, re.MULTILINE))
            if matches:
                warning = f"{message}: {len(matches)} occurrences"
                result.warnings.append(warning)
                if self.verbose:
                    print(f"  ⚠ {warning}")
    def should_process_file(self, file_path: Path) -> bool:
        """Determine if a file should be processed."""
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
    def convert_directory(self, directory_path: Path) -> List[ConversionResult]:
        """Recursively convert all files in a directory."""
        results = []
        for file_path in directory_path.rglob("*"):
            if file_path.is_file() and self.should_process_file(file_path):
                result = self.convert_file(file_path)
                if result.conversions > 0 or result.warnings or result.errors:
                    results.append(result)
        return results
    def convert_path(self, path: Path) -> List[ConversionResult]:
        """Convert namespace in a file or directory."""
        if path.is_file():
            return [self.convert_file(path)]
        elif path.is_dir():
            return self.convert_directory(path)
        else:
            print(f"Error: {path} is not a valid file or directory")
            return []
    def generate_report(self) -> str:
        """Generate a detailed conversion report."""
        total_files = len(self.results)
        total_conversions = sum(r.conversions for r in self.results)
        total_errors = sum(len(r.errors) for r in self.results)
        total_warnings = sum(len(r.warnings) for r in self.results)
        report = []
        report.append("=" * 80)
        report.append("MachineNativeOps Namespace Conversion Report")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(
            f"Mode: {'Validation Only' if self.validate_only else 'Dry Run' if self.dry_run else 'Live Conversion'}"
        )
        report.append("")
        report.append("Summary")
        report.append("-" * 80)
        report.append(f"Files processed:     {total_files}")
        report.append(f"Total conversions:   {total_conversions}")
        report.append(f"Errors:              {total_errors}")
        report.append(f"Warnings:            {total_warnings}")
        report.append("")
        if self.results:
            report.append("Detailed Results")
            report.append("-" * 80)
            for result in self.results:
                if result.conversions > 0 or result.errors or result.warnings:
                    report.append(f"\nFile: {result.file_path}")
                    report.append(f"  Conversions: {result.conversions}")
                    if result.patterns_matched:
                        report.append("  Patterns matched:")
                        for pattern, count in result.patterns_matched.items():
                            report.append(f"    - {pattern[:50]}...: {count}")
                    if result.warnings:
                        report.append("  Warnings:")
                        for warning in result.warnings:
                            report.append(f"    ⚠ {warning}")
                    if result.errors:
                        report.append("  Errors:")
                        for error in result.errors:
                            report.append(f"    ✗ {error}")
        report.append("")
        report.append("=" * 80)
        if total_conversions == 0 and total_warnings == 0:
            report.append(
                "✓ All files are compliant with MachineNativeOps namespace standards"
            )
        elif total_errors == 0:
            report.append("✓ Conversion completed successfully")
        else:
            report.append("✗ Conversion completed with errors")
        report.append("=" * 80)
        return "\n".join(report)
    def save_report(
        self, report: str, output_path: str = "namespace-conversion-report.txt"
    ):
        """Save report to file."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\nReport saved to: {output_path}")
        except Exception as e:
            print(f"Error saving report: {e}")
def main():
    """Main entry point for the namespace converter."""
    import argparse
    parser = argparse.ArgumentParser(
        description="MachineNativeOps Namespace Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run on current directory
  python namespace-converter.py --dry-run .
  # Convert all files in src/ directory
  python namespace-converter.py src/
  # Validate files without conversion
  python namespace-converter.py --validate config/
  # Verbose output with report
  python namespace-converter.py --verbose --report .
        """,
    )
    parser.add_argument("path", type=str, help="File or directory path to process")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate only, do not convert"
    )
    parser.add_argument(
        "--report", action="store_true", help="Generate detailed report file"
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="namespace-conversion-report.txt",
        help="Path for report file (default: namespace-conversion-report.txt)",
    )
    args = parser.parse_args()
    # Validate path exists
    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path {path} does not exist")
        sys.exit(1)
    # Create converter
    converter = NamespaceConverter(
        dry_run=args.dry_run or args.validate,
        verbose=args.verbose,
        validate_only=args.validate,
    )
    # Load namespace config if available
    converter.load_namespace_config()
    # Perform conversion
    print(f"{'Validating' if args.validate else 'Converting'} namespace in: {path}")
    if args.dry_run:
        print("(DRY RUN - no files will be modified)")
    print()
    converter.results = converter.convert_path(path)
    # Generate and display report
    report = converter.generate_report()
    print("\n" + report)
    # Save report if requested
    if args.report:
        converter.save_report(report, args.report_path)
    # Exit with appropriate code
    total_errors = sum(len(r.errors) for r in converter.results)
    sys.exit(1 if total_errors > 0 else 0)
