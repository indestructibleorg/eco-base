#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: axiom-namespace-migrator
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
AXIOM 到 MachineNativeOps 命名空間遷移工具 (Namespace Migration Tool)
智能命名空間轉換工具，用於將 AXIOM 命名空間引用遷移到 MachineNativeOps 標準。
支援多種檔案格式，提供批量處理與備份機制。
Usage:
    python axiom-namespace-migrator.py [--dry-run] [--verbose] [--backup] <path>
    python axiom-namespace-migrator.py --validate <path>
    python axiom-namespace-migrator.py --report <path>
Examples:
    python axiom-namespace-migrator.py --dry-run .
    python axiom-namespace-migrator.py --verbose --backup src/
    python axiom-namespace-migrator.py --validate config/
    python axiom-namespace-migrator.py --report --output report.json .
Version: 1.0.0
Author: MachineNativeOps Platform Team
Date: 2025-12-20
"""
# MNGA-002: Import organization needs review
import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
# Optional YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print(
        "Warning: PyYAML not installed. YAML validation will be limited.",
        file=sys.stderr,
    )
@dataclass
class ConversionMatch:
    """Represents a single conversion match."""
    pattern: str
    original: str
    replacement: str
    line_number: int
    column: int
    context: str
@dataclass
class ConversionResult:
    """Represents the result of a conversion operation."""
    file_path: str
    file_type: str
    total_matches: int = 0
    conversions: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    matches: List[ConversionMatch] = field(default_factory=list)
    patterns_matched: Dict[str, int] = field(default_factory=dict)
    backup_path: Optional[str] = None
@dataclass
class MigrationSummary:
    """Summary of the entire migration operation."""
    total_files_scanned: int = 0
    total_files_modified: int = 0
    total_conversions: int = 0
    total_errors: int = 0
    total_warnings: int = 0
    files_by_type: Dict[str, int] = field(default_factory=dict)
    patterns_summary: Dict[str, int] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
class AxiomNamespaceMigrator:
    """
    Advanced namespace migrator for converting AXIOM references to MachineNativeOps.
    Supports:
    - YAML, JSON, Python, Markdown, Shell, TypeScript/JavaScript files
    - Batch processing with progress tracking
    - Backup mechanism before modification
    - Dry-run mode for safe verification
    - Detailed conversion reports and logs
    """
    VERSION = "1.0.0"
    def __init__(
        self,
        dry_run: bool = False,
        verbose: bool = False,
        backup: bool = True,
        validate_only: bool = False,
        backup_dir: Optional[str] = None,
    ):
        self.dry_run = dry_run
        self.verbose = verbose
        self.backup = backup
        self.validate_only = validate_only
        self.backup_dir = backup_dir or ".axiom-migration-backup"
        self.results: List[ConversionResult] = []
        self.summary = MigrationSummary()
        # Comprehensive conversion rules (ordered by specificity - longest
        # first)
        self.conversion_rules = self._build_conversion_rules()
        # Validation patterns (must NOT exist after conversion)
        self.forbidden_patterns = self._build_forbidden_patterns()
        # File extensions to process by category
        self.file_categories = {
            "yaml": {".yaml", ".yml"},
            "json": {".json"},
            "python": {".py"},
            "javascript": {".js", ".ts", ".jsx", ".tsx"},
            "markdown": {".md", ".mdx"},
            "shell": {".sh", ".bash"},
            "config": {".conf", ".toml", ".ini"},
            "text": {".txt"},
        }
        # All processable extensions
        self.processable_extensions = set()
        for exts in self.file_categories.values():
            self.processable_extensions.update(exts)
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
            ".axiom-migration-backup",
            ".next",
            ".nuxt",
            "coverage",
            ".pytest_cache",
        }
    def _build_conversion_rules(self) -> List[Tuple[str, str, str]]:
        """
        Build comprehensive conversion rules.
        Returns list of (pattern, replacement, category) tuples.
        Rules are ordered by specificity (longest patterns first).
        """
        rules = [
            # API Version conversions (most specific first)
            (
                r"apiVersion:\s*axiom\.io/v(\d+)",
                r"apiVersion: machinenativeops.io/v\1",
                "api_version",
            ),
            (r"axiom\.io/v(\d+)", r"machinenativeops.io/v\1", "api_version"),
            # Kind/Type conversions (full type names first)
            (r"\bGLRuntimeGlobalBaseline\b", r"MachineNativeOpsGlobalBaseline", "kind"),
            (r"\bGLRuntimeNamespaceConfig\b", r"MachineNativeOpsNamespaceConfig", "kind"),
            (r"\bAxiomAutoMonitor\b", r"MachineNativeOpsAutoMonitor", "kind"),
            (r"\bAxiomConfigValidator\b", r"MachineNativeOpsConfigValidator", "kind"),
            (
                r"\bAxiomPerformanceBenchmarker\b",
                r"MachineNativeOpsPerformanceBenchmarker",
                "kind",
            ),
            (r"\bAxiomQuantumOptimizer\b", r"MachineNativeOpsQuantumOptimizer", "kind"),
            (r"\bAxiomSecurityScanner\b", r"MachineNativeOpsSecurityScanner", "kind"),
            (r"\bAxiom([A-Z][a-zA-Z0-9]*)\b", r"MachineNativeOps\1", "kind"),
            # URN conversions
            (r"urn:axiom:", r"urn:machinenativeops:", "urn"),
            # Domain/Label conversions
            (r"axiom\.io/", r"machinenativeops.io/", "domain"),
            # Registry conversions
            (r"registry\.axiom\.io", r"registry.machinenativeops.io", "registry"),
            (r"ghcr\.io/axiom", r"ghcr.io/machinenativeops", "registry"),
            # Path conversions (filesystem)
            (r"/etc/gl-runtime", r"/etc/machinenativeops", "path"),
            (r"/opt/gl-runtime", r"/opt/machinenativeops", "path"),
            (r"/var/lib/gl-runtime", r"/var/lib/machinenativeops", "path"),
            (r"/var/log/gl-runtime", r"/var/log/machinenativeops", "path"),
            (r"/var/cache/axiom", r"/var/cache/machinenativeops", "path"),
            (r"/tmp/axiom", r"/tmp/machinenativeops", "path"),
            # Cluster/Resource name conversions
            (r"\baxiom-etcd-cluster\b", r"super-agent-etcd-cluster", "cluster"),
            (r"\baxiom-cluster\b", r"machinenativeops-cluster", "cluster"),
            # Namespace conversions (YAML context)
            (r"namespace:\s*axiom\b", r"namespace: machinenativeops", "namespace"),
            # Resource prefix conversions
            (r"\baxiom-([a-z0-9-]+)", r"machinenativeops-\1", "resource_name"),
            # Python/Code specific conversions
            (r"axiom_system", r"machinenativeops_system", "python"),
            (r"axiom_monitor", r"machinenativeops_monitor", "python"),
            (r"axiom_validator", r"machinenativeops_validator", "python"),
            (r"axiom_config", r"machinenativeops_config", "python"),
            # Environment variable conversions
            (r"\bAXIOM_([A-Z_]+)", r"MNO_\1", "env_var"),
            # General namespace conversion (least specific, last)
            (r"\baxiom\b(?!\.io)(?![a-zA-Z0-9_-])", r"machinenativeops", "namespace"),
            # Case-sensitive AXIOM conversions
            (r"\bAXIOM\b", r"MachineNativeOps", "brand"),
            (r"\bAxiom\b(?![A-Z])", r"MachineNativeOps", "brand"),
        ]
        return rules
    def _build_forbidden_patterns(self) -> List[Tuple[str, str]]:
        """Build patterns that should NOT exist after successful conversion."""
        return [
            (r"\baxiom\.io/", "Legacy axiom.io domain found"),
            (r"\bAxiom[A-Z]", "Legacy Axiom class/type name found"),
            (r"urn:axiom:", "Legacy axiom URN found"),
            (r"/etc/gl-runtime/", "Legacy axiom filesystem path found"),
            (r"/opt/gl-runtime/", "Legacy axiom filesystem path found"),
            (r"apiVersion:\s*axiom\.io", "Legacy axiom API version found"),
            (r"namespace:\s*axiom\b", "Legacy axiom namespace found"),
        ]
    def _get_file_category(self, file_path: Path) -> str:
        """Determine the category of a file based on its extension."""
        suffix = file_path.suffix.lower()
        for category, extensions in self.file_categories.items():
            if suffix in extensions:
                return category
        return "unknown"
    def _create_backup(self, file_path: Path) -> Optional[str]:
        """Create a backup of the file before modification."""
        if not self.backup:
            return None
        backup_base = Path(self.backup_dir)
        backup_base.mkdir(parents=True, exist_ok=True)
        # Create timestamped backup path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        relative_path = (
            file_path.relative_to(Path.cwd()) if file_path.is_absolute() else file_path
        )
        backup_path = backup_base / f"{timestamp}" / relative_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, backup_path)
        return str(backup_path)
    def _find_matches(self, content: str, file_path: str) -> List[ConversionMatch]:
        """Find all matches in the content with line numbers and context."""
        matches = []
        lines = content.split("\n")
        for pattern, replacement, category in self.conversion_rules:
            for line_num, line in enumerate(lines, 1):
                for match in re.finditer(pattern, line):
                    context_start = max(0, line_num - 2)
                    context_end = min(len(lines), line_num + 2)
                    context_lines = lines[context_start:context_end]
                    context = "\n".join(
                        f"  {i+context_start+1}: {l}"
                        for i, l in enumerate(context_lines)
                    )
                    matches.append(
                        ConversionMatch(
                            pattern=pattern,
                            original=match.group(0),
                            replacement=re.sub(pattern, replacement, match.group(0)),
                            line_number=line_num,
                            column=match.start() + 1,
                            context=context,
                        )
                    )
        return matches
    def convert_file(self, file_path: Path) -> ConversionResult:
        """Convert namespace references in a single file."""
        result = ConversionResult(
            file_path=str(file_path), file_type=self._get_file_category(file_path)
        )
        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            original_content = content
            # Find all matches before conversion (for reporting)
            if self.verbose:
                result.matches = self._find_matches(content, str(file_path))
            # Apply conversion rules
            for pattern, replacement, category in self.conversion_rules:
                matches = list(re.finditer(pattern, content, re.MULTILINE))
                if matches:
                    match_count = len(matches)
                    result.total_matches += match_count
                    result.patterns_matched[category] = (
                        result.patterns_matched.get(category, 0) + match_count
                    )
                    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
                    if self.verbose:
                        print(
                            f"  [{category}] Pattern '{pattern[:40]}...': {match_count} matches"
                        )
            # Count actual conversions (characters changed)
            if content != original_content:
                result.conversions = sum(
                    1 for a, b in zip(original_content, content) if a != b
                ) + abs(len(content) - len(original_content))
            # Validate converted content
            self._validate_content(content, result)
            # Create backup and write converted content
            if result.total_matches > 0 and not self.dry_run and not self.validate_only:
                if self.backup:
                    result.backup_path = self._create_backup(file_path)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                if self.verbose:
                    print(f"✓ Updated {file_path}: {result.total_matches} conversions")
            elif result.total_matches > 0 and self.dry_run:
                if self.verbose:
                    print(
                        f"⊡ Would update {file_path}: {result.total_matches} conversions"
                    )
        except UnicodeDecodeError:
            result.errors.append("Unable to read file: not valid UTF-8")
        except PermissionError:
            result.errors.append("Permission denied: cannot read/write file")
        except Exception as e:
            error_msg = f"Error processing file: {str(e)}"
            result.errors.append(error_msg)
            if self.verbose:
                print(f"✗ {error_msg}")
        return result
    def _validate_content(self, content: str, result: ConversionResult):
        """Validate that no forbidden patterns exist in content."""
        for pattern, message in self.forbidden_patterns:
            matches = list(re.finditer(pattern, content, re.MULTILINE))
            if matches:
                # Find line numbers for warnings
                lines = content.split("\n")
                line_nums = []
                for match in matches[:5]:  # Limit to first 5 matches
                    pos = 0
                    for i, line in enumerate(lines, 1):
                        if pos <= match.start() < pos + len(line) + 1:
                            line_nums.append(str(i))
                            break
                        pos += len(line) + 1
                warning = f"{message}: {len(matches)} occurrences (lines: {', '.join(line_nums)})"
                result.warnings.append(warning)
                if self.verbose:
                    print(f"  ⚠ {warning}")
    def should_process_file(self, file_path: Path) -> bool:
        """Determine if a file should be processed."""
        # Check file extension
        if file_path.suffix.lower() not in self.processable_extensions:
            return False
        # Check if in excluded directory
        for parent in file_path.parents:
            if parent.name in self.excluded_dirs:
                return False
        # Check if file is readable
        if not os.access(file_path, os.R_OK):
            return False
        # Skip very large files (> 10MB)
        try:
            if file_path.stat().st_size > 10 * 1024 * 1024:
                return False
        except OSError:
            return False
        return True
    def convert_directory(self, directory_path: Path) -> List[ConversionResult]:
        """Recursively convert all files in a directory."""
        results = []
        files_to_process = []
        # Collect all files first
        for file_path in directory_path.rglob("*"):
            if file_path.is_file() and self.should_process_file(file_path):
                files_to_process.append(file_path)
        total_files = len(files_to_process)
        if self.verbose:
            print(f"\nProcessing {total_files} files...")
        for i, file_path in enumerate(files_to_process, 1):
            if self.verbose:
                print(f"\n[{i}/{total_files}] {file_path}")
            result = self.convert_file(file_path)
            if result.total_matches > 0 or result.warnings or result.errors:
                results.append(result)
            # Update summary
            self.summary.total_files_scanned += 1
            if result.total_matches > 0:
                self.summary.total_files_modified += 1
            self.summary.total_conversions += result.total_matches
            self.summary.total_errors += len(result.errors)
            self.summary.total_warnings += len(result.warnings)
            # Track by file type
            file_type = result.file_type
            self.summary.files_by_type[file_type] = (
                self.summary.files_by_type.get(file_type, 0) + 1
            )
            # Track pattern summary
            for category, count in result.patterns_matched.items():
                self.summary.patterns_summary[category] = (
                    self.summary.patterns_summary.get(category, 0) + count
                )
        return results
    def convert_path(self, path: Path) -> List[ConversionResult]:
        """Convert namespace in a file or directory."""
        self.summary.start_time = datetime.now()
        if path.is_file():
            self.summary.total_files_scanned = 1
            result = self.convert_file(path)
            if result.total_matches > 0:
                self.summary.total_files_modified = 1
            self.summary.total_conversions = result.total_matches
            self.summary.total_errors = len(result.errors)
            self.summary.total_warnings = len(result.warnings)
            self.results = (
                [result]
                if result.total_matches > 0 or result.warnings or result.errors
                else []
            )
        elif path.is_dir():
            self.results = self.convert_directory(path)
        else:
            print(f"Error: {path} is not a valid file or directory")
            return []
        self.summary.end_time = datetime.now()
        return self.results
    def generate_report(self) -> str:
        """Generate a detailed conversion report."""
        duration = (
            (self.summary.end_time - self.summary.start_time).total_seconds()
            if self.summary.end_time
            else 0
        )
        report = []
        report.append("=" * 80)
        report.append("  AXIOM → MachineNativeOps Namespace Migration Report")
        report.append("=" * 80)
        report.append(f"  Tool Version: {self.VERSION}")
        report.append(f"  Generated: {datetime.now().isoformat()}")
        report.append(f"  Duration: {duration:.2f} seconds")
        mode = (
            "Validation Only"
            if self.validate_only
            else "Dry Run" if self.dry_run else "Live Conversion"
        )
        report.append(f"  Mode: {mode}")
        report.append("")
        # Summary section
        report.append("─" * 80)
        report.append("  SUMMARY")
        report.append("─" * 80)
        report.append(f"  Files scanned:     {self.summary.total_files_scanned}")
        report.append(f"  Files modified:    {self.summary.total_files_modified}")
        report.append(f"  Total conversions: {self.summary.total_conversions}")
        report.append(f"  Errors:            {self.summary.total_errors}")
        report.append(f"  Warnings:          {self.summary.total_warnings}")
        report.append("")
        # Files by type
        if self.summary.files_by_type:
            report.append("─" * 80)
            report.append("  FILES BY TYPE")
            report.append("─" * 80)
            for file_type, count in sorted(self.summary.files_by_type.items()):
                report.append(f"  {file_type:15} {count:5} files")
            report.append("")
        # Patterns matched
        if self.summary.patterns_summary:
            report.append("─" * 80)
            report.append("  CONVERSIONS BY CATEGORY")
            report.append("─" * 80)
            for category, count in sorted(
                self.summary.patterns_summary.items(), key=lambda x: -x[1]
            ):
                report.append(f"  {category:20} {count:5} matches")
            report.append("")
        # Detailed results
        if self.results and self.verbose:
            report.append("─" * 80)
            report.append("  DETAILED RESULTS")
            report.append("─" * 80)
            for result in self.results:
                if result.total_matches > 0 or result.errors or result.warnings:
                    report.append(f"\n  File: {result.file_path}")
                    report.append(f"  Type: {result.file_type}")
                    report.append(f"  Matches: {result.total_matches}")
                    if result.patterns_matched:
                        report.append("  Categories:")
                        for cat, count in result.patterns_matched.items():
                            report.append(f"    - {cat}: {count}")
                    if result.backup_path:
                        report.append(f"  Backup: {result.backup_path}")
                    if result.warnings:
                        report.append("  Warnings:")
                        for warning in result.warnings:
                            report.append(f"    ⚠ {warning}")
                    if result.errors:
                        report.append("  Errors:")
                        for error in result.errors:
                            report.append(f"    ✗ {error}")
        # Final status
        report.append("")
        report.append("=" * 80)
        if self.summary.total_errors == 0 and self.summary.total_warnings == 0:
            if self.summary.total_conversions == 0:
                report.append(
                    "  ✓ All files are compliant with MachineNativeOps namespace standards"
                )
            else:
                report.append("  ✓ Conversion completed successfully")
        elif self.summary.total_errors == 0:
            report.append("  ⚠ Conversion completed with warnings")
        else:
            report.append("  ✗ Conversion completed with errors")
        report.append("=" * 80)
        return "\n".join(report)
    def generate_json_report(self) -> dict:
        """Generate a JSON-serializable report."""
        duration = (
            (self.summary.end_time - self.summary.start_time).total_seconds()
            if self.summary.end_time
            else 0
        )
        return {
            "version": self.VERSION,
            "generated_at": datetime.now().isoformat(),
            "duration_seconds": duration,
            "mode": (
                "validation"
                if self.validate_only
                else "dry_run" if self.dry_run else "live"
            ),
            "summary": {
                "files_scanned": self.summary.total_files_scanned,
                "files_modified": self.summary.total_files_modified,
                "total_conversions": self.summary.total_conversions,
                "errors": self.summary.total_errors,
                "warnings": self.summary.total_warnings,
            },
            "files_by_type": self.summary.files_by_type,
            "patterns_summary": self.summary.patterns_summary,
            "results": [
                {
                    "file_path": r.file_path,
                    "file_type": r.file_type,
                    "total_matches": r.total_matches,
                    "patterns_matched": r.patterns_matched,
                    "backup_path": r.backup_path,
                    "warnings": r.warnings,
                    "errors": r.errors,
                }
                for r in self.results
            ],
            "success": self.summary.total_errors == 0,
        }
    def save_report(self, report: str, output_path: str):
        """Save report to file."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\nReport saved to: {output_path}")
        except Exception as e:
            print(f"Error saving report: {e}")
def main():
    """Main entry point for the namespace migrator."""
    parser = argparse.ArgumentParser(
        description="AXIOM to MachineNativeOps Namespace Migration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AXIOM → MachineNativeOps Namespace Migration Tool v1.0.0                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Examples:                                                                   ║
║    # Dry run on current directory (safe preview)                             ║
║    python axiom-namespace-migrator.py --dry-run .                            ║
║                                                                              ║
║    # Convert all files with backup                                           ║
║    python axiom-namespace-migrator.py --backup .                             ║
║                                                                              ║
║    # Validate files without conversion                                       ║
║    python axiom-namespace-migrator.py --validate config/                     ║
║                                                                              ║
║    # Verbose output with JSON report                                         ║
║    python axiom-namespace-migrator.py --verbose --report --json --output r.json .  ║
║                                                                              ║
║  Conversion Categories:                                                      ║
║    - api_version:   gl-runtime.io/v* → machinenativeops.io/v*                    ║
║    - kind:          Axiom* → MachineNativeOps*                              ║
║    - urn:           urn:axiom: → urn:machinenativeops:                      ║
║    - domain:        gl-runtime.io/* → machinenativeops.io/*                      ║
║    - registry:      registry.gl-runtime.io → registry.machinenativeops.io        ║
║    - path:          /etc/gl-runtime → /etc/machinenativeops                      ║
║    - namespace:     namespace: axiom → namespace: machinenativeops          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """,
    )
    parser.add_argument("path", type=str, help="File or directory path to process")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output with detailed match information",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate only - check for legacy patterns without converting",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create backup before modifying files (default: True)",
    )
    parser.add_argument(
        "--no-backup", action="store_true", help="Disable backup creation"
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=".axiom-migration-backup",
        help="Directory for backup files (default: .axiom-migration-backup)",
    )
    parser.add_argument(
        "--report", action="store_true", help="Generate detailed report file"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output report in JSON format"
    )
    parser.add_argument("--output", "-o", type=str, help="Output path for report file")
    args = parser.parse_args()
    # Validate path exists
    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path '{path}' does not exist")
        sys.exit(1)
    # Determine backup setting
    do_backup = args.backup and not args.no_backup
    # Create migrator
    migrator = AxiomNamespaceMigrator(
        dry_run=args.dry_run or args.validate,
        verbose=args.verbose,
        backup=do_backup and not args.dry_run and not args.validate,
        validate_only=args.validate,
        backup_dir=args.backup_dir,
    )
    # Print header
    print("=" * 80)
    print("  AXIOM → MachineNativeOps Namespace Migration Tool")
    print("=" * 80)
    print(f"  Target: {path.absolute()}")
    print(
        f"  Mode: {'Validation Only' if args.validate else 'Dry Run' if args.dry_run else 'Live Conversion'}"
    )
    if do_backup and not args.dry_run and not args.validate:
        print(f"  Backup: {args.backup_dir}")
    print("=" * 80)
    print()
    # Perform conversion
    migrator.convert_path(path)
    # Generate and display report
    if args.json:
        report = json.dumps(
            migrator.generate_json_report(), indent=2, ensure_ascii=False
        )
    else:
        report = migrator.generate_report()
    print(report)
    # Save report if requested
    if args.report or args.output:
        output_path = args.output
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extension = ".json" if args.json else ".txt"
            output_path = f"axiom-migration-report-{timestamp}{extension}"
        migrator.save_report(report, output_path)
    # Exit with appropriate code
    sys.exit(1 if migrator.summary.total_errors > 0 else 0)
if __name__ == "__main__":
    main()
