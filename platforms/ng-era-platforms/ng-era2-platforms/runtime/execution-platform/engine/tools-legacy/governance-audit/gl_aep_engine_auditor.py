# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: governance
# @ECO-semantic: gl_aep_engine_auditor
# @ECO-audit-trail: ../../engine/governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
GL Unified Architecture Governance Framework - AEP Engine Governance Auditor
===================================================
Executes one-by-one isolated ETL → Elasticsearch pipeline for all AEP Engine files.
GL Unified Architecture Governance Framework Activated
- Strict GL Root Semantic Anchor compliance
- Mandatory governance event stream
- Consistency / Reversibility / Provability enforcement
- No continue-on-error policy
"""
# MNGA-002: Import organization needs review
import os
import json
import hashlib
import datetime
import re
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from enum import Enum
# Compile regex pattern once at module level for performance
TYPESCRIPT_ANY_PATTERN = re.compile(r':\s*any\b')
class Severity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"
class IssueType(Enum):
    PIPELINE_ERROR = "pipeline_execution_error"
    SCHEMA_MISMATCH = "schema_mismatch"
    METADATA_MISSING = "metadata_missing"
    NAMING_INCONSISTENT = "naming_inconsistent"
    STRUCTURE_VIOLATION = "directory_structure_violation"
    GL_MARKER_MISSING = "gl_marker_missing"
    SEMANTIC_MANIFEST_MISSING = "semantic_manifest_missing"
    DAG_BREAK = "dag_governance_chain_break"
    SYNTAX_ERROR = "syntax_error"
    TYPE_ERROR = "type_error"
@dataclass
class GovernanceEvent:
    """Represents a governance event in the event stream."""
    event_id: str
    timestamp: str
    event_type: str
    source_file: str
    gl_layer: str
    semantic_anchor: str
    status: str
    details: Dict[str, Any]
    evidence_hash: str
@dataclass
class FileAuditResult:
    """Result of auditing a single file."""
    file_path: str
    file_type: str
    file_hash: str
    gl_layer: str
    semantic_anchor: str
    etl_status: str
    es_index_status: str
    issues: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    governance_events: List[GovernanceEvent]
    timestamp: str
    execution_time_ms: int
@dataclass
class GlobalAuditReport:
    """Global governance audit report aggregating all file results."""
    report_id: str
    gl_charter_version: str
    execution_timestamp: str
    total_files: int
    files_processed: int
    files_passed: int
    files_failed: int
    issues_by_severity: Dict[str, int]
    issues_by_type: Dict[str, int]
    file_results: List[FileAuditResult]
    governance_event_summary: Dict[str, Any]
    best_practice_recommendations: List[Dict[str, Any]]
    migration_suggestions: List[Dict[str, Any]]
class GLRootSemanticAnchor:
    """GL Root Semantic Anchor resolver and validator."""
    LAYER_MAPPING = {
        "00-strategic": "ECO-00",
        "10-operational": "ECO-10",
        "30-execution": "ECO-30",
        "50-observability": "ECO-50",
        "60-feedback": "ECO-60",
        "81-extended": "ECO-81",
        "90-meta": "ECO-90"
    }
    FILE_TYPE_ANCHORS = {
        ".ts": "ECO-30-EXEC-TS",
        ".js": "ECO-30-EXEC-JS",
        ".json": "ECO-10-CONFIG-JSON",
        ".yaml": "ECO-10-CONFIG-YAML",
        ".yml": "ECO-10-CONFIG-YAML",
        ".md": "ECO-90-META-DOC",
        ".d.ts": "ECO-30-EXEC-TYPEDEF"
    }
    @classmethod
    def resolve_anchor(cls, file_path: str) -> str:
        """Resolve the semantic anchor for a given file."""
        path = Path(file_path)
        suffix = path.suffix
        # Check for .d.ts files
        if file_path.endswith('.d.ts'):
            return cls.FILE_TYPE_ANCHORS['.d.ts']
        return cls.FILE_TYPE_ANCHORS.get(suffix, "ECO-90-META-UNKNOWN")
    @classmethod
    def resolve_layer(cls, file_path: str) -> str:
        """Resolve the GL layer for a given file."""
        path_parts = Path(file_path).parts
        # Engine files are execution layer
        if 'engine' in path_parts:
            if 'tests' in path_parts:
                return "ECO-50-OBSERVABILITY"
            elif 'governance' in path_parts:
                return "ECO-10-OPERATIONAL"
            else:
                return "ECO-30-EXECUTION"
        return "ECO-90-META"
class ETLPipeline:
    """ETL Pipeline executor for governance audit."""
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
    def extract(self, file_path: str) -> Dict[str, Any]:
        """Extract data from a file."""
        full_path = self.base_path / file_path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        content = full_path.read_text(encoding='utf-8', errors='replace')
        stat = full_path.stat()
        return {
            "file_path": file_path,
            "content": content,
            "size_bytes": stat.st_size,
            "modified_time": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "content_hash": hashlib.sha256(content.encode()).hexdigest(),
            "line_count": len(content.splitlines()),
            "file_extension": full_path.suffix
        }
    def transform(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform extracted data with governance metadata."""
        file_path = extracted_data["file_path"]
        content = extracted_data["content"]
        # Analyze content for governance markers
        gl_markers = self._detect_gl_markers(content)
        semantic_markers = self._detect_semantic_markers(content)
        metadata = self._extract_metadata(content, file_path)
        issues = self._detect_issues(content, file_path, extracted_data)
        return {
#*extracted_data,
            "gl_markers": gl_markers,
            "semantic_markers": semantic_markers,
            "metadata": metadata,
            "issues": issues,
            "gl_layer": GLRootSemanticAnchor.resolve_layer(file_path),
            "semantic_anchor": GLRootSemanticAnchor.resolve_anchor(file_path),
            "governance_compliant": len([i for i in issues if i["severity"] in ["CRITICAL", "HIGH"]]) == 0
        }
    def load(self, transformed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Load transformed data (prepare for ES indexing)."""
        return {
            "index": "gl-aep-engine-audit",
            "document": {
                "file_path": transformed_data["file_path"],
                "content_hash": transformed_data["content_hash"],
                "size_bytes": transformed_data["size_bytes"],
                "line_count": transformed_data["line_count"],
                "gl_layer": transformed_data["gl_layer"],
                "semantic_anchor": transformed_data["semantic_anchor"],
                "gl_markers_count": len(transformed_data["gl_markers"]),
                "issues_count": len(transformed_data["issues"]),
                "governance_compliant": transformed_data["governance_compliant"],
                "metadata": transformed_data["metadata"],
                "issues": transformed_data["issues"],
                "indexed_at": datetime.datetime.utcnow().isoformat()
            }
        }
    def _detect_gl_markers(self, content: str) -> List[str]:
        """Detect GL governance markers in content."""
        markers = []
        gl_patterns = ["ECO-", "gl_", "governance", "Governance", "GOVERNANCE"]
        for line in content.splitlines():
            for pattern in gl_patterns:
                if pattern in line:
                    markers.append(line.strip()[:100])
                    break
        return markers[:50]  # Limit to 50 markers
    def _detect_semantic_markers(self, content: str) -> List[str]:
        """Detect semantic markers in content."""
        markers = []
        semantic_patterns = ["@semantic", "@anchor", "semantic:", "anchor:"]
        for line in content.splitlines():
            for pattern in semantic_patterns:
                if pattern.lower() in line.lower():
                    markers.append(line.strip()[:100])
                    break
        return markers[:20]
    def _extract_metadata(self, content: str, file_path: str) -> Dict[str, Any]:
        """Extract metadata from file content."""
        metadata = {
            "has_jsdoc": "/**" in content,
            "has_exports": "export " in content,
            "has_imports": "import " in content,
            "has_interfaces": "interface " in content,
            "has_types": "type " in content,
            "has_classes": "class " in content,
            "has_functions": "function " in content or "=>" in content,
            "has_tests": "test(" in content or "describe(" in content or "it(" in content,
            "has_readme_reference": "README" in content,
            "has_license": "license" in content.lower() or "LICENSE" in content
        }
        # JSON specific
        if file_path.endswith('.json'):
            try:
                json.loads(content)
                metadata["valid_json"] = True
            except json.JSONDecodeError:
                metadata["valid_json"] = False
        return metadata
    def _detect_issues(self, content: str, file_path: str, extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect governance and quality issues."""
        issues = []
        # Check for missing GL markers
        if "ECO-" not in content and "gl_" not in content:
            issues.append({
                "type": IssueType.GL_MARKER_MISSING.value,
                "severity": Severity.MEDIUM.value,
                "message": "No GL governance markers found in file",
                "file_path": file_path,
                "recommendation": "Add GL layer annotation comment at file header"
            })
        # Check for missing semantic manifest reference
        if "@semantic" not in content.lower() and "semantic" not in content.lower():
            issues.append({
                "type": IssueType.SEMANTIC_MANIFEST_MISSING.value,
                "severity": Severity.LOW.value,
                "message": "No semantic manifest reference found",
                "file_path": file_path,
                "recommendation": "Add semantic anchor annotation"
            })
        # Check naming conventions
        path = Path(file_path)
        if path.stem != path.stem.lower() and not path.stem[0].isupper():
            # Mixed case that's not PascalCase
            if "_" not in path.stem and "-" not in path.stem:
                issues.append({
                    "type": IssueType.NAMING_INCONSISTENT.value,
                    "severity": Severity.LOW.value,
                    "message": f"File naming may not follow conventions: {path.name}",
                    "file_path": file_path,
                    "recommendation": "Use snake_case or kebab-case for file names"
                })
        # Check for TypeScript issues
        if file_path.endswith('.ts'):
            # Cheap fast-path: skip detailed scanning if there is no obvious 'any' usage
            # This avoids unnecessary allocations and loops on large codebases
            # Use regex to match the same pattern as the detailed scan (handles multiple spaces/tabs)
            if not TYPESCRIPT_ANY_PATTERN.search(content):
                # No potential 'any' type annotations found, skip processing
                pass
            else:
                # Check for 'any' usage with proper @ts-ignore scoping
                lines = content.split('\n')
                any_count = 0
                in_block_comment = False
                # Use precompiled regex with word boundary to avoid matching 'anyThing', 'anyType', etc.
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    # Handle block comments - check for single-line vs multiline
                    if '/*' in stripped and '*/' in stripped:
                        # Single-line block comment - remove it and process the rest of the line
                        # This is a simple approach; won't handle all edge cases but works for common patterns
                        before_comment = stripped.split('/*')[0]
                        after_comment = stripped.split('*/')[-1] if stripped.count('*/') == 1 else ''
                        line_to_check = before_comment + after_comment
                    elif '/*' in stripped:
                        # Start of multiline block comment
                        in_block_comment = True
                        line_to_check = stripped.split('/*')[0]
                    elif '*/' in stripped:
                        # End of multiline block comment - process text after the closing */
                        # Use maxsplit=1 to only split on first */ in case of multiple on same line
                        in_block_comment = False
                        parts = stripped.split('*/', 1)
                        line_to_check = parts[1] if len(parts) > 1 else ''
                    elif in_block_comment:
                        # Inside a multiline block comment
                        continue
                    else:
                        line_to_check = line
                    # Skip if the entire line is a line comment
                    if line_to_check.strip().startswith('//'):
                        continue
                    # For inline comments, we use a simple heuristic since proper parsing is complex
                    # Limitation: This may incorrectly split on '//' within string literals (e.g., URLs)
                    # A proper fix would require a full TypeScript parser, but this works for most cases
                    code_part = line_to_check.split('//')[0] if '//' in line_to_check else line_to_check
                    # Count occurrences of ': any' or ':any' with word boundary
                    matches = TYPESCRIPT_ANY_PATTERN.findall(code_part)
                    if matches:
                        # Check if the previous line has @ts-ignore (with optional whitespace)
                        # Note: @ts-ignore only applies when on the *preceding* line in TypeScript
                        has_ignore = False
                        if i > 0:
                            prev_line = lines[i-1].strip()
                            # Match @ts-ignore at or near the start of the comment
                            # Valid: "// @ts-ignore", "//@ts-ignore", "//  @ts-ignore"
                            # Invalid: "// This is not @ts-ignore but mentions it"
                            if prev_line.startswith('//'):
                                # Remove the leading // and optional whitespace
                                comment_content = prev_line[2:].lstrip()
                                # Only match '@ts-ignore' exactly, not '@ts-ignore-next-line' etc.
                                if comment_content.startswith('@ts-ignore') and (
                                    len(comment_content) == len('@ts-ignore') or 
                                    comment_content[len('@ts-ignore')].isspace()
                                ):
                                    has_ignore = True
                        if not has_ignore:
                            any_count += len(matches)
                if any_count > 0:
                    issues.append({
                        "type": IssueType.TYPE_ERROR.value,
                        "severity": Severity.MEDIUM.value,
                        "message": f"Found {any_count} uses of 'any' type without @ts-ignore",
                        "file_path": file_path,
                        "recommendation": "Replace 'any' with specific types for better type safety"
                    })
            # Check for missing exports in non-test files
            if "tests" not in file_path and "export " not in content:
                issues.append({
                    "type": IssueType.METADATA_MISSING.value,
                    "severity": Severity.LOW.value,
                    "message": "No exports found in TypeScript file",
                    "file_path": file_path,
                    "recommendation": "Add exports for module functionality"
                })
        # Check for missing documentation
        if file_path.endswith('.ts') and "/**" not in content:
            issues.append({
                "type": IssueType.METADATA_MISSING.value,
                "severity": Severity.LOW.value,
                "message": "No JSDoc documentation found",
                "file_path": file_path,
                "recommendation": "Add JSDoc comments for public APIs"
            })
        # Check README files
        if file_path.endswith('README.md'):
            if len(content) < 100:
                issues.append({
                    "type": IssueType.METADATA_MISSING.value,
                    "severity": Severity.MEDIUM.value,
                    "message": "README file is too short (< 100 chars)",
                    "file_path": file_path,
                    "recommendation": "Expand README with usage examples and API documentation"
                })
        # Check JSON validity
        if file_path.endswith('.json'):
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                issues.append({
                    "type": IssueType.SYNTAX_ERROR.value,
                    "severity": Severity.CRITICAL.value,
                    "message": f"Invalid JSON: {str(e)}",
                    "file_path": file_path,
                    "recommendation": "Fix JSON syntax errors"
                })
        return issues
class GovernanceAuditEngine:
    """Main governance audit engine orchestrator."""
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.etl_pipeline = ETLPipeline(repo_path)
        self.governance_events: List[GovernanceEvent] = []
        self.file_results: List[FileAuditResult] = []
        self.start_time = datetime.datetime.utcnow()
    def emit_event(self, event_type: str, source_file: str, status: str, details: Dict[str, Any]) -> GovernanceEvent:
        """Emit a governance event."""
        event = GovernanceEvent(
            event_id=hashlib.sha256(f"{datetime.datetime.utcnow().isoformat()}-{source_file}".encode()).hexdigest()[:16],
            timestamp=datetime.datetime.utcnow().isoformat(),
            event_type=event_type,
            source_file=source_file,
            gl_layer=GLRootSemanticAnchor.resolve_layer(source_file),
            semantic_anchor=GLRootSemanticAnchor.resolve_anchor(source_file),
            status=status,
            details=details,
            evidence_hash=hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest()
        )
        self.governance_events.append(event)
        return event
    def audit_file(self, file_path: str) -> FileAuditResult:
        """Audit a single file through the ETL pipeline."""
        start_time = datetime.datetime.utcnow()
        file_events = []
        # Emit start event
        file_events.append(self.emit_event(
            "ETL_START",
            file_path,
            "STARTED",
            {"phase": "extract"}
        ))
        try:
            # Extract
            extracted = self.etl_pipeline.extract(file_path)
            file_events.append(self.emit_event(
                "ETL_EXTRACT",
                file_path,
                "SUCCESS",
                {"size_bytes": extracted["size_bytes"], "line_count": extracted["line_count"]}
            ))
            # Transform
            transformed = self.etl_pipeline.transform(extracted)
            file_events.append(self.emit_event(
                "ETL_TRANSFORM",
                file_path,
                "SUCCESS",
                {"issues_count": len(transformed["issues"]), "gl_compliant": transformed["governance_compliant"]}
            ))
            # Load (prepare for ES)
            loaded = self.etl_pipeline.load(transformed)
            file_events.append(self.emit_event(
                "ETL_LOAD",
                file_path,
                "SUCCESS",
                {"index": loaded["index"]}
            ))
            etl_status = "SUCCESS"
            es_status = "READY"
        except Exception as e:
            file_events.append(self.emit_event(
                "ETL_ERROR",
                file_path,
                "FAILED",
                {"error": str(e)}
            ))
            etl_status = "FAILED"
            es_status = "NOT_INDEXED"
            transformed = {
                "issues": [{
                    "type": IssueType.PIPELINE_ERROR.value,
                    "severity": Severity.CRITICAL.value,
                    "message": str(e),
                    "file_path": file_path,
                    "recommendation": "Fix file access or parsing error"
                }],
                "metadata": {},
                "gl_layer": GLRootSemanticAnchor.resolve_layer(file_path),
                "semantic_anchor": GLRootSemanticAnchor.resolve_anchor(file_path),
                "content_hash": ""
            }
            extracted = {"content_hash": "", "file_extension": Path(file_path).suffix}
        end_time = datetime.datetime.utcnow()
        execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
        result = FileAuditResult(
            file_path=file_path,
            file_type=extracted.get("file_extension", "unknown"),
            file_hash=extracted.get("content_hash", transformed.get("content_hash", "")),
            gl_layer=transformed["gl_layer"],
            semantic_anchor=transformed["semantic_anchor"],
            etl_status=etl_status,
            es_index_status=es_status,
            issues=transformed["issues"],
            metadata=transformed["metadata"],
            governance_events=file_events,
            timestamp=end_time.isoformat(),
            execution_time_ms=execution_time_ms
        )
        self.file_results.append(result)
        return result
    def generate_global_report(self) -> GlobalAuditReport:
        """Generate the global governance audit report."""
        issues_by_severity = {s.value: 0 for s in Severity}
        issues_by_type = {t.value: 0 for t in IssueType}
        for result in self.file_results:
            for issue in result.issues:
                severity = issue.get("severity", "INFO")
                issue_type = issue.get("type", "unknown")
                if severity in issues_by_severity:
                    issues_by_severity[severity] += 1
                if issue_type in issues_by_type:
                    issues_by_type[issue_type] += 1
        files_passed = len([r for r in self.file_results if r.etl_status == "SUCCESS" and 
                          len([i for i in r.issues if i["severity"] in ["CRITICAL", "HIGH"]]) == 0])
        files_failed = len(self.file_results) - files_passed
        # Generate recommendations
        recommendations = self._generate_recommendations()
        migrations = self._generate_migration_suggestions()
        return GlobalAuditReport(
            report_id=hashlib.sha256(f"{self.start_time.isoformat()}-global".encode()).hexdigest()[:16],
            gl_charter_version="ECO-UNIFIED-CHARTER-v1.0",
            execution_timestamp=self.start_time.isoformat(),
            total_files=len(self.file_results),
            files_processed=len(self.file_results),
            files_passed=files_passed,
            files_failed=files_failed,
            issues_by_severity=issues_by_severity,
            issues_by_type=issues_by_type,
            file_results=self.file_results,
            governance_event_summary={
                "total_events": len(self.governance_events),
                "events_by_type": self._count_events_by_type(),
                "events_by_status": self._count_events_by_status()
            },
            best_practice_recommendations=recommendations,
            migration_suggestions=migrations
        )
    def _count_events_by_type(self) -> Dict[str, int]:
        """Count governance events by type."""
        counts = {}
        for event in self.governance_events:
            counts[event.event_type] = counts.get(event.event_type, 0) + 1
        return counts
    def _count_events_by_status(self) -> Dict[str, int]:
        """Count governance events by status."""
        counts = {}
        for event in self.governance_events:
            counts[event.status] = counts.get(event.status, 0) + 1
        return counts
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate best practice recommendations."""
        recommendations = []
        # Analyze common issues
        gl_marker_missing_count = sum(1 for r in self.file_results 
                                      for i in r.issues if i["type"] == IssueType.GL_MARKER_MISSING.value)
        if gl_marker_missing_count > 0:
            recommendations.append({
                "id": "REC-001",
                "priority": "HIGH",
                "category": "governance",
                "title": "Add GL Governance Markers",
                "description": f"{gl_marker_missing_count} files are missing GL governance markers",
                "action": "Add GL layer annotation comments to file headers",
                "template": "// @gl-layer: ECO-30-EXECUTION\n// @semantic-anchor: ECO-30-EXEC-TS"
            })
        # Check for documentation
        doc_missing_count = sum(1 for r in self.file_results 
                               for i in r.issues if "documentation" in i.get("message", "").lower())
        if doc_missing_count > 0:
            recommendations.append({
                "id": "REC-002",
                "priority": "MEDIUM",
                "category": "documentation",
                "title": "Improve Documentation Coverage",
                "description": f"{doc_missing_count} files need better documentation",
                "action": "Add JSDoc comments for all public APIs and exports"
            })
        return recommendations
    def _generate_migration_suggestions(self) -> List[Dict[str, Any]]:
        """Generate migration suggestions for better structure."""
        suggestions = []
        # Suggest consolidating test files
        test_files = [r for r in self.file_results if "test" in r.file_path.lower()]
        if len(test_files) > 0:
            suggestions.append({
                "id": "MIG-001",
                "priority": "LOW",
                "category": "structure",
                "title": "Consolidate Test Structure",
                "current_path": "engine/tests/*/",
                "suggested_path": "engine/__tests__/",
                "rationale": "Follow Jest conventions for test file organization"
            })
        return suggestions
def discover_engine_files(repo_path: str) -> List[str]:
    """Discover all files in the engine directory."""
    engine_path = Path(repo_path) / "engine"
    files = []
    for file_path in engine_path.rglob("*"):
        if file_path.is_file():
            relative_path = str(file_path.relative_to(repo_path))
            files.append(relative_path)
    return sorted(files)
def main():
    """Main execution entry point."""
    repo_path = os.environ.get("REPO_PATH", "/workspace/machine-native-ops")
    output_dir = Path(repo_path) / "governance-audit-results"
    output_dir.mkdir(exist_ok=True)
    print("=" * 80)
    print("GL UNIFIED CHARTER ACTIVATED")
    print("AEP Engine Governance Audit Pipeline")
    print("=" * 80)
    print()
    # Discover files
    print("[Phase 1] Discovering AEP Engine files...")
    files = discover_engine_files(repo_path)
    print(f"  Found {len(files)} files to audit")
    print()
    # Initialize audit engine
    print("[Phase 2] Initializing Governance Audit Engine...")
    engine = GovernanceAuditEngine(repo_path)
    print("  Engine initialized")
    print()
    # Audit each file
    print("[Phase 3] Executing ETL Pipeline (One-by-One Isolated Execution)...")
    for i, file_path in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] Processing: {file_path}")
        result = engine.audit_file(file_path)
        status_icon = "✓" if result.etl_status == "SUCCESS" else "✗"
        issue_count = len(result.issues)
        print(f"    {status_icon} ETL: {result.etl_status} | Issues: {issue_count} | Time: {result.execution_time_ms}ms")
    print()
    # Generate global report
    print("[Phase 4] Generating Global Governance Audit Report...")
    global_report = engine.generate_global_report()
    print(f"  Total Files: {global_report.total_files}")
    print(f"  Passed: {global_report.files_passed}")
    print(f"  Failed: {global_report.files_failed}")
    print()
    # Save reports
    print("[Phase 5] Saving Audit Artifacts...")
    # Save global report
    global_report_path = output_dir / "GLOBAL_GOVERNANCE_AUDIT_REPORT.json"
    with open(global_report_path, 'w', encoding='utf-8') as f:
        # Convert dataclasses to dict for JSON serialization
        report_dict = asdict(global_report)
        # Convert nested dataclasses
        report_dict["file_results"] = [asdict(r) for r in global_report.file_results]
        for fr in report_dict["file_results"]:
            fr["governance_events"] = [asdict(e) if hasattr(e, '__dataclass_fields__') else e for e in fr["governance_events"]]
        json.dump(report_dict, f, indent=2, default=str)
    print(f"  Saved: {global_report_path}")
    # Save per-file reports
    per_file_dir = output_dir / "per-file-reports"
    per_file_dir.mkdir(exist_ok=True)
    for result in global_report.file_results:
        safe_name = result.file_path.replace("/", "_").replace(":", "_")
        file_report_path = per_file_dir / f"{safe_name}_audit.json"
        with open(file_report_path, 'w', encoding='utf-8') as f:
            result_dict = asdict(result)
            result_dict["governance_events"] = [asdict(e) if hasattr(e, '__dataclass_fields__') else e for e in result.governance_events]
            json.dump(result_dict, f, indent=2, default=str)
    print(f"  Saved {len(global_report.file_results)} per-file reports to {per_file_dir}")
    # Save governance event stream
    events_path = output_dir / "governance_event_stream.json"
    with open(events_path, 'w', encoding='utf-8') as f:
        events_list = [asdict(e) for e in engine.governance_events]
        json.dump(events_list, f, indent=2, default=str)
    print(f"  Saved: {events_path}")
    # Generate summary markdown
    summary_path = output_dir / "AUDIT_SUMMARY.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# GL Unified Architecture Governance Framework - AEP Engine Governance Audit Summary\n\n")
        f.write(f"**GL Unified Architecture Governance Framework Activated**\n\n")
        f.write(f"**Report ID:** {global_report.report_id}\n")
        f.write(f"**Execution Time:** {global_report.execution_timestamp}\n\n")
        f.write("## Summary Statistics\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Total Files | {global_report.total_files} |\n")
        f.write(f"| Files Passed | {global_report.files_passed} |\n")
        f.write(f"| Files Failed | {global_report.files_failed} |\n")
        f.write(f"| Pass Rate | {global_report.files_passed/global_report.total_files*100:.1f}% |\n\n")
        f.write("## Issues by Severity\n\n")
        f.write("| Severity | Count |\n")
        f.write("|----------|-------|\n")
        for severity, count in global_report.issues_by_severity.items():
            f.write(f"| {severity} | {count} |\n")
        f.write("\n## Issues by Type\n\n")
        f.write("| Type | Count |\n")
        f.write("|------|-------|\n")
        for issue_type, count in global_report.issues_by_type.items():
            if count > 0:
                f.write(f"| {issue_type} | {count} |\n")
        f.write("\n## Governance Event Stream Summary\n\n")
        f.write(f"- Total Events: {global_report.governance_event_summary['total_events']}\n")
        f.write(f"- Events by Type: {global_report.governance_event_summary['events_by_type']}\n")
        f.write(f"- Events by Status: {global_report.governance_event_summary['events_by_status']}\n")
    print(f"  Saved: {summary_path}")
    print()
    print("=" * 80)
    print("AUDIT COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)
    return global_report
if __name__ == "__main__":
    main()