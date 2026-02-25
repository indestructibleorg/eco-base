#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: aep-executor
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
AEP Engine - Architecture Execution Pipeline
GL Unified Architecture Governance Framework Activated
逐檔單一執行（one-by-one isolated execution）治理稽核引擎
"""
# MNGA-002: Import organization needs review
import os
import sys
import json
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import traceback
class Severity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"
class IssueCategory(Enum):
    PIPELINE_ERROR = "pipeline_error"
    SCHEMA_MISMATCH = "schema_mismatch"
    METADATA_MISSING = "metadata_missing"
    NAMING_INCONSISTENT = "naming_inconsistent"
    STRUCTURE_VIOLATION = "structure_violation"
    GL_MARKER_MISSING = "gl_marker_missing"
    SEMANTIC_MANIFEST_MISSING = "semantic_manifest_missing"
    DAG_INTEGRITY = "dag_integrity"
    GOVERNANCE_CHAIN = "gl-platform.governance_chain"
@dataclass
class GovernanceEvent:
    """治理事件"""
    event_id: str
    timestamp: str
    event_type: str
    file_path: str
    details: Dict[str, Any]
    severity: str
    reversible: bool = True
    provable: bool = True
@dataclass
class FileIssue:
    """檔案問題"""
    issue_id: str
    category: str
    severity: str
    file_path: str
    line_number: Optional[int]
    description: str
    suggestion: str
    evidence: Dict[str, Any]
@dataclass
class FileReport:
    """單一檔案報告"""
    file_path: str
    file_hash: str
    file_type: str
    file_size: int
    execution_timestamp: str
    execution_status: str
    issues: List[FileIssue] = field(default_factory=list)
    gl-platform.governance_events: List[GovernanceEvent] = field(default_factory=list)
    metadata_check: Dict[str, Any] = field(default_factory=dict)
    schema_check: Dict[str, Any] = field(default_factory=dict)
    gl_markers: List[str] = field(default_factory=list)
    semantic_anchors: List[str] = field(default_factory=list)
    best_practice_violations: List[str] = field(default_factory=list)
    suggested_path: Optional[str] = None
    suggested_name: Optional[str] = None
class AEPGovernanceAuditor:
    """AEP 治理稽核引擎"""
    # GL 標記正則表達式
    GL_MARKER_PATTERNS = [
        r'@gl-layer\s+(\S+)',
        r'@gl-module\s+(\S+)',
        r'@gl-semantic-anchor\s+(\S+)',
        r'@gl-evidence-required\s+(\S+)',
        r'@gl-gl-platform.governance\s+(\S+)',
        r'ECO-\d{2}-[A-Z]+',
        r'gl_layer:\s*(\S+)',
        r'gl_module:\s*(\S+)',
    ]
    # 最佳實踐目錄結構
    BEST_PRACTICE_STRUCTURE = {
        'docs': ['*.md', 'README.md', 'CHANGELOG.md'],
        'src': ['*.ts', '*.py', '*.js'],
        'config': ['*.yaml', '*.yml', '*.json'],
        'tests': ['test_*.py', '*.test.ts', '*.spec.ts'],
        'schemas': ['*.schema.yaml', '*.schema.json'],
        'manifests': ['manifest.yaml', '*.manifest.yaml'],
        'policies': ['*.policy.yaml', 'policies.yaml'],
        'gl-platform.governance': ['*.gl-platform.governance.yaml'],
    }
    # 命名規範
    NAMING_CONVENTIONS = {
        '.yaml': r'^[a-z][a-z0-9-]*\.yaml$',
        '.yml': r'^[a-z][a-z0-9-]*\.yml$',
        '.json': r'^[a-z][a-z0-9-]*\.json$',
        '.ts': r'^[a-z][a-z0-9-]*\.ts$',
        '.py': r'^[a-z_][a-z0-9_]*\.py$',
        '.md': r'^[A-Z][A-Z0-9_-]*\.md$|^README\.md$|^CHANGELOG\.md$',
    }
    def __init__(self, root_path: str, output_dir: str):
        self.root_path = Path(root_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports: List[FileReport] = []
        self.gl-platform.governance_events: List[GovernanceEvent] = []
        self.global_issues: List[FileIssue] = []
        self.execution_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    def generate_event_id(self) -> str:
        """生成唯一事件 ID"""
        return f"GE-{self.execution_id}-{len(self.gl-platform.governance_events):06d}"
    def generate_issue_id(self) -> str:
        """生成唯一問題 ID"""
        return f"ISS-{self.execution_id}-{len(self.global_issues):06d}"
    def emit_governance_event(self, event_type: str, file_path: str, 
                              details: Dict[str, Any], severity: str) -> GovernanceEvent:
        """發出治理事件"""
        event = GovernanceEvent(
            event_id=self.generate_event_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            file_path=str(file_path),
            details=details,
            severity=severity
        )
        self.gl-platform.governance_events.append(event)
        return event
    def calculate_file_hash(self, file_path: Path) -> str:
        """計算檔案 SHA256 雜湊"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return "HASH_ERROR"
    def detect_file_type(self, file_path: Path) -> str:
        """檢測檔案類型"""
        suffix = file_path.suffix.lower()
        type_map = {
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.ts': 'typescript',
            '.js': 'javascript',
            '.py': 'python',
            '.md': 'markdown',
            '.sh': 'shell',
            '.txt': 'text',
        }
        return type_map.get(suffix, 'unknown')
    def check_gl_markers(self, content: str, file_path: Path) -> List[str]:
        """檢查 GL 標記"""
        markers = []
        for pattern in self.GL_MARKER_PATTERNS:
            matches = re.findall(pattern, content, re.MULTILINE)
            markers.extend(matches)
        return markers
    def check_metadata(self, content: str, file_path: Path, file_type: str) -> Dict[str, Any]:
        """檢查 metadata"""
        result = {
            'has_metadata': False,
            'metadata_fields': [],
            'missing_fields': [],
            'issues': []
        }
        required_fields = ['version', 'name', 'description']
        if file_type == 'yaml':
            # 檢查 YAML metadata
            if 'metadata:' in content or 'apiVersion:' in content:
                result['has_metadata'] = True
            for field in required_fields:
                if f'{field}:' in content:
                    result['metadata_fields'].append(field)
                else:
                    result['missing_fields'].append(field)
        elif file_type == 'json':
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    result['has_metadata'] = True
                    for field in required_fields:
                        if field in data:
                            result['metadata_fields'].append(field)
                        else:
                            result['missing_fields'].append(field)
            except json.JSONDecodeError:
                result['issues'].append('Invalid JSON format')
        elif file_type == 'markdown':
            # 檢查 frontmatter
            if content.startswith('---'):
                result['has_metadata'] = True
        elif file_type in ['typescript', 'javascript', 'python']:
            # 檢查文件頭註釋
            if content.startswith('/**') or content.startswith('"""') or content.startswith("'''"):
                result['has_metadata'] = True
            # 檢查 @gl- 標記
            if '@gl-' in content:
                result['has_metadata'] = True
        return result
    def check_schema(self, content: str, file_path: Path, file_type: str) -> Dict[str, Any]:
        """檢查 schema"""
        result = {
            'has_schema_reference': False,
            'schema_type': None,
            'schema_valid': None,
            'issues': []
        }
        if file_type == 'yaml':
            if '$schema:' in content or 'schema:' in content:
                result['has_schema_reference'] = True
                result['schema_type'] = 'yaml-schema'
            if 'apiVersion:' in content and 'kind:' in content:
                result['has_schema_reference'] = True
                result['schema_type'] = 'kubernetes'
        elif file_type == 'json':
            if '"$schema"' in content:
                result['has_schema_reference'] = True
                result['schema_type'] = 'json-schema'
        elif file_type == 'typescript':
            if 'interface ' in content or 'type ' in content:
                result['has_schema_reference'] = True
                result['schema_type'] = 'typescript-types'
        return result
    def check_naming_convention(self, file_path: Path) -> List[FileIssue]:
        """檢查命名規範"""
        issues = []
        suffix = file_path.suffix.lower()
        filename = file_path.name
        if suffix in self.NAMING_CONVENTIONS:
            pattern = self.NAMING_CONVENTIONS[suffix]
            if not re.match(pattern, filename):
                issues.append(FileIssue(
                    issue_id=self.generate_issue_id(),
                    category=IssueCategory.NAMING_INCONSISTENT.value,
                    severity=Severity.LOW.value,
                    file_path=str(file_path),
                    line_number=None,
                    description=f"Filename '{filename}' does not match naming convention",
                    suggestion=f"Rename to match pattern: {pattern}",
                    evidence={'current_name': filename, 'expected_pattern': pattern}
                ))
        # 檢查空格和特殊字符
        if ' ' in filename:
            issues.append(FileIssue(
                issue_id=self.generate_issue_id(),
                category=IssueCategory.NAMING_INCONSISTENT.value,
                severity=Severity.MEDIUM.value,
                file_path=str(file_path),
                line_number=None,
                description=f"Filename contains spaces: '{filename}'",
                suggestion="Replace spaces with hyphens or underscores",
                evidence={'current_name': filename}
            ))
        return issues
    def check_structure(self, file_path: Path) -> List[FileIssue]:
        """檢查目錄結構"""
        issues = []
        relative_path = file_path.relative_to(self.root_path)
        parts = relative_path.parts
        # 檢查深度
        if len(parts) > 8:
            issues.append(FileIssue(
                issue_id=self.generate_issue_id(),
                category=IssueCategory.STRUCTURE_VIOLATION.value,
                severity=Severity.LOW.value,
                file_path=str(file_path),
                line_number=None,
                description=f"Directory depth exceeds recommended limit (8): {len(parts)}",
                suggestion="Consider flattening directory structure",
                evidence={'depth': len(parts), 'path': str(relative_path)}
            ))
        return issues
    def suggest_best_practice_path(self, file_path: Path, file_type: str) -> Optional[str]:
        """建議最佳實踐路徑"""
        filename = file_path.name
        suffix = file_path.suffix.lower()
        # 根據檔案類型建議目錄
        suggestions = {
            'markdown': 'docs',
            'yaml': 'config',
            'json': 'config',
            'typescript': 'src',
            'javascript': 'src',
            'python': 'src',
            'shell': 'scripts',
        }
        # 特殊檔案處理
        if 'test' in filename.lower() or 'spec' in filename.lower():
            return f"tests/{filename}"
        if 'schema' in filename.lower():
            return f"schemas/{filename}"
        if 'manifest' in filename.lower():
            return f"manifests/{filename}"
        if 'policy' in filename.lower() or 'policies' in filename.lower():
            return f"policies/{filename}"
        suggested_dir = suggestions.get(file_type)
        if suggested_dir:
            return f"{suggested_dir}/{filename}"
        return None
    def execute_file_audit(self, file_path: Path) -> FileReport:
        """執行單一檔案稽核"""
        # 發出開始事件
        self.emit_gl-platform.governance_event(
            event_type="FILE_AUDIT_START",
            file_path=str(file_path),
            details={'action': 'start_audit'},
            severity=Severity.INFO.value
        )
        try:
            # 讀取檔案
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
            except Exception as e:
                content = ""
            file_type = self.detect_file_type(file_path)
            file_hash = self.calculate_file_hash(file_path)
            file_size = file_path.stat().st_size
            # 建立報告
            report = FileReport(
                file_path=str(file_path.relative_to(self.root_path)),
                file_hash=file_hash,
                file_type=file_type,
                file_size=file_size,
                execution_timestamp=datetime.now(timezone.utc).isoformat(),
                execution_status="SUCCESS"
            )
            # 檢查 GL 標記
            report.gl_markers = self.check_gl_markers(content, file_path)
            if not report.gl_markers:
                issue = FileIssue(
                    issue_id=self.generate_issue_id(),
                    category=IssueCategory.GL_MARKER_MISSING.value,
                    severity=Severity.MEDIUM.value,
                    file_path=str(file_path),
                    line_number=None,
                    description="No GL gl-platform.governance markers found",
                    suggestion="Add @gl-layer, @gl-module, or @gl-semantic-anchor markers",
                    evidence={'searched_patterns': self.GL_MARKER_PATTERNS[:3]}
                )
                report.issues.append(issue)
                self.global_issues.append(issue)
            # 檢查 metadata
            report.metadata_check = self.check_metadata(content, file_path, file_type)
            if not report.metadata_check.get('has_metadata'):
                issue = FileIssue(
                    issue_id=self.generate_issue_id(),
                    category=IssueCategory.METADATA_MISSING.value,
                    severity=Severity.MEDIUM.value,
                    file_path=str(file_path),
                    line_number=None,
                    description="No metadata found in file",
                    suggestion="Add metadata header with version, name, description",
                    evidence=report.metadata_check
                )
                report.issues.append(issue)
                self.global_issues.append(issue)
            # 檢查 schema
            report.schema_check = self.check_schema(content, file_path, file_type)
            # 檢查命名規範
            naming_issues = self.check_naming_convention(file_path)
            report.issues.extend(naming_issues)
            self.global_issues.extend(naming_issues)
            # 檢查結構
            structure_issues = self.check_structure(file_path)
            report.issues.extend(structure_issues)
            self.global_issues.extend(structure_issues)
            # 建議最佳實踐路徑
            report.suggested_path = self.suggest_best_practice_path(file_path, file_type)
            # 發出完成事件
            self.emit_gl-platform.governance_event(
                event_type="FILE_AUDIT_COMPLETE",
                file_path=str(file_path),
                details={
                    'issues_count': len(report.issues),
                    'gl_markers_count': len(report.gl_markers),
                    'has_metadata': report.metadata_check.get('has_metadata', False)
                },
                severity=Severity.INFO.value
            )
            return report
        except Exception as e:
            # 發出錯誤事件
            self.emit_gl-platform.governance_event(
                event_type="FILE_AUDIT_ERROR",
                file_path=str(file_path),
                details={'error': str(e), 'traceback': traceback.format_exc()},
                severity=Severity.HIGH.value
            )
            return FileReport(
                file_path=str(file_path),
                file_hash="ERROR",
                file_type="unknown",
                file_size=0,
                execution_timestamp=datetime.now(timezone.utc).isoformat(),
                execution_status="ERROR",
                issues=[FileIssue(
                    issue_id=self.generate_issue_id(),
                    category=IssueCategory.PIPELINE_ERROR.value,
                    severity=Severity.CRITICAL.value,
                    file_path=str(file_path),
                    line_number=None,
                    description=f"Pipeline execution error: {str(e)}",
                    suggestion="Review file for encoding or access issues",
                    evidence={'error': str(e)}
                )]
            )
    def scan_directory(self) -> List[Path]:
        """掃描目錄獲取所有檔案"""
        files = []
        for file_path in self.root_path.rglob('*'):
            if file_path.is_file():
                # 排除特定目錄
                if any(part.startswith('.') for part in file_path.parts):
                    continue
                if 'node_modules' in file_path.parts:
                    continue
                if '__pycache__' in file_path.parts:
                    continue
                files.append(file_path)
        return sorted(files)
    def run_full_audit(self) -> Dict[str, Any]:
        """執行完整稽核"""
        print(f"\n{'='*60}")
        print("AEP Engine - Architecture Execution Pipeline")
        print("GL Unified Architecture Governance Framework Activated")
        print(f"{'='*60}\n")
        # 發出開始事件
        self.emit_gl-platform.governance_event(
            event_type="FULL_AUDIT_START",
            file_path=str(self.root_path),
            details={'execution_id': self.execution_id},
            severity=Severity.INFO.value
        )
        # 掃描檔案
        print(f"[Phase 1] Scanning directory: {self.root_path}")
        files = self.scan_directory()
        print(f"  Found {len(files)} files to audit\n")
        # 逐檔執行
        print(f"[Phase 2] Executing one-by-one isolated audit...")
        for i, file_path in enumerate(files, 1):
            if i % 50 == 0 or i == len(files):
                print(f"  Progress: {i}/{len(files)} files processed")
            report = self.execute_file_audit(file_path)
            self.reports.append(report)
        print(f"\n[Phase 3] Generating reports...")
        # 生成全域報告
        global_report = self.generate_global_report()
        # 保存報告
        self.save_reports()
        print(f"\n[Phase 4] Audit complete!")
        print(f"  Total files: {len(self.reports)}")
        print(f"  Total issues: {len(self.global_issues)}")
        print(f"  Total gl-platform.governance events: {len(self.gl-platform.governance_events)}")
        return global_report
    def generate_global_report(self) -> Dict[str, Any]:
        """生成全域治理稽核報告"""
        # 統計問題
        issues_by_severity = {}
        issues_by_category = {}
        for issue in self.global_issues:
            # 按嚴重度統計
            sev = issue.severity
            if sev not in issues_by_severity:
                issues_by_severity[sev] = []
            issues_by_severity[sev].append(asdict(issue))
            # 按類別統計
            cat = issue.category
            if cat not in issues_by_category:
                issues_by_category[cat] = []
            issues_by_category[cat].append(asdict(issue))
        # 統計檔案類型
        files_by_type = {}
        for report in self.reports:
            ft = report.file_type
            if ft not in files_by_type:
                files_by_type[ft] = 0
            files_by_type[ft] += 1
        # 統計 GL 標記覆蓋率
        files_with_gl_markers = sum(1 for r in self.reports if r.gl_markers)
        gl_coverage = files_with_gl_markers / len(self.reports) * 100 if self.reports else 0
        # 統計 metadata 覆蓋率
        files_with_metadata = sum(1 for r in self.reports if r.metadata_check.get('has_metadata'))
        metadata_coverage = files_with_metadata / len(self.reports) * 100 if self.reports else 0
        # 生成最佳實踐建議
        migration_suggestions = []
        for report in self.reports:
            if report.suggested_path and report.suggested_path != report.file_path:
                migration_suggestions.append({
                    'current_path': report.file_path,
                    'suggested_path': report.suggested_path,
                    'file_type': report.file_type
                })
        return {
            'gl_unified_charter': 'ACTIVATED',
            'execution_id': self.execution_id,
            'execution_timestamp': datetime.now(timezone.utc).isoformat(),
            'root_path': str(self.root_path),
            'summary': {
                'total_files': len(self.reports),
                'total_issues': len(self.global_issues),
                'total_gl-platform.governance_events': len(self.gl-platform.governance_events),
                'gl_marker_coverage_percent': round(gl_coverage, 2),
                'metadata_coverage_percent': round(metadata_coverage, 2),
                'files_with_errors': sum(1 for r in self.reports if r.execution_status == 'ERROR')
            },
            'issues_by_severity': {
                k: len(v) for k, v in issues_by_severity.items()
            },
            'issues_by_category': {
                k: len(v) for k, v in issues_by_category.items()
            },
            'files_by_type': files_by_type,
            'critical_issues': issues_by_severity.get(Severity.CRITICAL.value, []),
            'high_issues': issues_by_severity.get(Severity.HIGH.value, []),
            'medium_issues': issues_by_severity.get(Severity.MEDIUM.value, [])[:50],  # 限制數量
            'migration_suggestions': migration_suggestions[:100],  # 限制數量
            'gl-platform.governance_event_summary': {
                'total_events': len(self.gl-platform.governance_events),
                'event_types': list(set(e.event_type for e in self.gl-platform.governance_events))
            },
            'best_practice_recommendations': self.generate_best_practice_recommendations()
        }
    def generate_best_practice_recommendations(self) -> List[Dict[str, Any]]:
        """生成最佳實踐建議"""
        recommendations = []
        # 檢查 GL 標記覆蓋率
        files_with_gl = sum(1 for r in self.reports if r.gl_markers)
        if files_with_gl < len(self.reports) * 0.5:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'gl-platform.governance',
                'title': 'Improve GL Marker Coverage',
                'description': f'Only {files_with_gl}/{len(self.reports)} files have GL markers',
                'action': 'Add @gl-layer, @gl-module annotations to all configuration files'
            })
        # 檢查 metadata 覆蓋率
        files_with_meta = sum(1 for r in self.reports if r.metadata_check.get('has_metadata'))
        if files_with_meta < len(self.reports) * 0.5:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'documentation',
                'title': 'Improve Metadata Coverage',
                'description': f'Only {files_with_meta}/{len(self.reports)} files have metadata',
                'action': 'Add metadata headers with version, name, description fields'
            })
        # 檢查命名問題
        naming_issues = [i for i in self.global_issues if i.category == IssueCategory.NAMING_INCONSISTENT.value]
        if naming_issues:
            recommendations.append({
                'priority': 'LOW',
                'category': 'naming',
                'title': 'Standardize File Naming',
                'description': f'{len(naming_issues)} files have naming convention issues',
                'action': 'Rename files to follow kebab-case for configs, snake_case for Python'
            })
        return recommendations
    def save_reports(self):
        """保存所有報告"""
        # 保存全域報告
        global_report = self.generate_global_report()
        global_report_path = self.output_dir / f'ECO-GLOBAL-GOVERNANCE-AUDIT-{self.execution_id}.json'
        with open(global_report_path, 'w', encoding='utf-8') as f:
            json.dump(global_report, f, indent=2, ensure_ascii=False)
        print(f"  Global report saved: {global_report_path}")
        # 保存治理事件流
        events_path = self.output_dir / f'ECO-GOVERNANCE-EVENT-STREAM-{self.execution_id}.json'
        with open(events_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(e) for e in self.gl-platform.governance_events], f, indent=2, ensure_ascii=False)
        print(f"  Event stream saved: {events_path}")
        # 保存問題列表
        issues_path = self.output_dir / f'ECO-ISSUES-LIST-{self.execution_id}.json'
        with open(issues_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(i) for i in self.global_issues], f, indent=2, ensure_ascii=False)
        print(f"  Issues list saved: {issues_path}")
        # 保存個別檔案報告（壓縮版）
        file_reports_path = self.output_dir / f'ECO-FILE-REPORTS-{self.execution_id}.json'
        with open(file_reports_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in self.reports], f, indent=2, ensure_ascii=False)
        print(f"  File reports saved: {file_reports_path}")
def main():
    """主程式"""
    root_path = sys.argv[1] if len(sys.argv) > 1 else '/workspace/machine-native-ops/ns-root'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '/workspace/machine-native-ops/aep-gl-platform.governance-audit/reports'
    auditor = AEPGovernanceAuditor(root_path, output_dir)
    report = auditor.run_full_audit()
    # 輸出摘要
    print(f"\n{'='*60}")
    print("AUDIT SUMMARY")
    print(f"{'='*60}")
    print(f"Total Files Audited: {report['summary']['total_files']}")
    print(f"Total Issues Found: {report['summary']['total_issues']}")
    print(f"GL Marker Coverage: {report['summary']['gl_marker_coverage_percent']}%")
    print(f"Metadata Coverage: {report['summary']['metadata_coverage_percent']}%")
    print(f"\nIssues by Severity:")
    for sev, count in report['issues_by_severity'].items():
        print(f"  {sev}: {count}")
    print(f"\nIssues by Category:")
    for cat, count in report['issues_by_category'].items():
        print(f"  {cat}: {count}")
    return 0 if report['summary']['total_issues'] == 0 else 1
if __name__ == '__main__':
    sys.exit(main())