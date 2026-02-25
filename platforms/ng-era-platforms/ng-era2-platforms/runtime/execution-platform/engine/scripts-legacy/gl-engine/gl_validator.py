#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: gl_validator
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
GL Artifacts Validator - Comprehensive Governance Artifact Validation
MachineNativeOps GL Architecture Implementation
This module provides comprehensive validation for GL gl-platform.gl-platform.governance artifacts,
including schema validation, policy compliance, security checks, and
cross-layer consistency verification.
"""
# MNGA-002: Import organization needs review
import sys
import yaml
import json
import re
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('GLValidator')
class ValidationSeverity(Enum):
    """Validation finding severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    def __lt__(self, other):
        order = {self.ERROR: 0, self.WARNING: 1, self.INFO: 2}
        return order[self] < order[other]
@dataclass
class ValidationFinding:
    """Represents a validation finding."""
    rule_id: str
    rule_name: str
    severity: ValidationSeverity
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    field_path: Optional[str] = None
    suggestion: Optional[str] = None
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rule_id': self.rule_id,
            'rule_name': self.rule_name,
            'severity': self.severity.value,
            'message': self.message,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'field_path': self.field_path,
            'suggestion': self.suggestion
        }
@dataclass
class ValidationResult:
    """Result of validation operation."""
    passed: bool
    findings: List[ValidationFinding] = field(default_factory=list)
    files_validated: int = 0
    artifacts_validated: int = 0
    execution_time: float = 0.0
    @property
    def error_count(self) -> int:
        return len([f for f in self.findings if f.severity == ValidationSeverity.ERROR])
    @property
    def warning_count(self) -> int:
        return len([f for f in self.findings if f.severity == ValidationSeverity.WARNING])
    @property
    def info_count(self) -> int:
        return len([f for f in self.findings if f.severity == ValidationSeverity.INFO])
    def to_dict(self) -> Dict[str, Any]:
        return {
            'passed': self.passed,
            'findings': [f.to_dict() for f in self.findings],
            'files_validated': self.files_validated,
            'artifacts_validated': self.artifacts_validated,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'info_count': self.info_count,
            'execution_time': self.execution_time
        }
class ValidationRule:
    """Base class for validation rules."""
    def __init__(self, rule_id: str, rule_name: str, severity: ValidationSeverity = ValidationSeverity.ERROR):
        self.rule_id = rule_id
        self.rule_name = rule_name
        self.severity = severity
    def validate(self, artifact: Dict[str, Any], file_path: str) -> List[ValidationFinding]:
        """Validate artifact and return findings."""
        raise NotImplementedError
class RequiredFieldRule(ValidationRule):
    """Validates required fields exist."""
    def __init__(self, field_path: str, severity: ValidationSeverity = ValidationSeverity.ERROR):
        super().__init__(
            rule_id=f"REQ-{field_path.upper().replace('.', '-')}",
            rule_name=f"Required Field: {field_path}",
            severity=severity
        )
        self.field_path = field_path
    def validate(self, artifact: Dict[str, Any], file_path: str) -> List[ValidationFinding]:
        findings = []
        # Navigate to field
        parts = self.field_path.split('.')
        current = artifact
        for i, part in enumerate(parts):
            if not isinstance(current, dict) or part not in current:
                findings.append(ValidationFinding(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    severity=self.severity,
                    message=f"Missing required field: {self.field_path}",
                    file_path=file_path,
                    field_path='.'.join(parts[:i+1]),
                    suggestion=f"Add the '{part}' field to your artifact"
                ))
                break
            current = current[part]
        return findings
class FieldFormatRule(ValidationRule):
    """Validates field format using regex."""
    def __init__(self, field_path: str, pattern: str, description: str,
                 severity: ValidationSeverity = ValidationSeverity.WARNING):
        super().__init__(
            rule_id=f"FMT-{field_path.upper().replace('.', '-')}",
            rule_name=f"Field Format: {field_path}",
            severity=severity
        )
        self.field_path = field_path
        self.pattern = pattern
        self.description = description
    def validate(self, artifact: Dict[str, Any], file_path: str) -> List[ValidationFinding]:
        findings = []
        # Navigate to field
        parts = self.field_path.split('.')
        current = artifact
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return findings  # Field doesn't exist, handled by RequiredFieldRule
            current = current[part]
        if isinstance(current, str) and not re.match(self.pattern, current):
            findings.append(ValidationFinding(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                severity=self.severity,
                message=f"Field '{self.field_path}' has invalid format: {current}",
                file_path=file_path,
                field_path=self.field_path,
                suggestion=self.description
            ))
        return findings
class LayerValidationRule(ValidationRule):
    """Validates layer-specific requirements."""
    VALID_LAYERS = ['GL00-09', 'GL10-29', 'GL30-49', 'GL50-59', 'GL60-80', 'GL81-83', 'GL90-99']
    def __init__(self):
        super().__init__(
            rule_id="LAYER-001",
            rule_name="Valid GL Layer",
            severity=ValidationSeverity.ERROR
        )
    def validate(self, artifact: Dict[str, Any], file_path: str) -> List[ValidationFinding]:
        findings = []
        layer = artifact.get('metadata', {}).get('layer', '')
        if layer and layer not in self.VALID_LAYERS:
            findings.append(ValidationFinding(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                severity=self.severity,
                message=f"Invalid layer: {layer}",
                file_path=file_path,
                field_path='metadata.layer',
                suggestion=f"Use one of: {', '.join(self.VALID_LAYERS)}"
            ))
        return findings
class AgeValidationRule(ValidationRule):
    """Validates artifact age against policy."""
    LAYER_MAX_AGE = {
        'GL00-09': 365,
        'GL10-29': 90,
        'GL30-49': 30,
        'GL50-59': 180,
        'GL60-80': 90,
        'GL81-83': 180,
        'GL90-99': 365,
    }
    def __init__(self):
        super().__init__(
            rule_id="AGE-001",
            rule_name="Artifact Age Policy",
            severity=ValidationSeverity.WARNING
        )
    def validate(self, artifact: Dict[str, Any], file_path: str) -> List[ValidationFinding]:
        findings = []
        metadata = artifact.get('metadata', {})
        layer = metadata.get('layer', '')
        updated_at = metadata.get('updated_at', metadata.get('created_at', ''))
        if not updated_at or layer not in self.LAYER_MAX_AGE:
            return findings
        try:
            update_date = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
            age_days = (datetime.now(update_date.tzinfo) - update_date).days
            max_age = self.LAYER_MAX_AGE[layer]
            if age_days > max_age:
                findings.append(ValidationFinding(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    severity=self.severity,
                    message=f"Artifact is {age_days} days old, exceeds {max_age} day limit for {layer}",
                    file_path=file_path,
                    field_path='metadata.updated_at',
                    suggestion="Review and update the artifact to ensure it remains current"
                ))
        except Exception:
            pass
        return findings
class SecurityValidationRule(ValidationRule):
    """Validates for security issues."""
    SENSITIVE_PATTERNS = [
        (r'password\s*[:=]\s*["\']?[^"\'\s]+', 'Potential password exposure'),
        (r'api[_-]?key\s*[:=]\s*["\']?[a-zA-Z0-9]{20,}', 'Potential API key exposure'),
        (r'secret\s*[:=]\s*["\']?[^"\'\s]+', 'Potential secret exposure'),
        (r'token\s*[:=]\s*["\']?[a-zA-Z0-9]{20,}', 'Potential token exposure'),
        (r'private[_-]?key', 'Potential private key reference'),
        (r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----', 'Private key detected'),
        (r'aws[_-]?access[_-]?key[_-]?id', 'AWS access key reference'),
        (r'aws[_-]?secret[_-]?access[_-]?key', 'AWS secret key reference'),
    ]
    def __init__(self):
        super().__init__(
            rule_id="SEC-001",
            rule_name="Security Sensitive Data",
            severity=ValidationSeverity.ERROR
        )
    def validate(self, artifact: Dict[str, Any], file_path: str) -> List[ValidationFinding]:
        findings = []
        # Convert artifact to string for pattern matching
        content = yaml.dump(artifact, default_flow_style=False)
        for pattern, description in self.SENSITIVE_PATTERNS:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                findings.append(ValidationFinding(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    severity=self.severity,
                    message=f"{description}: {match.group()[:30]}...",
                    file_path=file_path,
                    suggestion="Remove sensitive data and use environment variables or secrets management"
                ))
        return findings
class DependencyValidationRule(ValidationRule):
    """Validates layer dependencies."""
    VALID_DEPENDENCIES = {
        'GL00-09': {'downstream': ['GL10-29'], 'upstream': ['GL90-99']},
        'GL10-29': {'downstream': ['GL30-49'], 'upstream': ['GL00-09']},
        'GL30-49': {'downstream': ['GL50-59'], 'upstream': ['GL10-29']},
        'GL50-59': {'downstream': ['GL60-80'], 'upstream': ['GL30-49']},
        'GL60-80': {'downstream': [], 'upstream': ['GL50-59']},
        'GL81-83': {'downstream': [], 'upstream': []},
        'GL90-99': {'downstream': ['all'], 'upstream': ['GL00-09']},
    }
    def __init__(self):
        super().__init__(
            rule_id="DEP-001",
            rule_name="Layer Dependency",
            severity=ValidationSeverity.WARNING
        )
    def validate(self, artifact: Dict[str, Any], file_path: str) -> List[ValidationFinding]:
        findings = []
        layer = artifact.get('metadata', {}).get('layer', '')
        if not layer or layer not in self.VALID_DEPENDENCIES:
            return findings
        # Find layer references in spec
        spec = artifact.get('spec', {})
        referenced_layers = self._find_layer_references(spec)
        valid_deps = self.VALID_DEPENDENCIES[layer]
        all_valid = set(valid_deps.get('downstream', [])) | set(valid_deps.get('upstream', [])) | {layer}
        if 'all' in valid_deps.get('downstream', []):
            return findings  # GL90-99 can reference all layers
        for ref_layer in referenced_layers:
            if ref_layer not in all_valid and ref_layer in self.VALID_DEPENDENCIES:
                findings.append(ValidationFinding(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    severity=self.severity,
                    message=f"Invalid dependency from {layer} to {ref_layer}",
                    file_path=file_path,
                    suggestion=f"Valid dependencies for {layer}: {all_valid}"
                ))
        return findings
    def _find_layer_references(self, obj: Any, refs: Set[str] = None) -> Set[str]:
        """Recursively find layer references."""
        if refs is None:
            refs = set()
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in ['layer', 'source_layer', 'target_layer'] and isinstance(value, str):
                    if value.startswith('GL'):
                        refs.add(value)
                else:
                    self._find_layer_references(value, refs)
        elif isinstance(obj, list):
            for item in obj:
                self._find_layer_references(item, refs)
        return refs
class SpecCompletenessRule(ValidationRule):
    """Validates spec completeness based on artifact kind."""
    REQUIRED_SPEC_FIELDS = {
        'VisionStatement': ['vision', 'mission'],
        'StrategicObjectives': ['objectives'],
        'GovernanceCharter': ['gl-platform.gl-platform.governance_structure'],
        'OperationalPlan': ['initiatives'],
        'StandardOperatingProcedure': ['procedure'],
        'MetricsDefinition': ['metrics'],
        'AlertRules': ['rules'],
        'ProjectPlan': ['overview', 'timeline'],
    }
    def __init__(self):
        super().__init__(
            rule_id="SPEC-001",
            rule_name="Spec Completeness",
            severity=ValidationSeverity.WARNING
        )
    def validate(self, artifact: Dict[str, Any], file_path: str) -> List[ValidationFinding]:
        findings = []
        kind = artifact.get('kind', '')
        spec = artifact.get('spec', {})
        if kind not in self.REQUIRED_SPEC_FIELDS:
            return findings
        required_fields = self.REQUIRED_SPEC_FIELDS[kind]
        for field in required_fields:
            if field not in spec:
                findings.append(ValidationFinding(
                    rule_id=self.rule_id,
                    rule_name=self.rule_name,
                    severity=self.severity,
                    message=f"Missing recommended spec field '{field}' for {kind}",
                    file_path=file_path,
                    field_path=f'spec.{field}',
                    suggestion=f"Add '{field}' to the spec section"
                ))
        return findings
class GLValidator:
    """Main GL Validator class."""
    def __init__(self, workspace_path: str = '.'):
        self.workspace_path = Path(workspace_path).resolve()
        self.rules: List[ValidationRule] = []
        self._register_default_rules()
    def _register_default_rules(self):
        """Register default validation rules."""
        # Required fields
        self.rules.append(RequiredFieldRule('apiVersion'))
        self.rules.append(RequiredFieldRule('kind'))
        self.rules.append(RequiredFieldRule('metadata'))
        self.rules.append(RequiredFieldRule('metadata.name'))
        self.rules.append(RequiredFieldRule('metadata.version'))
        self.rules.append(RequiredFieldRule('metadata.created_at'))
        self.rules.append(RequiredFieldRule('metadata.owner'))
        self.rules.append(RequiredFieldRule('metadata.layer'))
        self.rules.append(RequiredFieldRule('spec'))
        # Format rules
        self.rules.append(FieldFormatRule(
            'apiVersion',
            r'^gl-platform.gl-platform.governance\.machinenativeops\.io/v\d+$',
            "apiVersion should be 'gl-platform.gl-platform.governance.machinenativeops.io/v2'"
        ))
        self.rules.append(FieldFormatRule(
            'metadata.version',
            r'^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?(\+[a-zA-Z0-9]+)?$',
            "Version should follow semver format (e.g., 1.0.0)"
        ))
        # Layer validation
        self.rules.append(LayerValidationRule())
        # Age validation
        self.rules.append(AgeValidationRule())
        # Security validation
        self.rules.append(SecurityValidationRule())
        # Dependency validation
        self.rules.append(DependencyValidationRule())
        # Spec completeness
        self.rules.append(SpecCompletenessRule())
    def add_rule(self, rule: ValidationRule):
        """Add a custom validation rule."""
        self.rules.append(rule)
    def validate_artifact(self, artifact: Dict[str, Any], file_path: str) -> List[ValidationFinding]:
        """Validate a single artifact."""
        findings = []
        for rule in self.rules:
            try:
                rule_findings = rule.validate(artifact, file_path)
                findings.extend(rule_findings)
            except Exception as e:
                logger.warning(f"Rule {rule.rule_id} failed: {e}")
        return findings
    def validate_file(self, file_path: Path) -> Tuple[bool, List[ValidationFinding]]:
        """Validate a single file."""
        findings = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
            if content is None:
                return True, []  # Empty file
            if not isinstance(content, dict):
                findings.append(ValidationFinding(
                    rule_id="PARSE-001",
                    rule_name="YAML Structure",
                    severity=ValidationSeverity.ERROR,
                    message="File does not contain a valid YAML dictionary",
                    file_path=str(file_path)
                ))
                return False, findings
            # Check if it's a GL artifact
            if 'apiVersion' not in content:
                return True, []  # Not a GL artifact, skip
            findings = self.validate_artifact(content, str(file_path))
            has_errors = any(f.severity == ValidationSeverity.ERROR for f in findings)
            return not has_errors, findings
        except yaml.YAMLError as e:
            findings.append(ValidationFinding(
                rule_id="PARSE-002",
                rule_name="YAML Syntax",
                severity=ValidationSeverity.ERROR,
                message=f"YAML parsing error: {str(e)}",
                file_path=str(file_path)
            ))
            return False, findings
        except Exception as e:
            findings.append(ValidationFinding(
                rule_id="PARSE-003",
                rule_name="File Read",
                severity=ValidationSeverity.ERROR,
                message=f"Error reading file: {str(e)}",
                file_path=str(file_path)
            ))
            return False, findings
    def validate_directory(self, directory: Path, recursive: bool = True) -> ValidationResult:
        """Validate all artifacts in a directory."""
        import time
        start_time = time.time()
        all_findings = []
        files_validated = 0
        artifacts_validated = 0
        if recursive:
            yaml_files = list(directory.rglob('*.yaml')) + list(directory.rglob('*.yml'))
        else:
            yaml_files = list(directory.glob('*.yaml')) + list(directory.glob('*.yml'))
        for file_path in yaml_files:
            files_validated += 1
            passed, findings = self.validate_file(file_path)
            if findings:
                artifacts_validated += 1
                all_findings.extend(findings)
        execution_time = time.time() - start_time
        has_errors = any(f.severity == ValidationSeverity.ERROR for f in all_findings)
        return ValidationResult(
            passed=not has_errors,
            findings=all_findings,
            files_validated=files_validated,
            artifacts_validated=artifacts_validated,
            execution_time=execution_time
        )
    def validate_workspace(self) -> ValidationResult:
        """Validate the entire workspace."""
        gl-platform.gl-platform.governance_path = self.workspace_path / 'workspace' / 'gl-platform.gl-platform.governance'
        if not gl-platform.gl-platform.governance_path.exists():
            return ValidationResult(
                passed=False,
                findings=[ValidationFinding(
                    rule_id="PATH-001",
                    rule_name="Governance Path",
                    severity=ValidationSeverity.ERROR,
                    message=f"Governance path not found: {gl-platform.gl-platform.governance_path}"
                )]
            )
        return self.validate_directory(gl-platform.gl-platform.governance_path)
    def generate_report(self, result: ValidationResult, format: str = 'markdown') -> str:
        """Generate validation report."""
        if format == 'json':
            return json.dumps(result.to_dict(), indent=2)
        elif format == 'yaml':
            return yaml.dump(result.to_dict(), default_flow_style=False)
        else:
            return self._generate_markdown_report(result)
    def _generate_markdown_report(self, result: ValidationResult) -> str:
        """Generate markdown report."""
        report = []
        report.append("# GL Validation Report\n")
        report.append(f"**Generated**: {datetime.utcnow().isoformat()}Z\n")
        # Summary
        status = "✅ PASSED" if result.passed else "❌ FAILED"
        report.append(f"## Summary: {status}\n")
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        report.append(f"| Files Validated | {result.files_validated} |")
        report.append(f"| Artifacts Validated | {result.artifacts_validated} |")
        report.append(f"| Errors | {result.error_count} |")
        report.append(f"| Warnings | {result.warning_count} |")
        report.append(f"| Info | {result.info_count} |")
        report.append(f"| Execution Time | {result.execution_time:.2f}s |")
        report.append("")
        # Findings by severity
        if result.error_count > 0:
            report.append("## ❌ Errors\n")
            for finding in result.findings:
                if finding.severity == ValidationSeverity.ERROR:
                    report.append(f"### {finding.rule_id}: {finding.rule_name}")
                    report.append(f"- **File**: {finding.file_path}")
                    report.append(f"- **Message**: {finding.message}")
                    if finding.field_path:
                        report.append(f"- **Field**: {finding.field_path}")
                    if finding.suggestion:
                        report.append(f"- **Suggestion**: {finding.suggestion}")
                    report.append("")
        if result.warning_count > 0:
            report.append("## ⚠️ Warnings\n")
            for finding in result.findings:
                if finding.severity == ValidationSeverity.WARNING:
                    report.append(f"- **{finding.rule_id}** ({finding.file_path}): {finding.message}")
            report.append("")
        if result.info_count > 0:
            report.append("## ℹ️ Info\n")
            for finding in result.findings:
                if finding.severity == ValidationSeverity.INFO:
                    report.append(f"- **{finding.rule_id}**: {finding.message}")
            report.append("")
        # Recommendations
        report.append("## Recommendations\n")
        if result.error_count > 0:
            report.append("1. ⚠️ Fix all errors before proceeding")
        if result.warning_count > 5:
            report.append("2. 📋 Address warnings to improve gl-platform.gl-platform.governance quality")
        if result.passed:
            report.append("✅ All validations passed!")
        return '\n'.join(report)
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='GL Artifacts Validator - Comprehensive Governance Artifact Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('path', nargs='?', default='.',
                        help='Path to validate (default: current directory)')
    parser.add_argument('--workspace', '-w', default='.',
                        help='Workspace root path')
    parser.add_argument('--format', '-f', choices=['markdown', 'json', 'yaml'],
                        default='markdown', help='Output format')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--strict', '-s', action='store_true',
                        help='Treat warnings as errors')
    args = parser.parse_args()
    # Create validator
    validator = GLValidator(args.workspace)
    # Validate
    target_path = Path(args.path)
    if target_path.is_file():
        passed, findings = validator.validate_file(target_path)
        result = ValidationResult(
            passed=passed,
            findings=findings,
            files_validated=1,
            artifacts_validated=1 if findings else 0
        )
    elif target_path.is_dir():
        result = validator.validate_directory(target_path)
    else:
        result = validator.validate_workspace()
    # Apply strict mode
    if args.strict and result.warning_count > 0:
        result.passed = False
    # Generate report
    report = validator.generate_report(result, args.format)
    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")
    else:
        print(report)
    # Exit code
    sys.exit(0 if result.passed else 1)
if __name__ == '__main__':
    main()