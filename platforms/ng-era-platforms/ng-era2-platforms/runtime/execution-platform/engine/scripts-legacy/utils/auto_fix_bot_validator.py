#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: auto_fix_bot_validator
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Auto-Fix Bot Template Validator
Validates auto-fix templates for correctness, multi-language support,
and ensures no false positives or incorrect fixes.
GL Unified Architecture Governance Framework Activated - Governance Layer Validation
"""
# MNGA-002: Import organization needs review
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
class AutoFixBotValidator:
    """Validates Auto-Fix Bot configuration and templates."""
    # Supported programming languages
    SUPPORTED_LANGUAGES = [
        "python", "javascript", "typescript", "java", "go",
        "rust", "cpp", "c", "cs", "ruby", "php"
    ]
    # Required fields for security fix patterns
    REQUIRED_SECURITY_FIX_FIELDS = ["enabled", "strategy"]
    # Valid fix strategies
    VALID_STRATEGIES = [
        "replace", "add_validation", "remove_or_env",
        "mask_or_remove", "replace_or_sanitize"
    ]
    # Valid severity levels
    VALID_SEVERITIES = ["critical", "high", "medium", "low"]
    def __init__(self, config_path: Optional[str] = None):
        """Initialize validator with optional config path."""
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.validation_results: Dict[str, Any] = {}
    def load_config(self, config_path: Optional[str] = None) -> bool:
        """Load the auto-fix bot configuration file."""
        path = Path(config_path or self.config_path)
        if not path.exists():
            self.errors.append(f"Configuration file not found: {path}")
            return False
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            return True
        except yaml.YAMLError as e:
            self.errors.append(f"YAML parsing error: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Error loading config: {e}")
            return False
    def validate_all(self) -> Tuple[bool, Dict[str, Any]]:
        """Run all validation checks."""
        self.errors = []
        self.warnings = []
        self.validation_results = {
            "template_validation": False,
            "multi_language_support": False,
            "false_positive_check": False,
            "config_completeness": False,
            "security_rules_valid": False,
        }
        if not self.config:
            self.errors.append("No configuration loaded")
            return False, self.validation_results
        # Run all validations
        self.validation_results["template_validation"] = self._validate_templates()
        self.validation_results["multi_language_support"] = self._validate_multi_language_support()
        self.validation_results["false_positive_check"] = self._validate_no_false_positives()
        self.validation_results["config_completeness"] = self._validate_config_completeness()
        self.validation_results["security_rules_valid"] = self._validate_security_rules()
        all_valid = all(self.validation_results.values())
        return all_valid, self.validation_results
    def _validate_templates(self) -> bool:
        """Validate all auto-fix templates for correctness."""
        is_valid = True
        # Check fix configuration exists
        fix_config = self.config.get("fix", {})
        if not fix_config:
            self.errors.append("Missing 'fix' configuration section")
            return False
        # Validate security_fixes section
        security_fixes = fix_config.get("security_fixes", {})
        if not security_fixes:
            self.errors.append("Missing 'security_fixes' in fix configuration")
            return False
        # Validate each security fix type
        for fix_type, fix_config_item in security_fixes.items():
            if not self._validate_fix_type(fix_type, fix_config_item):
                is_valid = False
        return is_valid
    def _validate_fix_type(self, fix_type: str, fix_config: Dict[str, Any]) -> bool:
        """Validate a specific fix type configuration."""
        is_valid = True
        # Check required fields
        for field in self.REQUIRED_SECURITY_FIX_FIELDS:
            if field not in fix_config:
                self.errors.append(f"Missing required field '{field}' in {fix_type}")
                is_valid = False
        # Validate strategy
        strategy = fix_config.get("strategy", "")
        if strategy and strategy not in self.VALID_STRATEGIES:
            self.errors.append(f"Invalid strategy '{strategy}' in {fix_type}")
            is_valid = False
        # Validate replacements if present
        replacements = fix_config.get("replacements", [])
        for idx, replacement in enumerate(replacements):
            if not self._validate_replacement(fix_type, idx, replacement):
                is_valid = False
        # Validate patterns if present
        patterns = fix_config.get("patterns", [])
        for idx, pattern in enumerate(patterns):
            if not self._validate_pattern(fix_type, idx, pattern):
                is_valid = False
        return is_valid
    def _validate_replacement(
        self, fix_type: str, idx: int, replacement: Dict[str, Any]
    ) -> bool:
        """Validate a replacement rule."""
        is_valid = True
        # Check for 'from' and 'to' fields
        if "from" not in replacement:
            self.errors.append(
                f"Missing 'from' field in {fix_type} replacement #{idx + 1}"
            )
            is_valid = False
        if "to" not in replacement:
            self.errors.append(
                f"Missing 'to' field in {fix_type} replacement #{idx + 1}"
            )
            is_valid = False
        # Check language if present
        language = replacement.get("language", "")
        if language and language not in self.SUPPORTED_LANGUAGES:
            self.warnings.append(
                f"Unsupported language '{language}' in {fix_type} replacement #{idx + 1}"
            )
        return is_valid
    def _validate_pattern(
        self, fix_type: str, idx: int, pattern: Dict[str, Any]
    ) -> bool:
        """Validate a pattern rule."""
        is_valid = True
        # Check for required fields based on pattern type
        if "language" in pattern:
            language = pattern["language"]
            if language not in self.SUPPORTED_LANGUAGES:
                self.warnings.append(
                    f"Unsupported language '{language}' in {fix_type} pattern #{idx + 1}"
                )
        # Validate regex patterns if present
        if pattern.get("regex", False):
            from_pattern = pattern.get("from", "")
            if from_pattern:
                try:
                    re.compile(from_pattern)
                except re.error as e:
                    self.errors.append(
                        f"Invalid regex pattern in {fix_type} pattern #{idx + 1}: {e}"
                    )
                    is_valid = False
        # Validate validation_template if present
        if "validation_template" in pattern:
            template = pattern["validation_template"]
            if not template or not isinstance(template, str):
                self.errors.append(
                    f"Invalid validation_template in {fix_type} pattern #{idx + 1}"
                )
                is_valid = False
        return is_valid
    def _validate_multi_language_support(self) -> bool:
        """Validate multi-language support completeness."""
        is_valid = True
        supported_langs_in_config = set()
        # Check analysis include patterns
        analysis = self.config.get("analysis", {})
        include_patterns = analysis.get("include", [])
        # Map file extensions to languages
        ext_to_lang = {
            "*.py": "python",
            "*.js": "javascript",
            "*.ts": "typescript",
            "*.java": "java",
            "*.go": "go",
            "*.rs": "rust",
            "*.cpp": "cpp",
            "*.c": "c",
            "*.cs": "cs",
            "*.rb": "ruby",
            "*.php": "php",
        }
        for pattern in include_patterns:
            if pattern in ext_to_lang:
                supported_langs_in_config.add(ext_to_lang[pattern])
        # Check security_fixes for language coverage
        fix_config = self.config.get("fix", {})
        security_fixes = fix_config.get("security_fixes", {})
        languages_with_fixes = set()
        for fix_type, fix_data in security_fixes.items():
            # Check replacements
            for replacement in fix_data.get("replacements", []):
                lang = replacement.get("language", "")
                if lang:
                    languages_with_fixes.add(lang)
            # Check patterns
            for pattern in fix_data.get("patterns", []):
                lang = pattern.get("language", "")
                if lang:
                    languages_with_fixes.add(lang)
        # Verify critical languages have fix support
        critical_languages = {"python", "javascript", "java"}
        missing_critical = critical_languages - languages_with_fixes
        if missing_critical:
            self.warnings.append(
                f"Missing auto-fix support for critical languages: {missing_critical}"
            )
        # Check if at least 3 languages are supported
        if len(languages_with_fixes) < 3:
            self.errors.append(
                f"Insufficient multi-language support: only {len(languages_with_fixes)} languages"
            )
            is_valid = False
        self.validation_results["languages_supported"] = list(languages_with_fixes)
        return is_valid
    def _validate_no_false_positives(self) -> bool:
        """Validate that patterns won't produce false positives or incorrect fixes."""
        is_valid = True
        fix_config = self.config.get("fix", {})
        security_fixes = fix_config.get("security_fixes", {})
        for fix_type, fix_data in security_fixes.items():
            # Check for overly broad patterns
            for pattern in fix_data.get("patterns", []):
                from_pattern = pattern.get("from", "")
                detect_pattern = pattern.get("detect", "")
                check_pattern = from_pattern or detect_pattern
                if check_pattern:
                    # Check for overly broad patterns
                    if check_pattern in [".", ".*", ".+", "\\w+", "\\d+"]:
                        self.errors.append(
                            f"Overly broad pattern in {fix_type}: '{check_pattern}'"
                        )
                        is_valid = False
                    # Check for patterns that might match comments
                    if not pattern.get("skip_comments", True):
                        self.warnings.append(
                            f"Pattern in {fix_type} might match comments"
                        )
            # Validate replacements don't introduce new issues
            for replacement in fix_data.get("replacements", []):
                to_value = replacement.get("to", "")
                # Check if replacement introduces weak crypto
                weak_crypto = ["md5", "sha1", "des", "3des", "rc4"]
                for weak in weak_crypto:
                    if weak in to_value.lower() and fix_type != "weak_crypto":
                        self.errors.append(
                            f"Replacement in {fix_type} may introduce weak crypto: {weak}"
                        )
                        is_valid = False
        return is_valid
    def _validate_config_completeness(self) -> bool:
        """Validate configuration completeness."""
        is_valid = True
        required_sections = [
            "version",
            "metadata",
            "project",
            "settings",
            "analysis",
            "fix",
            "security",
        ]
        for section in required_sections:
            if section not in self.config:
                self.errors.append(f"Missing required section: {section}")
                is_valid = False
        # Validate settings
        settings = self.config.get("settings", {})
        if not settings.get("enabled"):
            self.warnings.append("Auto-fix bot is not enabled in settings")
        # Validate analysis rules
        analysis = self.config.get("analysis", {})
        rules = analysis.get("rules", [])
        if len(rules) < 3:
            self.warnings.append(f"Only {len(rules)} analysis rules defined")
        return is_valid
    def _validate_security_rules(self) -> bool:
        """Validate security scanning rules."""
        is_valid = True
        security = self.config.get("security", {})
        if not security:
            self.errors.append("Missing security configuration")
            return False
        # Check scan types
        scan_types = security.get("scan_types", [])
        required_scan_types = [
            "sql_injection",
            "xss",
            "weak_cryptography",
            "path_traversal",
        ]
        for scan_type in required_scan_types:
            if scan_type not in scan_types:
                self.warnings.append(f"Missing recommended scan type: {scan_type}")
        # Validate severity threshold
        severity = security.get("severity_threshold", "")
        if severity and severity not in self.VALID_SEVERITIES:
            self.errors.append(f"Invalid severity threshold: {severity}")
            is_valid = False
        # Validate cryptography policy
        crypto = security.get("cryptography", {})
        if crypto.get("enabled"):
            blocked = crypto.get("blocked_algorithms", [])
            required_blocked = ["md5", "sha1"]
            for algo in required_blocked:
                if algo not in blocked:
                    self.warnings.append(f"Weak algorithm not blocked: {algo}")
        return is_valid
    def get_validation_report(self) -> Dict[str, Any]:
        """Generate a comprehensive validation report."""
        return {
            "is_valid": all(self.validation_results.values()),
            "results": self.validation_results,
            "errors": self.errors,
            "warnings": self.warnings,
            "summary": {
                "total_errors": len(self.errors),
                "total_warnings": len(self.warnings),
                "validations_passed": sum(
                    1 for v in self.validation_results.values() if v
                ),
                "validations_failed": sum(
                    1 for v in self.validation_results.values() if not v
                ),
            },
        }
def validate_auto_fix_config(config_path: str) -> Tuple[bool, Dict[str, Any]]:
    """Convenience function to validate an auto-fix bot configuration file."""
    validator = AutoFixBotValidator(config_path)
    if not validator.load_config():
        return False, validator.get_validation_report()
    is_valid, _ = validator.validate_all()
    return is_valid, validator.get_validation_report()
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python auto_fix_bot_validator.py <config_path>")
        sys.exit(1)
    config_path = sys.argv[1]
    is_valid, report = validate_auto_fix_config(config_path)
    print("=" * 60)
    print("AUTO-FIX BOT CONFIGURATION VALIDATION REPORT")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Valid: {is_valid}")
    print()
    print("Validation Results:")
    for check, passed in report["results"].items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
    if report["errors"]:
        print("\nErrors:")
        for error in report["errors"]:
            print(f"  ❌ {error}")
    if report["warnings"]:
        print("\nWarnings:")
        for warning in report["warnings"]:
            print(f"  ⚠️ {warning}")
    print("\nSummary:")
    print(f"  Passed: {report['summary']['validations_passed']}")
    print(f"  Failed: {report['summary']['validations_failed']}")
    print(f"  Errors: {report['summary']['total_errors']}")
    print(f"  Warnings: {report['summary']['total_warnings']}")
    sys.exit(0 if is_valid else 1)
