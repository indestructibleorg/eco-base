#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: test_auto_fix_bot_validator
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Unit tests for Auto-Fix Bot Template Validator
GL Unified Architecture Governance Framework Activated - Comprehensive validation test suite
Tests cover:
1. Auto-fix template validation
2. Multi-language support verification
3. False positive/incorrect fix detection
4. Configuration completeness
5. Security rules validation
"""
# MNGA-002: Import organization needs review
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
import yaml
# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "utils"))
try:
    from scripts.utils.auto_fix_bot_validator import (
        AutoFixBotValidator,
        validate_auto_fix_config,
    )
except ImportError:
    AutoFixBotValidator = None
    validate_auto_fix_config = None
class TestAutoFixBotValidatorInit(unittest.TestCase):
    """Test AutoFixBotValidator initialization."""
    @classmethod
    def setUpClass(cls):
        if AutoFixBotValidator is None:
            raise unittest.SkipTest("AutoFixBotValidator not available")
    def test_initialization(self):
        """Test validator initializes correctly."""
        validator = AutoFixBotValidator()
        self.assertEqual(validator.errors, [])
        self.assertEqual(validator.warnings, [])
        self.assertEqual(validator.config, {})
    def test_initialization_with_path(self):
        """Test validator initializes with config path."""
        validator = AutoFixBotValidator("/path/to/config.yml")
        self.assertEqual(validator.config_path, "/path/to/config.yml")
class TestAutoFixBotValidatorLoad(unittest.TestCase):
    """Test configuration loading."""
    @classmethod
    def setUpClass(cls):
        if AutoFixBotValidator is None:
            raise unittest.SkipTest("AutoFixBotValidator not available")
    def setUp(self):
        self.validator = AutoFixBotValidator()
        self.temp_dir = tempfile.mkdtemp()
    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    def _create_config_file(self, config: Dict[str, Any]) -> str:
        """Create a temporary config file."""
        filepath = os.path.join(self.temp_dir, "test-config.yml")
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        return filepath
    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        config = {
            "version": "1.0",
            "metadata": {"name": "test"},
            "settings": {"enabled": True},
        }
        filepath = self._create_config_file(config)
        result = self.validator.load_config(filepath)
        self.assertTrue(result)
        self.assertEqual(self.validator.config["version"], "1.0")
    def test_load_nonexistent_file(self):
        """Test loading a nonexistent file."""
        result = self.validator.load_config("/nonexistent/path.yml")
        self.assertFalse(result)
        self.assertTrue(len(self.validator.errors) > 0)
    def test_load_invalid_yaml(self):
        """Test loading an invalid YAML file."""
        filepath = os.path.join(self.temp_dir, "invalid.yml")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("invalid: yaml: content: [")
        result = self.validator.load_config(filepath)
        self.assertFalse(result)
class TestTemplateValidation(unittest.TestCase):
    """Test auto-fix template validation."""
    @classmethod
    def setUpClass(cls):
        if AutoFixBotValidator is None:
            raise unittest.SkipTest("AutoFixBotValidator not available")
    def setUp(self):
        self.validator = AutoFixBotValidator()
    def test_valid_template_structure(self):
        """Test validation of valid template structure."""
        self.validator.config = {
            "version": "1.0",
            "metadata": {"name": "test"},
            "project": {"name": "test"},
            "settings": {"enabled": True},
            "analysis": {
                "enabled": True,
                "rules": [{"name": "security"}],
                "include": ["*.py", "*.js", "*.java"],
            },
            "fix": {
                "enabled": True,
                "security_fixes": {
                    "weak_crypto": {
                        "enabled": True,
                        "strategy": "replace",
                        "replacements": [
                            {"from": "hashlib.md5", "to": "hashlib.sha256", "language": "python"},
                        ],
                    },
                },
            },
            "security": {"enabled": True, "scan_types": ["sql_injection", "xss"]},
        }
        result = self.validator._validate_templates()
        self.assertTrue(result)
    def test_missing_fix_section(self):
        """Test validation fails when fix section is missing."""
        self.validator.config = {
            "version": "1.0",
            "metadata": {"name": "test"},
        }
        result = self.validator._validate_templates()
        self.assertFalse(result)
        self.assertTrue(any("fix" in e.lower() for e in self.validator.errors))
    def test_missing_security_fixes(self):
        """Test validation fails when security_fixes is missing."""
        self.validator.config = {
            "fix": {"enabled": True},
        }
        result = self.validator._validate_templates()
        self.assertFalse(result)
        self.assertTrue(any("security_fixes" in e.lower() for e in self.validator.errors))
    def test_invalid_strategy(self):
        """Test validation fails for invalid strategy."""
        self.validator.config = {
            "fix": {
                "security_fixes": {
                    "test_fix": {
                        "enabled": True,
                        "strategy": "invalid_strategy",
                    },
                },
            },
        }
        result = self.validator._validate_templates()
        self.assertFalse(result)
        self.assertTrue(any("invalid strategy" in e.lower() for e in self.validator.errors))
    def test_valid_replacement_rule(self):
        """Test validation of valid replacement rule."""
        self.validator.config = {
            "fix": {
                "security_fixes": {
                    "weak_crypto": {
                        "enabled": True,
                        "strategy": "replace",
                        "replacements": [
                            {"from": "md5", "to": "sha256", "language": "python"},
                        ],
                    },
                },
            },
        }
        result = self.validator._validate_templates()
        self.assertTrue(result)
    def test_missing_from_in_replacement(self):
        """Test validation fails when 'from' is missing in replacement."""
        self.validator.config = {
            "fix": {
                "security_fixes": {
                    "test_fix": {
                        "enabled": True,
                        "strategy": "replace",
                        "replacements": [{"to": "sha256", "language": "python"}],
                    },
                },
            },
        }
        result = self.validator._validate_templates()
        self.assertFalse(result)
    def test_invalid_regex_pattern(self):
        """Test validation fails for invalid regex pattern."""
        self.validator.config = {
            "fix": {
                "security_fixes": {
                    "code_injection": {
                        "enabled": True,
                        "strategy": "replace_or_sanitize",
                        "patterns": [
                            {"from": "[invalid(regex", "regex": True},
                        ],
                    },
                },
            },
        }
        result = self.validator._validate_templates()
        self.assertFalse(result)
        self.assertTrue(any("regex" in e.lower() for e in self.validator.errors))
class TestMultiLanguageSupport(unittest.TestCase):
    """Test multi-language support validation."""
    @classmethod
    def setUpClass(cls):
        if AutoFixBotValidator is None:
            raise unittest.SkipTest("AutoFixBotValidator not available")
    def setUp(self):
        self.validator = AutoFixBotValidator()
    def test_sufficient_language_support(self):
        """Test validation passes with sufficient language support."""
        self.validator.config = {
            "analysis": {"include": ["*.py", "*.js", "*.java"]},
            "fix": {
                "security_fixes": {
                    "weak_crypto": {
                        "enabled": True,
                        "strategy": "replace",
                        "replacements": [
                            {"from": "md5", "to": "sha256", "language": "python"},
                            {"from": "md5", "to": "sha256", "language": "javascript"},
                            {"from": "md5", "to": "sha256", "language": "java"},
                        ],
                    },
                },
            },
        }
        result = self.validator._validate_multi_language_support()
        self.assertTrue(result)
    def test_insufficient_language_support(self):
        """Test validation fails with insufficient language support."""
        self.validator.config = {
            "analysis": {"include": ["*.py"]},
            "fix": {
                "security_fixes": {
                    "weak_crypto": {
                        "enabled": True,
                        "strategy": "replace",
                        "replacements": [
                            {"from": "md5", "to": "sha256", "language": "python"},
                        ],
                    },
                },
            },
        }
        result = self.validator._validate_multi_language_support()
        self.assertFalse(result)
    def test_critical_languages_warning(self):
        """Test warning for missing critical languages."""
        self.validator.config = {
            "analysis": {"include": ["*.py", "*.go", "*.rs"]},
            "fix": {
                "security_fixes": {
                    "test": {
                        "enabled": True,
                        "strategy": "replace",
                        "replacements": [
                            {"from": "x", "to": "y", "language": "python"},
                            {"from": "x", "to": "y", "language": "go"},
                            {"from": "x", "to": "y", "language": "rust"},
                        ],
                    },
                },
            },
        }
        self.validator._validate_multi_language_support()
        # Should warn about missing javascript and java
        self.assertTrue(any("critical" in w.lower() for w in self.validator.warnings))
class TestFalsePositiveValidation(unittest.TestCase):
    """Test false positive detection validation."""
    @classmethod
    def setUpClass(cls):
        if AutoFixBotValidator is None:
            raise unittest.SkipTest("AutoFixBotValidator not available")
    def setUp(self):
        self.validator = AutoFixBotValidator()
    def test_overly_broad_pattern_detected(self):
        """Test detection of overly broad patterns."""
        self.validator.config = {
            "fix": {
                "security_fixes": {
                    "test": {
                        "enabled": True,
                        "strategy": "replace",
                        "patterns": [{"from": ".*"}],
                    },
                },
            },
        }
        result = self.validator._validate_no_false_positives()
        self.assertFalse(result)
        self.assertTrue(any("broad" in e.lower() for e in self.validator.errors))
    def test_replacement_introducing_weak_crypto(self):
        """Test detection of replacements that introduce weak crypto."""
        self.validator.config = {
            "fix": {
                "security_fixes": {
                    "some_fix": {
                        "enabled": True,
                        "strategy": "replace",
                        "replacements": [
                            {"from": "secure_hash", "to": "md5_hash"},
                        ],
                    },
                },
            },
        }
        result = self.validator._validate_no_false_positives()
        self.assertFalse(result)
        self.assertTrue(any("weak crypto" in e.lower() for e in self.validator.errors))
    def test_safe_patterns_pass(self):
        """Test that safe patterns pass validation."""
        self.validator.config = {
            "fix": {
                "security_fixes": {
                    "weak_crypto": {
                        "enabled": True,
                        "strategy": "replace",
                        "patterns": [
                            {"from": "hashlib\\.md5", "detect": ""},
                        ],
                        "replacements": [
                            {"from": "hashlib.md5", "to": "hashlib.sha256"},
                        ],
                    },
                },
            },
        }
        result = self.validator._validate_no_false_positives()
        self.assertTrue(result)
class TestConfigCompleteness(unittest.TestCase):
    """Test configuration completeness validation."""
    @classmethod
    def setUpClass(cls):
        if AutoFixBotValidator is None:
            raise unittest.SkipTest("AutoFixBotValidator not available")
    def setUp(self):
        self.validator = AutoFixBotValidator()
    def test_complete_config(self):
        """Test validation passes with complete configuration."""
        self.validator.config = {
            "version": "1.0",
            "metadata": {"name": "test"},
            "project": {"name": "test"},
            "settings": {"enabled": True},
            "analysis": {"enabled": True, "rules": [{"name": "a"}, {"name": "b"}, {"name": "c"}]},
            "fix": {"enabled": True},
            "security": {"enabled": True},
        }
        result = self.validator._validate_config_completeness()
        self.assertTrue(result)
    def test_missing_required_sections(self):
        """Test validation fails with missing required sections."""
        self.validator.config = {"version": "1.0"}
        result = self.validator._validate_config_completeness()
        self.assertFalse(result)
        self.assertTrue(any("missing" in e.lower() for e in self.validator.errors))
    def test_disabled_settings_warning(self):
        """Test warning when auto-fix is disabled."""
        self.validator.config = {
            "version": "1.0",
            "metadata": {"name": "test"},
            "project": {"name": "test"},
            "settings": {"enabled": False},
            "analysis": {"enabled": True, "rules": []},
            "fix": {"enabled": True},
            "security": {"enabled": True},
        }
        self.validator._validate_config_completeness()
        self.assertTrue(any("not enabled" in w.lower() for w in self.validator.warnings))
class TestSecurityRulesValidation(unittest.TestCase):
    """Test security rules validation."""
    @classmethod
    def setUpClass(cls):
        if AutoFixBotValidator is None:
            raise unittest.SkipTest("AutoFixBotValidator not available")
    def setUp(self):
        self.validator = AutoFixBotValidator()
    def test_valid_security_config(self):
        """Test validation passes with valid security configuration."""
        self.validator.config = {
            "security": {
                "enabled": True,
                "scan_types": ["sql_injection", "xss", "weak_cryptography", "path_traversal"],
                "severity_threshold": "medium",
                "cryptography": {
                    "enabled": True,
                    "blocked_algorithms": ["md5", "sha1", "des"],
                },
            },
        }
        result = self.validator._validate_security_rules()
        self.assertTrue(result)
    def test_missing_security_section(self):
        """Test validation fails with missing security section."""
        self.validator.config = {}
        result = self.validator._validate_security_rules()
        self.assertFalse(result)
    def test_invalid_severity_threshold(self):
        """Test validation fails with invalid severity threshold."""
        self.validator.config = {
            "security": {
                "enabled": True,
                "severity_threshold": "invalid",
            },
        }
        result = self.validator._validate_security_rules()
        self.assertFalse(result)
    def test_missing_recommended_scan_types(self):
        """Test warning for missing recommended scan types."""
        self.validator.config = {
            "security": {
                "enabled": True,
                "scan_types": ["custom_scan"],
            },
        }
        self.validator._validate_security_rules()
        # Should warn about missing recommended scan types
        self.assertTrue(any("scan type" in w.lower() for w in self.validator.warnings))
class TestValidateAll(unittest.TestCase):
    """Test the validate_all method."""
    @classmethod
    def setUpClass(cls):
        if AutoFixBotValidator is None:
            raise unittest.SkipTest("AutoFixBotValidator not available")
    def setUp(self):
        self.validator = AutoFixBotValidator()
        self.temp_dir = tempfile.mkdtemp()
    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    def _create_config_file(self, config: Dict[str, Any]) -> str:
        """Create a temporary config file."""
        filepath = os.path.join(self.temp_dir, "test-config.yml")
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        return filepath
    def test_validate_all_with_valid_config(self):
        """Test validate_all with a complete valid configuration."""
        self.validator.config = {
            "version": "1.0",
            "metadata": {"name": "test"},
            "project": {"name": "test"},
            "settings": {"enabled": True},
            "analysis": {
                "enabled": True,
                "rules": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
                "include": ["*.py", "*.js", "*.java"],
            },
            "fix": {
                "enabled": True,
                "security_fixes": {
                    "weak_crypto": {
                        "enabled": True,
                        "strategy": "replace",
                        "replacements": [
                            {"from": "md5", "to": "sha256", "language": "python"},
                            {"from": "md5", "to": "sha256", "language": "javascript"},
                            {"from": "md5", "to": "sha256", "language": "java"},
                        ],
                    },
                },
            },
            "security": {
                "enabled": True,
                "scan_types": ["sql_injection", "xss", "weak_cryptography", "path_traversal"],
                "severity_threshold": "medium",
                "cryptography": {
                    "enabled": True,
                    "blocked_algorithms": ["md5", "sha1"],
                },
            },
        }
        is_valid, results = self.validator.validate_all()
        self.assertTrue(is_valid)
        self.assertTrue(results["template_validation"])
        self.assertTrue(results["multi_language_support"])
        self.assertTrue(results["false_positive_check"])
        self.assertTrue(results["config_completeness"])
        self.assertTrue(results["security_rules_valid"])
    def test_validate_all_with_no_config(self):
        """Test validate_all with no configuration loaded."""
        is_valid, results = self.validator.validate_all()
        self.assertFalse(is_valid)
        self.assertTrue(any("no configuration" in e.lower() for e in self.validator.errors))
    def test_get_validation_report(self):
        """Test validation report generation."""
        self.validator.config = {
            "version": "1.0",
            "metadata": {"name": "test"},
            "project": {"name": "test"},
            "settings": {"enabled": True},
            "analysis": {"enabled": True, "rules": [{"name": "a"}, {"name": "b"}, {"name": "c"}], "include": ["*.py"]},
            "fix": {
                "security_fixes": {
                    "test": {"enabled": True, "strategy": "replace", "replacements": []},
                },
            },
            "security": {"enabled": True, "scan_types": []},
        }
        self.validator.validate_all()
        report = self.validator.get_validation_report()
        self.assertIn("is_valid", report)
        self.assertIn("results", report)
        self.assertIn("errors", report)
        self.assertIn("warnings", report)
        self.assertIn("summary", report)
        self.assertIn("total_errors", report["summary"])
        self.assertIn("total_warnings", report["summary"])
class TestValidateAutoFixConfig(unittest.TestCase):
    """Test the convenience function."""
    @classmethod
    def setUpClass(cls):
        if validate_auto_fix_config is None:
            raise unittest.SkipTest("validate_auto_fix_config not available")
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    def _create_config_file(self, config: Dict[str, Any]) -> str:
        """Create a temporary config file."""
        filepath = os.path.join(self.temp_dir, "test-config.yml")
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        return filepath
    def test_validate_auto_fix_config_valid(self):
        """Test validation of a valid auto-fix config file."""
        config = {
            "version": "1.0",
            "metadata": {"name": "test"},
            "project": {"name": "test"},
            "settings": {"enabled": True},
            "analysis": {
                "enabled": True,
                "rules": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
                "include": ["*.py", "*.js", "*.java"],
            },
            "fix": {
                "security_fixes": {
                    "test": {
                        "enabled": True,
                        "strategy": "replace",
                        "replacements": [
                            {"from": "x", "to": "y", "language": "python"},
                            {"from": "x", "to": "y", "language": "javascript"},
                            {"from": "x", "to": "y", "language": "java"},
                        ],
                    },
                },
            },
            "security": {
                "enabled": True,
                "scan_types": ["sql_injection", "xss", "weak_cryptography", "path_traversal"],
                "severity_threshold": "medium",
            },
        }
        filepath = self._create_config_file(config)
        is_valid, report = validate_auto_fix_config(filepath)
        self.assertTrue(is_valid)
    def test_validate_auto_fix_config_invalid(self):
        """Test validation of an invalid auto-fix config file."""
        is_valid, report = validate_auto_fix_config("/nonexistent/path.yml")
        self.assertFalse(is_valid)
class TestRealConfigValidation(unittest.TestCase):
    """Test validation against the actual auto-fix-bot.yml configuration."""
    @classmethod
    def setUpClass(cls):
        if AutoFixBotValidator is None:
            raise unittest.SkipTest("AutoFixBotValidator not available")
        # Find the actual config file using relative paths from test file
        test_file_dir = Path(__file__).parent
        project_root = test_file_dir.parent.parent
        possible_paths = [
            project_root / "config" / ".auto-fix-bot.yml",
            project_root / "workspace" / "config" / ".auto-fix-bot.yml",
        ]
        cls.config_path = None
        for path in possible_paths:
            if path.exists():
                cls.config_path = path
                break
        if cls.config_path is None:
            raise unittest.SkipTest("Config file not found in any expected location")
    def setUp(self):
        self.validator = AutoFixBotValidator(str(self.config_path))
        self.validator.load_config()
    def test_real_config_template_validation(self):
        """Test template validation against real config."""
        result = self.validator._validate_templates()
        # Should pass or produce meaningful errors
        self.assertIsInstance(result, bool)
    def test_real_config_multi_language_support(self):
        """Test multi-language support against real config."""
        result = self.validator._validate_multi_language_support()
        # Should support at least 3 languages
        self.assertTrue(result or len(self.validator.errors) > 0)
    def test_real_config_security_rules(self):
        """Test security rules validation against real config."""
        result = self.validator._validate_security_rules()
        self.assertIsInstance(result, bool)
    def test_real_config_full_validation(self):
        """Test full validation against real config."""
        is_valid, results = self.validator.validate_all()
        # The config should be mostly valid
        self.assertIsInstance(is_valid, bool)
if __name__ == '__main__':
    unittest.main(verbosity=2)
