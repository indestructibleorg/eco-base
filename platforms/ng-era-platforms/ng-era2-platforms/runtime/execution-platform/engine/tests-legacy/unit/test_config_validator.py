#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: test_config_validator
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Unit tests for Configuration Validator
Tests YAML/JSON configuration validation functionality
"""
# MNGA-002: Import organization needs review
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
import yaml
# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from scripts.utils.config_validator import ConfigValidator
except ImportError:
    ConfigValidator = None
class TestConfigValidator(unittest.TestCase):
    """Test cases for ConfigValidator."""
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        if ConfigValidator is None:
            raise unittest.SkipTest("ConfigValidator not available")
    def setUp(self):
        """Set up test instance."""
        self.validator = ConfigValidator(strict=True)
        self.temp_dir = tempfile.mkdtemp()
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    def _create_yaml_file(self, filename: str, content: dict) -> str:
        """Create a temporary YAML file."""
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, 'w') as f:
            yaml.dump(content, f)
        return filepath
    def _create_json_file(self, filename: str, content: dict) -> str:
        """Create a temporary JSON file."""
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(content, f)
        return filepath
    def test_initialization(self):
        """Test ConfigValidator initialization."""
        validator = ConfigValidator(strict=True)
        self.assertTrue(validator.strict)
        self.assertEqual(validator.errors, [])
        self.assertEqual(validator.warnings, [])
    def test_valid_config(self):
        """Test validation of a valid configuration."""
        config = {
            'apiVersion': 'v1',
            'kind': 'Config',
            'metadata': {
                'name': 'test-config',
                'version': '1.0.0'
            },
            'spec': {
                'enabled': True
            }
        }
        filepath = self._create_yaml_file("valid.yaml", config)
        is_valid, errors, warnings = self.validator.validate_file(filepath)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    def test_missing_api_version(self):
        """Test validation fails when apiVersion is missing."""
        config = {
            'kind': 'Config',
            'metadata': {
                'name': 'test-config',
                'version': '1.0.0'
            }
        }
        filepath = self._create_yaml_file("missing_api.yaml", config)
        is_valid, errors, warnings = self.validator.validate_file(filepath)
        self.assertFalse(is_valid)
        self.assertTrue(any('apiVersion' in e for e in errors))
    def test_missing_kind(self):
        """Test validation fails when kind is missing."""
        config = {
            'apiVersion': 'v1',
            'metadata': {
                'name': 'test-config',
                'version': '1.0.0'
            }
        }
        filepath = self._create_yaml_file("missing_kind.yaml", config)
        is_valid, errors, warnings = self.validator.validate_file(filepath)
        self.assertFalse(is_valid)
        self.assertTrue(any('kind' in e for e in errors))
    def test_missing_metadata(self):
        """Test validation fails when metadata is missing."""
        config = {
            'apiVersion': 'v1',
            'kind': 'Config'
        }
        filepath = self._create_yaml_file("missing_metadata.yaml", config)
        is_valid, errors, warnings = self.validator.validate_file(filepath)
        self.assertFalse(is_valid)
        self.assertTrue(any('metadata' in e for e in errors))
    def test_invalid_name_format(self):
        """Test validation fails for invalid name format."""
        config = {
            'apiVersion': 'v1',
            'kind': 'Config',
            'metadata': {
                'name': 'Invalid_Name_With_Caps',  # Should be kebab-case
                'version': '1.0.0'
            }
        }
        filepath = self._create_yaml_file("invalid_name.yaml", config)
        is_valid, errors, warnings = self.validator.validate_file(filepath)
        self.assertFalse(is_valid)
    def test_invalid_version_format(self):
        """Test validation warns for invalid version format."""
        config = {
            'apiVersion': 'v1',
            'kind': 'Config',
            'metadata': {
                'name': 'test-config',
                'version': 'invalid'  # Should be semver
            }
        }
        filepath = self._create_yaml_file("invalid_version.yaml", config)
        is_valid, errors, warnings = self.validator.validate_file(filepath)
        # Invalid version generates a warning, not an error (design decision)
        self.assertTrue(is_valid)  # Still valid, but with warnings
        self.assertTrue(any('version' in w.lower() or 'semver' in w.lower() for w in warnings))
    def test_json_file_validation(self):
        """Test validation of JSON configuration files."""
        config = {
            'apiVersion': 'v1',
            'kind': 'Config',
            'metadata': {
                'name': 'json-config',
                'version': '1.0.0'
            }
        }
        filepath = self._create_json_file("config.json", config)
        is_valid, errors, warnings = self.validator.validate_file(filepath)
        self.assertTrue(is_valid)
    def test_nonexistent_file(self):
        """Test validation of non-existent file."""
        is_valid, errors, warnings = self.validator.validate_file("/nonexistent/file.yaml")
        self.assertFalse(is_valid)
        self.assertTrue(len(errors) > 0)
    def test_invalid_yaml_syntax(self):
        """Test validation of file with invalid YAML syntax."""
        filepath = os.path.join(self.temp_dir, "invalid.yaml")
        with open(filepath, 'w') as f:
            f.write("invalid: yaml: content: [")
        is_valid, errors, warnings = self.validator.validate_file(filepath)
        self.assertFalse(is_valid)
    def test_non_strict_mode(self):
        """Test validation in non-strict mode."""
        validator = ConfigValidator(strict=False)
        config = {
            'apiVersion': 'v1',
            'kind': 'Config',
            'metadata': {
                'name': 'test-config',
                'version': '1.0.0'
            }
        }
        filepath = self._create_yaml_file("non_strict.yaml", config)
        is_valid, errors, warnings = validator.validate_file(filepath)
        self.assertTrue(is_valid)
class TestConfigValidatorPatterns(unittest.TestCase):
    """Test ConfigValidator pattern matching."""
    @classmethod
    def setUpClass(cls):
        if ConfigValidator is None:
            raise unittest.SkipTest("ConfigValidator not available")
    def test_name_pattern_valid(self):
        """Test valid name patterns."""
        valid_names = [
            'my-config',
            'test-config-v1',
            'a',
            'config123',
            'my-long-config-name'
        ]
        for name in valid_names:
            self.assertTrue(
                ConfigValidator.NAME_PATTERN.match(name),
                f"Name '{name}' should be valid"
            )
    def test_name_pattern_invalid(self):
        """Test invalid name patterns."""
        invalid_names = [
            'MyConfig',  # Uppercase
            '123config',  # Starts with number
            'my_config',  # Underscore
            'my config',  # Space
            '-config',  # Starts with dash
        ]
        for name in invalid_names:
            self.assertIsNone(
                ConfigValidator.NAME_PATTERN.match(name),
                f"Name '{name}' should be invalid"
            )
    def test_version_pattern_valid(self):
        """Test valid version patterns."""
        valid_versions = [
            '1.0.0',
            '0.1.0',
            '10.20.30',
            '1.0.0-alpha',
            '2.0.0-beta1'
        ]
        for version in valid_versions:
            self.assertTrue(
                ConfigValidator.VERSION_PATTERN.match(version),
                f"Version '{version}' should be valid"
            )
    def test_version_pattern_invalid(self):
        """Test invalid version patterns."""
        invalid_versions = [
            'v1.0.0',  # Prefix
            '1.0',  # Missing patch
            '1',  # Only major
            'latest',  # Not semver
            '1.0.0.0',  # Too many parts
        ]
        for version in invalid_versions:
            self.assertIsNone(
                ConfigValidator.VERSION_PATTERN.match(version),
                f"Version '{version}' should be invalid"
            )
if __name__ == '__main__':
    unittest.main()