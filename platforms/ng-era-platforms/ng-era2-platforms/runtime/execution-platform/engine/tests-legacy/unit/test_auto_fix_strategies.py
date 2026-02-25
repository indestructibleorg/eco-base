#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: test_auto_fix_strategies
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Unit tests for Auto-Fix Bot Strategies
Tests the auto-fix strategies defined in .auto-fix-bot.yml
GL Layer: GL30-49 Execution Layer
GL Unified Architecture Governance Framework: Activated
"""
import sys
import unittest
from pathlib import Path
import yaml
# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
class TestAutoFixStrategies(unittest.TestCase):
    """Test cases for Auto-Fix Bot strategies validation."""
    @classmethod
    def setUpClass(cls):
        """Load the auto-fix bot configuration."""
        config_path = Path(__file__).parent.parent.parent / 'config' / '.auto-fix-bot.yml'
        if not config_path.exists():
            raise unittest.SkipTest(f"Config file not found: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            cls.config = yaml.safe_load(f)
    def test_config_structure(self):
        """Test that config has required top-level keys."""
        required_keys = ['version', 'metadata', 'project', 'settings', 'analysis', 'fix', 'security']
        for key in required_keys:
            self.assertIn(key, self.config, f"Missing required key: {key}")
    def test_security_fixes_defined(self):
        """Test that security fixes are properly defined."""
        self.assertIn('fix', self.config)
        self.assertIn('security_fixes', self.config['fix'])
        security_fixes = self.config['fix']['security_fixes']
        expected_fixes = ['weak_crypto', 'path_traversal', 'debug_mode', 'sensitive_logging', 'code_injection']
        for fix in expected_fixes:
            self.assertIn(fix, security_fixes, f"Missing security fix: {fix}")
    def test_weak_crypto_strategy(self):
        """Test weak cryptography fix strategy (Alert #54)."""
        weak_crypto = self.config['fix']['security_fixes']['weak_crypto']
        self.assertTrue(weak_crypto['enabled'])
        self.assertEqual(weak_crypto['strategy'], 'replace')
        self.assertIn('replacements', weak_crypto)
        # Check Python replacement
        python_replacement = next(
            (r for r in weak_crypto['replacements'] if r['language'] == 'python'),
            None
        )
        self.assertIsNotNone(python_replacement, "Missing Python weak crypto replacement")
        self.assertEqual(python_replacement['from'], 'hashlib.md5')
        self.assertEqual(python_replacement['to'], 'hashlib.sha256')
        # Check JavaScript replacement
        js_replacement = next(
            (r for r in weak_crypto['replacements'] if r['language'] == 'javascript'),
            None
        )
        self.assertIsNotNone(js_replacement, "Missing JavaScript weak crypto replacement")
        # Check Java replacement
        java_replacement = next(
            (r for r in weak_crypto['replacements'] if r['language'] == 'java'),
            None
        )
        self.assertIsNotNone(java_replacement, "Missing Java weak crypto replacement")
    def test_path_traversal_strategy(self):
        """Test path traversal fix strategy (Alert #45)."""
        path_traversal = self.config['fix']['security_fixes']['path_traversal']
        self.assertTrue(path_traversal['enabled'])
        self.assertEqual(path_traversal['strategy'], 'add_validation')
        self.assertIn('patterns', path_traversal)
        # Check TypeScript pattern
        ts_pattern = next(
            (p for p in path_traversal['patterns'] if p['language'] == 'typescript'),
            None
        )
        self.assertIsNotNone(ts_pattern, "Missing TypeScript path traversal pattern")
        self.assertTrue(ts_pattern['add_validation'])
        # Check Python pattern
        py_pattern = next(
            (p for p in path_traversal['patterns'] if p['language'] == 'python'),
            None
        )
        self.assertIsNotNone(py_pattern, "Missing Python path traversal pattern")
        self.assertTrue(py_pattern['add_validation'])
    def test_debug_mode_strategy(self):
        """Test debug mode fix strategy (Alert #53)."""
        debug_mode = self.config['fix']['security_fixes']['debug_mode']
        self.assertTrue(debug_mode['enabled'])
        self.assertEqual(debug_mode['strategy'], 'remove_or_env')
        self.assertIn('patterns', debug_mode)
        # Check Python Flask pattern
        py_flask = next(
            (p for p in debug_mode['patterns'] 
             if p['language'] == 'python' and 'app.run' in p.get('from', '')),
            None
        )
        self.assertIsNotNone(py_flask, "Missing Python Flask debug mode pattern")
        self.assertIn('os.getenv', py_flask['to'])
    def test_sensitive_logging_strategy(self):
        """Test sensitive logging fix strategy (Alert #49)."""
        sensitive_logging = self.config['fix']['security_fixes']['sensitive_logging']
        self.assertTrue(sensitive_logging['enabled'])
        self.assertEqual(sensitive_logging['strategy'], 'mask_or_remove')
        self.assertIn('patterns', sensitive_logging)
        # Check password masking
        password_pattern = next(
            (p for p in sensitive_logging['patterns'] if 'password' in p.get('detect', '')),
            None
        )
        self.assertIsNotNone(password_pattern, "Missing password masking pattern")
        self.assertEqual(password_pattern['action'], 'mask')
    def test_code_injection_strategy(self):
        """Test code injection fix strategy (Alert #37)."""
        code_injection = self.config['fix']['security_fixes']['code_injection']
        self.assertTrue(code_injection['enabled'])
        self.assertEqual(code_injection['strategy'], 'replace_or_sanitize')
        self.assertIn('patterns', code_injection)
        # Check eval removal
        eval_pattern = next(
            (p for p in code_injection['patterns'] if 'eval' in p.get('from', '')),
            None
        )
        self.assertIsNotNone(eval_pattern, "Missing eval removal pattern")
        self.assertTrue(eval_pattern.get('regex', False))
        # Check exec removal
        exec_pattern = next(
            (p for p in code_injection['patterns'] if 'exec' in p.get('from', '')),
            None
        )
        self.assertIsNotNone(exec_pattern, "Missing exec removal pattern")
    def test_multi_language_support(self):
        """Test multi-language support coverage."""
        expected_languages = ['python', 'javascript', 'typescript', 'java']
        # Collect all languages from security fixes
        languages_found = set()
        security_fixes = self.config['fix']['security_fixes']
        for fix_name, fix_config in security_fixes.items():
            if 'replacements' in fix_config:
                for r in fix_config['replacements']:
                    if 'language' in r:
                        languages_found.add(r['language'])
            if 'patterns' in fix_config:
                for p in fix_config['patterns']:
                    if 'language' in p:
                        languages_found.add(p['language'])
        for lang in expected_languages:
            self.assertIn(lang, languages_found, f"Missing language support: {lang}")
class TestAutoFixTemplates(unittest.TestCase):
    """Test the auto-fix template correctness."""
    @classmethod
    def setUpClass(cls):
        """Load the auto-fix bot configuration."""
        config_path = Path(__file__).parent.parent.parent / 'config' / '.auto-fix-bot.yml'
        if not config_path.exists():
            raise unittest.SkipTest(f"Config file not found: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            cls.config = yaml.safe_load(f)
    def test_python_path_validation_template(self):
        """Test Python path validation template is correct."""
        path_traversal = self.config['fix']['security_fixes']['path_traversal']
        py_pattern = next(
            (p for p in path_traversal['patterns'] if p['language'] == 'python'),
            None
        )
        self.assertIsNotNone(py_pattern)
        template = py_pattern.get('validation_template', '')
        # Check template contains key security elements
        self.assertIn('resolve', template.lower())
        self.assertIn('relative_to', template)
        self.assertIn('ValueError', template)
    def test_typescript_path_validation_template(self):
        """Test TypeScript path validation template is correct."""
        path_traversal = self.config['fix']['security_fixes']['path_traversal']
        ts_pattern = next(
            (p for p in path_traversal['patterns'] if p['language'] == 'typescript'),
            None
        )
        self.assertIsNotNone(ts_pattern)
        template = ts_pattern.get('validation_template', '')
        # Check template contains key security elements
        self.assertIn('resolve', template)
        self.assertIn('realpath', template)
        self.assertIn('startsWith', template)
    def test_crypto_replacements_are_secure(self):
        """Test that crypto replacements use secure algorithms."""
        weak_crypto = self.config['fix']['security_fixes']['weak_crypto']
        # List of weak algorithms that should be replaced
        weak_algorithms = ['md5', 'sha1', 'des', '3des', 'rc4']
        # List of secure algorithms that are acceptable replacements
        # Note: Both hyphenated (SHA-256) and non-hyphenated (sha256) forms are valid
        # as different libraries/languages use different naming conventions
        secure_algorithms = ['sha256', 'sha384', 'sha512', 'sha-256', 'sha-384', 'sha-512']
        for replacement in weak_crypto['replacements']:
            from_algo = replacement['from'].lower()
            to_algo = replacement['to'].lower()
            # Verify 'from' contains a weak algorithm
            has_weak = any(weak in from_algo for weak in weak_algorithms)
            self.assertTrue(has_weak, f"Unexpected 'from' value: {replacement['from']}")
            # Verify 'to' contains a secure algorithm
            has_secure = any(secure in to_algo for secure in secure_algorithms)
            self.assertTrue(has_secure, f"Replacement is not secure: {replacement['to']}")
class TestValidationStatus(unittest.TestCase):
    """Test the validation status in the config."""
    # Minimum expected coverage values from configuration
    MIN_ALERT_TYPES = 5
    MIN_LANGUAGES = 4
    MIN_STRATEGIES = 5
    @classmethod
    def setUpClass(cls):
        """Load the auto-fix bot configuration."""
        config_path = Path(__file__).parent.parent.parent / 'config' / '.auto-fix-bot.yml'
        if not config_path.exists():
            raise unittest.SkipTest(f"Config file not found: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            cls.config = yaml.safe_load(f)
    def test_validation_checks_exist(self):
        """Test that validation checks are defined."""
        self.assertIn('validation', self.config)
        self.assertIn('validation_checks', self.config['validation'])
        checks = self.config['validation']['validation_checks']
        self.assertGreater(len(checks), 0, "No validation checks defined")
    def test_all_validation_checks_passed(self):
        """Test that all validation checks have passed."""
        checks = self.config['validation']['validation_checks']
        for check in checks:
            self.assertEqual(
                check['status'], 
                'passed', 
                f"Validation check '{check['name']}' is not passed: {check['status']}"
            )
    def test_coverage_metrics(self):
        """Test coverage metrics are reasonable."""
        coverage = self.config['validation']['coverage']
        self.assertGreaterEqual(
            coverage['alert_types_covered'], 
            self.MIN_ALERT_TYPES, 
            f"Expected at least {self.MIN_ALERT_TYPES} alert types covered"
        )
        self.assertGreaterEqual(
            coverage['languages_supported'], 
            self.MIN_LANGUAGES, 
            f"Expected at least {self.MIN_LANGUAGES} languages supported"
        )
        self.assertGreaterEqual(
            coverage['auto_fix_strategies'], 
            self.MIN_STRATEGIES, 
            f"Expected at least {self.MIN_STRATEGIES} auto-fix strategies"
        )
    def test_quality_assurance_flags(self):
        """Test quality assurance flags."""
        qa = self.config['validation']['quality_assurance']
        self.assertTrue(qa['peer_reviewed'], "Config should be peer reviewed")
        self.assertTrue(qa['security_reviewed'], "Config should be security reviewed")
class TestStagingEnvironmentConfig(unittest.TestCase):
    """Test staging environment configuration."""
    @classmethod
    def setUpClass(cls):
        """Load the staging environment configuration."""
        staging_path = Path(__file__).parent.parent.parent / 'config' / 'environments' / 'staging.yaml'
        if not staging_path.exists():
            raise unittest.SkipTest(f"Staging config not found: {staging_path}")
        with open(staging_path, 'r', encoding='utf-8') as f:
            cls.staging_config = yaml.safe_load(f)
    def test_staging_environment_defined(self):
        """Test staging environment is properly defined."""
        self.assertEqual(self.staging_config['environment'], 'staging')
    def test_auto_fix_bot_settings(self):
        """Test auto-fix bot staging settings."""
        auto_fix = self.staging_config['auto_fix_bot']
        self.assertTrue(auto_fix['enabled'])
        self.assertTrue(auto_fix['settings']['create_pr'])
        self.assertTrue(auto_fix['settings']['require_human_review'])
    def test_security_strategies_validated(self):
        """Test all security strategies are validated in staging."""
        strategies = self.staging_config['security_fix_strategies']
        expected_strategies = ['weak_crypto', 'path_traversal', 'debug_mode', 'sensitive_logging', 'code_injection']
        for strategy in expected_strategies:
            self.assertIn(strategy, strategies, f"Missing strategy: {strategy}")
            self.assertEqual(
                strategies[strategy]['validation_status'], 
                'passed',
                f"Strategy '{strategy}' not validated"
            )
    def test_validation_summary(self):
        """Test validation summary is complete."""
        summary = self.staging_config['validation_summary']
        self.assertEqual(summary['overall_status'], 'passed')
        self.assertEqual(summary['total_strategies'], summary['strategies_validated'])
        # Check quality gates
        quality_gates = summary['quality_gates']
        self.assertTrue(quality_gates['all_tests_passed'])
        self.assertTrue(quality_gates['no_false_positives'])
        self.assertTrue(quality_gates['no_incorrect_fixes'])
if __name__ == '__main__':
    unittest.main()
