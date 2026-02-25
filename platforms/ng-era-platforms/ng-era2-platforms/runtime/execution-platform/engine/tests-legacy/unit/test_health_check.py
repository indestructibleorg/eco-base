#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: test_health_check
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Unit tests for Health Check System
Tests repository health check functionality
"""
# MNGA-002: Import organization needs review
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from scripts.monitoring.health_check import (
        HealthChecker,
        HealthCheckResult,
        HealthStatus
    )
    HealthCheckSystem = HealthChecker  # Alias for test compatibility
except ImportError:
    HealthCheckSystem = None
    HealthChecker = None
    HealthCheckResult = None
    HealthStatus = None
class TestHealthStatus(unittest.TestCase):
    """Test cases for HealthStatus enum."""
    @classmethod
    def setUpClass(cls):
        if HealthStatus is None:
            raise unittest.SkipTest("HealthStatus not available")
    def test_health_status_values(self):
        """Test HealthStatus enum values."""
        self.assertEqual(HealthStatus.HEALTHY.value, "healthy")
        self.assertEqual(HealthStatus.DEGRADED.value, "degraded")
        self.assertEqual(HealthStatus.UNHEALTHY.value, "unhealthy")
        self.assertEqual(HealthStatus.UNKNOWN.value, "unknown")
    def test_health_status_comparison(self):
        """Test HealthStatus comparison."""
        self.assertEqual(HealthStatus.HEALTHY, HealthStatus.HEALTHY)
        self.assertNotEqual(HealthStatus.HEALTHY, HealthStatus.UNHEALTHY)
class TestHealthCheckResult(unittest.TestCase):
    """Test cases for HealthCheckResult dataclass."""
    @classmethod
    def setUpClass(cls):
        if HealthCheckResult is None:
            raise unittest.SkipTest("HealthCheckResult not available")
    def test_result_creation(self):
        """Test creating a HealthCheckResult."""
        result = HealthCheckResult(
            name="test_check",
            status=HealthStatus.HEALTHY,
            message="All good"
        )
        self.assertEqual(result.name, "test_check")
        self.assertEqual(result.status, HealthStatus.HEALTHY)
        self.assertEqual(result.message, "All good")
    def test_result_with_details(self):
        """Test HealthCheckResult with details."""
        result = HealthCheckResult(
            name="detailed_check",
            status=HealthStatus.DEGRADED,
            message="Some issues found",
            details={"issues": ["issue1", "issue2"]}
        )
        self.assertEqual(len(result.details["issues"]), 2)
    def test_result_default_details(self):
        """Test HealthCheckResult default details."""
        result = HealthCheckResult(
            name="simple_check",
            status=HealthStatus.HEALTHY,
            message="OK"
        )
        self.assertIsInstance(result.details, dict)
class TestHealthCheckSystem(unittest.TestCase):
    """Test cases for HealthCheckSystem (HealthChecker)."""
    @classmethod
    def setUpClass(cls):
        if HealthCheckSystem is None:
            raise unittest.SkipTest("HealthCheckSystem not available")
    def setUp(self):
        """Set up test instance."""
        self.repo_root = Path(__file__).parent.parent.parent
        self.health_check = HealthCheckSystem(str(self.repo_root))
    def test_initialization(self):
        """Test HealthCheckSystem initialization."""
        hc = HealthCheckSystem(".")
        self.assertEqual(hc.repo_root, Path("."))
        self.assertIsInstance(hc.results, list)
    def test_run_all_checks(self):
        """Test running all health checks."""
        results = self.health_check.run_all_checks()
        self.assertIsInstance(results, list)
        # Each result should be a HealthCheckResult
        for result in results:
            self.assertIsInstance(result, HealthCheckResult)
            self.assertIn(result.status, list(HealthStatus))
    def test_get_overall_status(self):
        """Test getting overall health status."""
        self.health_check.run_all_checks()
        overall = self.health_check.get_overall_status()
        self.assertIn(overall, list(HealthStatus))
    def test_export_json(self):
        """Test exporting health report to JSON."""
        self.health_check.run_all_checks()
        # Export to temp file
        temp_file = os.path.join(tempfile.mkdtemp(), "health.json")
        self.health_check.export_json(temp_file)
        # Verify file was created and is valid JSON
        self.assertTrue(os.path.exists(temp_file))
        with open(temp_file, 'r') as f:
            report = json.load(f)
        self.assertIsInstance(report, dict)
        self.assertIn('overall_status', report)
        self.assertIn('checks', report)
        self.assertIn('timestamp', report)
    def test_results_to_dict(self):
        """Test that results can be converted to dict."""
        self.health_check.run_all_checks()
        for result in self.health_check.results:
            result_dict = result.to_dict()
            self.assertIsInstance(result_dict, dict)
            self.assertIn('name', result_dict)
            self.assertIn('status', result_dict)
            self.assertIn('message', result_dict)
class TestHealthChecks(unittest.TestCase):
    """Test individual health checks."""
    @classmethod
    def setUpClass(cls):
        if HealthCheckSystem is None:
            raise unittest.SkipTest("HealthCheckSystem not available")
    def setUp(self):
        self.repo_root = Path(__file__).parent.parent.parent
        self.health_check = HealthCheckSystem(str(self.repo_root))
    def test_git_status_check(self):
        """Test git status health check."""
        results = self.health_check.run_all_checks()
        git_check = next((r for r in results if 'git' in r.name.lower()), None)
        if git_check:
            self.assertIn(git_check.status, list(HealthStatus))
            self.assertIsInstance(git_check.message, str)
    def test_workflows_check(self):
        """Test workflows health check."""
        results = self.health_check.run_all_checks()
        workflow_check = next((r for r in results if 'workflow' in r.name.lower()), None)
        if workflow_check:
            self.assertIn(workflow_check.status, list(HealthStatus))
    def test_documentation_check(self):
        """Test documentation health check."""
        results = self.health_check.run_all_checks()
        doc_check = next((r for r in results if 'doc' in r.name.lower()), None)
        if doc_check:
            self.assertIn(doc_check.status, list(HealthStatus))
    def test_security_config_check(self):
        """Test security configuration health check."""
        results = self.health_check.run_all_checks()
        security_check = next((r for r in results if 'security' in r.name.lower()), None)
        if security_check:
            self.assertIn(security_check.status, list(HealthStatus))
class TestHealthCheckSystemEdgeCases(unittest.TestCase):
    """Edge case tests for HealthCheckSystem."""
    @classmethod
    def setUpClass(cls):
        if HealthCheckSystem is None:
            raise unittest.SkipTest("HealthCheckSystem not available")
    def test_invalid_repo_path(self):
        """Test health check with invalid repository path."""
        hc = HealthCheckSystem("/nonexistent/path")
        results = hc.run_all_checks()
        # Should still return results, possibly with unhealthy status
        self.assertIsInstance(results, list)
    def test_empty_results_status(self):
        """Test getting status before running checks."""
        hc = HealthCheckSystem(".")
        status = hc.get_overall_status()
        # Should return UNKNOWN when no checks have been run
        self.assertEqual(status, HealthStatus.UNKNOWN)
    def test_multiple_runs(self):
        """Test running health checks multiple times."""
        hc = HealthCheckSystem(".")
        results1 = hc.run_all_checks()
        results2 = hc.run_all_checks()
        # Should return consistent structure
        self.assertEqual(len(results1), len(results2))
class TestHealthCheckIntegration(unittest.TestCase):
    """Integration tests for HealthCheckSystem."""
    @classmethod
    def setUpClass(cls):
        if HealthCheckSystem is None:
            raise unittest.SkipTest("HealthCheckSystem not available")
    def test_full_health_check_cycle(self):
        """Test complete health check cycle."""
        repo_root = Path(__file__).parent.parent.parent
        hc = HealthCheckSystem(str(repo_root))
        # Run checks
        results = hc.run_all_checks()
        self.assertGreater(len(results), 0)
        # Get overall status
        overall = hc.get_overall_status()
        self.assertIn(overall, list(HealthStatus))
        # Verify results structure
        for result in results:
            result_dict = result.to_dict()
            self.assertIn('status', result_dict)
            self.assertEqual(result_dict['status'], result.status.value)
    def test_export_report_to_file(self):
        """Test exporting health report to file."""
        temp_dir = tempfile.mkdtemp()
        try:
            hc = HealthCheckSystem(".")
            hc.run_all_checks()
            output_path = os.path.join(temp_dir, "health_check.json")
            hc.export_json(output_path)
            # Verify file
            self.assertTrue(os.path.exists(output_path))
            with open(output_path, 'r') as f:
                loaded = json.load(f)
            self.assertIn('overall_status', loaded)
            self.assertIn('checks', loaded)
        finally:
            import shutil
            shutil.rmtree(temp_dir)
if __name__ == '__main__':
    unittest.main()