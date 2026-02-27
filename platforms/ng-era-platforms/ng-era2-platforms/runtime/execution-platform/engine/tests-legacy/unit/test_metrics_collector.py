#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: test_metrics_collector
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Unit tests for Metrics Collector
Tests observability metrics collection functionality
"""
# MNGA-002: Import organization needs review
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from scripts.monitoring.metrics_collector import MetricsCollector
except ImportError:
    MetricsCollector = None
class TestMetricsCollector(unittest.TestCase):
    """Test cases for MetricsCollector."""
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        if MetricsCollector is None:
            raise unittest.SkipTest("MetricsCollector not available")
    def setUp(self):
        """Set up test instance."""
        self.repo_root = Path(__file__).parent.parent.parent
        self.collector = MetricsCollector(str(self.repo_root))
    def test_initialization(self):
        """Test MetricsCollector initialization."""
        collector = MetricsCollector(".")
        self.assertEqual(collector.repo_root, Path("."))
        self.assertIsInstance(collector.metrics, dict)
    def test_collect_all_returns_dict(self):
        """Test that collect_all returns a dictionary."""
        metrics = self.collector.collect_all()
        self.assertIsInstance(metrics, dict)
    def test_metrics_has_timestamp(self):
        """Test that metrics include timestamp."""
        metrics = self.collector.collect_all()
        self.assertIn('timestamp', metrics)
        # Verify timestamp is ISO format
        try:
            datetime.fromisoformat(metrics['timestamp'].replace('Z', '+00:00'))
        except ValueError:
            self.fail("Timestamp is not in valid ISO format")
    def test_metrics_has_repository_section(self):
        """Test that metrics include repository section."""
        metrics = self.collector.collect_all()
        self.assertIn('repository', metrics)
        self.assertIsInstance(metrics['repository'], dict)
    def test_metrics_has_code_quality_section(self):
        """Test that metrics include code quality section."""
        metrics = self.collector.collect_all()
        self.assertIn('code_quality', metrics)
        self.assertIsInstance(metrics['code_quality'], dict)
    def test_metrics_has_ci_cd_section(self):
        """Test that metrics include CI/CD section."""
        metrics = self.collector.collect_all()
        self.assertIn('ci_cd', metrics)
        self.assertIsInstance(metrics['ci_cd'], dict)
    def test_metrics_has_security_section(self):
        """Test that metrics include security section."""
        metrics = self.collector.collect_all()
        self.assertIn('security', metrics)
        self.assertIsInstance(metrics['security'], dict)
    def test_metrics_json_serializable(self):
        """Test that metrics are JSON serializable."""
        metrics = self.collector.collect_all()
        try:
            json_str = json.dumps(metrics)
            self.assertIsInstance(json_str, str)
        except (TypeError, ValueError) as e:
            self.fail(f"Metrics are not JSON serializable: {e}")
    def test_repo_metrics_structure(self):
        """Test repository metrics structure."""
        metrics = self.collector.collect_all()
        repo_metrics = metrics.get('repository', {})
        # Should have common repo metrics
        expected_keys = ['total_files', 'total_lines', 'languages']
        for key in expected_keys:
            if key in repo_metrics:
                self.assertIsNotNone(repo_metrics[key])
    def test_code_quality_metrics_structure(self):
        """Test code quality metrics structure."""
        metrics = self.collector.collect_all()
        quality_metrics = metrics.get('code_quality', {})
        # Should have quality-related metrics
        self.assertIsInstance(quality_metrics, dict)
    def test_collect_with_invalid_repo_path(self):
        """Test collection with invalid repository path."""
        collector = MetricsCollector("/nonexistent/path")
        metrics = collector.collect_all()
        # Should still return a dict, possibly with error info
        self.assertIsInstance(metrics, dict)
class TestMetricsCollectorOutput(unittest.TestCase):
    """Test MetricsCollector output functionality."""
    @classmethod
    def setUpClass(cls):
        if MetricsCollector is None:
            raise unittest.SkipTest("MetricsCollector not available")
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.collector = MetricsCollector(".")
    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    def test_export_to_json(self):
        """Test exporting metrics to JSON file."""
        metrics = self.collector.collect_all()
        output_path = os.path.join(self.temp_dir, "metrics.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        # Verify file was created and is valid JSON
        self.assertTrue(os.path.exists(output_path))
        with open(output_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        self.assertEqual(loaded, metrics)
    def test_prometheus_format(self):
        """Test metrics can be formatted for Prometheus."""
        metrics = self.collector.collect_all()
        # Simple Prometheus format conversion
        prometheus_lines = []
        def flatten_metrics(data, prefix=""):
            for key, value in data.items():
                metric_name = f"{prefix}_{key}" if prefix else key
                if isinstance(value, (int, float)):
                    prometheus_lines.append(f"{metric_name} {value}")
                elif isinstance(value, dict):
                    flatten_metrics(value, metric_name)
        flatten_metrics(metrics)
        # Should have some metrics
        self.assertGreater(len(prometheus_lines), 0)
class TestMetricsCollectorIntegration(unittest.TestCase):
    """Integration tests for MetricsCollector."""
    @classmethod
    def setUpClass(cls):
        if MetricsCollector is None:
            raise unittest.SkipTest("MetricsCollector not available")
    def test_collect_from_actual_repo(self):
        """Test collecting metrics from actual repository."""
        repo_root = Path(__file__).parent.parent.parent
        if not (repo_root / ".git").exists():
            self.skipTest("Not in a git repository")
        collector = MetricsCollector(str(repo_root))
        metrics = collector.collect_all()
        # Should have collected real data
        self.assertIn('timestamp', metrics)
        self.assertIn('repository', metrics)
    def test_metrics_consistency(self):
        """Test that metrics are consistent across multiple collections."""
        collector = MetricsCollector(".")
        metrics1 = collector.collect_all()
        metrics2 = collector.collect_all()
        # Structure should be the same
        self.assertEqual(set(metrics1.keys()), set(metrics2.keys()))
if __name__ == '__main__':
    unittest.main()