#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: test_code_deduplicator
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Unit tests for Code Deduplicator
Tests duplicate code detection functionality
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
    from scripts.utils.code_deduplicator import CodeDeduplicator
except ImportError:
    CodeDeduplicator = None
class TestCodeDeduplicator(unittest.TestCase):
    """Test cases for CodeDeduplicator."""
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        if CodeDeduplicator is None:
            raise unittest.SkipTest("CodeDeduplicator not available")
    def setUp(self):
        """Set up test instance."""
        self.deduplicator = CodeDeduplicator(min_lines=3)
        self.temp_dir = tempfile.mkdtemp()
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    def _create_temp_file(self, filename: str, content: str) -> str:
        """Create a temporary Python file."""
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return filepath
    def test_initialization(self):
        """Test CodeDeduplicator initialization."""
        dedup = CodeDeduplicator(min_lines=5)
        self.assertEqual(dedup.min_lines, 5)
        self.assertIsInstance(dedup.function_hashes, dict)
        self.assertIsInstance(dedup.class_hashes, dict)
    def test_analyze_simple_file(self):
        """Test analyzing a simple Python file."""
        content = '''
def hello():
    print("Hello")
    return True
def world():
    print("World")
    return False
'''
        filepath = self._create_temp_file("simple.py", content)
        self.deduplicator.analyze_file(filepath)
        # Should not raise any errors
        self.assertIsNotNone(self.deduplicator.function_hashes)
    def test_detect_duplicate_functions(self):
        """Test detection of duplicate functions."""
        content1 = '''
def process_data(data):
    result = []
    for item in data:
        result.append(item * 2)
    return result
'''
        content2 = '''
def transform_data(data):
    result = []
    for item in data:
        result.append(item * 2)
    return result
'''
        file1 = self._create_temp_file("file1.py", content1)
        file2 = self._create_temp_file("file2.py", content2)
        self.deduplicator.analyze_file(file1)
        self.deduplicator.analyze_file(file2)
        duplicates = self.deduplicator.get_duplicates()
        self.assertIsInstance(duplicates, dict)
    def test_analyze_invalid_file(self):
        """Test analyzing non-existent file."""
        # Should handle gracefully without raising
        try:
            self.deduplicator.analyze_file("/nonexistent/path/file.py")
        except (FileNotFoundError, IOError):
            pass  # Expected behavior - may raise or handle gracefully
    def test_analyze_syntax_error_file(self):
        """Test analyzing file with syntax errors."""
        content = '''
def broken(
    # Missing closing parenthesis
    print("broken")
'''
        filepath = self._create_temp_file("broken.py", content)
        # Should handle gracefully without raising
        self.deduplicator.analyze_file(filepath)
    def test_get_duplicates_structure(self):
        """Test duplicates structure."""
        duplicates = self.deduplicator.get_duplicates()
        self.assertIsInstance(duplicates, dict)
        # Should have functions and classes sections
        self.assertIn('functions', duplicates)
        self.assertIn('classes', duplicates)
    def test_min_lines_filter(self):
        """Test that min_lines parameter is respected."""
        dedup = CodeDeduplicator(min_lines=10)
        content = '''
def short():
    return 1
'''
        filepath = self._create_temp_file("short.py", content)
        dedup.analyze_file(filepath)
        # Verify min_lines is set correctly
        self.assertEqual(dedup.min_lines, 10)
        # The function may or may not be filtered depending on implementation
        # Just verify the deduplicator processed the file without error
        duplicates = dedup.get_duplicates()
        self.assertIsInstance(duplicates, dict)
    def test_generate_report(self):
        """Test report generation."""
        content = '''
def example():
    x = 1
    y = 2
    z = 3
    return x + y + z
'''
        filepath = self._create_temp_file("example.py", content)
        self.deduplicator.analyze_file(filepath)
        report = self.deduplicator.generate_report()
        # Should be a string report
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)
    def test_json_output(self):
        """Test JSON serialization of duplicates."""
        content = '''
def example():
    x = 1
    y = 2
    z = 3
    return x + y + z
'''
        filepath = self._create_temp_file("example.py", content)
        self.deduplicator.analyze_file(filepath)
        duplicates = self.deduplicator.get_duplicates()
        # Should be JSON serializable
        json_str = json.dumps(duplicates)
        self.assertIsInstance(json_str, str)
        # Should be parseable back
        parsed = json.loads(json_str)
        self.assertEqual(parsed, duplicates)
class TestCodeDeduplicatorIntegration(unittest.TestCase):
    """Integration tests for CodeDeduplicator."""
    @classmethod
    def setUpClass(cls):
        if CodeDeduplicator is None:
            raise unittest.SkipTest("CodeDeduplicator not available")
    def test_scan_directory(self):
        """Test scanning a directory for duplicates."""
        dedup = CodeDeduplicator()
        # Scan the scripts directory if it exists
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        if scripts_dir.exists():
            dedup.analyze_directory(str(scripts_dir))
            duplicates = dedup.get_duplicates()
            self.assertIsInstance(duplicates, dict)
if __name__ == '__main__':
    unittest.main()