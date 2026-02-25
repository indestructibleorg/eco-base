#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: test_gl-platform.gl-platform.governance_audit_script
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Unit tests for GL Governance Audit Script.
"""
import pytest
import importlib.util
from pathlib import Path
# Import the gl-platform.gl-platform.governance-audit-script module (hyphenated filename)
script_path = Path(__file__).parent.parent.parent / "gl-platform.gl-platform.governance-audit-script.py"
spec = importlib.util.spec_from_file_location("gl-platform.gl-platform.governance_audit_script", script_path)
gl-platform.gl-platform.governance_audit_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gl-platform.gl-platform.governance_audit_script)
GLGovernanceAudit = gl-platform.gl-platform.governance_audit_script.GLGovernanceAudit
class TestGLGovernanceAuditInit:
    """Tests for GLGovernanceAudit initialization."""
    def test_init_default_path(self):
        """Test initialization with default path."""
        auditor = GLGovernanceAudit()
        assert auditor.base_path.exists() or auditor.base_path == Path(".").resolve()
    def test_init_custom_path(self, tmp_path):
        """Test initialization with custom path."""
        auditor = GLGovernanceAudit(base_path=str(tmp_path))
        assert auditor.base_path == tmp_path
        assert auditor.root_path == tmp_path / "root"
    def test_init_audit_results_structure(self, tmp_path):
        """Test that audit results are properly initialized."""
        auditor = GLGovernanceAudit(base_path=str(tmp_path))
        assert "audit_metadata" in auditor.audit_results
        assert "summary" in auditor.audit_results
        assert "file_reports" in auditor.audit_results
        assert "issues" in auditor.audit_results
        assert auditor.audit_results["audit_metadata"]["gl_unified_charter"] == "activated"
class TestExtensionNormalization:
    """Tests for extension normalization."""
    def test_normalize_lowercase(self, tmp_path):
        """Test normalizing lowercase extension."""
        auditor = GLGovernanceAudit(base_path=str(tmp_path))
        assert auditor.normalize_extension(".yaml") == ".yaml"
    def test_normalize_uppercase(self, tmp_path):
        """Test normalizing uppercase extension."""
        auditor = GLGovernanceAudit(base_path=str(tmp_path))
        assert auditor.normalize_extension(".YAML") == ".yaml"
    def test_normalize_mixed_case(self, tmp_path):
        """Test normalizing mixed case extension."""
        auditor = GLGovernanceAudit(base_path=str(tmp_path))
        assert auditor.normalize_extension(".YaMl") == ".yaml"
class TestGLMarkerCheck:
    """Tests for GL marker detection."""
    @pytest.fixture
    def auditor(self, tmp_path):
        """Create auditor instance."""
        return GLGovernanceAudit(base_path=str(tmp_path))
    def test_no_gl_marker_yaml(self, auditor):
        """Test detection of missing GL marker in YAML."""
        content = "key: value\nname: test"
        issues = auditor.check_gl_markers(content, ".yaml")
        assert len(issues) == 1
        assert issues[0]["type"] == "missing_gl_marker"
        assert issues[0]["severity"] == "MEDIUM"
    def test_has_gl_marker_machinenativeops(self, auditor):
        """Test that machinenativeops.io marker is detected."""
        content = "apiVersion: machinenativeops.io/v1\nname: test"
        issues = auditor.check_gl_markers(content, ".yaml")
        assert len(issues) == 0
    def test_has_gl_marker_prefix(self, auditor):
        """Test that GL: marker is detected."""
        content = "GL: layer-10\nname: test"
        issues = auditor.check_gl_markers(content, ".yaml")
        assert len(issues) == 0
    def test_no_marker_check_for_md(self, auditor):
        """Test that markdown files don't require GL markers."""
        content = "# Heading\nSome content"
        issues = auditor.check_gl_markers(content, ".md")
        assert len(issues) == 0
    def test_no_marker_check_for_sh(self, auditor):
        """Test that shell scripts don't require GL markers."""
        content = "#!/bin/bash\necho hello"
        issues = auditor.check_gl_markers(content, ".sh")
        assert len(issues) == 0
    def test_uppercase_extension(self, auditor):
        """Test GL marker check with uppercase extension."""
        content = "key: value"
        issues = auditor.check_gl_markers(content, ".YAML")
        assert len(issues) == 1
class TestMetadataCheck:
    """Tests for metadata validation."""
    @pytest.fixture
    def auditor(self, tmp_path):
        """Create auditor instance."""
        return GLGovernanceAudit(base_path=str(tmp_path))
    def test_valid_metadata(self, auditor):
        """Test that valid metadata passes."""
        content = "apiVersion: test"
        parsed = {"apiVersion": "v1", "metadata": {"name": "test"}}
        issues = auditor.check_metadata(content, ".yaml", parsed)
        assert len(issues) == 0
    def test_missing_metadata_section(self, auditor):
        """Test detection of missing metadata section."""
        content = "apiVersion: test"
        parsed = {"apiVersion": "v1", "spec": {}}
        issues = auditor.check_metadata(content, ".yaml", parsed)
        assert len(issues) == 1
        assert issues[0]["type"] == "missing_metadata"
class TestFileExecution:
    """Tests for file type execution handlers."""
    @pytest.fixture
    def setup_test_files(self, tmp_path):
        """Setup test files."""
        root_path = tmp_path / "root"
        root_path.mkdir()
        # Create YAML file with GL marker
        yaml_file = root_path / "test.yaml"
        yaml_file.write_text("apiVersion: machinenativeops.io/v1\nmetadata:\n  name: test")
        # Create JSON file
        json_file = root_path / "test.json"
        json_file.write_text('{"name": "test", "version": "1.0.0"}')
        # Create shell script with set -e
        shell_file = root_path / "test.sh"
        shell_file.write_text("#!/bin/bash\nset -e\necho hello")
        # Create shell script without set -e
        shell_no_set_e = root_path / "test_no_set_e.sh"
        shell_no_set_e.write_text("#!/bin/bash\necho hello")
        # Create shell script with set -euo pipefail
        shell_pipefail = root_path / "test_pipefail.sh"
        shell_pipefail.write_text("#!/bin/bash\nset -euo pipefail\necho hello")
        # Create markdown file
        md_file = root_path / "test.md"
        md_file.write_text("# Test\n## Section\nContent")
        return tmp_path
    def test_execute_yaml_file(self, setup_test_files):
        """Test YAML file execution."""
        auditor = GLGovernanceAudit(base_path=str(setup_test_files))
        yaml_file = setup_test_files / "root" / "test.yaml"
        result = auditor.execute_yaml_file(yaml_file)
        assert result["status"] == "success"
        assert result["type"] == "yaml"
        assert len(result["issues"]) == 0  # Has GL marker
    def test_execute_json_file(self, setup_test_files):
        """Test JSON file execution."""
        auditor = GLGovernanceAudit(base_path=str(setup_test_files))
        json_file = setup_test_files / "root" / "test.json"
        result = auditor.execute_json_file(json_file)
        assert result["status"] == "success"
        assert result["type"] == "json"
    def test_execute_shell_file_with_set_e(self, setup_test_files):
        """Test shell file with set -e."""
        auditor = GLGovernanceAudit(base_path=str(setup_test_files))
        shell_file = setup_test_files / "root" / "test.sh"
        result = auditor.execute_shell_file(shell_file)
        assert result["status"] == "success"
        assert result["type"] == "shell"
        assert result["structure"]["has_shebang"] is True
        assert result["structure"]["uses_set_e"] is True
        assert len(result["issues"]) == 0
    def test_execute_shell_file_without_set_e(self, setup_test_files):
        """Test shell file without set -e."""
        auditor = GLGovernanceAudit(base_path=str(setup_test_files))
        shell_file = setup_test_files / "root" / "test_no_set_e.sh"
        result = auditor.execute_shell_file(shell_file)
        assert result["status"] == "success"
        assert result["structure"]["uses_set_e"] is False
        assert len(result["issues"]) == 1
        assert result["issues"][0]["type"] == "missing_safety_flag"
    def test_execute_shell_file_with_pipefail(self, setup_test_files):
        """Test shell file with set -euo pipefail."""
        auditor = GLGovernanceAudit(base_path=str(setup_test_files))
        shell_file = setup_test_files / "root" / "test_pipefail.sh"
        result = auditor.execute_shell_file(shell_file)
        assert result["status"] == "success"
        assert result["structure"]["uses_set_e"] is True
        assert len(result["issues"]) == 0
    def test_execute_markdown_file(self, setup_test_files):
        """Test markdown file execution."""
        auditor = GLGovernanceAudit(base_path=str(setup_test_files))
        md_file = setup_test_files / "root" / "test.md"
        result = auditor.execute_markdown_file(md_file)
        assert result["status"] == "success"
        assert result["type"] == "markdown"
        assert result["structure"]["header_count"] == 2
class TestFileScanning:
    """Tests for file scanning and exclusion."""
    def test_scan_excludes_audit_reports(self, tmp_path):
        """Test that .audit-reports directory is excluded."""
        root_path = tmp_path / "root"
        root_path.mkdir()
        # Create regular file
        (root_path / "test.yaml").write_text("test: value")
        # Create audit reports directory
        audit_dir = root_path / ".audit-reports"
        audit_dir.mkdir()
        (audit_dir / "report.json").write_text("{}")
        auditor = GLGovernanceAudit(base_path=str(tmp_path))
        files = auditor.scan_root_files()
        # Should only find test.yaml, not report.json
        assert len(files) == 1
        assert files[0].name == "test.yaml"
    def test_scan_excludes_generated_files(self, tmp_path):
        """Test that generated audit files are excluded."""
        root_path = tmp_path / "root"
        root_path.mkdir()
        # Create regular file
        (root_path / "test.yaml").write_text("test: value")
        # Create generated files
        (root_path / "GLOBAL_GOVERNANCE_AUDIT_REPORT.json").write_text("{}")
        (root_path / "file-inventory.json").write_text("{}")
        auditor = GLGovernanceAudit(base_path=str(tmp_path))
        files = auditor.scan_root_files()
        # Should only find test.yaml
        assert len(files) == 1
        assert files[0].name == "test.yaml"
class TestExitCodes:
    """Tests for exit code logic."""
    def test_exit_code_no_issues(self, tmp_path):
        """Test exit code when no critical/high issues."""
        root_path = tmp_path / "root"
        root_path.mkdir()
        # Create file with GL marker (no issues)
        (root_path / "test.yaml").write_text("apiVersion: machinenativeops.io/v1\nmetadata:\n  name: test")
        auditor = GLGovernanceAudit(base_path=str(tmp_path))
        results = auditor.run_audit()
        # Should return 0 (no critical/high)
        assert results["summary"]["severity_breakdown"]["CRITICAL"] == 0
        assert results["summary"]["severity_breakdown"]["HIGH"] == 0
    def test_audit_generates_reports(self, tmp_path):
        """Test that audit generates all expected reports."""
        root_path = tmp_path / "root"
        root_path.mkdir()
        (root_path / "test.yaml").write_text("apiVersion: machinenativeops.io/v1\nmetadata:\n  name: test")
        auditor = GLGovernanceAudit(base_path=str(tmp_path))
        auditor.run_audit()
        # Check generated files
        assert (root_path / "GLOBAL_GOVERNANCE_AUDIT_REPORT.json").exists()
        assert (root_path / "file-inventory.json").exists()
        assert (root_path / ".audit-reports").exists()
class TestFileInventory:
    """Tests for file inventory generation."""
    def test_inventory_includes_hash(self, tmp_path):
        """Test that file inventory includes SHA-256 hash."""
        root_path = tmp_path / "root"
        root_path.mkdir()
        test_content = "test content"
        (root_path / "test.txt").write_text(test_content)
        auditor = GLGovernanceAudit(base_path=str(tmp_path))
        files = [root_path / "test.txt"]
        inventory = auditor.generate_file_inventory(files)
        assert len(inventory["files"]) == 1
        assert "sha256" in inventory["files"][0]
        assert len(inventory["files"][0]["sha256"]) == 64  # SHA-256 hex length
class TestEmptyFileHandling:
    """Tests for empty file handling."""
    def test_empty_shell_file_shebang_check(self, tmp_path):
        """Test shebang check on empty shell file."""
        root_path = tmp_path / "root"
        root_path.mkdir()
        # Create empty shell file
        shell_file = root_path / "empty.sh"
        shell_file.write_text("")
        auditor = GLGovernanceAudit(base_path=str(tmp_path))
        result = auditor.execute_shell_file(shell_file)
        assert result["status"] == "success"
        assert result["structure"]["has_shebang"] is False
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
