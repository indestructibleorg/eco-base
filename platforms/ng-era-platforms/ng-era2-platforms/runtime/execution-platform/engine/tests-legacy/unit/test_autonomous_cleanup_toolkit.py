#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: test_autonomous_cleanup_toolkit
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Unit tests for Autonomous Cleanup Toolkit.
GL Layer: GL30-49 Execution Layer
GL Unified Architecture Governance Framework: Activated
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))
from autonomous_cleanup_toolkit import (
    AutonomousCleanupEngine,
    CleanupReport,
    DuplicateGroup,
    NotImplementedStub,
    TodoItem,
)
class TestAutonomousCleanupEngine:
    """Tests for AutonomousCleanupEngine class."""
    @pytest.fixture
    def temp_repo(self, tmp_path):
        """Create a temporary repository for testing."""
        # Create test files
        (tmp_path / ".automation_logs").mkdir()
        (tmp_path / "test.py").write_text("# TODO: Fix this\nprint('hello')\n")
        (tmp_path / "test2.py").write_text("# TODO: Fix this\nprint('hello')\n")
        (tmp_path / ".github" / "workflows").mkdir(parents=True)
        (tmp_path / ".github" / "workflows" / "test.yml").write_text("name: Test\n")
        return tmp_path
    @pytest.fixture
    def engine(self, temp_repo):
        """Create engine instance for testing."""
        return AutonomousCleanupEngine(temp_repo)
    def test_engine_initialization(self, engine, temp_repo):
        """Test engine initializes correctly."""
        assert engine.repo_path == temp_repo
        assert engine.logger is not None
        assert engine.stats["scans_performed"] == 0
    def test_find_duplicates(self, engine, temp_repo):
        """Test duplicate file detection."""
        duplicates = engine.find_duplicates(extensions=[".py"])
        # Should find duplicates since test.py and test2.py have same content
        assert len(duplicates) >= 1
        assert all(isinstance(d, DuplicateGroup) for d in duplicates)
    def test_find_todos(self, engine, temp_repo):
        """Test TODO marker detection."""
        todos = engine.find_todos()
        assert len(todos) >= 1
        assert all(isinstance(t, TodoItem) for t in todos)
        # Check the TODO we created
        todo_found = any(t.todo_type == "TODO" for t in todos)
        assert todo_found
    def test_find_not_implemented_stubs(self, engine, temp_repo):
        """Test NotImplementedError stub detection."""
        # Create a file with NotImplementedError
        (temp_repo / "stub.py").write_text(
            "def my_function():\n    raise NotImplementedError\n"
        )
        stubs = engine.find_not_implemented_stubs()
        assert all(isinstance(s, NotImplementedStub) for s in stubs)
    def test_verify_p0_safety(self, engine, temp_repo):
        """Test P0 safety verification."""
        results = engine.verify_p0_safety()
        assert "checks" in results
        assert "passed" in results
        assert "failed" in results
        assert "warnings" in results
        # Should have at least the CI/CD check pass since we created a workflow
        assert results["passed"] >= 1
    def test_execute_cleanup_dry_run(self, engine, temp_repo):
        """Test cleanup execution in dry-run mode."""
        results = engine.execute_cleanup(phase="todos", dry_run=True)
        assert results["dry_run"] is True
        assert results["files_modified"] == 0
        assert "actions" in results
    def test_run_git_workflow_status(self, engine, temp_repo):
        """Test Git workflow status action."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=" M test.py\n", stderr=""
            )
            results = engine.run_git_workflow(action="status")
            assert results["success"] is True
            assert "status" in results["action"]
    def test_run_git_workflow_branch(self, engine, temp_repo):
        """Test Git workflow branch action."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="main\n", stderr=""
            )
            results = engine.run_git_workflow(action="branch")
            assert results["success"] is True
            assert results["current_branch"] == "main"
    def test_generate_report(self, engine, temp_repo):
        """Test report generation."""
        output_path = temp_repo / "test_report.json"
        report = engine.generate_report(output_path=output_path)
        assert isinstance(report, CleanupReport)
        assert report.phase == "Analysis"
        assert output_path.exists()
    def test_determine_todo_severity_high(self, engine):
        """Test high severity detection for TODOs."""
        severity = engine._determine_todo_severity("FIXME", "critical bug")
        assert severity == "HIGH"
    def test_determine_todo_severity_medium(self, engine):
        """Test medium severity detection for TODOs."""
        severity = engine._determine_todo_severity("TODO", "implement feature")
        assert severity == "MEDIUM"
    def test_determine_todo_severity_low(self, engine):
        """Test low severity detection for TODOs."""
        severity = engine._determine_todo_severity("TODO", "minor note")
        assert severity == "LOW"
    def test_identify_removable_duplicates(self, engine):
        """Test identification of removable duplicates."""
        files = ["legacy/old.py", "services/agents/new.py"]
        removable = engine._identify_removable_duplicates(files)
        assert "legacy/old.py" in removable
    def test_generate_todo_suggestion(self, engine):
        """Test TODO suggestion generation."""
        todo = TodoItem(
            file_path="test.py",
            line_number=1,
            todo_type="HACK",
            message="Quick fix",
            severity="MEDIUM",  # HACK type but MEDIUM severity triggers refactor suggestion
            context="# HACK: Quick fix",
        )
        suggestion = engine._generate_todo_suggestion(todo)
        assert "Refactor" in suggestion
class TestDataClasses:
    """Tests for data classes."""
    def test_todo_item_creation(self):
        """Test TodoItem dataclass."""
        item = TodoItem(
            file_path="test.py",
            line_number=10,
            todo_type="TODO",
            message="Fix this",
            severity="HIGH",
            context="# TODO: Fix this",
        )
        assert item.file_path == "test.py"
        assert item.line_number == 10
    def test_duplicate_group_creation(self):
        """Test DuplicateGroup dataclass."""
        group = DuplicateGroup(
            file_hash="abc123",
            files=["a.py", "b.py"],
            size_bytes=100,
            removable=["b.py"],
        )
        assert len(group.files) == 2
        assert group.size_bytes == 100
    def test_not_implemented_stub_creation(self):
        """Test NotImplementedStub dataclass."""
        stub = NotImplementedStub(
            file_path="test.py",
            function_name="my_func",
            line_number=5,
            class_name="MyClass",
        )
        assert stub.function_name == "my_func"
        assert stub.class_name == "MyClass"
    def test_cleanup_report_creation(self):
        """Test CleanupReport dataclass."""
        report = CleanupReport(
            timestamp="2026-01-21T00:00:00",
            phase="Analysis",
            items_found=10,
            items_fixed=5,
            items_remaining=5,
            files_modified=2,
            lines_added=10,
            lines_removed=5,
        )
        assert report.items_found == 10
        assert report.items_remaining == 5
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
