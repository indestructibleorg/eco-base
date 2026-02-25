#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: test_marker_detection
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
# GL Layer: GL30-49 Execution Layer
# Purpose: Unit tests for Marker Detection (Phase 6.2)
"""
Unit tests for marker detection functionality in autonomous_cleanup_toolkit.
Tests the _determine_todo_severity method and related TODO marker
detection functionality, ensuring alignment with GL gl-platform.gl-platform.governance layer
and scan_tech_debt.py severity determination logic.
"""
import sys
from pathlib import Path
import pytest
# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))
from autonomous_cleanup_toolkit import (
    AutonomousCleanupEngine,
    TodoItem,
)
class TestDetermineTodoSeverity:
    """Tests for the _determine_todo_severity method."""
    @pytest.fixture
    def engine(self, tmp_path):
        """Create an engine instance for testing."""
        return AutonomousCleanupEngine(tmp_path)
    # HIGH severity tests
    def test_security_keyword_returns_high(self, engine):
        """Security keyword should return HIGH severity."""
        result = engine._determine_todo_severity("TODO", "Fix security vulnerability")
        assert result == "HIGH"
    def test_critical_keyword_returns_high(self, engine):
        """Critical keyword should return HIGH severity."""
        result = engine._determine_todo_severity("TODO", "Critical bug fix needed")
        assert result == "HIGH"
    def test_urgent_keyword_returns_high(self, engine):
        """Urgent keyword should return HIGH severity."""
        result = engine._determine_todo_severity("TODO", "Urgent: fix this issue")
        assert result == "HIGH"
    def test_bug_keyword_returns_high(self, engine):
        """Bug keyword should return HIGH severity."""
        result = engine._determine_todo_severity("TODO", "Bug in calculation logic")
        assert result == "HIGH"
    def test_broken_keyword_returns_high(self, engine):
        """Broken keyword should return HIGH severity (aligned with scan_tech_debt.py)."""
        result = engine._determine_todo_severity("TODO", "This feature is broken")
        assert result == "HIGH"
    def test_fix_immediately_keyword_returns_high(self, engine):
        """'Fix immediately' phrase should return HIGH severity."""
        result = engine._determine_todo_severity("TODO", "Fix immediately before release")
        assert result == "HIGH"
    def test_hack_type_returns_high(self, engine):
        """HACK type should return HIGH severity."""
        result = engine._determine_todo_severity("HACK", "Simple message")
        assert result == "HIGH"
    def test_deprecated_type_returns_high(self, engine):
        """DEPRECATED type should return HIGH severity."""
        result = engine._determine_todo_severity("DEPRECATED", "Old API")
        assert result == "HIGH"
    # MEDIUM severity tests
    def test_important_keyword_returns_medium(self, engine):
        """Important keyword should return MEDIUM severity (aligned with scan_tech_debt.py)."""
        result = engine._determine_todo_severity("TODO", "Important refactoring needed")
        assert result == "MEDIUM"
    def test_should_keyword_returns_medium(self, engine):
        """'Should' keyword should return MEDIUM severity (aligned with scan_tech_debt.py)."""
        result = engine._determine_todo_severity("TODO", "Should add more tests")
        assert result == "MEDIUM"
    def test_refactor_keyword_returns_medium(self, engine):
        """Refactor keyword should return MEDIUM severity."""
        result = engine._determine_todo_severity("TODO", "Refactor this function")
        assert result == "MEDIUM"
    def test_improve_keyword_returns_medium(self, engine):
        """Improve keyword should return MEDIUM severity (aligned with scan_tech_debt.py)."""
        result = engine._determine_todo_severity("TODO", "Improve performance")
        assert result == "MEDIUM"
    def test_fixme_type_returns_medium(self, engine):
        """FIXME type should return MEDIUM severity (aligned with scan_tech_debt.py)."""
        result = engine._determine_todo_severity("FIXME", "Simple message")
        assert result == "MEDIUM"
    def test_xxx_type_returns_medium(self, engine):
        """XXX type should return MEDIUM severity (aligned with scan_tech_debt.py)."""
        result = engine._determine_todo_severity("XXX", "Something to check")
        assert result == "MEDIUM"
    # LOW severity tests
    def test_simple_todo_returns_low(self, engine):
        """Simple TODO without keywords should return LOW severity."""
        result = engine._determine_todo_severity("TODO", "Clean up later")
        assert result == "LOW"
    def test_todo_type_no_keywords_returns_low(self, engine):
        """TODO type without priority keywords should return LOW severity."""
        result = engine._determine_todo_severity("TODO", "Maybe consider this")
        assert result == "LOW"
    # Case insensitivity tests
    def test_keywords_case_insensitive(self, engine):
        """Keywords should be case insensitive."""
        result_lower = engine._determine_todo_severity("TODO", "security issue")
        result_upper = engine._determine_todo_severity("TODO", "SECURITY ISSUE")
        result_mixed = engine._determine_todo_severity("TODO", "Security Issue")
        assert result_lower == "HIGH"
        assert result_upper == "HIGH"
        assert result_mixed == "HIGH"
class TestFindTodos:
    """Tests for the find_todos method."""
    @pytest.fixture
    def temp_repo(self, tmp_path):
        """Create a temporary repository structure for testing."""
        # Create a Python file with various TODO markers
        test_file = tmp_path / "test_file.py"
        test_file.write_text("""
# TODO: Add more tests
def some_function():
    # FIXME: This needs to be fixed
    pass
# HACK: Workaround for issue #123
class SomeClass:
    # XXX: Check this logic
    def method(self):
        # TODO: Security vulnerability here
        pass
""")
        return tmp_path
    def test_find_todos_finds_all_markers(self, temp_repo):
        """Test that find_todos finds all TODO markers."""
        engine = AutonomousCleanupEngine(temp_repo)
        todos = engine.find_todos()
        # Expected markers in test file: 2x TODO, 1x FIXME, 1x HACK, 1x XXX
        expected_marker_count = 5
        assert len(todos) == expected_marker_count
    def test_find_todos_correct_types(self, temp_repo):
        """Test that find_todos identifies correct marker types."""
        engine = AutonomousCleanupEngine(temp_repo)
        todos = engine.find_todos()
        types = {t.todo_type for t in todos}
        assert "TODO" in types
        assert "FIXME" in types
        assert "HACK" in types
        assert "XXX" in types
    def test_find_todos_returns_todoitem_instances(self, temp_repo):
        """Test that find_todos returns TodoItem instances."""
        engine = AutonomousCleanupEngine(temp_repo)
        todos = engine.find_todos()
        for todo in todos:
            assert isinstance(todo, TodoItem)
    def test_find_todos_severity_assignment(self, temp_repo):
        """Test that find_todos assigns severities correctly."""
        engine = AutonomousCleanupEngine(temp_repo)
        todos = engine.find_todos()
        severities = {t.severity for t in todos}
        assert severities.issubset({"HIGH", "MEDIUM", "LOW"})
class TestTodoItemDataclass:
    """Tests for the TodoItem dataclass."""
    def test_todoitem_creation(self):
        """Test creating a TodoItem instance."""
        item = TodoItem(
            file_path="test.py",
            line_number=10,
            todo_type="TODO",
            message="Test message",
            severity="MEDIUM",
            context="# TODO: Test message"
        )
        assert item.file_path == "test.py"
        assert item.line_number == 10
        assert item.todo_type == "TODO"
        assert item.message == "Test message"
        assert item.severity == "MEDIUM"
        assert item.context == "# TODO: Test message"
    def test_todoitem_required_fields(self):
        """Test that TodoItem requires all fields."""
        with pytest.raises(TypeError):
            TodoItem(file_path="test.py")  # Missing required fields
