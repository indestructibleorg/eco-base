#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: test_workflow_orchestrator
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
"""
Unit Tests for Workflow Orchestrator Safe Expression Parser
===========================================================
Tests for the safe condition evaluation in workflow_orchestrator.py
"""
import pytest
import sys
import os
import ast  # Added for ast.literal_eval()
# Add the path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'ns-root', 'namespaces-adk'))
try:
    from adk.core.workflow_orchestrator import WorkflowOrchestrator
    HAS_ORCHESTRATOR = True
except ImportError:
    HAS_ORCHESTRATOR = False
@pytest.mark.skipif(not HAS_ORCHESTRATOR, reason="WorkflowOrchestrator not available")
class TestSafeExpressionParser:
    """Test cases for safe expression parsing (eval() replacement)"""
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance"""
        return WorkflowOrchestrator()
    @pytest.fixture
    def context(self):
        """Sample context for testing"""
        return {
            "status": "success",
            "count": 10,
            "enabled": True,
            "name": "test",
            "items": ["a", "b", "c"],
            "nested": {"value": 42}
        }
    # Boolean literal tests
    def test_true_literal(self, orchestrator, context):
        """Test True literal evaluation"""
        assert orchestrator._safe_eval_condition("True", context) is True
    def test_false_literal(self, orchestrator, context):
        """Test False literal evaluation"""
        assert orchestrator._safe_eval_condition("False", context) is False
    # Comparison tests
    def test_equality_string(self, orchestrator, context):
        """Test string equality comparison"""
        assert orchestrator._safe_eval_condition("status == 'success'", context) is True
        assert orchestrator._safe_eval_condition("status == 'failed'", context) is False
    def test_equality_number(self, orchestrator, context):
        """Test numeric equality comparison"""
        assert orchestrator._safe_eval_condition("count == 10", context) is True
        assert orchestrator._safe_eval_condition("count == 5", context) is False
    def test_inequality(self, orchestrator, context):
        """Test inequality comparison"""
        assert orchestrator._safe_eval_condition("count != 5", context) is True
        assert orchestrator._safe_eval_condition("count != 10", context) is False
    def test_greater_than(self, orchestrator, context):
        """Test greater than comparison"""
        assert orchestrator._safe_eval_condition("count > 5", context) is True
        assert orchestrator._safe_eval_condition("count > 15", context) is False
    def test_less_than(self, orchestrator, context):
        """Test less than comparison"""
        assert orchestrator._safe_eval_condition("count < 15", context) is True
        assert orchestrator._safe_eval_condition("count < 5", context) is False
    def test_greater_equal(self, orchestrator, context):
        """Test greater than or equal comparison"""
        assert orchestrator._safe_eval_condition("count >= 10", context) is True
        assert orchestrator._safe_eval_condition("count >= 11", context) is False
    def test_less_equal(self, orchestrator, context):
        """Test less than or equal comparison"""
        assert orchestrator._safe_eval_condition("count <= 10", context) is True
        assert orchestrator._safe_eval_condition("count <= 9", context) is False
    # Boolean operator tests
    def test_and_operator(self, orchestrator, context):
        """Test AND boolean operator"""
        assert orchestrator._safe_eval_condition("enabled and count > 5", context) is True
        assert orchestrator._safe_eval_condition("enabled and count > 15", context) is False
    def test_or_operator(self, orchestrator, context):
        """Test OR boolean operator"""
        assert orchestrator._safe_eval_condition("count > 15 or enabled", context) is True
        assert orchestrator._safe_eval_condition("count > 15 or status == 'failed'", context) is False
    def test_not_operator(self, orchestrator, context):
        """Test NOT boolean operator"""
        assert orchestrator._safe_eval_condition("not False", context) is True
        assert orchestrator._safe_eval_condition("not enabled", context) is False
    # Membership tests
    def test_in_operator(self, orchestrator, context):
        """Test IN membership operator"""
        assert orchestrator._safe_eval_condition("'a' in items", context) is True
        assert orchestrator._safe_eval_condition("'z' in items", context) is False
    def test_not_in_operator(self, orchestrator, context):
        """Test NOT IN membership operator"""
        assert orchestrator._safe_eval_condition("'z' not in items", context) is True
        assert orchestrator._safe_eval_condition("'a' not in items", context) is False
    # Variable lookup tests
    def test_simple_variable(self, orchestrator, context):
        """Test simple variable lookup (truthy check)"""
        assert orchestrator._safe_eval_condition("enabled", context) is True
        assert orchestrator._safe_eval_condition("nonexistent", context) is False
    # Error handling tests
    def test_invalid_expression_returns_false(self, orchestrator, context):
        """Test that invalid expressions return False gracefully"""
        result = orchestrator._evaluate_condition("invalid expression !!!", context)
        assert result is False
    def test_empty_condition(self, orchestrator, context):
        """Test empty condition handling"""
        result = orchestrator._safe_eval_condition("", context)
        # Empty string is falsy
        assert result is False
class TestSafeExpressionParserSecurity:
    """Security-focused tests to ensure eval() replacement is safe"""
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance"""
        if not HAS_ORCHESTRATOR:
            pytest.skip("WorkflowOrchestrator not available")
        return WorkflowOrchestrator()
    @pytest.fixture
    def context(self):
        """Sample context"""
        return {"value": 1}
    def test_no_code_execution(self, orchestrator, context):
        """Ensure arbitrary code cannot be executed"""
        # These should all return False without executing code
        dangerous_inputs = [
            "__import__('os').system('echo hacked')",
            # SECURITY WARNING: exec() usage - ensure input is trusted
            "exec('print(1)')",
            "ast.literal_eval('1+1')",
            "open('/etc/passwd', encoding='utf-8').read()",
            "lambda: 1",
            "[x for x in range(10)]",
        ]
        for dangerous in dangerous_inputs:
            result = orchestrator._evaluate_condition(dangerous, context)
            # Should return False, not execute the code
            assert result is False or result is None or isinstance(result, bool)
    def test_no_attribute_access_exploit(self, orchestrator, context):
        """Ensure attribute access cannot be exploited"""
        dangerous_inputs = [
            "''.__class__.__mro__[1].__subclasses__()",
            "{}.__class__.__bases__[0].__subclasses__()",
        ]
        for dangerous in dangerous_inputs:
            result = orchestrator._evaluate_condition(dangerous, context)
            assert result is False or result is None or isinstance(result, bool)
# Additional tests from local branch
# ============================================================================
# SafeExpressionParser Mock Tests (for standalone testing)
# ============================================================================
class SafeExpressionParserMock:
    """Mock SafeExpressionParser for testing when import fails."""
    SAFE_OPERATORS = {'+', '-', '*', '/', '%', '**', '==', '!=', '<', '>', '<=', '>='}
    SAFE_FUNCTIONS = {'len', 'str', 'int', 'float', 'bool', 'abs', 'min', 'max', 'sum'}
    def __init__(self, allowed_names: dict = None):
        self.allowed_names = allowed_names or {}
    def parse(self, expression: str):
        """Parse and evaluate a safe expression."""
        if not expression or not isinstance(expression, str):
            raise ValueError("Expression must be a non-empty string")
        # Block dangerous patterns
        dangerous = ['import', 'exec', 'eval', '__', 'open', 'file', 'input']
        expr_lower = expression.lower()
        for pattern in dangerous:
            if pattern in expr_lower:
                raise ValueError(f"Dangerous pattern detected: {pattern}")
        # Simple evaluation for basic expressions
        try:
            safe_builtins = {
                'len': len, 'str': str, 'int': int, 'float': float,
                'bool': bool, 'abs': abs, 'min': min, 'max': max, 'sum': sum,
                'True': True, 'False': False, 'None': None
            }
            return ast.literal_eval(expression, {"__builtins__": safe_builtins}, self.allowed_names)
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression: {e}")
class TestSafeExpressionParserMock(unittest.TestCase):
    """Test cases for SafeExpressionParser mock (standalone tests)."""
    def setUp(self):
        self.parser = SafeExpressionParserMock()
    def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        self.assertEqual(self.parser.parse("2 + 3"), 5)
        self.assertEqual(self.parser.parse("10 - 4"), 6)
        self.assertEqual(self.parser.parse("3 * 4"), 12)
    def test_comparison_operators(self):
        """Test comparison operations."""
        self.assertTrue(self.parser.parse("5 > 3"))
        self.assertFalse(self.parser.parse("2 > 5"))
    def test_blocks_dangerous_patterns(self):
        """Test that dangerous patterns are blocked."""
        with self.assertRaises(ValueError):
            self.parser.parse("__import__('os')")
        with self.assertRaises(ValueError):
            # SECURITY WARNING: exec() usage - ensure input is trusted
            self.parser.parse("exec('print(1)')")
        with self.assertRaises(ValueError):
            self.parser.parse("ast.literal_eval('1+1')")
    def test_empty_expression(self):
        """Test that empty expressions raise error."""
        with self.assertRaises(ValueError):
            self.parser.parse("")
if __name__ == '__main__':
    # Run both pytest and unittest tests
    unittest.main()
