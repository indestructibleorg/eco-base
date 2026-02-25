"""Tests for semantic platform sandbox."""

import pytest

from semantic_platform.sandbox.semantic_sandbox import SemanticSandbox, SandboxResult
from semantic_platform.domain.value_objects import InferenceStrategy


class TestSemanticSandbox:
    def test_create_sandbox(self):
        sb = SemanticSandbox()
        assert sb.sandbox_id.startswith("sem-sb-")
        assert sb.is_active is True

    def test_index_document(self):
        sb = SemanticSandbox()
        result = sb.index_document(
            content="hello world",
            title="Test",
            tags=["test"],
        )
        assert result.success is True
        assert result.operation == "index_document"
        assert result.result["title"] == "Test"
        assert result.result["dimensions"] == 64

    def test_index_empty_content(self):
        sb = SemanticSandbox()
        result = sb.index_document(content="  ")
        assert result.success is False
        assert any("empty" in e for e in result.errors)

    def test_query_after_indexing(self):
        sb = SemanticSandbox()
        sb.index_document(content="python machine learning")
        sb.index_document(content="web development javascript")

        result = sb.query("python programming")
        assert result.success is True
        assert result.result["total_candidates"] == 2

    def test_query_empty_text(self):
        sb = SemanticSandbox()
        result = sb.query("  ")
        assert result.success is False

    def test_query_with_strategy(self):
        sb = SemanticSandbox()
        sb.index_document(content="test content")
        result = sb.query("test", strategy=InferenceStrategy.CENTROID)
        assert result.success is True

    def test_operations_log(self):
        sb = SemanticSandbox()
        sb.index_document(content="doc one")
        sb.query("search term")
        log = sb.get_operations_log()
        assert len(log) == 2
        assert log[0].operation == "index_document"
        assert log[1].operation == "query"

    def test_destroy(self):
        sb = SemanticSandbox()
        sb.destroy()
        assert sb.is_active is False

    def test_operations_after_destroy_raise(self):
        sb = SemanticSandbox()
        sb.destroy()
        with pytest.raises(RuntimeError, match="destroyed"):
            sb.index_document(content="test")

    def test_custom_dimensions(self):
        sb = SemanticSandbox(dimensions=32)
        result = sb.index_document(content="test content")
        assert result.success is True
        assert result.result["dimensions"] == 32
