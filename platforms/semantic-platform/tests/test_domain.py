"""Tests for semantic platform domain entities, value objects, events, and exceptions."""

import pytest

from semantic_platform.domain.entities import (
    Document,
    SemanticVector,
    InferenceRequest,
    InferenceResult,
)
from semantic_platform.domain.value_objects import (
    VectorSpace,
    InferenceStrategy,
    SimilarityMetric,
)
from semantic_platform.domain.events import (
    DocumentIndexed,
    VectorComputed,
    InferenceCompleted,
)
from semantic_platform.domain.exceptions import (
    SemanticError,
    DocumentNotFoundError,
    IndexingError,
    InferenceError,
)


# ============================================================================
# Document
# ============================================================================


class TestDocument:
    def test_defaults(self):
        doc = Document(content="hello world")
        assert doc.content == "hello world"
        assert len(doc.doc_id) == 16
        assert doc.title == ""
        assert doc.tags == []

    def test_custom_fields(self):
        doc = Document(
            content="test content",
            title="Test Doc",
            tags=["python", "ml"],
            metadata={"author": "admin"},
        )
        assert doc.title == "Test Doc"
        assert doc.tags == ["python", "ml"]
        assert doc.metadata["author"] == "admin"


# ============================================================================
# SemanticVector
# ============================================================================


class TestSemanticVector:
    def test_defaults(self):
        sv = SemanticVector(doc_id="d1", vector=[0.1, 0.2, 0.3])
        assert sv.doc_id == "d1"
        assert sv.dimensions == 3
        assert sv.space == VectorSpace.DENSE

    def test_auto_dimensions(self):
        sv = SemanticVector(doc_id="d1", vector=[1.0] * 128)
        assert sv.dimensions == 128

    def test_explicit_dimensions(self):
        sv = SemanticVector(doc_id="d1", vector=[1.0, 2.0], dimensions=2)
        assert sv.dimensions == 2


# ============================================================================
# InferenceRequest
# ============================================================================


class TestInferenceRequest:
    def test_defaults(self):
        req = InferenceRequest(query="test query")
        assert req.query == "test query"
        assert req.strategy == InferenceStrategy.NEAREST_NEIGHBOR
        assert req.metric == SimilarityMetric.COSINE
        assert req.top_k == 5

    def test_custom(self):
        req = InferenceRequest(
            query="search",
            strategy=InferenceStrategy.WEIGHTED,
            metric=SimilarityMetric.EUCLIDEAN,
            top_k=10,
            threshold=0.5,
        )
        assert req.strategy == InferenceStrategy.WEIGHTED
        assert req.threshold == 0.5


# ============================================================================
# InferenceResult
# ============================================================================


class TestInferenceResult:
    def test_defaults(self):
        result = InferenceResult(request_id="r1")
        assert result.matches == []
        assert result.duration_ms == 0.0
        assert result.total_candidates == 0


# ============================================================================
# Events
# ============================================================================


class TestEvents:
    def test_document_indexed(self):
        evt = DocumentIndexed(doc_id="d1", dimensions=128)
        assert evt.doc_id == "d1"
        assert evt.dimensions == 128

    def test_vector_computed(self):
        evt = VectorComputed(doc_id="d1", vector_id="v1", dimensions=64)
        assert evt.vector_id == "v1"

    def test_inference_completed(self):
        evt = InferenceCompleted(
            request_id="r1",
            match_count=5,
            duration_ms=10.5,
            strategy="nearest_neighbor",
        )
        assert evt.match_count == 5


# ============================================================================
# Exceptions
# ============================================================================


class TestExceptions:
    def test_semantic_error(self):
        err = SemanticError("test error")
        assert "test error" in str(err)

    def test_document_not_found(self):
        err = DocumentNotFoundError("doc-123")
        assert err.doc_id == "doc-123"
        assert "doc-123" in str(err)

    def test_indexing_error(self):
        err = IndexingError("bad index", doc_id="d1")
        assert err.doc_id == "d1"

    def test_inference_error(self):
        err = InferenceError("failed", request_id="r1")
        assert err.request_id == "r1"
