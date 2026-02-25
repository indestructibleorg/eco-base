"""Tests for semantic platform engines — FoldingEngine, IndexingEngine, InferenceEngine."""

import math

import pytest

from semantic_platform.domain.entities import Document, SemanticVector, InferenceRequest
from semantic_platform.domain.value_objects import (
    VectorSpace,
    InferenceStrategy,
    SimilarityMetric,
)
from semantic_platform.domain.exceptions import (
    SemanticError,
    DocumentNotFoundError,
    IndexingError,
    InferenceError,
)
from semantic_platform.engines.folding_engine import FoldingEngine
from semantic_platform.engines.indexing_engine import IndexingEngine
from semantic_platform.engines.inference_engine import InferenceEngine


# ============================================================================
# FoldingEngine
# ============================================================================


class TestFoldingEngine:
    def test_fold_document(self, folding_engine: FoldingEngine):
        doc = Document(content="hello world")
        sv = folding_engine.fold(doc)
        assert sv.doc_id == doc.doc_id
        assert sv.dimensions == 64
        assert sv.space == VectorSpace.DENSE
        assert len(sv.vector) == 64

    def test_fold_is_normalized(self, folding_engine: FoldingEngine):
        doc = Document(content="test content for normalization")
        sv = folding_engine.fold(doc)
        magnitude = math.sqrt(sum(v * v for v in sv.vector))
        assert abs(magnitude - 1.0) < 1e-6

    def test_fold_deterministic(self, folding_engine: FoldingEngine):
        doc1 = Document(content="same content", doc_id="d1")
        doc2 = Document(content="same content", doc_id="d2")
        sv1 = folding_engine.fold(doc1)
        sv2 = folding_engine.fold(doc2)
        assert sv1.vector == sv2.vector

    def test_fold_different_content_different_vectors(self, folding_engine: FoldingEngine):
        doc1 = Document(content="hello world")
        doc2 = Document(content="completely different text about python programming")
        sv1 = folding_engine.fold(doc1)
        sv2 = folding_engine.fold(doc2)
        assert sv1.vector != sv2.vector

    def test_fold_empty_content_raises(self, folding_engine: FoldingEngine):
        doc = Document(content="   ")
        with pytest.raises(SemanticError, match="empty"):
            folding_engine.fold(doc)

    def test_fold_text(self, folding_engine: FoldingEngine):
        vector = folding_engine.fold_text("hello world")
        assert len(vector) == 64
        magnitude = math.sqrt(sum(v * v for v in vector))
        assert abs(magnitude - 1.0) < 1e-6

    def test_fold_text_empty_raises(self, folding_engine: FoldingEngine):
        with pytest.raises(SemanticError, match="empty"):
            folding_engine.fold_text("")

    def test_events_emitted(self, folding_engine: FoldingEngine):
        doc = Document(content="test")
        folding_engine.fold(doc)
        assert len(folding_engine.events) == 1
        assert folding_engine.events[0].doc_id == doc.doc_id

    def test_custom_dimensions(self):
        engine = FoldingEngine(dimensions=32)
        assert engine.dimensions == 32
        doc = Document(content="test")
        sv = engine.fold(doc)
        assert len(sv.vector) == 32

    def test_invalid_dimensions_raises(self):
        with pytest.raises(SemanticError):
            FoldingEngine(dimensions=0)

    def test_name_property(self, folding_engine: FoldingEngine):
        assert folding_engine.name == "folding-engine"


# ============================================================================
# IndexingEngine
# ============================================================================


class TestIndexingEngine:
    def test_index_and_get(self, indexing_engine: IndexingEngine, folding_engine: FoldingEngine):
        doc = Document(content="test document")
        vector = folding_engine.fold(doc)
        indexing_engine.index(doc, vector)

        retrieved = indexing_engine.get_document(doc.doc_id)
        assert retrieved.content == "test document"

    def test_get_vector(self, indexing_engine: IndexingEngine, folding_engine: FoldingEngine):
        doc = Document(content="test")
        vector = folding_engine.fold(doc)
        indexing_engine.index(doc, vector)

        sv = indexing_engine.get_vector(doc.doc_id)
        assert sv.doc_id == doc.doc_id

    def test_get_nonexistent_raises(self, indexing_engine: IndexingEngine):
        with pytest.raises(DocumentNotFoundError):
            indexing_engine.get_document("missing")

    def test_id_mismatch_raises(self, indexing_engine: IndexingEngine):
        doc = Document(content="test", doc_id="d1")
        vector = SemanticVector(doc_id="d2", vector=[1.0, 2.0])
        with pytest.raises(IndexingError, match="mismatch"):
            indexing_engine.index(doc, vector)

    def test_remove(self, indexing_engine: IndexingEngine, folding_engine: FoldingEngine):
        doc = Document(content="to remove")
        vector = folding_engine.fold(doc)
        indexing_engine.index(doc, vector)

        removed = indexing_engine.remove(doc.doc_id)
        assert removed.doc_id == doc.doc_id

        with pytest.raises(DocumentNotFoundError):
            indexing_engine.get_document(doc.doc_id)

    def test_document_count(self, indexing_engine: IndexingEngine, folding_engine: FoldingEngine):
        for i in range(3):
            doc = Document(content=f"doc {i}")
            vector = folding_engine.fold(doc)
            indexing_engine.index(doc, vector)
        assert indexing_engine.document_count == 3

    def test_search_cosine(self, indexing_engine: IndexingEngine, folding_engine: FoldingEngine):
        # Index some documents
        docs = [
            Document(content="python programming language", title="Python"),
            Document(content="java programming language", title="Java"),
            Document(content="cooking recipes food", title="Cooking"),
        ]
        for doc in docs:
            vector = folding_engine.fold(doc)
            indexing_engine.index(doc, vector)

        # Search for programming-related content
        query_vector = folding_engine.fold_text("programming language")
        results = indexing_engine.search(query_vector, metric=SimilarityMetric.COSINE, top_k=2)
        assert len(results) <= 2
        assert all("score" in r for r in results)

    def test_search_euclidean(self, indexing_engine: IndexingEngine, folding_engine: FoldingEngine):
        doc = Document(content="test content")
        vector = folding_engine.fold(doc)
        indexing_engine.index(doc, vector)

        query_vector = folding_engine.fold_text("test content")
        results = indexing_engine.search(query_vector, metric=SimilarityMetric.EUCLIDEAN)
        assert len(results) == 1
        assert results[0]["score"] > 0

    def test_search_dot_product(self, indexing_engine: IndexingEngine, folding_engine: FoldingEngine):
        doc = Document(content="test content")
        vector = folding_engine.fold(doc)
        indexing_engine.index(doc, vector)

        query_vector = folding_engine.fold_text("test content")
        results = indexing_engine.search(query_vector, metric=SimilarityMetric.DOT_PRODUCT)
        assert len(results) >= 1

    def test_search_with_threshold(self, indexing_engine: IndexingEngine, folding_engine: FoldingEngine):
        doc = Document(content="very specific content about quantum physics")
        vector = folding_engine.fold(doc)
        indexing_engine.index(doc, vector)

        query_vector = folding_engine.fold_text("completely unrelated topic about cooking")
        results = indexing_engine.search(query_vector, threshold=0.99)
        # High threshold should filter out low-similarity results
        assert len(results) <= 1

    def test_list_documents(self, indexing_engine: IndexingEngine, folding_engine: FoldingEngine):
        doc = Document(content="test")
        vector = folding_engine.fold(doc)
        indexing_engine.index(doc, vector)
        docs = indexing_engine.list_documents()
        assert len(docs) == 1

    def test_clear(self, indexing_engine: IndexingEngine, folding_engine: FoldingEngine):
        doc = Document(content="test")
        vector = folding_engine.fold(doc)
        indexing_engine.index(doc, vector)
        indexing_engine.clear()
        assert indexing_engine.document_count == 0

    def test_events_emitted(self, indexing_engine: IndexingEngine, folding_engine: FoldingEngine):
        doc = Document(content="test")
        vector = folding_engine.fold(doc)
        indexing_engine.index(doc, vector)
        assert len(indexing_engine.events) == 1

    def test_name_property(self, indexing_engine: IndexingEngine):
        assert indexing_engine.name == "indexing-engine"

    def test_similarity_dimension_mismatch(self, indexing_engine: IndexingEngine):
        score = IndexingEngine._compute_similarity([1.0, 2.0], [1.0], SimilarityMetric.COSINE)
        assert score == 0.0


# ============================================================================
# InferenceEngine
# ============================================================================


class TestInferenceEngine:
    def _index_docs(self, folding_engine: FoldingEngine, indexing_engine: IndexingEngine):
        """Helper to index a few test documents."""
        docs = [
            Document(content="python machine learning artificial intelligence", title="ML"),
            Document(content="web development javascript react framework", title="Web"),
            Document(content="database sql postgres optimization", title="DB"),
        ]
        for doc in docs:
            vector = folding_engine.fold(doc)
            indexing_engine.index(doc, vector)

    def test_infer_nearest_neighbor(
        self,
        inference_engine: InferenceEngine,
        folding_engine: FoldingEngine,
        indexing_engine: IndexingEngine,
    ):
        self._index_docs(folding_engine, indexing_engine)
        request = InferenceRequest(query="machine learning python")
        result = inference_engine.infer(request)
        assert result.request_id == request.request_id
        assert result.total_candidates == 3
        assert len(result.matches) <= 5

    def test_infer_centroid(
        self,
        inference_engine: InferenceEngine,
        folding_engine: FoldingEngine,
        indexing_engine: IndexingEngine,
    ):
        self._index_docs(folding_engine, indexing_engine)
        request = InferenceRequest(
            query="python programming",
            strategy=InferenceStrategy.CENTROID,
        )
        result = inference_engine.infer(request)
        assert result.strategy == InferenceStrategy.CENTROID

    def test_infer_weighted(
        self,
        inference_engine: InferenceEngine,
        folding_engine: FoldingEngine,
        indexing_engine: IndexingEngine,
    ):
        self._index_docs(folding_engine, indexing_engine)
        request = InferenceRequest(
            query="web framework",
            strategy=InferenceStrategy.WEIGHTED,
        )
        result = inference_engine.infer(request)
        assert result.strategy == InferenceStrategy.WEIGHTED

    def test_infer_empty_query_raises(self, inference_engine: InferenceEngine):
        request = InferenceRequest(query="  ")
        with pytest.raises(InferenceError, match="empty"):
            inference_engine.infer(request)

    def test_infer_with_top_k(
        self,
        inference_engine: InferenceEngine,
        folding_engine: FoldingEngine,
        indexing_engine: IndexingEngine,
    ):
        self._index_docs(folding_engine, indexing_engine)
        request = InferenceRequest(query="python", top_k=1)
        result = inference_engine.infer(request)
        assert len(result.matches) <= 1

    def test_events_emitted(
        self,
        inference_engine: InferenceEngine,
        folding_engine: FoldingEngine,
        indexing_engine: IndexingEngine,
    ):
        self._index_docs(folding_engine, indexing_engine)
        request = InferenceRequest(query="test query")
        inference_engine.infer(request)
        assert len(inference_engine.events) == 1
        assert inference_engine.events[0].request_id == request.request_id

    def test_name_property(self, inference_engine: InferenceEngine):
        assert inference_engine.name == "inference-engine"

    def test_duration_recorded(
        self,
        inference_engine: InferenceEngine,
        folding_engine: FoldingEngine,
        indexing_engine: IndexingEngine,
    ):
        self._index_docs(folding_engine, indexing_engine)
        request = InferenceRequest(query="test")
        result = inference_engine.infer(request)
        assert result.duration_ms >= 0
