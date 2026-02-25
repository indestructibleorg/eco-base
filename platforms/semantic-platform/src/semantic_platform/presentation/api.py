"""Semantic Platform API.

Provides the following endpoints (in-memory, framework-agnostic):
  POST /documents         — index a document
  GET  /documents/{id}    — retrieve a document
  POST /search            — semantic similarity search
  GET  /health            — platform health check
"""

from __future__ import annotations

from typing import Any

from platform_shared.protocols.health import HealthReport, HealthStatus

from semantic_platform.domain.entities import Document, InferenceRequest
from semantic_platform.domain.value_objects import InferenceStrategy, SimilarityMetric
from semantic_platform.engines.folding_engine import FoldingEngine
from semantic_platform.engines.indexing_engine import IndexingEngine
from semantic_platform.engines.inference_engine import InferenceEngine


class SemanticAPI:
    """Framework-agnostic API facade for semantic operations.

    Each method returns a tuple of (response_dict, http_status_code).
    """

    def __init__(
        self,
        folding_engine: FoldingEngine | None = None,
        indexing_engine: IndexingEngine | None = None,
        inference_engine: InferenceEngine | None = None,
    ) -> None:
        self._folding = folding_engine or FoldingEngine()
        self._indexing = indexing_engine or IndexingEngine()
        self._inference = inference_engine or InferenceEngine(self._folding, self._indexing)

    # POST /documents
    def post_document(self, body: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """Index a new document."""
        try:
            content = body["content"]
            title = body.get("title", "")
            tags = body.get("tags", [])

            doc = Document(content=content, title=title, tags=tags)
            vector = self._folding.fold(doc)
            self._indexing.index(doc, vector)

            return {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "dimensions": vector.dimensions,
            }, 201
        except KeyError as exc:
            return {"error": f"Missing required field: {exc}"}, 400
        except Exception as exc:
            return {"error": str(exc)}, 400

    # GET /documents/{id}
    def get_document(self, doc_id: str) -> tuple[dict[str, Any], int]:
        """Retrieve a document by ID."""
        try:
            doc = self._indexing.get_document(doc_id)
            return {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "content": doc.content,
                "tags": doc.tags,
            }, 200
        except Exception:
            return {"error": f"Document not found: {doc_id}"}, 404

    # POST /search
    def post_search(self, body: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """Perform semantic similarity search."""
        try:
            query = body["query"]
            top_k = body.get("top_k", 5)
            strategy = InferenceStrategy(body.get("strategy", "nearest_neighbor"))
            metric = SimilarityMetric(body.get("metric", "cosine"))
            threshold = body.get("threshold", 0.0)

            request = InferenceRequest(
                query=query,
                strategy=strategy,
                metric=metric,
                top_k=top_k,
                threshold=threshold,
            )
            result = self._inference.infer(request)

            return {
                "matches": result.matches,
                "match_count": len(result.matches),
                "duration_ms": result.duration_ms,
                "strategy": result.strategy.value,
            }, 200
        except KeyError as exc:
            return {"error": f"Missing required field: {exc}"}, 400
        except Exception as exc:
            return {"error": str(exc)}, 400

    # GET /health
    def get_health(self) -> tuple[dict[str, Any], int]:
        """Return platform health status."""
        report = HealthReport(
            component="semantic-platform",
            status=HealthStatus.HEALTHY,
            message="All engines operational",
            details={
                "folding_engine": self._folding.status.value,
                "indexing_engine": self._indexing.status.value,
                "inference_engine": self._inference.status.value,
                "indexed_documents": self._indexing.document_count,
                "vector_dimensions": self._folding.dimensions,
            },
        )
        return {
            "component": report.component,
            "status": report.status.value,
            "message": report.message,
            "details": report.details,
            "is_healthy": report.is_healthy,
        }, 200
