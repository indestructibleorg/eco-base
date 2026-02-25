"""IndexingEngine — in-memory vector index with document storage and search."""

from __future__ import annotations

import math
from typing import Any

from platform_shared.protocols.engine import EngineStatus

from semantic_platform.domain.entities import Document, SemanticVector
from semantic_platform.domain.value_objects import SimilarityMetric
from semantic_platform.domain.events import DocumentIndexed
from semantic_platform.domain.exceptions import DocumentNotFoundError, IndexingError


class IndexingEngine:
    """In-memory vector index that stores documents and their embeddings.

    Supports similarity search using cosine, euclidean, or dot product metrics.
    """

    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}
        self._vectors: dict[str, SemanticVector] = {}  # keyed by doc_id
        self._status = EngineStatus.RUNNING
        self._events: list[DocumentIndexed] = []

    @property
    def name(self) -> str:
        return "indexing-engine"

    @property
    def status(self) -> EngineStatus:
        return self._status

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def events(self) -> list[DocumentIndexed]:
        return list(self._events)

    def index(self, document: Document, vector: SemanticVector) -> None:
        """Index a document with its vector embedding."""
        if document.doc_id != vector.doc_id:
            raise IndexingError(
                f"Document ID mismatch: {document.doc_id} != {vector.doc_id}",
                doc_id=document.doc_id,
            )

        self._documents[document.doc_id] = document
        self._vectors[document.doc_id] = vector

        self._events.append(DocumentIndexed(
            doc_id=document.doc_id,
            dimensions=vector.dimensions,
        ))

    def get_document(self, doc_id: str) -> Document:
        """Retrieve a document by ID."""
        try:
            return self._documents[doc_id]
        except KeyError:
            raise DocumentNotFoundError(doc_id)

    def get_vector(self, doc_id: str) -> SemanticVector:
        """Retrieve a vector embedding by document ID."""
        try:
            return self._vectors[doc_id]
        except KeyError:
            raise DocumentNotFoundError(doc_id)

    def remove(self, doc_id: str) -> Document:
        """Remove a document and its vector from the index."""
        doc = self.get_document(doc_id)
        del self._documents[doc_id]
        self._vectors.pop(doc_id, None)
        return doc

    def search(
        self,
        query_vector: list[float],
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Search for similar documents by vector.

        Returns a list of matches sorted by similarity (descending).
        Each match contains: doc_id, score, title, tags.
        """
        results: list[dict[str, Any]] = []

        for doc_id, sv in self._vectors.items():
            score = self._compute_similarity(query_vector, sv.vector, metric)
            if score >= threshold:
                doc = self._documents[doc_id]
                results.append({
                    "doc_id": doc_id,
                    "score": score,
                    "title": doc.title,
                    "tags": doc.tags,
                })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def list_documents(self) -> list[Document]:
        """Return all indexed documents."""
        return list(self._documents.values())

    def clear(self) -> None:
        """Clear all indexed data."""
        self._documents.clear()
        self._vectors.clear()

    @staticmethod
    def _compute_similarity(
        a: list[float],
        b: list[float],
        metric: SimilarityMetric,
    ) -> float:
        """Compute similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        if metric == SimilarityMetric.COSINE:
            dot = sum(x * y for x, y in zip(a, b))
            mag_a = math.sqrt(sum(x * x for x in a))
            mag_b = math.sqrt(sum(x * x for x in b))
            if mag_a == 0 or mag_b == 0:
                return 0.0
            return dot / (mag_a * mag_b)

        elif metric == SimilarityMetric.DOT_PRODUCT:
            return sum(x * y for x, y in zip(a, b))

        elif metric == SimilarityMetric.EUCLIDEAN:
            dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
            # Convert distance to similarity: 1 / (1 + dist)
            return 1.0 / (1.0 + dist)

        return 0.0
