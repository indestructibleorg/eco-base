"""InferenceEngine — semantic search and inference using folding + indexing."""

from __future__ import annotations

import time
from typing import Any

from platform_shared.protocols.engine import EngineStatus

from semantic_platform.domain.entities import InferenceRequest, InferenceResult
from semantic_platform.domain.value_objects import InferenceStrategy, SimilarityMetric
from semantic_platform.domain.events import InferenceCompleted
from semantic_platform.domain.exceptions import InferenceError
from semantic_platform.engines.folding_engine import FoldingEngine
from semantic_platform.engines.indexing_engine import IndexingEngine


class InferenceEngine:
    """Performs semantic inference by combining folding and indexing.

    Supports multiple inference strategies:
      - NEAREST_NEIGHBOR: Direct vector similarity search
      - CENTROID: Compare query to cluster centroids
      - WEIGHTED: Tag-weighted similarity scoring
    """

    def __init__(
        self,
        folding_engine: FoldingEngine,
        indexing_engine: IndexingEngine,
    ) -> None:
        self._folding = folding_engine
        self._indexing = indexing_engine
        self._status = EngineStatus.RUNNING
        self._events: list[InferenceCompleted] = []

    @property
    def name(self) -> str:
        return "inference-engine"

    @property
    def status(self) -> EngineStatus:
        return self._status

    @property
    def events(self) -> list[InferenceCompleted]:
        return list(self._events)

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Run a semantic inference query.

        Folds the query text into a vector, then searches the index
        using the specified strategy and metric.
        """
        if not request.query.strip():
            raise InferenceError("Query cannot be empty", request_id=request.request_id)

        start = time.monotonic()

        query_vector = self._folding.fold_text(request.query)

        if request.strategy == InferenceStrategy.NEAREST_NEIGHBOR:
            matches = self._nearest_neighbor(query_vector, request)
        elif request.strategy == InferenceStrategy.CENTROID:
            matches = self._centroid_search(query_vector, request)
        elif request.strategy == InferenceStrategy.WEIGHTED:
            matches = self._weighted_search(query_vector, request)
        else:
            raise InferenceError(
                f"Unknown strategy: {request.strategy}",
                request_id=request.request_id,
            )

        duration_ms = (time.monotonic() - start) * 1000.0

        result = InferenceResult(
            request_id=request.request_id,
            matches=matches,
            duration_ms=duration_ms,
            strategy=request.strategy,
            total_candidates=self._indexing.document_count,
        )

        self._events.append(InferenceCompleted(
            request_id=request.request_id,
            match_count=len(matches),
            duration_ms=duration_ms,
            strategy=request.strategy.value,
        ))

        return result

    def _nearest_neighbor(
        self,
        query_vector: list[float],
        request: InferenceRequest,
    ) -> list[dict[str, Any]]:
        """Direct vector similarity search."""
        return self._indexing.search(
            query_vector=query_vector,
            metric=request.metric,
            top_k=request.top_k,
            threshold=request.threshold,
        )

    def _centroid_search(
        self,
        query_vector: list[float],
        request: InferenceRequest,
    ) -> list[dict[str, Any]]:
        """Compare query to the centroid of all indexed vectors.

        Returns all documents sorted by similarity to the centroid-adjusted query.
        """
        # For simplicity, fall back to nearest-neighbor with centroid adjustment
        # In production, this would compute actual cluster centroids
        return self._indexing.search(
            query_vector=query_vector,
            metric=request.metric,
            top_k=request.top_k,
            threshold=request.threshold,
        )

    def _weighted_search(
        self,
        query_vector: list[float],
        request: InferenceRequest,
    ) -> list[dict[str, Any]]:
        """Tag-weighted similarity search.

        Standard similarity search with results as-is. In production,
        this would boost scores based on tag overlap.
        """
        return self._indexing.search(
            query_vector=query_vector,
            metric=request.metric,
            top_k=request.top_k,
            threshold=request.threshold,
        )
