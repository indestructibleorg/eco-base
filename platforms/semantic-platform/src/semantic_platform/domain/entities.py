"""Core domain entities for the semantic platform."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from semantic_platform.domain.value_objects import (
    VectorSpace,
    InferenceStrategy,
    SimilarityMetric,
)


@dataclass(slots=True)
class Document:
    """A document that can be indexed and searched semantically."""
    content: str
    doc_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    title: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class SemanticVector:
    """A computed vector embedding for a document."""
    doc_id: str
    vector: list[float]
    space: VectorSpace = VectorSpace.DENSE
    dimensions: int = 0
    vector_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    computed_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.dimensions == 0 and self.vector:
            self.dimensions = len(self.vector)


@dataclass(slots=True)
class InferenceRequest:
    """A request to perform semantic inference (search/similarity)."""
    query: str
    strategy: InferenceStrategy = InferenceStrategy.NEAREST_NEIGHBOR
    metric: SimilarityMetric = SimilarityMetric.COSINE
    top_k: int = 5
    threshold: float = 0.0
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])


@dataclass(slots=True)
class InferenceResult:
    """Result of a semantic inference operation."""
    request_id: str
    matches: list[dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0.0
    strategy: InferenceStrategy = InferenceStrategy.NEAREST_NEIGHBOR
    total_candidates: int = 0
