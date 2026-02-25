"""Semantic platform domain layer."""

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
    InferenceCompleted,
    VectorComputed,
)
from semantic_platform.domain.exceptions import (
    SemanticError,
    DocumentNotFoundError,
    IndexingError,
    InferenceError,
)

__all__ = [
    "Document",
    "SemanticVector",
    "InferenceRequest",
    "InferenceResult",
    "VectorSpace",
    "InferenceStrategy",
    "SimilarityMetric",
    "DocumentIndexed",
    "InferenceCompleted",
    "VectorComputed",
    "SemanticError",
    "DocumentNotFoundError",
    "IndexingError",
    "InferenceError",
]
