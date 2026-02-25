"""Value objects for the semantic domain."""

from __future__ import annotations

from enum import Enum


class VectorSpace(str, Enum):
    """Type of vector space for embeddings."""
    DENSE = "dense"
    SPARSE = "sparse"
    BINARY = "binary"


class InferenceStrategy(str, Enum):
    """Strategy used for inference operations."""
    NEAREST_NEIGHBOR = "nearest_neighbor"
    CENTROID = "centroid"
    WEIGHTED = "weighted"


class SimilarityMetric(str, Enum):
    """Metric used for computing vector similarity."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
