"""FoldingEngine — compute vector embeddings from text content.

Uses a simple hash-based folding approach for deterministic, dependency-free
vector generation. Production systems would replace this with a real
embedding model.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from platform_shared.protocols.engine import EngineStatus

from semantic_platform.domain.entities import Document, SemanticVector
from semantic_platform.domain.value_objects import VectorSpace
from semantic_platform.domain.events import VectorComputed
from semantic_platform.domain.exceptions import SemanticError


class FoldingEngine:
    """Compute vector embeddings from document text using semantic folding.

    This implementation uses a deterministic hash-based approach that maps
    text tokens to fixed-dimensional vectors. No external dependencies.
    """

    def __init__(self, dimensions: int = 128) -> None:
        if dimensions < 1:
            raise SemanticError(f"Dimensions must be positive, got {dimensions}")
        self._dimensions = dimensions
        self._status = EngineStatus.RUNNING
        self._events: list[VectorComputed] = []

    @property
    def name(self) -> str:
        return "folding-engine"

    @property
    def status(self) -> EngineStatus:
        return self._status

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def events(self) -> list[VectorComputed]:
        return list(self._events)

    def fold(self, document: Document) -> SemanticVector:
        """Compute a vector embedding for a document.

        Tokenizes the content, hashes each token to generate dimensional
        values, then normalizes the resulting vector to unit length.
        """
        if not document.content.strip():
            raise SemanticError("Cannot fold empty document content")

        raw_vector = self._compute_raw_vector(document.content)
        normalized = self._normalize(raw_vector)

        sv = SemanticVector(
            doc_id=document.doc_id,
            vector=normalized,
            space=VectorSpace.DENSE,
            dimensions=self._dimensions,
        )

        self._events.append(VectorComputed(
            doc_id=document.doc_id,
            vector_id=sv.vector_id,
            dimensions=self._dimensions,
        ))

        return sv

    def fold_text(self, text: str) -> list[float]:
        """Compute a normalized vector for raw text (no document wrapper)."""
        if not text.strip():
            raise SemanticError("Cannot fold empty text")
        raw = self._compute_raw_vector(text)
        return self._normalize(raw)

    def _compute_raw_vector(self, text: str) -> list[float]:
        """Hash-based token folding into a fixed-dimensional vector."""
        vector = [0.0] * self._dimensions
        tokens = text.lower().split()

        for token in tokens:
            h = hashlib.sha256(token.encode()).hexdigest()
            for i in range(self._dimensions):
                # Use pairs of hex chars to generate float values
                start = (i * 2) % len(h)
                byte_val = int(h[start:start + 2], 16)
                # Map to [-1, 1] range
                vector[i] += (byte_val / 127.5) - 1.0

        return vector

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        """L2-normalize a vector to unit length."""
        magnitude = math.sqrt(sum(v * v for v in vector))
        if magnitude == 0:
            return vector
        return [v / magnitude for v in vector]
