"""Domain events for the semantic platform."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DocumentIndexed:
    """Emitted when a document is successfully indexed."""
    doc_id: str
    dimensions: int
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class VectorComputed:
    """Emitted when a vector embedding is computed for a document."""
    doc_id: str
    vector_id: str
    dimensions: int
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class InferenceCompleted:
    """Emitted when an inference query completes."""
    request_id: str
    match_count: int
    duration_ms: float
    strategy: str = "nearest_neighbor"
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)
