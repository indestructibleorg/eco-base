"""SemanticSandbox — isolated environment for testing semantic operations.

Provides a self-contained sandbox with its own folding, indexing, and
inference engines to validate documents and queries without affecting
production state.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from semantic_platform.domain.entities import Document, InferenceRequest
from semantic_platform.domain.value_objects import InferenceStrategy, SimilarityMetric
from semantic_platform.engines.folding_engine import FoldingEngine
from semantic_platform.engines.indexing_engine import IndexingEngine
from semantic_platform.engines.inference_engine import InferenceEngine


@dataclass(slots=True)
class SandboxResult:
    """Result of a sandbox operation."""
    sandbox_id: str
    operation: str
    success: bool
    result: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class SemanticSandbox:
    """Isolated sandbox for testing semantic operations.

    Creates its own engine instances so that indexing and inference
    can be tested without modifying real indices.
    """

    def __init__(self, dimensions: int = 64) -> None:
        self._sandbox_id = f"sem-sb-{uuid.uuid4().hex[:12]}"
        self._folding = FoldingEngine(dimensions=dimensions)
        self._indexing = IndexingEngine()
        self._inference = InferenceEngine(self._folding, self._indexing)
        self._operations: list[SandboxResult] = []
        self._active = True

    @property
    def sandbox_id(self) -> str:
        return self._sandbox_id

    @property
    def is_active(self) -> bool:
        return self._active

    def index_document(
        self,
        content: str,
        title: str = "",
        tags: list[str] | None = None,
    ) -> SandboxResult:
        """Index a document in the sandbox."""
        self._check_active()
        errors: list[str] = []
        result_data: dict[str, Any] = {}

        if not content.strip():
            errors.append("Document content cannot be empty")

        if not errors:
            try:
                doc = Document(content=content, title=title, tags=tags or [])
                vector = self._folding.fold(doc)
                self._indexing.index(doc, vector)
                result_data = {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "dimensions": vector.dimensions,
                }
            except Exception as exc:
                errors.append(str(exc))

        result = SandboxResult(
            sandbox_id=self._sandbox_id,
            operation="index_document",
            success=len(errors) == 0,
            result=result_data,
            errors=errors,
        )
        self._operations.append(result)
        return result

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        strategy: InferenceStrategy = InferenceStrategy.NEAREST_NEIGHBOR,
    ) -> SandboxResult:
        """Run a semantic query in the sandbox."""
        self._check_active()
        errors: list[str] = []
        result_data: dict[str, Any] = {}

        if not query_text.strip():
            errors.append("Query text cannot be empty")

        if not errors:
            try:
                request = InferenceRequest(
                    query=query_text,
                    strategy=strategy,
                    top_k=top_k,
                )
                inference_result = self._inference.infer(request)
                result_data = {
                    "matches": inference_result.matches,
                    "match_count": len(inference_result.matches),
                    "duration_ms": inference_result.duration_ms,
                    "total_candidates": inference_result.total_candidates,
                }
            except Exception as exc:
                errors.append(str(exc))

        result = SandboxResult(
            sandbox_id=self._sandbox_id,
            operation="query",
            success=len(errors) == 0,
            result=result_data,
            errors=errors,
        )
        self._operations.append(result)
        return result

    def get_operations_log(self) -> list[SandboxResult]:
        """Return log of all sandbox operations."""
        return list(self._operations)

    def destroy(self) -> None:
        """Tear down the sandbox."""
        self._active = False
        self._indexing.clear()
        self._operations.clear()

    def _check_active(self) -> None:
        if not self._active:
            raise RuntimeError("Sandbox has been destroyed")
