"""Custom exceptions for the semantic platform."""

from __future__ import annotations

from platform_shared.domain.errors import PlatformError


class SemanticError(PlatformError):
    """Base error for all semantic platform operations."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, code="SEMANTIC_ERROR", **kwargs)


class DocumentNotFoundError(SemanticError):
    """Raised when a queried document does not exist."""

    def __init__(self, doc_id: str):
        super().__init__(f"Document not found: {doc_id}")
        self.doc_id = doc_id


class IndexingError(SemanticError):
    """Raised when document indexing fails."""

    def __init__(self, message: str, doc_id: str = ""):
        super().__init__(message)
        self.doc_id = doc_id


class InferenceError(SemanticError):
    """Raised when an inference operation fails."""

    def __init__(self, message: str, request_id: str = ""):
        super().__init__(message)
        self.request_id = request_id
