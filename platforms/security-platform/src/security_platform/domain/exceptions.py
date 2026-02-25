"""Custom exceptions for the security platform."""

from __future__ import annotations

from platform_shared.domain.errors import PlatformError


class SecurityError(PlatformError):
    """Base error for all security platform operations."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, code="SECURITY_ERROR", **kwargs)


class PolicyViolationError(SecurityError):
    """Raised when a critical policy violation blocks an operation."""

    def __init__(self, message: str, violations: list | None = None):
        super().__init__(message)
        self.violations = violations or []


class AuditIntegrityError(SecurityError):
    """Raised when the audit log hash chain is broken."""

    def __init__(self, message: str, entry_id: str = ""):
        super().__init__(message)
        self.entry_id = entry_id


class IsolationError(SecurityError):
    """Raised when an isolation constraint is violated."""

    def __init__(self, message: str, sandbox_id: str = ""):
        super().__init__(message)
        self.sandbox_id = sandbox_id
