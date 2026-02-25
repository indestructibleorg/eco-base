"""Value objects for the security domain."""

from __future__ import annotations

from enum import Enum


class ThreatLevel(str, Enum):
    """Severity classification for security threats."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class EnforcementAction(str, Enum):
    """Action to take when a policy violation is detected."""
    BLOCK = "block"
    WARN = "warn"
    LOG = "log"


class AuditVerdict(str, Enum):
    """Outcome of an audited action."""
    ALLOWED = "allowed"
    DENIED = "denied"
    ESCALATED = "escalated"
