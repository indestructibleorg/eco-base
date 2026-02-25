"""Core domain entities for the security platform."""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from security_platform.domain.value_objects import (
    ThreatLevel,
    EnforcementAction,
    AuditVerdict,
)


@dataclass(slots=True)
class PolicyCondition:
    """A single condition within a security policy."""
    field: str
    operator: str  # "contains", "eq", "ne", "in", "not_in", "exists"
    value: Any = None

    def evaluate(self, request: dict[str, Any]) -> bool:
        """Check whether the request matches this condition."""
        actual = request.get(self.field)

        if self.operator == "contains":
            return self.value in str(actual) if actual is not None else False
        elif self.operator == "eq":
            return actual == self.value
        elif self.operator == "ne":
            return actual != self.value
        elif self.operator == "in":
            return actual in self.value if self.value else False
        elif self.operator == "not_in":
            return actual not in self.value if self.value else True
        elif self.operator == "exists":
            return actual is not None
        return False


@dataclass(slots=True)
class SecurityPolicy:
    """A security policy with conditions that determine enforcement."""
    name: str
    description: str = ""
    threat_level: ThreatLevel = ThreatLevel.MEDIUM
    action: EnforcementAction = EnforcementAction.BLOCK
    conditions: list[PolicyCondition] = field(default_factory=list)
    enabled: bool = True
    policy_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    def matches(self, request: dict[str, Any]) -> bool:
        """Check whether a request matches ALL conditions of this policy."""
        if not self.enabled:
            return False
        if not self.conditions:
            return False
        return all(c.evaluate(request) for c in self.conditions)


@dataclass(slots=True)
class Violation:
    """A detected policy violation."""
    policy_name: str
    field: str
    message: str
    threat_level: ThreatLevel
    action: EnforcementAction
    violation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)


@dataclass(slots=True)
class AuditEntry:
    """An entry in the immutable audit log."""
    actor: str
    action: str
    resource: str
    verdict: AuditVerdict
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)
    previous_hash: str = ""
    hash: str = ""

    def compute_hash(self, previous_hash: str = "") -> str:
        """Compute SHA-256 hash for this entry in the chain."""
        payload = (
            f"{self.entry_id}:{self.timestamp}:{self.actor}:{self.action}:"
            f"{self.resource}:{self.verdict.value}:{previous_hash}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(slots=True)
class IsolationRule:
    """Rule defining isolation constraints for a sandbox."""
    sandbox_id: str
    no_network: bool = True
    readonly_fs: bool = True
    max_memory_mb: int = 256
    allowed_paths: list[str] = field(default_factory=list)
    rule_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
