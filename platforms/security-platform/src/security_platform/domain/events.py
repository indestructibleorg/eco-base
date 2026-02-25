"""Domain events for the security platform."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from security_platform.domain.value_objects import (
    ThreatLevel,
    EnforcementAction,
    AuditVerdict,
)


@dataclass(frozen=True, slots=True)
class ViolationDetected:
    """Emitted when a policy violation is detected."""
    policy_name: str
    threat_level: ThreatLevel
    action: EnforcementAction
    message: str = ""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class PolicyEnforced:
    """Emitted when a policy enforcement action is taken."""
    policy_name: str
    action: EnforcementAction
    request_blocked: bool = False
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class AuditRecorded:
    """Emitted when a new audit entry is recorded."""
    entry_id: str = ""
    actor: str = ""
    action: str = ""
    verdict: AuditVerdict = AuditVerdict.ALLOWED
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)
