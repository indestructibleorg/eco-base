"""Value objects for registry domain."""

from __future__ import annotations

from enum import Enum


class PlatformState(str, Enum):
    """Lifecycle state of a registered platform."""

    REGISTERED = "registered"
    ACTIVE = "active"
    DEGRADED = "degraded"
    DECOMMISSIONED = "decommissioned"


class NamespaceScope(str, Enum):
    """Scope / visibility of a namespace."""

    PUBLIC = "public"
    PRIVATE = "private"
    INTERNAL = "internal"
