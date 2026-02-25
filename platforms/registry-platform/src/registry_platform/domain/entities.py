"""Domain entities for registry platform."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from registry_platform.domain.value_objects import NamespaceScope, PlatformState


@dataclass(slots=True)
class PlatformEntry:
    """Represents a registered platform in the catalog."""

    id: str = field(default_factory=lambda: f"plat-{uuid.uuid4().hex[:12]}")
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    tags: list[str] = field(default_factory=list)
    endpoints: dict[str, str] = field(default_factory=dict)
    state: PlatformState = PlatformState.REGISTERED
    health: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "tags": self.tags,
            "endpoints": self.endpoints,
            "state": self.state.value,
            "health": self.health,
            "metadata": self.metadata,
            "registered_at": self.registered_at,
        }


@dataclass(slots=True)
class NamespaceEntry:
    """Represents a hierarchical namespace (e.g., org.team.service)."""

    id: str = field(default_factory=lambda: f"nsentry-{uuid.uuid4().hex[:12]}")
    path: str = ""
    scope: NamespaceScope = NamespaceScope.PUBLIC
    description: str = ""
    owner: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def depth(self) -> int:
        """Number of segments in the namespace path."""
        return len(self.path.split(".")) if self.path else 0

    @property
    def parent_path(self) -> str:
        """Parent namespace path, or empty string for root-level."""
        parts = self.path.split(".")
        return ".".join(parts[:-1]) if len(parts) > 1 else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "scope": self.scope.value,
            "description": self.description,
            "owner": self.owner,
            "depth": self.depth,
            "parent_path": self.parent_path,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class DependencyEdge:
    """A directed dependency edge from one platform to another."""

    from_id: str
    to_id: str
    dependency_type: str = "runtime"  # runtime, build, optional
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_id": self.from_id,
            "to_id": self.to_id,
            "dependency_type": self.dependency_type,
        }


@dataclass(slots=True)
class RegistryCatalog:
    """Aggregate root holding the full registry state."""

    platforms: dict[str, PlatformEntry] = field(default_factory=dict)
    namespaces: dict[str, NamespaceEntry] = field(default_factory=dict)
    dependencies: list[DependencyEdge] = field(default_factory=list)

    @property
    def platform_count(self) -> int:
        return len(self.platforms)

    @property
    def namespace_count(self) -> int:
        return len(self.namespaces)

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform_count": self.platform_count,
            "namespace_count": self.namespace_count,
            "dependency_count": len(self.dependencies),
        }
