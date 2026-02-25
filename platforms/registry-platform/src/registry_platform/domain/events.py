"""Domain events for registry platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from platform_shared.protocols.event_bus import Event


def _make_registry_event(topic: str, payload: dict[str, Any], source: str = "registry-platform") -> Event:
    """Helper to create registry events."""
    return Event(topic=topic, payload=payload, source=source)


@dataclass(frozen=True, slots=True)
class PlatformRegistered:
    platform_id: str
    name: str
    version: str

    def to_event(self) -> Event:
        return _make_registry_event(
            "registry.platform.registered",
            {
                "platform_id": self.platform_id,
                "name": self.name,
                "version": self.version,
            },
        )


@dataclass(frozen=True, slots=True)
class PlatformDeregistered:
    platform_id: str
    name: str
    reason: str = ""

    def to_event(self) -> Event:
        return _make_registry_event(
            "registry.platform.deregistered",
            {
                "platform_id": self.platform_id,
                "name": self.name,
                "reason": self.reason,
            },
        )


@dataclass(frozen=True, slots=True)
class DependencyAdded:
    from_id: str
    to_id: str
    dependency_type: str = "runtime"

    def to_event(self) -> Event:
        return _make_registry_event(
            "registry.dependency.added",
            {
                "from_id": self.from_id,
                "to_id": self.to_id,
                "dependency_type": self.dependency_type,
            },
        )


@dataclass(frozen=True, slots=True)
class HealthAggregated:
    platform_id: str
    health_status: str
    timestamp: float = 0.0

    def to_event(self) -> Event:
        return _make_registry_event(
            "registry.health.aggregated",
            {
                "platform_id": self.platform_id,
                "health_status": self.health_status,
                "timestamp": self.timestamp,
            },
        )
