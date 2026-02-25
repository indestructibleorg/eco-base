"""CatalogEngine -- register/deregister platforms, list, search by tags."""

from __future__ import annotations

import re
from typing import Any

from platform_shared.protocols.engine import EngineStatus
from platform_shared.protocols.event_bus import LocalEventBus

from registry_platform.domain.entities import PlatformEntry
from registry_platform.domain.exceptions import PlatformNotFoundError
from registry_platform.domain.value_objects import PlatformState


class CatalogEngine:
    """In-memory platform catalog -- register, search, and manage platform entries.

    Supports tag-based search and name pattern matching.
    """

    def __init__(self, event_bus: LocalEventBus | None = None) -> None:
        self._platforms: dict[str, PlatformEntry] = {}
        self._event_bus = event_bus or LocalEventBus()
        self._status = EngineStatus.RUNNING

    @property
    def name(self) -> str:
        return "catalog-engine"

    @property
    def status(self) -> EngineStatus:
        return self._status

    # -- Core operations ------------------------------------------------------

    def register(self, entry: PlatformEntry) -> PlatformEntry:
        """Register a new platform entry in the catalog."""
        entry.state = PlatformState.REGISTERED
        self._platforms[entry.id] = entry
        return entry

    def deregister(self, platform_id: str) -> PlatformEntry:
        """Deregister a platform (marks as DECOMMISSIONED and removes)."""
        entry = self.get(platform_id)
        entry.state = PlatformState.DECOMMISSIONED
        del self._platforms[platform_id]
        return entry

    def get(self, platform_id: str) -> PlatformEntry:
        """Retrieve a platform entry by id."""
        try:
            return self._platforms[platform_id]
        except KeyError:
            raise PlatformNotFoundError(platform_id)

    def list_all(self) -> list[PlatformEntry]:
        """Return all registered platform entries."""
        return list(self._platforms.values())

    def search(
        self,
        tags: list[str] | None = None,
        name_pattern: str = "",
    ) -> list[PlatformEntry]:
        """Search platforms by tags and/or name pattern.

        - tags: if provided, a platform must contain ALL specified tags.
        - name_pattern: if provided, platform name must match the regex pattern.
        """
        results = list(self._platforms.values())

        if tags:
            results = [
                p for p in results
                if all(t in p.tags for t in tags)
            ]

        if name_pattern:
            compiled = re.compile(name_pattern, re.IGNORECASE)
            results = [p for p in results if compiled.search(p.name)]

        return results

    def update_health(self, platform_id: str, health_status: str) -> PlatformEntry:
        """Update the health status of a platform.

        Also transitions state based on health:
          - "healthy" -> ACTIVE
          - "degraded" -> DEGRADED
          - "unhealthy" -> DEGRADED
        """
        entry = self.get(platform_id)
        entry.health = health_status

        if health_status == "healthy":
            entry.state = PlatformState.ACTIVE
        elif health_status in ("degraded", "unhealthy"):
            entry.state = PlatformState.DEGRADED

        return entry

    def update_state(self, platform_id: str, state: PlatformState) -> PlatformEntry:
        """Directly update a platform's state."""
        entry = self.get(platform_id)
        entry.state = state
        return entry
