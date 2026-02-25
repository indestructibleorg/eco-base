"""AuditEngine — append-only audit log with SHA-256 hash chain integrity."""

from __future__ import annotations

import hashlib
import time
from typing import Any

from security_platform.domain.entities import AuditEntry
from security_platform.domain.value_objects import AuditVerdict
from security_platform.domain.events import AuditRecorded
from security_platform.domain.exceptions import AuditIntegrityError


class AuditEngine:
    """Append-only audit log with hash chain integrity verification.

    Each entry's hash is computed from its fields plus the previous entry's hash,
    forming an immutable chain that can be verified for tampering.
    """

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []
        self._events: list[AuditRecorded] = []

    def record(
        self,
        actor: str,
        action: str,
        resource: str,
        verdict: AuditVerdict,
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Record a new audit entry.

        The entry is appended to the log and its hash is computed
        as part of the hash chain.
        """
        previous_hash = self._entries[-1].hash if self._entries else ""

        entry = AuditEntry(
            actor=actor,
            action=action,
            resource=resource,
            verdict=verdict,
            details=details or {},
            previous_hash=previous_hash,
        )
        entry.hash = entry.compute_hash(previous_hash)

        self._entries.append(entry)

        self._events.append(AuditRecorded(
            entry_id=entry.entry_id,
            actor=entry.actor,
            action=entry.action,
            verdict=entry.verdict,
        ))

        return entry

    def query(
        self,
        actor: str | None = None,
        action: str | None = None,
        resource: str | None = None,
        verdict: AuditVerdict | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> list[AuditEntry]:
        """Query audit entries with optional filters.

        All filters are AND-combined. Only entries matching all specified
        criteria are returned.
        """
        results: list[AuditEntry] = []

        for entry in self._entries:
            if actor is not None and entry.actor != actor:
                continue
            if action is not None and entry.action != action:
                continue
            if resource is not None and entry.resource != resource:
                continue
            if verdict is not None and entry.verdict != verdict:
                continue
            if since is not None and entry.timestamp < since:
                continue
            if until is not None and entry.timestamp > until:
                continue
            results.append(entry)

        return results

    def verify_integrity(self) -> bool:
        """Verify the integrity of the entire audit log hash chain.

        Returns True if the chain is valid. Raises AuditIntegrityError
        if any entry's hash does not match the expected value.
        """
        previous_hash = ""

        for i, entry in enumerate(self._entries):
            expected_hash = entry.compute_hash(previous_hash)
            if entry.hash != expected_hash:
                raise AuditIntegrityError(
                    f"Hash chain broken at entry {i} ({entry.entry_id}): "
                    f"expected {expected_hash[:16]}..., got {entry.hash[:16]}...",
                    entry_id=entry.entry_id,
                )
            if entry.previous_hash != previous_hash:
                raise AuditIntegrityError(
                    f"Previous hash mismatch at entry {i} ({entry.entry_id})",
                    entry_id=entry.entry_id,
                )
            previous_hash = entry.hash

        return True

    @property
    def entries(self) -> list[AuditEntry]:
        """All audit entries in order."""
        return list(self._entries)

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def events(self) -> list[AuditRecorded]:
        return list(self._events)
