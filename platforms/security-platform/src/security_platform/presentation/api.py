"""Security API — programmatic interface for enforcement, audit, and health.

Provides:
  POST /enforce       — evaluate a request against policies
  GET  /audit         — query audit log
  POST /audit/verify  — verify audit log integrity
  GET  /health        — security platform health
"""

from __future__ import annotations

from typing import Any

from security_platform.engines.enforcement_engine import EnforcementEngine
from security_platform.engines.audit_engine import AuditEngine
from security_platform.domain.value_objects import AuditVerdict
from security_platform.domain.exceptions import PolicyViolationError


class SecurityAPI:
    """In-process API facade for the security platform."""

    def __init__(
        self,
        enforcement_engine: EnforcementEngine | None = None,
        audit_engine: AuditEngine | None = None,
    ) -> None:
        self._enforcement = enforcement_engine or EnforcementEngine()
        self._audit = audit_engine or AuditEngine()

    async def enforce(self, data: dict[str, Any]) -> dict[str, Any]:
        """POST /enforce — evaluate a request against registered policies.

        Expected data:
            request: dict of fields to check
        """
        request = data.get("request", data)
        try:
            result = self._enforcement.evaluate_request(request)
            # Record audit entry
            self._audit.record(
                actor=request.get("actor", "unknown"),
                action="enforce",
                resource=request.get("resource", "unknown"),
                verdict=AuditVerdict.ALLOWED if result.allowed else AuditVerdict.DENIED,
            )
            return {
                "allowed": result.allowed,
                "action": result.action_taken.value,
                "violations": [
                    {
                        "policy": v.policy_name,
                        "field": v.field,
                        "message": v.message,
                        "threat_level": v.threat_level.value,
                    }
                    for v in result.violations
                ],
            }
        except PolicyViolationError as exc:
            self._audit.record(
                actor=request.get("actor", "unknown"),
                action="enforce",
                resource=request.get("resource", "unknown"),
                verdict=AuditVerdict.DENIED,
            )
            return {
                "allowed": False,
                "action": "block",
                "violations": [
                    {
                        "policy": v.policy_name,
                        "field": v.field,
                        "message": v.message,
                        "threat_level": v.threat_level.value,
                    }
                    for v in exc.violations
                ],
                "error": str(exc),
            }

    async def get_audit(
        self,
        actor: str | None = None,
        action: str | None = None,
    ) -> dict[str, Any]:
        """GET /audit — query audit log."""
        entries = self._audit.query(actor=actor, action=action)
        return {
            "entries": [
                {
                    "entry_id": e.entry_id,
                    "actor": e.actor,
                    "action": e.action,
                    "resource": e.resource,
                    "verdict": e.verdict.value,
                    "timestamp": e.timestamp,
                }
                for e in entries
            ],
            "count": len(entries),
        }

    async def verify_audit(self) -> dict[str, Any]:
        """POST /audit/verify — verify audit log hash chain integrity."""
        try:
            valid = self._audit.verify_integrity()
            return {"valid": valid, "entry_count": self._audit.entry_count}
        except Exception as exc:
            return {"valid": False, "error": str(exc)}

    async def get_health(self) -> dict[str, Any]:
        """GET /health — security platform health."""
        return {
            "status": "healthy",
            "policies_count": len(self._enforcement.policies),
            "audit_entries": self._audit.entry_count,
            "zero_tolerance": self._enforcement.zero_tolerance,
        }
