"""EnforcementEngine — evaluate requests against security policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from security_platform.domain.entities import SecurityPolicy, Violation
from security_platform.domain.value_objects import (
    ThreatLevel,
    EnforcementAction,
)
from security_platform.domain.events import ViolationDetected, PolicyEnforced
from security_platform.domain.exceptions import PolicyViolationError


@dataclass(slots=True)
class EnforcementResult:
    """Result of evaluating a request against policies."""
    allowed: bool
    violations: list[Violation] = field(default_factory=list)
    action_taken: EnforcementAction = EnforcementAction.LOG


class EnforcementEngine:
    """Evaluates requests against registered security policies.

    In zero-tolerance mode, ANY CRITICAL violation results in BLOCK.
    Otherwise, actions are determined by each policy's configured enforcement action.
    """

    def __init__(self, zero_tolerance: bool = True) -> None:
        self._policies: dict[str, SecurityPolicy] = {}
        self._zero_tolerance = zero_tolerance
        self._events: list[ViolationDetected | PolicyEnforced] = []

    def register_policy(self, policy: SecurityPolicy) -> None:
        """Register a security policy."""
        self._policies[policy.name] = policy

    def unregister_policy(self, name: str) -> None:
        """Remove a policy."""
        self._policies.pop(name, None)

    def evaluate_request(
        self,
        request: dict[str, Any],
        policies: list[SecurityPolicy] | None = None,
    ) -> EnforcementResult:
        """Evaluate a request against policies.

        Args:
            request: Dict of request fields to check.
            policies: Optional list of policies to check. If None, use all registered.

        Returns:
            EnforcementResult with allowed flag and list of violations.

        Raises:
            PolicyViolationError: If zero_tolerance mode and CRITICAL violation found.
        """
        check_policies = policies if policies is not None else list(self._policies.values())
        violations: list[Violation] = []

        for policy in check_policies:
            if not policy.enabled:
                continue

            if policy.matches(request):
                # Build violation for each matched condition
                matched_fields = [c.field for c in policy.conditions]
                violation = Violation(
                    policy_name=policy.name,
                    field=", ".join(matched_fields),
                    message=f"Request violates policy '{policy.name}': {policy.description}",
                    threat_level=policy.threat_level,
                    action=policy.action,
                )
                violations.append(violation)

                self._events.append(ViolationDetected(
                    policy_name=policy.name,
                    threat_level=policy.threat_level,
                    action=policy.action,
                    message=violation.message,
                ))

        if not violations:
            return EnforcementResult(allowed=True)

        # Determine overall action
        has_critical = any(v.threat_level == ThreatLevel.CRITICAL for v in violations)
        has_block = any(v.action == EnforcementAction.BLOCK for v in violations)

        if self._zero_tolerance and has_critical:
            action = EnforcementAction.BLOCK
            self._events.append(PolicyEnforced(
                policy_name=violations[0].policy_name,
                action=EnforcementAction.BLOCK,
                request_blocked=True,
            ))
            raise PolicyViolationError(
                f"CRITICAL violation detected: {violations[0].message}",
                violations=violations,
            )
        elif has_block:
            action = EnforcementAction.BLOCK
            allowed = False
        elif any(v.action == EnforcementAction.WARN for v in violations):
            action = EnforcementAction.WARN
            allowed = True
        else:
            action = EnforcementAction.LOG
            allowed = True

        self._events.append(PolicyEnforced(
            policy_name=violations[0].policy_name,
            action=action,
            request_blocked=not allowed,
        ))

        return EnforcementResult(
            allowed=allowed,
            violations=violations,
            action_taken=action,
        )

    @property
    def policies(self) -> dict[str, SecurityPolicy]:
        return dict(self._policies)

    @property
    def events(self) -> list[ViolationDetected | PolicyEnforced]:
        return list(self._events)

    @property
    def zero_tolerance(self) -> bool:
        return self._zero_tolerance

    @zero_tolerance.setter
    def zero_tolerance(self, value: bool) -> None:
        self._zero_tolerance = value
