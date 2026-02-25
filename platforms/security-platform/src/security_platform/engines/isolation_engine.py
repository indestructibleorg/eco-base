"""IsolationEngine — sandbox isolation rule management and verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from platform_shared.sandbox import (
    SandboxConfig,
    SandboxRuntime,
    ProcessSandbox,
)
from platform_shared.sandbox.network import POLICY_NO_NETWORK, POLICY_ALLOW_ALL
from platform_shared.sandbox.resource import ResourceLimits

from security_platform.domain.entities import IsolationRule
from security_platform.domain.exceptions import IsolationError


@dataclass(slots=True)
class IsolationStatus:
    """Status of isolation for a sandbox."""
    sandbox_id: str
    isolated: bool = True
    no_network: bool = True
    readonly_fs: bool = True
    memory_limited: bool = True
    violations: list[str] = field(default_factory=list)


class IsolationEngine:
    """Manages isolation rules for sandboxes.

    Applies security constraints (no network, read-only FS, memory limits)
    and verifies that sandboxes comply with their isolation rules.
    """

    def __init__(self) -> None:
        self._rules: dict[str, IsolationRule] = {}
        self._sandbox_configs: dict[str, SandboxConfig] = {}

    def apply_isolation(self, rule: IsolationRule) -> SandboxConfig:
        """Apply isolation rules and produce a SandboxConfig.

        Creates a sandbox configuration that enforces all constraints
        specified in the IsolationRule.

        Returns:
            SandboxConfig reflecting the isolation constraints.
        """
        self._rules[rule.sandbox_id] = rule

        network_policy = POLICY_NO_NETWORK if rule.no_network else POLICY_ALLOW_ALL
        resource_limits = ResourceLimits(
            memory_mb=rule.max_memory_mb,
            cpu_cores=0.5,
            max_processes=16,
        )

        config = SandboxConfig(
            name=f"isolated-{rule.sandbox_id}",
            resource_limits=resource_limits,
            network_policy=network_policy,
            filesystem_readonly=rule.readonly_fs,
            timeout_seconds=300.0,
        )

        self._sandbox_configs[rule.sandbox_id] = config
        return config

    def verify_isolation(self, sandbox_id: str) -> IsolationStatus:
        """Verify that a sandbox complies with its isolation rules.

        Args:
            sandbox_id: ID of the sandbox to check.

        Returns:
            IsolationStatus with details about compliance.

        Raises:
            IsolationError: If the sandbox has no isolation rules registered.
        """
        if sandbox_id not in self._rules:
            raise IsolationError(
                f"No isolation rules registered for sandbox '{sandbox_id}'",
                sandbox_id=sandbox_id,
            )

        rule = self._rules[sandbox_id]
        config = self._sandbox_configs.get(sandbox_id)
        violations: list[str] = []

        if config is None:
            violations.append("No sandbox config found — isolation not applied")
            return IsolationStatus(
                sandbox_id=sandbox_id,
                isolated=False,
                violations=violations,
            )

        # Check network isolation
        no_network_ok = True
        if rule.no_network:
            from platform_shared.sandbox.network import NetworkAction
            no_network_ok = config.network_policy.default_action == NetworkAction.DENY
            if not no_network_ok:
                violations.append("Network access is not blocked")

        # Check filesystem
        readonly_ok = True
        if rule.readonly_fs and not config.filesystem_readonly:
            readonly_ok = False
            violations.append("Filesystem is not read-only")

        # Check memory
        memory_ok = config.resource_limits.memory_mb <= rule.max_memory_mb
        if not memory_ok:
            violations.append(
                f"Memory limit {config.resource_limits.memory_mb}MB "
                f"exceeds rule maximum {rule.max_memory_mb}MB"
            )

        isolated = len(violations) == 0

        return IsolationStatus(
            sandbox_id=sandbox_id,
            isolated=isolated,
            no_network=no_network_ok,
            readonly_fs=readonly_ok if rule.readonly_fs else True,
            memory_limited=memory_ok,
            violations=violations,
        )

    def get_rule(self, sandbox_id: str) -> IsolationRule | None:
        """Get the isolation rule for a sandbox."""
        return self._rules.get(sandbox_id)

    def remove_rule(self, sandbox_id: str) -> None:
        """Remove isolation rule and config for a sandbox."""
        self._rules.pop(sandbox_id, None)
        self._sandbox_configs.pop(sandbox_id, None)

    @property
    def rules(self) -> dict[str, IsolationRule]:
        return dict(self._rules)

    @property
    def isolated_sandboxes(self) -> list[str]:
        return list(self._rules.keys())
