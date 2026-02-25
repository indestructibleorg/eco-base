"""Domain exceptions for registry platform."""

from __future__ import annotations

from platform_shared.domain.errors import PlatformError


class RegistryError(PlatformError):
    """Base error for all registry platform operations."""

    def __init__(self, message: str, **kw):
        super().__init__(message, code="REGISTRY_ERROR", **kw)


class PlatformNotFoundError(RegistryError):
    """Raised when a platform entry cannot be found."""

    def __init__(self, platform_id: str):
        super().__init__(f"Platform not found: {platform_id}")
        self.platform_id = platform_id


class NamespaceNotFoundError(RegistryError):
    """Raised when a namespace cannot be found."""

    def __init__(self, path: str):
        super().__init__(f"Namespace not found: {path}")
        self.namespace_path = path


class DependencyNotFoundError(RegistryError):
    """Raised when a dependency edge cannot be found."""

    def __init__(self, from_id: str, to_id: str):
        super().__init__(f"Dependency not found: {from_id} -> {to_id}")
        self.from_id = from_id
        self.to_id = to_id


class CyclicDependencyError(RegistryError):
    """Raised when adding a dependency would create a cycle."""

    def __init__(self, cycle: list[str]):
        cycle_str = " -> ".join(cycle)
        super().__init__(f"Cyclic dependency detected: {cycle_str}")
        self.cycle = cycle
