"""Registry platform domain layer."""

from registry_platform.domain.entities import (
    DependencyEdge,
    NamespaceEntry,
    PlatformEntry,
    RegistryCatalog,
)
from registry_platform.domain.value_objects import (
    NamespaceScope,
    PlatformState,
)
from registry_platform.domain.events import (
    DependencyAdded,
    HealthAggregated,
    PlatformDeregistered,
    PlatformRegistered,
)
from registry_platform.domain.exceptions import (
    CyclicDependencyError,
    DependencyNotFoundError,
    NamespaceNotFoundError,
    PlatformNotFoundError,
    RegistryError,
)

__all__ = [
    "DependencyEdge",
    "NamespaceEntry",
    "PlatformEntry",
    "RegistryCatalog",
    "NamespaceScope",
    "PlatformState",
    "DependencyAdded",
    "HealthAggregated",
    "PlatformDeregistered",
    "PlatformRegistered",
    "CyclicDependencyError",
    "DependencyNotFoundError",
    "NamespaceNotFoundError",
    "PlatformNotFoundError",
    "RegistryError",
]
