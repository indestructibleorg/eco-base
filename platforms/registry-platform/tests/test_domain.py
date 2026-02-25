"""Tests for registry platform domain entities, value objects, events, and exceptions."""

import pytest

from registry_platform.domain.entities import (
    PlatformEntry,
    NamespaceEntry,
    DependencyEdge,
    RegistryCatalog,
)
from registry_platform.domain.value_objects import PlatformState, NamespaceScope
from registry_platform.domain.events import (
    PlatformRegistered,
    PlatformDeregistered,
    DependencyAdded,
    HealthAggregated,
)
from registry_platform.domain.exceptions import (
    RegistryError,
    PlatformNotFoundError,
    NamespaceNotFoundError,
    DependencyNotFoundError,
    CyclicDependencyError,
)


# ============================================================================
# PlatformEntry
# ============================================================================


class TestPlatformEntry:
    def test_defaults(self):
        entry = PlatformEntry(name="test-platform")
        assert entry.name == "test-platform"
        assert entry.version == "1.0.0"
        assert entry.state == PlatformState.REGISTERED
        assert entry.health == "unknown"
        assert entry.id.startswith("plat-")

    def test_to_dict(self):
        entry = PlatformEntry(name="svc", version="2.0.0", tags=["core"])
        d = entry.to_dict()
        assert d["name"] == "svc"
        assert d["version"] == "2.0.0"
        assert d["tags"] == ["core"]
        assert d["state"] == "registered"

    def test_custom_fields(self):
        entry = PlatformEntry(
            name="custom",
            description="A custom platform",
            endpoints={"api": "http://localhost:8080"},
            metadata={"team": "infra"},
        )
        assert entry.description == "A custom platform"
        assert entry.endpoints["api"] == "http://localhost:8080"
        assert entry.metadata["team"] == "infra"


# ============================================================================
# NamespaceEntry
# ============================================================================


class TestNamespaceEntry:
    def test_depth(self):
        entry = NamespaceEntry(path="org.team.service")
        assert entry.depth == 3

    def test_parent_path(self):
        entry = NamespaceEntry(path="org.team.service")
        assert entry.parent_path == "org.team"

    def test_root_parent_path(self):
        entry = NamespaceEntry(path="org")
        assert entry.parent_path == ""

    def test_scope(self):
        entry = NamespaceEntry(path="internal", scope=NamespaceScope.PRIVATE)
        assert entry.scope == NamespaceScope.PRIVATE

    def test_to_dict(self):
        entry = NamespaceEntry(path="org.team", owner="admin")
        d = entry.to_dict()
        assert d["path"] == "org.team"
        assert d["owner"] == "admin"
        assert d["depth"] == 2
        assert d["parent_path"] == "org"


# ============================================================================
# DependencyEdge
# ============================================================================


class TestDependencyEdge:
    def test_defaults(self):
        edge = DependencyEdge(from_id="a", to_id="b")
        assert edge.dependency_type == "runtime"

    def test_to_dict(self):
        edge = DependencyEdge(from_id="a", to_id="b", dependency_type="build")
        d = edge.to_dict()
        assert d["from_id"] == "a"
        assert d["to_id"] == "b"
        assert d["dependency_type"] == "build"

    def test_frozen(self):
        edge = DependencyEdge(from_id="a", to_id="b")
        with pytest.raises(AttributeError):
            edge.from_id = "c"


# ============================================================================
# RegistryCatalog
# ============================================================================


class TestRegistryCatalog:
    def test_empty_catalog(self):
        catalog = RegistryCatalog()
        assert catalog.platform_count == 0
        assert catalog.namespace_count == 0

    def test_to_dict(self):
        catalog = RegistryCatalog()
        catalog.platforms["a"] = PlatformEntry(name="a")
        d = catalog.to_dict()
        assert d["platform_count"] == 1
        assert d["namespace_count"] == 0
        assert d["dependency_count"] == 0


# ============================================================================
# Events
# ============================================================================


class TestEvents:
    def test_platform_registered_event(self):
        evt = PlatformRegistered(platform_id="p1", name="test", version="1.0.0")
        event = evt.to_event()
        assert event.topic == "registry.platform.registered"
        assert event.payload["platform_id"] == "p1"

    def test_platform_deregistered_event(self):
        evt = PlatformDeregistered(platform_id="p1", name="test", reason="deprecated")
        event = evt.to_event()
        assert event.topic == "registry.platform.deregistered"
        assert event.payload["reason"] == "deprecated"

    def test_dependency_added_event(self):
        evt = DependencyAdded(from_id="a", to_id="b", dependency_type="runtime")
        event = evt.to_event()
        assert event.topic == "registry.dependency.added"

    def test_health_aggregated_event(self):
        evt = HealthAggregated(platform_id="p1", health_status="healthy")
        event = evt.to_event()
        assert event.topic == "registry.health.aggregated"


# ============================================================================
# Exceptions
# ============================================================================


class TestExceptions:
    def test_registry_error(self):
        err = RegistryError("test error")
        assert "test error" in str(err)

    def test_platform_not_found(self):
        err = PlatformNotFoundError("plat-123")
        assert err.platform_id == "plat-123"
        assert "plat-123" in str(err)

    def test_namespace_not_found(self):
        err = NamespaceNotFoundError("org.missing")
        assert err.namespace_path == "org.missing"

    def test_dependency_not_found(self):
        err = DependencyNotFoundError("a", "b")
        assert err.from_id == "a"
        assert err.to_id == "b"

    def test_cyclic_dependency(self):
        err = CyclicDependencyError(["a", "b", "a"])
        assert err.cycle == ["a", "b", "a"]
        assert "a -> b -> a" in str(err)
