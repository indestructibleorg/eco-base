"""Registry Platform API.

Provides the following endpoints (in-memory, framework-agnostic):
  POST /platforms
  GET  /platforms
  GET  /platforms/{id}
  POST /namespaces
  GET  /dependencies/graph
  GET  /health
"""

from __future__ import annotations

from typing import Any

from platform_shared.protocols.health import HealthReport, HealthStatus

from registry_platform.domain.entities import PlatformEntry
from registry_platform.domain.value_objects import PlatformState, NamespaceScope
from registry_platform.engines.catalog_engine import CatalogEngine
from registry_platform.engines.namespace_engine import NamespaceEngine
from registry_platform.engines.dependency_engine import DependencyEngine


class RegistryAPI:
    """Framework-agnostic API facade for registry operations.

    Each method returns a tuple of (response_dict, http_status_code).
    """

    def __init__(
        self,
        catalog_engine: CatalogEngine | None = None,
        namespace_engine: NamespaceEngine | None = None,
        dependency_engine: DependencyEngine | None = None,
    ) -> None:
        self._catalog = catalog_engine or CatalogEngine()
        self._namespaces = namespace_engine or NamespaceEngine()
        self._dependencies = dependency_engine or DependencyEngine()

    # POST /platforms
    def post_platform(self, body: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """Register a new platform."""
        try:
            name = body["name"]
            version = body.get("version", "1.0.0")
            description = body.get("description", "")
            tags = body.get("tags", [])
            endpoints = body.get("endpoints", {})

            entry = PlatformEntry(
                name=name,
                version=version,
                description=description,
                tags=tags,
                endpoints=endpoints,
            )
            registered = self._catalog.register(entry)
            return registered.to_dict(), 201
        except KeyError as exc:
            return {"error": f"Missing required field: {exc}"}, 400
        except Exception as exc:
            return {"error": str(exc)}, 400

    # GET /platforms
    def get_platforms(
        self,
        tags: list[str] | None = None,
        name_pattern: str = "",
    ) -> tuple[list[dict[str, Any]], int]:
        """List or search platforms."""
        if tags or name_pattern:
            results = self._catalog.search(tags=tags, name_pattern=name_pattern)
        else:
            results = self._catalog.list_all()
        return [p.to_dict() for p in results], 200

    # GET /platforms/{id}
    def get_platform(self, platform_id: str) -> tuple[dict[str, Any], int]:
        """Get a specific platform by id."""
        try:
            entry = self._catalog.get(platform_id)
            return entry.to_dict(), 200
        except Exception:
            return {"error": f"Platform not found: {platform_id}"}, 404

    # POST /namespaces
    def post_namespace(self, body: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """Create a new namespace."""
        try:
            path = body["path"]
            scope_str = body.get("scope", "public")
            scope = NamespaceScope(scope_str)
            description = body.get("description", "")
            owner = body.get("owner", "")

            entry = self._namespaces.create(
                path=path,
                scope=scope,
                description=description,
                owner=owner,
            )
            return entry.to_dict(), 201
        except KeyError as exc:
            return {"error": f"Missing required field: {exc}"}, 400
        except Exception as exc:
            return {"error": str(exc)}, 400

    # GET /dependencies/graph
    def get_dependency_graph(self) -> tuple[dict[str, Any], int]:
        """Return the full dependency graph."""
        edges = self._dependencies.get_all_edges()
        graph = self._dependencies.get_graph_dict()

        try:
            order = self._dependencies.topological_sort()
        except Exception:
            order = []

        return {
            "edges": [e.to_dict() for e in edges],
            "graph": graph,
            "topological_order": order,
        }, 200

    # GET /health
    def get_health(self) -> tuple[dict[str, Any], int]:
        """Return platform health status."""
        report = HealthReport(
            component="registry-platform",
            status=HealthStatus.HEALTHY,
            message="All engines operational",
            details={
                "catalog_engine": self._catalog.status.value,
                "namespace_engine": self._namespaces.status.value,
                "dependency_engine": self._dependencies.status.value,
                "platforms_count": len(self._catalog._platforms),
                "namespaces_count": len(self._namespaces._namespaces),
            },
        )
        return {
            "component": report.component,
            "status": report.status.value,
            "message": report.message,
            "details": report.details,
            "is_healthy": report.is_healthy,
        }, 200
