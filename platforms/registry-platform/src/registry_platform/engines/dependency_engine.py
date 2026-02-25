"""DependencyEngine -- directed graph of platform dependencies with cycle detection."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from platform_shared.protocols.engine import EngineStatus

from registry_platform.domain.entities import DependencyEdge
from registry_platform.domain.exceptions import (
    CyclicDependencyError,
    DependencyNotFoundError,
    RegistryError,
)


class DependencyEngine:
    """Manages a directed dependency graph between platforms.

    Provides cycle detection (DFS) and topological sorting for
    deployment ordering.
    """

    def __init__(self) -> None:
        # adjacency list: from_id -> set of to_ids
        self._graph: dict[str, set[str]] = defaultdict(set)
        # reverse adjacency: to_id -> set of from_ids (dependents)
        self._reverse: dict[str, set[str]] = defaultdict(set)
        # edge metadata
        self._edges: dict[tuple[str, str], DependencyEdge] = {}
        # all known nodes
        self._nodes: set[str] = set()
        self._status = EngineStatus.RUNNING

    @property
    def name(self) -> str:
        return "dependency-engine"

    @property
    def status(self) -> EngineStatus:
        return self._status

    # -- Core operations ------------------------------------------------------

    def add_dependency(
        self,
        from_id: str,
        to_id: str,
        dependency_type: str = "runtime",
    ) -> DependencyEdge:
        """Add a dependency edge: from_id depends on to_id.

        Raises CyclicDependencyError if the edge would create a cycle.
        """
        if from_id == to_id:
            raise CyclicDependencyError([from_id, to_id])

        # Check if adding this edge would create a cycle
        # A cycle exists if to_id can already reach from_id
        if self._has_path(to_id, from_id):
            # Reconstruct a minimal cycle path for the error
            cycle = self._find_cycle_path(to_id, from_id)
            raise CyclicDependencyError([from_id] + cycle + [from_id])

        self._nodes.add(from_id)
        self._nodes.add(to_id)
        self._graph[from_id].add(to_id)
        self._reverse[to_id].add(from_id)

        edge = DependencyEdge(
            from_id=from_id,
            to_id=to_id,
            dependency_type=dependency_type,
        )
        self._edges[(from_id, to_id)] = edge
        return edge

    def remove_dependency(self, from_id: str, to_id: str) -> None:
        """Remove a dependency edge."""
        if (from_id, to_id) not in self._edges:
            raise DependencyNotFoundError(from_id, to_id)

        self._graph[from_id].discard(to_id)
        self._reverse[to_id].discard(from_id)
        del self._edges[(from_id, to_id)]

    def get_dependencies(self, platform_id: str) -> list[str]:
        """Get the list of platform IDs that platform_id depends on."""
        return sorted(self._graph.get(platform_id, set()))

    def get_dependents(self, platform_id: str) -> list[str]:
        """Get the list of platform IDs that depend on platform_id."""
        return sorted(self._reverse.get(platform_id, set()))

    def detect_cycles(self) -> list[list[str]]:
        """Detect all cycles in the dependency graph using DFS.

        Returns a list of cycles, where each cycle is a list of node IDs.
        Returns an empty list if the graph is acyclic.
        """
        visited: set[str] = set()
        rec_stack: set[str] = set()
        cycles: list[list[str]] = []

        def _dfs(node: str, path: list[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self._graph.get(node, set()):
                if neighbor not in visited:
                    _dfs(neighbor, path)
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            path.pop()
            rec_stack.discard(node)

        for node in self._nodes:
            if node not in visited:
                _dfs(node, [])

        return cycles

    def topological_sort(self) -> list[str]:
        """Return nodes in topological order (deployment ordering).

        Nodes with no dependencies come first.
        Raises CyclicDependencyError if a cycle exists.
        """
        # Kahn's algorithm
        in_degree: dict[str, int] = {n: 0 for n in self._nodes}
        for node in self._nodes:
            for dep in self._graph.get(node, set()):
                in_degree[dep] = in_degree.get(dep, 0) + 1

        # Start with nodes that have no incoming edges
        queue = sorted([n for n, d in in_degree.items() if d == 0])
        result: list[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for dep in sorted(self._graph.get(node, set())):
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)
            queue.sort()  # deterministic ordering

        if len(result) != len(self._nodes):
            raise CyclicDependencyError(
                [n for n in self._nodes if n not in result]
            )

        return result

    def get_all_edges(self) -> list[DependencyEdge]:
        """Return all dependency edges."""
        return list(self._edges.values())

    def get_graph_dict(self) -> dict[str, list[str]]:
        """Return the adjacency list as a plain dict."""
        return {k: sorted(v) for k, v in self._graph.items() if v}

    # -- Internals ------------------------------------------------------------

    def _has_path(self, source: str, target: str) -> bool:
        """Check if there is a directed path from source to target."""
        visited: set[str] = set()
        stack = [source]
        while stack:
            node = stack.pop()
            if node == target:
                return True
            if node in visited:
                continue
            visited.add(node)
            stack.extend(self._graph.get(node, set()))
        return False

    def _find_cycle_path(self, source: str, target: str) -> list[str]:
        """Find a path from source to target for cycle reporting."""
        visited: set[str] = set()
        parent: dict[str, str] = {}
        stack = [source]

        while stack:
            node = stack.pop()
            if node == target:
                # Reconstruct path
                path = [target]
                current = target
                while current in parent and current != source:
                    current = parent[current]
                    path.append(current)
                path.reverse()
                return path
            if node in visited:
                continue
            visited.add(node)
            for neighbor in self._graph.get(node, set()):
                if neighbor not in visited:
                    parent[neighbor] = node
                    stack.append(neighbor)

        return [source, target]
