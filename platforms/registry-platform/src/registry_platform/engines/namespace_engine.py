"""NamespaceEngine -- hierarchical namespace management with dot-separated paths."""

from __future__ import annotations

import re
from typing import Any

from platform_shared.protocols.engine import EngineStatus

from registry_platform.domain.entities import NamespaceEntry
from registry_platform.domain.exceptions import NamespaceNotFoundError, RegistryError
from registry_platform.domain.value_objects import NamespaceScope

# Namespace naming rules:
# - Each segment: lowercase alphanumeric, hyphens allowed, 1-63 chars
# - Segments separated by dots
# - Maximum depth of 10 levels
_SEGMENT_RE = re.compile(r"^[a-z][a-z0-9-]{0,62}$")
_MAX_DEPTH = 10


class NamespaceEngine:
    """Manages hierarchical namespaces stored as dot-separated keys.

    Example paths: "org", "org.team", "org.team.service"
    """

    def __init__(self) -> None:
        self._namespaces: dict[str, NamespaceEntry] = {}  # keyed by path
        self._status = EngineStatus.RUNNING

    @property
    def name(self) -> str:
        return "namespace-engine"

    @property
    def status(self) -> EngineStatus:
        return self._status

    # -- Core operations ------------------------------------------------------

    def create(
        self,
        path: str,
        scope: NamespaceScope = NamespaceScope.PUBLIC,
        description: str = "",
        owner: str = "",
    ) -> NamespaceEntry:
        """Create a new namespace at the given path.

        Parent namespaces are automatically created if they do not exist.
        """
        self.validate(path)

        if path in self._namespaces:
            raise RegistryError(f"Namespace already exists: {path}")

        # Ensure all parent namespaces exist
        parts = path.split(".")
        for i in range(1, len(parts)):
            parent_path = ".".join(parts[:i])
            if parent_path not in self._namespaces:
                parent = NamespaceEntry(
                    path=parent_path,
                    scope=scope,
                    description=f"Auto-created parent namespace",
                )
                self._namespaces[parent_path] = parent

        entry = NamespaceEntry(
            path=path,
            scope=scope,
            description=description,
            owner=owner,
        )
        self._namespaces[path] = entry
        return entry

    def validate(self, path: str) -> bool:
        """Validate a namespace path against naming rules.

        Returns True if valid, raises RegistryError if not.
        """
        if not path:
            raise RegistryError("Namespace path cannot be empty")

        segments = path.split(".")
        if len(segments) > _MAX_DEPTH:
            raise RegistryError(
                f"Namespace depth exceeds maximum ({len(segments)} > {_MAX_DEPTH})"
            )

        for seg in segments:
            if not _SEGMENT_RE.match(seg):
                raise RegistryError(
                    f"Invalid namespace segment '{seg}': "
                    "must be lowercase alphanumeric with optional hyphens, 1-63 chars, "
                    "starting with a letter"
                )

        return True

    def get(self, path: str) -> NamespaceEntry:
        """Retrieve a namespace by path."""
        try:
            return self._namespaces[path]
        except KeyError:
            raise NamespaceNotFoundError(path)

    def list_all(self) -> list[NamespaceEntry]:
        """Return all namespaces."""
        return list(self._namespaces.values())

    def list_children(self, parent: str) -> list[NamespaceEntry]:
        """Return direct children of a parent namespace.

        For parent "org", children are "org.X" but not "org.X.Y".
        """
        parent_depth = len(parent.split("."))
        results: list[NamespaceEntry] = []
        for path, entry in self._namespaces.items():
            if path.startswith(parent + "."):
                entry_depth = len(path.split("."))
                if entry_depth == parent_depth + 1:
                    results.append(entry)
        return results

    def search(self, pattern: str) -> list[NamespaceEntry]:
        """Search namespaces by regex pattern on the path."""
        compiled = re.compile(pattern, re.IGNORECASE)
        return [
            entry for path, entry in self._namespaces.items()
            if compiled.search(path)
        ]

    def delete(self, path: str) -> NamespaceEntry:
        """Delete a namespace and all its children."""
        entry = self.get(path)
        # Remove children first
        children = [p for p in self._namespaces if p.startswith(path + ".")]
        for child in children:
            del self._namespaces[child]
        del self._namespaces[path]
        return entry
