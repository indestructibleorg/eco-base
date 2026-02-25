"""RegistrySandbox -- validate platform manifests in sandbox.

Provides an isolated environment to validate platform registration
manifests, namespace configurations, and dependency declarations
without modifying the real registry.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from registry_platform.domain.entities import PlatformEntry
from registry_platform.domain.value_objects import PlatformState, NamespaceScope
from registry_platform.engines.catalog_engine import CatalogEngine
from registry_platform.engines.namespace_engine import NamespaceEngine
from registry_platform.engines.dependency_engine import DependencyEngine


@dataclass(slots=True)
class ValidationResult:
    """Result of a sandbox validation operation."""

    sandbox_id: str
    operation: str
    valid: bool
    result: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class RegistrySandbox:
    """Validate platform manifests in an isolated sandbox environment.

    Creates isolated engine instances to test registration, namespace
    creation, and dependency declarations without affecting production state.
    """

    def __init__(self) -> None:
        self._sandbox_id = f"reg-sb-{uuid.uuid4().hex[:12]}"
        self._catalog = CatalogEngine()
        self._namespaces = NamespaceEngine()
        self._dependencies = DependencyEngine()
        self._validations: list[ValidationResult] = []
        self._active = True

    @property
    def sandbox_id(self) -> str:
        return self._sandbox_id

    @property
    def is_active(self) -> bool:
        return self._active

    def validate_platform_manifest(
        self,
        name: str,
        version: str = "1.0.0",
        description: str = "",
        tags: list[str] | None = None,
        endpoints: dict[str, str] | None = None,
    ) -> ValidationResult:
        """Validate a platform registration manifest.

        Checks:
          - Name is non-empty and reasonable length
          - Version follows semver-like format
          - Tags are non-empty strings
          - Endpoints have valid keys
        """
        self._check_active()
        errors: list[str] = []
        warnings: list[str] = []
        result_data: dict[str, Any] = {}

        # Validate name
        if not name:
            errors.append("Platform name is required")
        elif len(name) > 128:
            errors.append("Platform name exceeds 128 characters")

        # Validate version
        if not version:
            errors.append("Version is required")
        else:
            parts = version.split(".")
            if len(parts) < 2:
                warnings.append(f"Version '{version}' does not follow semver format")

        # Validate tags
        if tags:
            for tag in tags:
                if not tag.strip():
                    errors.append("Tags cannot be empty strings")
                    break

        # Validate endpoints
        if endpoints:
            for key, url in endpoints.items():
                if not key.strip():
                    errors.append("Endpoint keys cannot be empty")
                    break
                if not url.strip():
                    errors.append(f"Endpoint '{key}' has empty URL")
                    break

        if not errors:
            entry = PlatformEntry(
                name=name,
                version=version,
                description=description,
                tags=tags or [],
                endpoints=endpoints or {},
            )
            self._catalog.register(entry)
            result_data = entry.to_dict()

        result = ValidationResult(
            sandbox_id=self._sandbox_id,
            operation="validate_platform_manifest",
            valid=len(errors) == 0,
            result=result_data,
            errors=errors,
            warnings=warnings,
        )
        self._validations.append(result)
        return result

    def validate_namespace(
        self,
        path: str,
        scope: NamespaceScope = NamespaceScope.PUBLIC,
    ) -> ValidationResult:
        """Validate a namespace path and attempt creation in sandbox."""
        self._check_active()
        errors: list[str] = []
        warnings: list[str] = []
        result_data: dict[str, Any] = {}

        if not path:
            errors.append("Namespace path is required")
        else:
            try:
                self._namespaces.validate(path)
                entry = self._namespaces.create(path=path, scope=scope)
                result_data = entry.to_dict()
            except Exception as exc:
                errors.append(str(exc))

        result = ValidationResult(
            sandbox_id=self._sandbox_id,
            operation="validate_namespace",
            valid=len(errors) == 0,
            result=result_data,
            errors=errors,
            warnings=warnings,
        )
        self._validations.append(result)
        return result

    def validate_dependency(
        self,
        from_id: str,
        to_id: str,
        dependency_type: str = "runtime",
    ) -> ValidationResult:
        """Validate a dependency declaration -- checks for cycles."""
        self._check_active()
        errors: list[str] = []
        warnings: list[str] = []
        result_data: dict[str, Any] = {}

        if not from_id:
            errors.append("from_id is required")
        if not to_id:
            errors.append("to_id is required")

        if not errors:
            try:
                edge = self._dependencies.add_dependency(
                    from_id=from_id,
                    to_id=to_id,
                    dependency_type=dependency_type,
                )
                result_data = edge.to_dict()
            except Exception as exc:
                errors.append(str(exc))

        result = ValidationResult(
            sandbox_id=self._sandbox_id,
            operation="validate_dependency",
            valid=len(errors) == 0,
            result=result_data,
            errors=errors,
            warnings=warnings,
        )
        self._validations.append(result)
        return result

    def get_validations_log(self) -> list[ValidationResult]:
        """Return a log of all validations performed in this sandbox."""
        return list(self._validations)

    def destroy(self) -> None:
        """Tear down the sandbox."""
        self._active = False
        self._validations.clear()

    def _check_active(self) -> None:
        if not self._active:
            raise RuntimeError("Sandbox has been destroyed")
