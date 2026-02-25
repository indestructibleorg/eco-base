"""Tests for registry platform sandbox."""

import pytest

from registry_platform.sandbox.registry_sandbox import RegistrySandbox, ValidationResult
from registry_platform.domain.value_objects import NamespaceScope


class TestRegistrySandbox:
    def test_create_sandbox(self):
        sb = RegistrySandbox()
        assert sb.sandbox_id.startswith("reg-sb-")
        assert sb.is_active is True

    def test_validate_platform_manifest_valid(self):
        sb = RegistrySandbox()
        result = sb.validate_platform_manifest(
            name="test-platform",
            version="1.0.0",
            description="A test",
            tags=["core"],
        )
        assert result.valid is True
        assert result.operation == "validate_platform_manifest"
        assert result.result["name"] == "test-platform"

    def test_validate_platform_manifest_empty_name(self):
        sb = RegistrySandbox()
        result = sb.validate_platform_manifest(name="", version="1.0.0")
        assert result.valid is False
        assert any("required" in e for e in result.errors)

    def test_validate_platform_manifest_long_name(self):
        sb = RegistrySandbox()
        result = sb.validate_platform_manifest(name="x" * 200)
        assert result.valid is False
        assert any("128" in e for e in result.errors)

    def test_validate_platform_manifest_bad_version(self):
        sb = RegistrySandbox()
        result = sb.validate_platform_manifest(name="svc", version="1")
        assert result.valid is True  # valid but with warning
        assert any("semver" in w for w in result.warnings)

    def test_validate_platform_manifest_empty_tag(self):
        sb = RegistrySandbox()
        result = sb.validate_platform_manifest(name="svc", tags=["valid", ""])
        assert result.valid is False

    def test_validate_platform_manifest_empty_endpoint_key(self):
        sb = RegistrySandbox()
        result = sb.validate_platform_manifest(
            name="svc", endpoints={"": "http://localhost"}
        )
        assert result.valid is False

    def test_validate_namespace_valid(self):
        sb = RegistrySandbox()
        result = sb.validate_namespace("org.team")
        assert result.valid is True
        assert result.result["path"] == "org.team"

    def test_validate_namespace_empty(self):
        sb = RegistrySandbox()
        result = sb.validate_namespace("")
        assert result.valid is False

    def test_validate_dependency_valid(self):
        sb = RegistrySandbox()
        result = sb.validate_dependency("a", "b")
        assert result.valid is True

    def test_validate_dependency_empty_ids(self):
        sb = RegistrySandbox()
        result = sb.validate_dependency("", "b")
        assert result.valid is False

    def test_validate_dependency_cycle(self):
        sb = RegistrySandbox()
        sb.validate_dependency("a", "b")
        sb.validate_dependency("b", "c")
        result = sb.validate_dependency("c", "a")
        assert result.valid is False

    def test_validations_log(self):
        sb = RegistrySandbox()
        sb.validate_platform_manifest(name="svc")
        sb.validate_namespace("org")
        log = sb.get_validations_log()
        assert len(log) == 2

    def test_destroy(self):
        sb = RegistrySandbox()
        sb.destroy()
        assert sb.is_active is False

    def test_operations_after_destroy_raise(self):
        sb = RegistrySandbox()
        sb.destroy()
        with pytest.raises(RuntimeError, match="destroyed"):
            sb.validate_platform_manifest(name="svc")
