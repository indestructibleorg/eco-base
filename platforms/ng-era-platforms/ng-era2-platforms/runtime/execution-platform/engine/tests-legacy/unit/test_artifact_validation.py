#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: test_artifact_validation
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
"""
Unit Tests for Artifact Validation
==================================
Tests for artifact structure and validation logic
"""
import pytest
from tests.helpers import TestHelper
class TestArtifactValidation:
    """Test cases for artifact validation"""
    @pytest.fixture
    def helper(self):
        """Test helper fixture"""
        return TestHelper()
    @pytest.fixture
    def valid_artifact(self):
        """Valid artifact fixture"""
        return {
            "id": "artifact-001",
            "type": "service",
            "version": "1.0.0",
            "name": "test-service",
            "namespace": "test-namespace",
            "metadata": {"created_at": "2025-01-16T00:00:00Z", "author": "test-user"},
        }
    def test_valid_artifact_structure(self, helper, valid_artifact):
        """Test that valid artifact passes validation"""
        # Should not raise any assertion
        helper.assert_valid_artifact(valid_artifact)
    def test_missing_required_field(self, helper, valid_artifact):
        """Test that missing required fields raise assertion"""
        # Remove id field
        artifact = valid_artifact.copy()
        del artifact["id"]
        with pytest.raises(AssertionError, match="Missing required field: id"):
            helper.assert_valid_artifact(artifact)
    def test_missing_all_required_fields(self, helper):
        """Test that missing all fields raises appropriate error"""
        artifact = {}
        with pytest.raises(AssertionError, match="Missing required field"):
            helper.assert_valid_artifact(artifact)
    def test_invalid_id_type(self, helper, valid_artifact):
        """Test that non-string id raises assertion"""
        artifact = valid_artifact.copy()
        artifact["id"] = 12345
        with pytest.raises(AssertionError, match="id must be a string"):
            helper.assert_valid_artifact(artifact)
    def test_invalid_type_type(self, helper, valid_artifact):
        """Test that non-string type raises assertion"""
        artifact = valid_artifact.copy()
        artifact["type"] = ["service"]
        with pytest.raises(AssertionError, match="type must be a string"):
            helper.assert_valid_artifact(artifact)
    def test_invalid_version_type(self, helper, valid_artifact):
        """Test that non-string version raises assertion"""
        artifact = valid_artifact.copy()
        artifact["version"] = 1.0
        with pytest.raises(AssertionError, match="version must be a string"):
            helper.assert_valid_artifact(artifact)
    def test_artifact_metadata_optional(self, helper):
        """Test that metadata is optional"""
        artifact = {
            "id": "artifact-001",
            "type": "service",
            "version": "1.0.0",
            "name": "test-service",
            "namespace": "test-namespace",
        }
        # Should not raise - metadata is optional
        helper.assert_valid_artifact(artifact)
    def test_artifact_types(self, helper):
        """Test various artifact types"""
        valid_types = ["service", "component", "library", "tool", "system"]
        for artifact_type in valid_types:
            artifact = helper.create_sample_artifact(type=artifact_type)
            helper.assert_valid_artifact(artifact)
    def test_artifact_version_formats(self, helper):
        """Test various version formats"""
        versions = ["1.0.0", "2.1.3", "0.0.1", "10.20.30", "1.0.0-beta", "2.1.0-rc1"]
        for version in versions:
            artifact = helper.create_sample_artifact(version=version)
            helper.assert_valid_artifact(artifact)
    def test_artifact_id_generation(self, helper):
        """Test that artifact IDs are generated uniquely"""
        artifact1 = helper.create_sample_artifact()
        artifact2 = helper.create_sample_artifact()
        assert artifact1["id"] != artifact2["id"]
class TestNamespaceValidation:
    """Test cases for namespace validation"""
    @pytest.fixture
    def helper(self):
        """Test helper fixture"""
        return TestHelper()
    @pytest.fixture
    def valid_namespace(self):
        """Valid namespace fixture"""
        return {
            "id": "namespace-001",
            "name": "test-namespace",
            "owner": "test-owner",
            "status": "active",
            "created_at": "2025-01-16T00:00:00Z",
        }
    def test_valid_namespace_structure(self, helper, valid_namespace):
        """Test that valid namespace passes validation"""
        # Should not raise any assertion
        helper.assert_valid_namespace(valid_namespace)
    def test_missing_required_field(self, helper, valid_namespace):
        """Test that missing required fields raise assertion"""
        # Remove name field
        namespace = valid_namespace.copy()
        del namespace["name"]
        with pytest.raises(AssertionError, match="Missing required field: name"):
            helper.assert_valid_namespace(namespace)
    def test_invalid_id_type(self, helper, valid_namespace):
        """Test that non-string id raises assertion"""
        namespace = valid_namespace.copy()
        namespace["id"] = 12345
        with pytest.raises(AssertionError, match="id must be a string"):
            helper.assert_valid_namespace(namespace)
    def test_invalid_name_type(self, helper, valid_namespace):
        """Test that non-string name raises assertion"""
        namespace = valid_namespace.copy()
        namespace["name"] = ["test-namespace"]
        with pytest.raises(AssertionError, match="name must be a string"):
            helper.assert_valid_namespace(namespace)
    def test_valid_status_active(self, helper):
        """Test that 'active' status is valid"""
        namespace = helper.create_sample_namespace(status="active")
        helper.assert_valid_namespace(namespace)
    def test_valid_status_inactive(self, helper):
        """Test that 'inactive' status is valid"""
        namespace = helper.create_sample_namespace(status="inactive")
        helper.assert_valid_namespace(namespace)
    def test_valid_status_pending(self, helper):
        """Test that 'pending' status is valid"""
        namespace = helper.create_sample_namespace(status="pending")
        helper.assert_valid_namespace(namespace)
    def test_invalid_status(self, helper):
        """Test that invalid status raises assertion"""
        namespace = helper.create_sample_namespace(status="invalid-status")
        with pytest.raises(AssertionError, match="Invalid status"):
            helper.assert_valid_namespace(namespace)
    def test_namespace_id_generation(self, helper):
        """Test that namespace IDs are generated uniquely"""
        namespace1 = helper.create_sample_namespace()
        namespace2 = helper.create_sample_namespace()
        assert namespace1["id"] != namespace2["id"]
class TestArtifactNamespaceRelationships:
    """Test cases for artifact and namespace relationships"""
    @pytest.fixture
    def helper(self):
        """Test helper fixture"""
        return TestHelper()
    def test_artifact_references_namespace(self, helper):
        """Test that artifact can reference a namespace"""
        namespace = helper.create_sample_namespace(name="my-namespace")
        artifact = helper.create_sample_artifact(namespace="my-namespace")
        assert artifact["namespace"] == namespace["name"]
    def test_multiple_artifacts_same_namespace(self, helper):
        """Test multiple artifacts in the same namespace"""
        namespace = "shared-namespace"
        artifact1 = helper.create_sample_artifact(name="artifact1", namespace=namespace)
        artifact2 = helper.create_sample_artifact(name="artifact2", namespace=namespace)
        assert artifact1["namespace"] == namespace
        assert artifact2["namespace"] == namespace
        assert artifact1["id"] != artifact2["id"]
    def test_artifacts_different_namespaces(self, helper):
        """Test artifacts in different namespaces"""
        artifact1 = helper.create_sample_artifact(
            name="artifact1", namespace="namespace-1"
        )
        artifact2 = helper.create_sample_artifact(
            name="artifact2", namespace="namespace-2"
        )
        assert artifact1["namespace"] == "namespace-1"
        assert artifact2["namespace"] == "namespace-2"
        assert artifact1["namespace"] != artifact2["namespace"]
