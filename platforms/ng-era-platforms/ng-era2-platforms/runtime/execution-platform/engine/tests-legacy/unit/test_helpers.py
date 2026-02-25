#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: test_helpers
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
"""
Unit Tests for Test Helpers
===========================
Tests for the base test helper functionality
"""
import os
import pytest
from tests.helpers import TestHelper, TestLogger
class TestTestHelper:
    """Test cases for TestHelper class"""
    def test_initialization(self):
        """Test TestHelper initialization"""
        helper = TestHelper()
        assert helper.project_root is not None
        assert helper.test_data_dir is not None
        assert helper.temp_dir is None
    def test_setup_temp_dir(self):
        """Test temporary directory setup"""
        helper = TestHelper()
        temp_dir = helper.setup_temp_dir()
        assert temp_dir is not None
        assert os.path.exists(temp_dir)
        assert helper.temp_dir == temp_dir
        # Cleanup
        helper.cleanup_temp_dir()
        assert not os.path.exists(temp_dir)
    def test_cleanup_temp_dir(self):
        """Test temporary directory cleanup"""
        helper = TestHelper()
        temp_dir = helper.setup_temp_dir()
        assert os.path.exists(temp_dir)
        helper.cleanup_temp_dir()
        assert not os.path.exists(temp_dir)
        assert helper.temp_dir is None
    def test_create_sample_artifact(self):
        """Test sample artifact creation"""
        helper = TestHelper()
        artifact = helper.create_sample_artifact()
        assert "id" in artifact
        assert "type" in artifact
        assert "version" in artifact
        assert "name" in artifact
        assert "namespace" in artifact
        assert "metadata" in artifact
    def test_create_sample_artifact_with_custom_fields(self):
        """Test sample artifact creation with custom fields"""
        helper = TestHelper()
        artifact = helper.create_sample_artifact(
            id="custom-id", type="custom-type", version="2.0.0", name="custom-name"
        )
        assert artifact["id"] == "custom-id"
        assert artifact["type"] == "custom-type"
        assert artifact["version"] == "2.0.0"
        assert artifact["name"] == "custom-name"
    def test_create_sample_namespace(self):
        """Test sample namespace creation"""
        helper = TestHelper()
        namespace = helper.create_sample_namespace()
        assert "id" in namespace
        assert "name" in namespace
        assert "owner" in namespace
        assert "status" in namespace
        assert "created_at" in namespace
    def test_assert_valid_artifact(self):
        """Test artifact validation"""
        helper = TestHelper()
        # Valid artifact should pass
        artifact = helper.create_sample_artifact()
        helper.assert_valid_artifact(artifact)  # Should not raise
        # Missing field should raise
        invalid_artifact = {"id": "test", "type": "service"}
        with pytest.raises(AssertionError, match="Missing required field"):
            helper.assert_valid_artifact(invalid_artifact)
    def test_assert_valid_namespace(self):
        """Test namespace validation"""
        helper = TestHelper()
        # Valid namespace should pass
        namespace = helper.create_sample_namespace()
        helper.assert_valid_namespace(namespace)  # Should not raise
        # Invalid status should raise
        invalid_namespace = helper.create_sample_namespace(status="invalid")
        with pytest.raises(AssertionError, match="Invalid status"):
            helper.assert_valid_namespace(invalid_namespace)
    def test_compare_dicts(self):
        """Test dictionary comparison"""
        helper = TestHelper()
        dict1 = {"a": 1, "b": 2, "c": 3}
        dict2 = {"a": 1, "b": 2, "c": 3}
        # Identical dicts should pass
        helper.compare_dicts(dict1, dict2)
        # Different dicts should raise
        dict3 = {"a": 1, "b": 2, "c": 4}
        with pytest.raises(AssertionError, match="Dictionaries differ"):
            helper.compare_dicts(dict1, dict3)
        # Ignoring fields should pass
        dict4 = {"a": 1, "b": 2, "c": 4, "d": 5}
        helper.compare_dicts(dict1, dict4, ignore_fields=["c", "d"])
    def test_wait_for_condition(self):
        """Test waiting for condition"""
        helper = TestHelper()
        # Condition immediately true
        assert helper.wait_for_condition(lambda: True, timeout=1)
        # Condition becomes true after delay
        counter = {"count": 0}
        def condition():
            counter["count"] += 1
            return counter["count"] >= 3
        assert helper.wait_for_condition(condition, timeout=2, interval=0.1)
        # Condition never true should timeout
        with pytest.raises(TimeoutError):
            helper.wait_for_condition(lambda: False, timeout=0.5, interval=0.1)
class TestTestLogger:
    """Test cases for TestLogger class"""
    def test_initialization(self):
        """Test TestLogger initialization"""
        logger = TestLogger("test-logger")
        assert logger.name == "test-logger"
        assert len(logger.logs) == 0
    def test_logging(self):
        """Test logging functionality"""
        logger = TestLogger()
        logger.info("Info message")
        logger.debug("Debug message")
        logger.error("Error message")
        assert len(logger.logs) == 3
        assert logger.logs[0]["level"] == "INFO"
        assert logger.logs[1]["level"] == "DEBUG"
        assert logger.logs[2]["level"] == "ERROR"
    def test_assert_logged(self):
        """Test log assertion"""
        logger = TestLogger()
        logger.info("Test message")
        # Should pass
        logger.assert_logged("INFO", "Test message")
        # Should fail with wrong message
        with pytest.raises(AssertionError):
            logger.assert_logged("INFO", "Wrong message")
        # Should fail with wrong level
        with pytest.raises(AssertionError):
            logger.assert_logged("ERROR", "Test message")
    def test_log_structure(self):
        """Test log entry structure"""
        logger = TestLogger()
        logger.info("Test message")
        log_entry = logger.logs[0]
        assert "level" in log_entry
        assert "message" in log_entry
        assert "timestamp" in log_entry
        assert log_entry["level"] == "INFO"
        assert log_entry["message"] == "Test message"
@pytest.fixture
def test_helper():
    """Fixture providing TestHelper instance"""
    return TestHelper()
@pytest.fixture
def test_logger():
    """Fixture providing TestLogger instance"""
    return TestLogger("test-logger")
