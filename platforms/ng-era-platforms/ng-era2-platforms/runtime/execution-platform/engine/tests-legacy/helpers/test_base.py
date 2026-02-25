#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: test_base
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
"""
Base Test Helper Class
======================
Provides common utilities and helpers for all test types
"""
# MNGA-002: Import organization needs review
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import yaml
class TestHelper:
    """Base test helper with common utilities"""
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.test_data_dir = self.project_root / "tests" / "fixtures"
        self.temp_dir = None
    def setup_temp_dir(self):
        """Create temporary directory for tests"""
        self.temp_dir = tempfile.mkdtemp()
        return self.temp_dir
    def cleanup_temp_dir(self):
        """Clean up temporary directory"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
    def load_yaml(self, file_path: str) -> Dict:
        """Load YAML file"""
        full_path = self.test_data_dir / file_path
        with open(full_path, "r") as f:
            return yaml.safe_load(f)
    def load_json(self, file_path: str) -> Dict:
        """Load JSON file"""
        full_path = self.test_data_dir / file_path
        with open(full_path, "r") as f:
            return json.load(f)
    def save_yaml(self, data: Dict, file_path: str):
        """Save data to YAML file"""
        full_path = self.test_data_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
    def save_json(self, data: Dict, file_path: str):
        """Save data to JSON file"""
        full_path = self.test_data_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w") as f:
            json.dump(data, f, indent=2)
    def create_sample_artifact(self, **kwargs) -> Dict:
        """Create a sample artifact for testing"""
        default_artifact = {
            "id": kwargs.get("id", f"artifact-{datetime.now().timestamp()}"),
            "type": kwargs.get("type", "service"),
            "version": kwargs.get("version", "1.0.0"),
            "name": kwargs.get("name", "test-artifact"),
            "namespace": kwargs.get("namespace", "test-namespace"),
            "metadata": {
                "created_at": kwargs.get("created_at", datetime.now().isoformat()),
                "author": kwargs.get("author", "test-user"),
                "description": kwargs.get("description", "Test artifact"),
            },
        }
        return default_artifact
    def create_sample_namespace(self, **kwargs) -> Dict:
        """Create a sample namespace for testing"""
        default_namespace = {
            "id": kwargs.get("id", f"namespace-{datetime.now().timestamp()}"),
            "name": kwargs.get("name", "test-namespace"),
            "owner": kwargs.get("owner", "test-owner"),
            "status": kwargs.get("status", "active"),
            "created_at": kwargs.get("created_at", datetime.now().isoformat()),
            "description": kwargs.get("description", "Test namespace"),
        }
        return default_namespace
    def assert_valid_artifact(self, artifact: Dict):
        """Validate artifact structure"""
        required_fields = ["id", "type", "version", "name", "namespace"]
        for field in required_fields:
            assert field in artifact, f"Missing required field: {field}"
        assert isinstance(artifact["id"], str), "id must be a string"
        assert isinstance(artifact["type"], str), "type must be a string"
        assert isinstance(artifact["version"], str), "version must be a string"
    def assert_valid_namespace(self, namespace: Dict):
        """Validate namespace structure"""
        required_fields = ["id", "name", "owner", "status"]
        for field in required_fields:
            assert field in namespace, f"Missing required field: {field}"
        assert isinstance(namespace["id"], str), "id must be a string"
        assert isinstance(namespace["name"], str), "name must be a string"
        assert isinstance(namespace["status"], str), "status must be a string"
        assert namespace["status"] in [
            "active",
            "inactive",
            "pending",
        ], f"Invalid status: {namespace['status']}"
    def compare_dicts(self, dict1: Dict, dict2: Dict, ignore_fields: List[str] = None):
        """Compare two dictionaries, optionally ignoring certain fields"""
        if ignore_fields is None:
            ignore_fields = []
        filtered1 = {k: v for k, v in dict1.items() if k not in ignore_fields}
        filtered2 = {k: v for k, v in dict2.items() if k not in ignore_fields}
        assert (
            filtered1 == filtered2
        ), f"Dictionaries differ:\n{json.dumps(filtered1, indent=2)}\nvs\n{json.dumps(filtered2, indent=2)}"
    def wait_for_condition(
        self, condition_func, timeout: int = 10, interval: float = 0.5
    ):
        """Wait for a condition to become true"""
        import time
        elapsed = 0
        while elapsed < timeout:
            if condition_func():
                return True
            time.sleep(interval)
            elapsed += interval
        raise TimeoutError(f"Condition not met within {timeout} seconds")
class MockServer:
    """Mock server for testing HTTP clients"""
    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self.server = None
    def start(self):
        """Start mock server"""
        # Placeholder for mock server implementation
        pass
    def stop(self):
        """Stop mock server"""
        # Placeholder for mock server implementation
        pass
    def add_response(self, path: str, response: Dict, status_code: int = 200):
        """Add a mock response"""
        # Placeholder for adding mock responses
        pass
class TestLogger:
    """Test logging helper"""
    def __init__(self, name: str = "test"):
        self.name = name
        self.logs = []
    def log(self, level: str, message: str):
        """Log a message"""
        self.logs.append(
            {
                "level": level,
                "message": message,
                "timestamp": datetime.now().isoformat(),
            }
        )
    def info(self, message: str):
        """Log info message"""
        self.log("INFO", message)
    def debug(self, message: str):
        """Log debug message"""
        self.log("DEBUG", message)
    def error(self, message: str):
        """Log error message"""
        self.log("ERROR", message)
    def assert_logged(self, level: str, message: str = None):
        """Assert that a message was logged"""
        for log in self.logs:
            if log["level"] == level:
                if message is None or message in log["message"]:
                    return True
        raise AssertionError(f"No {level} log found with message: {message}")
