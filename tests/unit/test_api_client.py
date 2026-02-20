"""Unit tests for API client (Step 29)."""
import os
import pytest

CLIENT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "packages", "api-client", "src")


class TestApiClientStructure:
    def test_client_file_exists(self):
        assert os.path.isfile(os.path.join(CLIENT_DIR, "client.ts"))

    def test_index_file_exists(self):
        assert os.path.isfile(os.path.join(CLIENT_DIR, "index.ts"))

    def test_client_has_retry(self):
        content = open(os.path.join(CLIENT_DIR, "client.ts")).read()
        assert "RetryConfig" in content
        assert "maxRetries" in content

    def test_client_has_interceptors(self):
        content = open(os.path.join(CLIENT_DIR, "client.ts")).read()
        assert "RequestInterceptor" in content
        assert "onRequest" in content
        assert "onResponse" in content

    def test_client_has_typed_methods(self):
        content = open(os.path.join(CLIENT_DIR, "client.ts")).read()
        for method in ["health", "listModels", "chatCompletion", "embed", "generateYAML", "validateYAML"]:
            assert f"async {method}" in content, f"Missing method: {method}"

    def test_no_any_return_types(self):
        content = open(os.path.join(CLIENT_DIR, "client.ts")).read()
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if "async " in line and "Promise<" in line:
                assert "Promise<any>" not in line, f"Line {i+1} has untyped return: {line.strip()}"

    def test_index_exports_client(self):
        content = open(os.path.join(CLIENT_DIR, "index.ts")).read()
        assert "EcoApiClient" in content
        assert "ClientConfig" in content
