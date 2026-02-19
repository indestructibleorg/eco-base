"""Unit tests for web frontend integration.

Tests verify that the API endpoints consumed by the React frontend
return correct response structures. This ensures the frontend
components (Dashboard, AIPlayground, Models, Platforms, Settings)
receive the data shapes they expect.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def ai_app():
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from backend.ai.src.app import app
    return app


@pytest.fixture(scope="module")
def ai_client(ai_app):
    with TestClient(ai_app) as c:
        yield c


@pytest.fixture(scope="module")
def gateway_app():
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.app import create_app
    return create_app()


@pytest.fixture(scope="module")
def gateway_client(gateway_app):
    with TestClient(gateway_app) as c:
        yield c


class TestDashboardEndpoints:
    """Endpoints consumed by Dashboard page."""

    def test_health_returns_expected_shape(self, ai_client):
        resp = ai_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "service" in data
        assert "version" in data
        assert isinstance(data.get("engines", []), list)

    def test_models_returns_list(self, ai_client):
        resp = ai_client.get("/api/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        if len(data) > 0:
            m = data[0]
            assert "id" in m
            assert "status" in m

    def test_metrics_returns_data(self, ai_client):
        resp = ai_client.get("/metrics")
        assert resp.status_code == 200


class TestAIPlaygroundEndpoints:
    """Endpoints consumed by AI Playground page."""

    def test_generate_returns_content_and_metadata(self, ai_client):
        resp = ai_client.post("/api/v1/generate", json={
            "prompt": "Hello from playground",
            "model_id": "default",
            "max_tokens": 256,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "request_id" in data
        assert "content" in data
        assert "engine" in data
        assert "usage" in data
        assert "latency_ms" in data
        assert "uri" in data

    def test_generate_with_different_models(self, ai_client):
        for model in ["default", "vllm"]:
            resp = ai_client.post("/api/v1/generate", json={
                "prompt": f"Test {model}",
                "model_id": model,
                "max_tokens": 128,
            })
            assert resp.status_code == 200
            assert "content" in resp.json()


class TestModelsPageEndpoints:
    """Endpoints consumed by Models page."""

    def test_models_have_required_fields(self, ai_client):
        resp = ai_client.get("/api/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        for m in data:
            assert "id" in m
            assert "status" in m
            assert "provider" in m
            assert "capabilities" in m
            assert "uri" in m

    def test_openai_models_endpoint(self, ai_client):
        resp = ai_client.get("/api/v1/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1
        for m in data["data"]:
            assert "id" in m
            assert "owned_by" in m


class TestGatewayEndpoints:
    """Gateway endpoints consumed by frontend."""

    def test_gateway_health(self, gateway_client):
        resp = gateway_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_gateway_requires_auth_for_models(self, gateway_client):
        resp = gateway_client.get("/v1/models")
        assert resp.status_code == 401

    def test_gateway_requires_auth_for_chat(self, gateway_client):
        resp = gateway_client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert resp.status_code == 401


class TestFrontendFileStructure:
    """Verify frontend source files exist."""

    def test_required_pages_exist(self):
        import os
        base = os.path.join(os.path.dirname(__file__), "..", "..", "platforms", "web", "app", "src")
        pages = ["pages/Dashboard.tsx", "pages/Login.tsx", "pages/AIPlayground.tsx",
                 "pages/Models.tsx", "pages/Platforms.tsx", "pages/Settings.tsx"]
        for page in pages:
            path = os.path.join(base, page)
            assert os.path.exists(path), f"Missing page: {page}"

    def test_required_components_exist(self):
        import os
        base = os.path.join(os.path.dirname(__file__), "..", "..", "platforms", "web", "app", "src")
        components = ["components/Layout.tsx", "components/AIChat.tsx",
                      "components/ModelSelector.tsx", "components/ProtectedRoute.tsx"]
        for comp in components:
            path = os.path.join(base, comp)
            assert os.path.exists(path), f"Missing component: {comp}"

    def test_required_stores_exist(self):
        import os
        base = os.path.join(os.path.dirname(__file__), "..", "..", "platforms", "web", "app", "src")
        stores = ["store/auth.ts", "store/ai.ts", "store/platform.ts"]
        for store in stores:
            path = os.path.join(base, store)
            assert os.path.exists(path), f"Missing store: {store}"

    def test_required_hooks_exist(self):
        import os
        base = os.path.join(os.path.dirname(__file__), "..", "..", "platforms", "web", "app", "src")
        hooks = ["hooks/useAuth.ts", "hooks/useAI.ts", "hooks/useWebSocket.ts"]
        for hook in hooks:
            path = os.path.join(base, hook)
            assert os.path.exists(path), f"Missing hook: {hook}"

    def test_required_lib_exist(self):
        import os
        base = os.path.join(os.path.dirname(__file__), "..", "..", "platforms", "web", "app", "src")
        libs = ["lib/api.ts", "lib/ws.ts"]
        for lib in libs:
            path = os.path.join(base, lib)
            assert os.path.exists(path), f"Missing lib: {lib}"