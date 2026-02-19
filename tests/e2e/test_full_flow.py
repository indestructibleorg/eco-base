"""End-to-end tests for IndestructibleEco API flow.

Tests the complete request lifecycle:
  auth -> generate -> models -> vector/align -> health -> metrics

Uses the src.app.create_app() factory for the core gateway,
and backend.ai.src.app for the AI service.
"""

import pytest
from fastapi.testclient import TestClient


# --- AI Service E2E Tests ---

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


class TestAIServiceE2E:
    """Full flow through AI service endpoints."""

    def test_health_returns_engines(self, ai_client):
        resp = ai_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "ai"
        assert "engines" in data
        assert isinstance(data["engines"], list)
        assert "uptime_seconds" in data

    def test_engine_health_detail(self, ai_client):
        resp = ai_client.get("/health/engines")
        assert resp.status_code == 200
        data = resp.json()
        assert "engines" in data
        assert "circuits" in data
        assert "pool" in data
        # Verify all 7 engines are tracked
        assert len(data["engines"]) == 7
        assert len(data["circuits"]) == 7
        for name in ["vllm", "tgi", "ollama", "sglang", "tensorrt", "deepspeed", "lmdeploy"]:
            assert name in data["engines"]
            assert name in data["circuits"]

    def test_metrics_has_engine_data(self, ai_client):
        resp = ai_client.get("/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "eco_uptime_seconds" in text
        assert "eco_memory_maxrss_bytes" in text

    def test_generate_with_engine_routing(self, ai_client):
        resp = ai_client.post("/api/v1/generate", json={
            "prompt": "Explain circuit breakers in distributed systems",
            "model_id": "default",
            "max_tokens": 512,
            "temperature": 0.5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "request_id" in data
        assert "content" in data
        assert "engine" in data
        assert "uri" in data
        assert data["uri"].startswith("indestructibleeco://")
        assert "urn" in data
        assert data["urn"].startswith("urn:indestructibleeco:")
        assert "usage" in data
        assert "latency_ms" in data

    def test_generate_specific_model(self, ai_client):
        resp = ai_client.post("/api/v1/generate", json={
            "prompt": "Hello",
            "model_id": "vllm",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == "vllm"

    def test_vector_align(self, ai_client):
        resp = ai_client.post("/api/v1/vector/align", json={
            "tokens": ["kubernetes", "deployment", "service"],
            "target_dim": 1024,
            "alignment_model": "quantum-bert-xxl-v1",
            "tolerance": 0.001,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["dimension"] == 1024
        assert data["alignment_model"] == "quantum-bert-xxl-v1"
        assert 0.0 < data["alignment_score"] <= 1.0
        assert len(data["coherence_vector"]) == 10
        assert data["uri"].startswith("indestructibleeco://")

    def test_vector_align_invalid_dim(self, ai_client):
        resp = ai_client.post("/api/v1/vector/align", json={
            "tokens": ["test"],
            "target_dim": 100,
        })
        assert resp.status_code == 400

    def test_models_list(self, ai_client):
        resp = ai_client.get("/api/v1/models")
        assert resp.status_code == 200
        models = resp.json()
        assert isinstance(models, list)
        assert len(models) >= 1
        for m in models:
            assert "id" in m
            assert "provider" in m
            assert "status" in m
            assert "capabilities" in m
            assert m["uri"].startswith("indestructibleeco://")

    def test_qyaml_descriptor(self, ai_client):
        resp = ai_client.post("/api/v1/qyaml/descriptor", json={
            "service_name": "test-service",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        desc = data["descriptor"]
        assert "document_metadata" in desc
        assert "governance_info" in desc
        assert "registry_binding" in desc
        assert "vector_alignment_map" in desc
        assert desc["document_metadata"]["schema_version"] == "v1"


# --- Core Gateway E2E Tests ---

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


class TestGatewayE2E:
    """Full flow through core gateway."""

    def test_health(self, gateway_client):
        resp = gateway_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"

    def test_metrics(self, gateway_client):
        resp = gateway_client.get("/metrics")
        assert resp.status_code == 200

    def test_models_requires_auth(self, gateway_client):
        resp = gateway_client.get("/v1/models")
        assert resp.status_code == 401

    def test_chat_completions_requires_auth(self, gateway_client):
        resp = gateway_client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert resp.status_code == 401


# --- Circuit Breaker Unit Verification ---

class TestCircuitBreakerIntegration:
    """Verify circuit breaker state machine in isolation."""

    def test_breaker_lifecycle(self):
        from backend.ai.src.services.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(name="test-engine", failure_threshold=2, recovery_timeout=0.1)
        assert cb.state == CircuitState.CLOSED

        # Two failures should open the circuit
        assert cb.allow_request() is True
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

        assert cb.allow_request() is True
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Should reject while open
        assert cb.allow_request() is False
        assert cb.total_rejections == 1

        # Wait for recovery
        import time
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        # Probe succeeds -> closed
        assert cb.allow_request() is True
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_breaker_half_open_failure(self):
        from backend.ai.src.services.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(name="test-fail", failure_threshold=1, recovery_timeout=0.1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        import time
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        assert cb.allow_request() is True
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
