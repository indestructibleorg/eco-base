"""Unit tests for Grafana dashboard (Step 30)."""
import json
import os
import pytest

DASHBOARD = os.path.join(
    os.path.dirname(__file__), "..", "..", "ecosystem", "monitoring", "dashboards", "ai-service.json"
)


class TestGrafanaDashboard:
    @pytest.fixture
    def dashboard(self):
        with open(DASHBOARD, encoding='utf-8') as f:
            return json.load(f)

    def test_dashboard_exists(self):
        assert os.path.isfile(DASHBOARD)

    def test_valid_json(self):
        with open(DASHBOARD, encoding='utf-8') as f:
            data = json.load(f)
        assert "panels" in data

    def test_has_inference_latency_panel(self, dashboard):
        titles = [p["title"] for p in dashboard["panels"]]
        assert "Inference Latency (p50/p95/p99)" in titles

    def test_has_engine_health_panel(self, dashboard):
        titles = [p["title"] for p in dashboard["panels"]]
        assert "Engine Health Status" in titles

    def test_has_circuit_breaker_panel(self, dashboard):
        titles = [p["title"] for p in dashboard["panels"]]
        assert "Circuit Breaker State" in titles

    def test_has_queue_depth_panel(self, dashboard):
        titles = [p["title"] for p in dashboard["panels"]]
        assert "Request Queue Depth" in titles

    def test_has_throughput_panel(self, dashboard):
        titles = [p["title"] for p in dashboard["panels"]]
        assert "Inference Throughput (req/s)" in titles

    def test_has_error_rate_panel(self, dashboard):
        titles = [p["title"] for p in dashboard["panels"]]
        assert "Error Rate by Engine" in titles

    def test_panel_count(self, dashboard):
        assert len(dashboard["panels"]) >= 9

    def test_has_eco_prefix_metrics(self, dashboard):
        all_exprs = []
        for p in dashboard["panels"]:
            for t in p.get("targets", []):
                all_exprs.append(t.get("expr", ""))
        eco_count = sum(1 for e in all_exprs if "eco_" in e)
        assert eco_count >= 8
