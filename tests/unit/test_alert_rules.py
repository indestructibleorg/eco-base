"""Unit tests for Prometheus alert rules (Step 31)."""
import os
import pytest
import yaml

ALERT_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "ecosystem", "monitoring", "alertmanager", "alert-rules.yaml"
)


class TestAlertRules:
    @pytest.fixture
    def rules(self):
        with open(ALERT_FILE, encoding='utf-8') as f:
            return yaml.safe_load(f)

    def test_file_exists(self):
        assert os.path.isfile(ALERT_FILE)

    def test_valid_yaml(self):
        with open(ALERT_FILE, encoding='utf-8') as f:
            data = yaml.safe_load(f)
        assert "groups" in data

    def test_has_inference_group(self, rules):
        names = [g["name"] for g in rules["groups"]]
        assert "eco-inference-alerts" in names

    def test_has_queue_group(self, rules):
        names = [g["name"] for g in rules["groups"]]
        assert "eco-queue-alerts" in names

    def test_has_error_group(self, rules):
        names = [g["name"] for g in rules["groups"]]
        assert "eco-error-alerts" in names

    def test_has_resource_group(self, rules):
        names = [g["name"] for g in rules["groups"]]
        assert "eco-resource-alerts" in names

    def test_high_latency_alert(self, rules):
        inference = [g for g in rules["groups"] if g["name"] == "eco-inference-alerts"][0]
        alerts = [r["alert"] for r in inference["rules"]]
        assert "HighInferenceLatency" in alerts

    def test_engine_down_alert(self, rules):
        inference = [g for g in rules["groups"] if g["name"] == "eco-inference-alerts"][0]
        alerts = [r["alert"] for r in inference["rules"]]
        assert "EngineDown" in alerts
        assert "AllEnginesDown" in alerts

    def test_queue_overflow_alert(self, rules):
        queue = [g for g in rules["groups"] if g["name"] == "eco-queue-alerts"][0]
        alerts = [r["alert"] for r in queue["rules"]]
        assert "QueueOverflow" in alerts

    def test_error_rate_alert(self, rules):
        errors = [g for g in rules["groups"] if g["name"] == "eco-error-alerts"][0]
        alerts = [r["alert"] for r in errors["rules"]]
        assert "HighErrorRate" in alerts
        assert "CriticalErrorRate" in alerts

    def test_all_alerts_have_severity(self, rules):
        for group in rules["groups"]:
            for rule in group["rules"]:
                assert "severity" in rule["labels"], f"Alert {rule['alert']} missing severity"

    def test_all_alerts_have_annotations(self, rules):
        for group in rules["groups"]:
            for rule in group["rules"]:
                assert "summary" in rule["annotations"], f"Alert {rule['alert']} missing summary"
                assert "description" in rule["annotations"], f"Alert {rule['alert']} missing description"

    def test_eco_prefix_in_exprs(self, rules):
        eco_count = 0
        for group in rules["groups"]:
            for rule in group["rules"]:
                if "eco_" in rule["expr"]:
                    eco_count += 1
        assert eco_count >= 10
