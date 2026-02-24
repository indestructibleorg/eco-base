"""Tests for PR blocked-response governance behavior."""
import importlib.util
import os


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIAGNOSE_PATH = os.path.join(REPO_ROOT, "tools", "pr-blocked-response", "diagnose.py")
ENGINE_PATH = os.path.join(REPO_ROOT, "tools", "autoecops-governance", "engine.py")


def _load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diagnose = _load_module(DIAGNOSE_PATH, "diagnose_module")
engine = _load_module(ENGINE_PATH, "engine_module")


def test_classify_checks_treats_required_skipped_as_pending():
    passing, failing, pending, all_pass, any_pending = diagnose.classify_checks([
        {"id": 10, "name": "validate", "status": "completed", "conclusion": "success"},
        {"id": 11, "name": "lint", "status": "completed", "conclusion": "skipped"},
    ])
    assert "validate" in passing
    assert "lint" in pending
    assert not failing
    assert not all_pass
    assert any_pending


def test_retrigger_ci_rerequests_skipped_required_checks(monkeypatch):
    calls = []

    monkeypatch.setattr(
        diagnose,
        "get_check_runs",
        lambda _: [{"id": 99, "name": "lint", "status": "completed", "conclusion": "skipped"}],
    )
    monkeypatch.setattr(
        diagnose,
        "gh_api",
        lambda path, method="GET", data=None: calls.append((path, method)) or {},
    )

    diagnose.retrigger_ci(123, "deadbeef", {"lint"})

    assert (f"/repos/{diagnose.REPO}/check-runs/99/rerequest", "POST") in calls


def test_engine_classify_marks_skipped_required_as_pending():
    summary = engine.classify_check_runs([
        {"name": "validate", "status": "completed", "conclusion": "success"},
        {"name": "lint", "status": "completed", "conclusion": "skipped"},
    ])
    assert summary["all_required_pass"] is False
    assert summary["any_required_pending"] is True
    assert summary["any_required_failed"] is False
    assert "lint" in summary["unexpected_skips"]
    assert "test" in summary["missing_required"]


def test_collect_non_required_gate_anomalies_detects_failed_and_key_skipped():
    anomalies = diagnose.collect_non_required_gate_anomalies([
        {"name": "Security Gates — Trivy + Checkov + Gitleaks", "status": "completed", "conclusion": "startup_failure"},
        {"name": "canary-gate", "status": "completed", "conclusion": "failure"},
        {"name": "optional-ci-health", "status": "completed", "conclusion": "skipped"},
        {"name": "optional-ci-health-queued", "status": "queued", "conclusion": "skipped"},
        {"name": "request-codacy-review", "status": "completed", "conclusion": "skipped"},
        {"name": "lint", "status": "completed", "conclusion": "failure"},
    ])
    assert ("Security Gates — Trivy + Checkov + Gitleaks", "startup_failure") in anomalies
    assert ("canary-gate", "failure") in anomalies
    assert ("optional-ci-health", "skipped") in anomalies
    assert ("optional-ci-health-queued", "skipped") not in anomalies
    assert ("request-codacy-review", "skipped") not in anomalies
    assert ("lint", "failure") not in anomalies


def test_build_dedup_key_uses_pr_sha_and_source(monkeypatch):
    monkeypatch.setattr(diagnose, "TRIGGER_EVENT", "workflow_run")
    monkeypatch.setattr(diagnose, "SOURCE_WORKFLOW", "Security Gates — Trivy + Checkov + Gitleaks")
    assert (
        diagnose.build_dedup_key(280, "abc123")
        == "280:abc123:Security Gates — Trivy + Checkov + Gitleaks"
    )


def test_upsert_anomaly_issue_includes_source_observed_at_and_markers(monkeypatch):
    captured = {}
    monkeypatch.setattr(diagnose, "TRIGGER_EVENT", "workflow_run")
    monkeypatch.setattr(diagnose, "SOURCE_WORKFLOW", "Security Gates — Trivy + Checkov + Gitleaks")
    monkeypatch.setattr(diagnose, "find_open_auto_anomaly_issue", lambda _: None)

    def fake_gh_api(path, method="GET", data=None):
        if path.endswith("/issues") and method == "POST":
            captured["body"] = data["body"]
            return {"number": 321}
        return {}

    monkeypatch.setattr(diagnose, "gh_api", fake_gh_api)

    result = diagnose.upsert_auto_anomaly_issue(280, [("canary-gate", "failure")], "abc123")

    assert result["issue_number"] == 321
    assert result["anomaly_count"] == 1
    assert result["dedup_skipped"] is False
    assert "Trigger Source" in captured["body"]
    assert "workflow_run: Security Gates — Trivy + Checkov + Gitleaks" in captured["body"]
    assert "Observed At" in captured["body"]
    assert "<!-- autoecoops:dkey=280:abc123:Security Gates — Trivy + Checkov + Gitleaks -->" in captured["body"]


def test_upsert_anomaly_issue_dedups_same_key(monkeypatch):
    monkeypatch.setattr(diagnose, "TRIGGER_EVENT", "schedule")
    monkeypatch.setattr(diagnose, "SOURCE_WORKFLOW", "")
    dedup_key = diagnose.build_dedup_key(280, "headsha")
    issue_body = (
        "x\n"
        f"<!-- autoecoops:dkey={dedup_key} -->\n"
        "<!-- autoecoops:asig=canary-gate::failure -->\n"
        "<!-- autoecoops:acount=2 -->\n"
    )
    monkeypatch.setattr(
        diagnose,
        "find_open_auto_anomaly_issue",
        lambda _: {"number": 42, "body": issue_body},
    )
    calls = []
    monkeypatch.setattr(diagnose, "gh_api", lambda *args, **kwargs: calls.append((args, kwargs)) or {})

    result = diagnose.upsert_auto_anomaly_issue(280, [("canary-gate", "failure")], "headsha")

    assert result["issue_number"] == 42
    assert result["anomaly_count"] == 2
    assert result["dedup_skipped"] is True
    assert not calls


def test_apply_anomaly_escalation_adds_labels(monkeypatch):
    added_labels = []
    patched_labels = []

    monkeypatch.setattr(diagnose, "add_label", lambda pr, label: added_labels.append((pr, label)))

    def fake_gh_api(path, method="GET", data=None):
        if path.endswith("/issues/88") and method == "GET":
            return {"labels": [{"name": "blocked"}]}
        if path.endswith("/issues/88") and method == "PATCH":
            patched_labels.append(data["labels"])
            return {}
        return {}

    monkeypatch.setattr(diagnose, "gh_api", fake_gh_api)

    diagnose.apply_anomaly_escalation(280, 88, diagnose.ANOMALY_ESCALATION_THRESHOLD)

    assert (280, diagnose.HUMAN_REVIEW_LABEL) in added_labels
    assert patched_labels
    assert diagnose.ANOMALY_SEVERITY_LABEL in patched_labels[0]
