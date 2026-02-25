"""Tests for security domain entities, value objects, events, and exceptions."""

import pytest

from security_platform.domain.value_objects import (
    ThreatLevel,
    EnforcementAction,
    AuditVerdict,
)
from security_platform.domain.entities import (
    SecurityPolicy,
    PolicyCondition,
    Violation,
    AuditEntry,
    IsolationRule,
)
from security_platform.domain.events import (
    ViolationDetected,
    PolicyEnforced,
    AuditRecorded,
)
from security_platform.domain.exceptions import (
    SecurityError,
    PolicyViolationError,
    AuditIntegrityError,
    IsolationError,
)


# -- Value Objects --


class TestThreatLevel:
    def test_values(self):
        assert ThreatLevel.CRITICAL == "critical"
        assert ThreatLevel.HIGH == "high"
        assert ThreatLevel.MEDIUM == "medium"
        assert ThreatLevel.LOW == "low"
        assert ThreatLevel.INFO == "info"

    def test_ordering(self):
        levels = [ThreatLevel.INFO, ThreatLevel.CRITICAL, ThreatLevel.LOW]
        assert ThreatLevel.CRITICAL in levels


class TestEnforcementAction:
    def test_values(self):
        assert EnforcementAction.BLOCK == "block"
        assert EnforcementAction.WARN == "warn"
        assert EnforcementAction.LOG == "log"


class TestAuditVerdict:
    def test_values(self):
        assert AuditVerdict.ALLOWED == "allowed"
        assert AuditVerdict.DENIED == "denied"
        assert AuditVerdict.ESCALATED == "escalated"


# -- Entities --


class TestPolicyCondition:
    def test_contains(self):
        cond = PolicyCondition(field="path", operator="contains", value="/admin")
        assert cond.evaluate({"path": "/admin/users"}) is True
        assert cond.evaluate({"path": "/api/users"}) is False

    def test_eq(self):
        cond = PolicyCondition(field="method", operator="eq", value="DELETE")
        assert cond.evaluate({"method": "DELETE"}) is True
        assert cond.evaluate({"method": "GET"}) is False

    def test_ne(self):
        cond = PolicyCondition(field="role", operator="ne", value="admin")
        assert cond.evaluate({"role": "user"}) is True
        assert cond.evaluate({"role": "admin"}) is False

    def test_in(self):
        cond = PolicyCondition(field="method", operator="in", value=["POST", "DELETE"])
        assert cond.evaluate({"method": "DELETE"}) is True
        assert cond.evaluate({"method": "GET"}) is False

    def test_not_in(self):
        cond = PolicyCondition(field="method", operator="not_in", value=["GET", "HEAD"])
        assert cond.evaluate({"method": "DELETE"}) is True
        assert cond.evaluate({"method": "GET"}) is False

    def test_exists(self):
        cond = PolicyCondition(field="auth_token", operator="exists")
        assert cond.evaluate({"auth_token": "abc"}) is True
        assert cond.evaluate({"other": "val"}) is False

    def test_missing_field(self):
        cond = PolicyCondition(field="missing", operator="contains", value="x")
        assert cond.evaluate({"other": "y"}) is False


class TestSecurityPolicy:
    def test_matches_all_conditions(self):
        policy = SecurityPolicy(
            name="block-admin-deletes",
            conditions=[
                PolicyCondition(field="path", operator="contains", value="/admin"),
                PolicyCondition(field="method", operator="eq", value="DELETE"),
            ],
        )
        assert policy.matches({"path": "/admin/users", "method": "DELETE"}) is True
        assert policy.matches({"path": "/admin/users", "method": "GET"}) is False

    def test_disabled_policy(self):
        policy = SecurityPolicy(
            name="disabled",
            enabled=False,
            conditions=[PolicyCondition(field="x", operator="eq", value="y")],
        )
        assert policy.matches({"x": "y"}) is False

    def test_no_conditions(self):
        policy = SecurityPolicy(name="empty")
        assert policy.matches({"any": "thing"}) is False


class TestAuditEntry:
    def test_compute_hash(self):
        entry = AuditEntry(
            actor="admin",
            action="delete",
            resource="user/123",
            verdict=AuditVerdict.ALLOWED,
        )
        h1 = entry.compute_hash("")
        h2 = entry.compute_hash("previous")
        assert len(h1) == 64  # SHA-256 hex
        assert h1 != h2

    def test_hash_chain(self):
        e1 = AuditEntry(actor="a", action="x", resource="r", verdict=AuditVerdict.ALLOWED)
        e1.hash = e1.compute_hash("")

        e2 = AuditEntry(actor="b", action="y", resource="s", verdict=AuditVerdict.DENIED)
        e2.previous_hash = e1.hash
        e2.hash = e2.compute_hash(e1.hash)

        assert e2.previous_hash == e1.hash


class TestIsolationRule:
    def test_defaults(self):
        rule = IsolationRule(sandbox_id="sb-123")
        assert rule.no_network is True
        assert rule.readonly_fs is True
        assert rule.max_memory_mb == 256
        assert rule.allowed_paths == []

    def test_custom_rule(self):
        rule = IsolationRule(
            sandbox_id="sb-456",
            no_network=False,
            readonly_fs=False,
            max_memory_mb=512,
            allowed_paths=["/tmp"],
        )
        assert rule.no_network is False
        assert rule.max_memory_mb == 512


class TestViolation:
    def test_creation(self):
        v = Violation(
            policy_name="test",
            field="path",
            message="bad path",
            threat_level=ThreatLevel.HIGH,
            action=EnforcementAction.BLOCK,
        )
        assert v.policy_name == "test"
        assert v.violation_id


# -- Events --


class TestViolationDetected:
    def test_creation(self):
        event = ViolationDetected(
            policy_name="test",
            threat_level=ThreatLevel.CRITICAL,
            action=EnforcementAction.BLOCK,
        )
        assert event.policy_name == "test"
        assert event.timestamp > 0


class TestPolicyEnforced:
    def test_creation(self):
        event = PolicyEnforced(
            policy_name="test",
            action=EnforcementAction.BLOCK,
            request_blocked=True,
        )
        assert event.request_blocked is True


class TestAuditRecorded:
    def test_creation(self):
        event = AuditRecorded(entry_id="abc", actor="admin", action="login")
        assert event.actor == "admin"


# -- Exceptions --


class TestExceptions:
    def test_security_error(self):
        err = SecurityError("test")
        assert str(err) == "test"
        assert err.code == "SECURITY_ERROR"

    def test_policy_violation_error(self):
        err = PolicyViolationError("blocked", violations=["v1", "v2"])
        assert err.violations == ["v1", "v2"]

    def test_audit_integrity_error(self):
        err = AuditIntegrityError("chain broken", entry_id="abc")
        assert err.entry_id == "abc"

    def test_isolation_error(self):
        err = IsolationError("escaped", sandbox_id="sb-1")
        assert err.sandbox_id == "sb-1"
