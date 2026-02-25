"""Tests for security engines — EnforcementEngine, AuditEngine, IsolationEngine."""

import pytest

from security_platform.domain.entities import (
    SecurityPolicy,
    PolicyCondition,
    IsolationRule,
)
from security_platform.domain.value_objects import (
    ThreatLevel,
    EnforcementAction,
    AuditVerdict,
)
from security_platform.domain.exceptions import (
    PolicyViolationError,
    AuditIntegrityError,
    IsolationError,
)
from security_platform.engines.enforcement_engine import EnforcementEngine
from security_platform.engines.audit_engine import AuditEngine
from security_platform.engines.isolation_engine import IsolationEngine


# ============================================================================
# EnforcementEngine
# ============================================================================


class TestEnforcementEngine:
    def test_no_violations(self, enforcement_engine: EnforcementEngine):
        policy = SecurityPolicy(
            name="block-admin",
            conditions=[PolicyCondition(field="path", operator="contains", value="/admin")],
        )
        enforcement_engine.register_policy(policy)
        result = enforcement_engine.evaluate_request({"path": "/api/users"})
        assert result.allowed is True
        assert result.violations == []

    def test_block_violation(self, enforcement_engine_permissive: EnforcementEngine):
        policy = SecurityPolicy(
            name="block-deletes",
            threat_level=ThreatLevel.HIGH,
            action=EnforcementAction.BLOCK,
            conditions=[PolicyCondition(field="method", operator="eq", value="DELETE")],
        )
        enforcement_engine_permissive.register_policy(policy)
        result = enforcement_engine_permissive.evaluate_request({"method": "DELETE"})
        assert result.allowed is False
        assert len(result.violations) == 1
        assert result.action_taken == EnforcementAction.BLOCK

    def test_warn_violation(self, enforcement_engine_permissive: EnforcementEngine):
        policy = SecurityPolicy(
            name="warn-admin",
            threat_level=ThreatLevel.MEDIUM,
            action=EnforcementAction.WARN,
            conditions=[PolicyCondition(field="path", operator="contains", value="/admin")],
        )
        enforcement_engine_permissive.register_policy(policy)
        result = enforcement_engine_permissive.evaluate_request({"path": "/admin/dashboard"})
        assert result.allowed is True
        assert result.action_taken == EnforcementAction.WARN
        assert len(result.violations) == 1

    def test_log_violation(self, enforcement_engine_permissive: EnforcementEngine):
        policy = SecurityPolicy(
            name="log-access",
            threat_level=ThreatLevel.LOW,
            action=EnforcementAction.LOG,
            conditions=[PolicyCondition(field="path", operator="exists")],
        )
        enforcement_engine_permissive.register_policy(policy)
        result = enforcement_engine_permissive.evaluate_request({"path": "/anything"})
        assert result.allowed is True
        assert result.action_taken == EnforcementAction.LOG

    def test_zero_tolerance_critical_raises(self, enforcement_engine: EnforcementEngine):
        policy = SecurityPolicy(
            name="critical-block",
            threat_level=ThreatLevel.CRITICAL,
            action=EnforcementAction.BLOCK,
            conditions=[
                PolicyCondition(field="action", operator="eq", value="rm_rf"),
            ],
        )
        enforcement_engine.register_policy(policy)

        with pytest.raises(PolicyViolationError, match="CRITICAL"):
            enforcement_engine.evaluate_request({"action": "rm_rf"})

    def test_zero_tolerance_non_critical_no_raise(self, enforcement_engine: EnforcementEngine):
        policy = SecurityPolicy(
            name="high-warn",
            threat_level=ThreatLevel.HIGH,
            action=EnforcementAction.WARN,
            conditions=[PolicyCondition(field="x", operator="eq", value="y")],
        )
        enforcement_engine.register_policy(policy)
        result = enforcement_engine.evaluate_request({"x": "y"})
        assert result.allowed is True  # WARN doesn't block

    def test_multiple_policies(self, enforcement_engine_permissive: EnforcementEngine):
        p1 = SecurityPolicy(
            name="warn-admin",
            threat_level=ThreatLevel.MEDIUM,
            action=EnforcementAction.WARN,
            conditions=[PolicyCondition(field="path", operator="contains", value="/admin")],
        )
        p2 = SecurityPolicy(
            name="block-delete",
            threat_level=ThreatLevel.HIGH,
            action=EnforcementAction.BLOCK,
            conditions=[PolicyCondition(field="method", operator="eq", value="DELETE")],
        )
        enforcement_engine_permissive.register_policy(p1)
        enforcement_engine_permissive.register_policy(p2)

        result = enforcement_engine_permissive.evaluate_request({
            "path": "/admin/users",
            "method": "DELETE",
        })
        assert result.allowed is False
        assert len(result.violations) == 2

    def test_custom_policies_override(self, enforcement_engine: EnforcementEngine):
        registered = SecurityPolicy(
            name="registered",
            conditions=[PolicyCondition(field="x", operator="eq", value="y")],
        )
        enforcement_engine.register_policy(registered)

        custom = SecurityPolicy(
            name="custom",
            threat_level=ThreatLevel.LOW,
            action=EnforcementAction.LOG,
            conditions=[PolicyCondition(field="a", operator="eq", value="b")],
        )
        result = enforcement_engine.evaluate_request({"a": "b"}, policies=[custom])
        assert len(result.violations) == 1
        assert result.violations[0].policy_name == "custom"

    def test_events_emitted(self, enforcement_engine_permissive: EnforcementEngine):
        policy = SecurityPolicy(
            name="test",
            action=EnforcementAction.BLOCK,
            conditions=[PolicyCondition(field="x", operator="eq", value="y")],
        )
        enforcement_engine_permissive.register_policy(policy)
        enforcement_engine_permissive.evaluate_request({"x": "y"})
        assert len(enforcement_engine_permissive.events) >= 1

    def test_unregister_policy(self, enforcement_engine: EnforcementEngine):
        policy = SecurityPolicy(
            name="temp",
            conditions=[PolicyCondition(field="x", operator="eq", value="y")],
        )
        enforcement_engine.register_policy(policy)
        enforcement_engine.unregister_policy("temp")
        assert "temp" not in enforcement_engine.policies


# ============================================================================
# AuditEngine
# ============================================================================


class TestAuditEngine:
    def test_record_entry(self, audit_engine: AuditEngine):
        entry = audit_engine.record(
            actor="admin",
            action="login",
            resource="system",
            verdict=AuditVerdict.ALLOWED,
        )
        assert entry.actor == "admin"
        assert entry.hash != ""
        assert entry.previous_hash == ""

    def test_hash_chain(self, audit_engine: AuditEngine):
        e1 = audit_engine.record("alice", "read", "file.txt", AuditVerdict.ALLOWED)
        e2 = audit_engine.record("bob", "write", "file.txt", AuditVerdict.DENIED)

        assert e2.previous_hash == e1.hash
        assert e2.hash != e1.hash

    def test_verify_integrity_valid(self, audit_engine: AuditEngine):
        audit_engine.record("a", "x", "r1", AuditVerdict.ALLOWED)
        audit_engine.record("b", "y", "r2", AuditVerdict.DENIED)
        audit_engine.record("c", "z", "r3", AuditVerdict.ALLOWED)

        assert audit_engine.verify_integrity() is True

    def test_verify_integrity_tampered(self, audit_engine: AuditEngine):
        audit_engine.record("a", "x", "r", AuditVerdict.ALLOWED)
        audit_engine.record("b", "y", "r", AuditVerdict.DENIED)

        # Tamper with the first entry's hash
        audit_engine._entries[0].hash = "tampered_hash_value"

        with pytest.raises(AuditIntegrityError, match="Hash chain broken"):
            audit_engine.verify_integrity()

    def test_verify_integrity_empty(self, audit_engine: AuditEngine):
        assert audit_engine.verify_integrity() is True

    def test_query_by_actor(self, audit_engine: AuditEngine):
        audit_engine.record("alice", "read", "r1", AuditVerdict.ALLOWED)
        audit_engine.record("bob", "write", "r2", AuditVerdict.DENIED)
        audit_engine.record("alice", "delete", "r3", AuditVerdict.DENIED)

        results = audit_engine.query(actor="alice")
        assert len(results) == 2
        assert all(e.actor == "alice" for e in results)

    def test_query_by_action(self, audit_engine: AuditEngine):
        audit_engine.record("a", "read", "r1", AuditVerdict.ALLOWED)
        audit_engine.record("b", "write", "r2", AuditVerdict.DENIED)

        results = audit_engine.query(action="read")
        assert len(results) == 1
        assert results[0].action == "read"

    def test_query_by_verdict(self, audit_engine: AuditEngine):
        audit_engine.record("a", "x", "r1", AuditVerdict.ALLOWED)
        audit_engine.record("b", "y", "r2", AuditVerdict.DENIED)

        results = audit_engine.query(verdict=AuditVerdict.DENIED)
        assert len(results) == 1
        assert results[0].verdict == AuditVerdict.DENIED

    def test_query_combined_filters(self, audit_engine: AuditEngine):
        audit_engine.record("alice", "read", "r1", AuditVerdict.ALLOWED)
        audit_engine.record("alice", "write", "r2", AuditVerdict.DENIED)
        audit_engine.record("bob", "read", "r3", AuditVerdict.ALLOWED)

        results = audit_engine.query(actor="alice", action="read")
        assert len(results) == 1
        assert results[0].actor == "alice"
        assert results[0].action == "read"

    def test_entry_count(self, audit_engine: AuditEngine):
        audit_engine.record("a", "x", "r", AuditVerdict.ALLOWED)
        audit_engine.record("b", "y", "r", AuditVerdict.DENIED)
        assert audit_engine.entry_count == 2

    def test_events_emitted(self, audit_engine: AuditEngine):
        audit_engine.record("a", "x", "r", AuditVerdict.ALLOWED)
        assert len(audit_engine.events) == 1
        assert audit_engine.events[0].actor == "a"


# ============================================================================
# IsolationEngine
# ============================================================================


class TestIsolationEngine:
    def test_apply_isolation(self, isolation_engine: IsolationEngine):
        rule = IsolationRule(
            sandbox_id="sb-test",
            no_network=True,
            readonly_fs=True,
            max_memory_mb=128,
        )
        config = isolation_engine.apply_isolation(rule)
        assert config.filesystem_readonly is True
        assert config.resource_limits.memory_mb == 128

    def test_verify_isolation_passes(self, isolation_engine: IsolationEngine):
        rule = IsolationRule(sandbox_id="sb-ok", no_network=True, readonly_fs=True)
        isolation_engine.apply_isolation(rule)
        status = isolation_engine.verify_isolation("sb-ok")
        assert status.isolated is True
        assert status.violations == []

    def test_verify_isolation_no_rules(self, isolation_engine: IsolationEngine):
        with pytest.raises(IsolationError, match="No isolation rules"):
            isolation_engine.verify_isolation("sb-unknown")

    def test_get_rule(self, isolation_engine: IsolationEngine):
        rule = IsolationRule(sandbox_id="sb-1")
        isolation_engine.apply_isolation(rule)
        retrieved = isolation_engine.get_rule("sb-1")
        assert retrieved is not None
        assert retrieved.sandbox_id == "sb-1"

    def test_get_rule_nonexistent(self, isolation_engine: IsolationEngine):
        assert isolation_engine.get_rule("missing") is None

    def test_remove_rule(self, isolation_engine: IsolationEngine):
        rule = IsolationRule(sandbox_id="sb-rm")
        isolation_engine.apply_isolation(rule)
        isolation_engine.remove_rule("sb-rm")
        assert "sb-rm" not in isolation_engine.rules

    def test_isolated_sandboxes(self, isolation_engine: IsolationEngine):
        isolation_engine.apply_isolation(IsolationRule(sandbox_id="sb-a"))
        isolation_engine.apply_isolation(IsolationRule(sandbox_id="sb-b"))
        assert sorted(isolation_engine.isolated_sandboxes) == ["sb-a", "sb-b"]

    def test_network_isolation_verified(self, isolation_engine: IsolationEngine):
        rule = IsolationRule(sandbox_id="sb-net", no_network=True)
        isolation_engine.apply_isolation(rule)
        status = isolation_engine.verify_isolation("sb-net")
        assert status.no_network is True

    def test_memory_limit_verified(self, isolation_engine: IsolationEngine):
        rule = IsolationRule(sandbox_id="sb-mem", max_memory_mb=256)
        isolation_engine.apply_isolation(rule)
        status = isolation_engine.verify_isolation("sb-mem")
        assert status.memory_limited is True
