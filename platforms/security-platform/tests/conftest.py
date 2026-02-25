"""Test fixtures for security platform tests."""

import pytest

from security_platform.engines.enforcement_engine import EnforcementEngine
from security_platform.engines.audit_engine import AuditEngine
from security_platform.engines.isolation_engine import IsolationEngine


@pytest.fixture
def enforcement_engine() -> EnforcementEngine:
    return EnforcementEngine(zero_tolerance=True)


@pytest.fixture
def enforcement_engine_permissive() -> EnforcementEngine:
    return EnforcementEngine(zero_tolerance=False)


@pytest.fixture
def audit_engine() -> AuditEngine:
    return AuditEngine()


@pytest.fixture
def isolation_engine() -> IsolationEngine:
    return IsolationEngine()
