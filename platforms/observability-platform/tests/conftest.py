"""Test fixtures for observability platform tests."""

import pytest

from observability_platform.engines.metrics_engine import MetricsEngine
from observability_platform.engines.alert_engine import AlertEngine
from observability_platform.engines.health_engine import HealthEngine


@pytest.fixture
def metrics_engine() -> MetricsEngine:
    return MetricsEngine()


@pytest.fixture
def alert_engine(metrics_engine: MetricsEngine) -> AlertEngine:
    return AlertEngine(metrics_engine)


@pytest.fixture
def health_engine() -> HealthEngine:
    return HealthEngine()
