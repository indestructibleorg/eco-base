"""Test fixtures for registry platform tests."""

import sys
from pathlib import Path

# Ensure src is on path for imports
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

_shared_src = Path(__file__).resolve().parent.parent.parent / "_shared" / "src"
if str(_shared_src) not in sys.path:
    sys.path.insert(0, str(_shared_src))

import pytest

from registry_platform.engines.catalog_engine import CatalogEngine
from registry_platform.engines.namespace_engine import NamespaceEngine
from registry_platform.engines.dependency_engine import DependencyEngine


@pytest.fixture
def catalog_engine() -> CatalogEngine:
    return CatalogEngine()


@pytest.fixture
def namespace_engine() -> NamespaceEngine:
    return NamespaceEngine()


@pytest.fixture
def dependency_engine() -> DependencyEngine:
    return DependencyEngine()
