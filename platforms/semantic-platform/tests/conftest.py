"""Test fixtures for semantic platform tests."""

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

from semantic_platform.engines.folding_engine import FoldingEngine
from semantic_platform.engines.indexing_engine import IndexingEngine
from semantic_platform.engines.inference_engine import InferenceEngine


@pytest.fixture
def folding_engine() -> FoldingEngine:
    return FoldingEngine(dimensions=64)


@pytest.fixture
def indexing_engine() -> IndexingEngine:
    return IndexingEngine()


@pytest.fixture
def inference_engine(folding_engine: FoldingEngine, indexing_engine: IndexingEngine) -> InferenceEngine:
    return InferenceEngine(folding_engine, indexing_engine)
