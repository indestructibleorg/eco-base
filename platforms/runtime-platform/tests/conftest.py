"""Test fixtures for the runtime platform."""

import pytest

from runtime_platform.domain.entities import ETLPipeline, Job, Step, Workflow
from runtime_platform.domain.value_objects import StepType


@pytest.fixture
def simple_step():
    """A single shell step that echoes hello."""
    return Step(name="echo-step", type=StepType.SHELL, command="echo hello", timeout=10.0)


@pytest.fixture
def python_step():
    """A Python step that evaluates an expression."""
    return Step(name="calc-step", type=StepType.PYTHON, command="2 + 2", timeout=5.0)


@pytest.fixture
def simple_job(simple_step):
    """A job with one shell step and no dependencies."""
    return Job(id="job-1", name="build", steps=[simple_step])


@pytest.fixture
def simple_workflow(simple_job):
    """A workflow with a single job."""
    return Workflow(id="wf-1", name="test-workflow", jobs=[simple_job])


@pytest.fixture
def diamond_workflow():
    """A diamond-shaped DAG: A -> B,C -> D."""
    return Workflow(
        id="wf-diamond",
        name="diamond",
        jobs=[
            Job(
                id="job-a", name="A",
                steps=[Step(name="a-step", type=StepType.SHELL, command="echo A")],
            ),
            Job(
                id="job-b", name="B",
                steps=[Step(name="b-step", type=StepType.SHELL, command="echo B")],
                depends_on=["A"],
            ),
            Job(
                id="job-c", name="C",
                steps=[Step(name="c-step", type=StepType.SHELL, command="echo C")],
                depends_on=["A"],
            ),
            Job(
                id="job-d", name="D",
                steps=[Step(name="d-step", type=StepType.SHELL, command="echo D")],
                depends_on=["B", "C"],
            ),
        ],
    )


@pytest.fixture
def etl_pipeline():
    """A simple ETL pipeline with in-memory data."""
    async def extract(source):
        return [{"id": i, "value": i * 10} for i in range(5)]

    async def transform(data):
        return [dict(row, doubled=row["value"] * 2) for row in data]

    async def load(target, data):
        return len(data)

    return ETLPipeline(
        id="etl-test",
        source="test-source",
        target="test-target",
        transforms=["double"],
        extract_fn=extract,
        transform_fns=[transform],
        load_fn=load,
    )
