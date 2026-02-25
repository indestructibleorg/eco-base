<!-- @ECO-governed -->
<!-- @ECO-layer: GL90-99 -->
<!-- @ECO-semantic: governed-documentation -->
<!-- @ECO-audit-trail: engine/governance/GL_SEMANTIC_ANCHOR.json -->

# Test Suite Documentation

## Overview

This test suite provides comprehensive testing infrastructure for the MachineNativeOps project. It includes unit tests, integration tests, and end-to-end tests.

## Directory Structure

```
tests/
├── __init__.py
├── conftest.py           # Pytest configuration and fixtures
├── pytest.ini            # Pytest settings
├── requirements-test.txt # Test dependencies
├── helpers/              # Test helper utilities
│   ├── __init__.py
│   └── test_base.py
├── fixtures/             # Test data and mock data
├── unit/                 # Unit tests
│   ├── test_helpers.py
│   └── test_artifact_validation.py
├── integration/          # Integration tests
└── e2e/                  # End-to-end tests
```

## Test Types

### Unit Tests (`tests/unit/`)
- Test individual functions and classes in isolation
- Fast execution (< 1 second per test)
- No external dependencies
- Use mocks and fixtures

### Integration Tests (`tests/integration/`)
- Test interactions between components
- May require external services (database, API)
- Medium execution time (1-10 seconds per test)
- Test real integrations with test databases/services

### End-to-End Tests (`tests/e2e/`)
- Test complete workflows from start to finish
- Slowest execution (> 10 seconds per test)
- Simulate real user scenarios
- Test full system behavior

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test Type
```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# E2E tests only
pytest -m e2e
```

### Run Specific Test File
```bash
pytest tests/unit/test_helpers.py
```

### Run Specific Test Function
```bash
pytest tests/unit/test_helpers.py::TestTestHelper::test_initialization
```

### Run with Coverage
```bash
pytest --cov=workspace/src --cov-report=html
```

### Run with Verbose Output
```bash
pytest -v
```

### Run in Parallel (faster execution)
```bash
pytest -n auto
```

### Skip Slow Tests
```bash
pytest -m "not slow"
```

## Fixtures

Available fixtures (defined in `conftest.py`):

- `project_root`: Project root directory
- `test_data_dir`: Test data directory
- `source_dir`: Source code directory
- `sample_config`: Sample configuration dictionary
- `sample_artifact`: Sample artifact dictionary
- `sample_namespace`: Sample namespace dictionary
- `test_helper`: TestHelper instance
- `test_logger`: TestLogger instance

## Test Helpers

### TestHelper
Base test helper with common utilities:

```python
from tests.helpers import TestHelper

helper = TestHelper()

# Create sample data
artifact = helper.create_sample_artifact()
namespace = helper.create_sample_namespace()

# Validate data
helper.assert_valid_artifact(artifact)
helper.assert_valid_namespace(namespace)

# Compare dictionaries
helper.compare_dicts(dict1, dict2, ignore_fields=["timestamp"])

# Wait for condition
helper.wait_for_condition(condition_func, timeout=10)
```

### TestLogger
Test logging utility:

```python
from tests.helpers import TestLogger

logger = TestLogger("test-name")
logger.info("Info message")
logger.error("Error message")

# Assert log entries
logger.assert_logged("INFO", "Info message")
```

## Test Markers

Use markers to categorize tests:

```python
@pytest.mark.unit
def test_something():
    pass

@pytest.mark.integration
def test_integration():
    pass

@pytest.mark.e2e
def test_e2e():
    pass

@pytest.mark.slow
def test_slow_operation():
    pass
```

## Coverage Goals

- **Unit Tests**: > 90% coverage
- **Integration Tests**: > 80% coverage
- **E2E Tests**: > 70% coverage
- **Overall**: > 80% coverage

## Continuous Integration

Tests are automatically run on:
- Every push to main branch
- Every pull request
- Scheduled daily runs

## Writing New Tests

### Unit Test Template

```python
import pytest
from tests.helpers import TestHelper

class TestMyFeature:
    """Test cases for MyFeature"""
    
    @pytest.fixture
    def helper(self):
        """Test helper fixture"""
        return TestHelper()
    
    def test_basic_functionality(self, helper):
        """Test basic functionality"""
        # Arrange
        artifact = helper.create_sample_artifact()
        
        # Act
        result = some_function(artifact)
        
        # Assert
        assert result is not None
        helper.assert_valid_artifact(result)
```

### Integration Test Template

```python
import pytest

@pytest.mark.integration
class TestMyIntegration:
    """Integration test cases"""
    
    def test_database_integration(self):
        """Test database integration"""
        # Test real database connection
        pass
    
    def test_api_integration(self):
        """Test API integration"""
        # Test real API calls
        pass
```

## Best Practices

1. **Test Isolation**: Each test should be independent
2. **Clear Naming**: Use descriptive test names
3. **AAA Pattern**: Arrange, Act, Assert
4. **Use Fixtures**: Reuse fixtures for common setup
5. **Mock External Dependencies**: Don't rely on external services in unit tests
6. **Fast Execution**: Keep unit tests fast
7. **Comprehensive Coverage**: Test both happy path and edge cases
8. **Clear Assertions**: Use specific assertion messages

## Troubleshooting

### Tests Failing on CI
- Check for environment-specific issues
- Ensure all dependencies are installed
- Verify test data files are committed

### Slow Tests
- Use `pytest --durations=10` to identify slow tests
- Consider marking slow tests with `@pytest.mark.slow`
- Optimize test setup/teardown

### Import Errors
- Ensure project root is in Python path
- Check `PYTHONPATH` environment variable
- Verify `conftest.py` is properly configured

## Resources

- [Pytest Documentation]([EXTERNAL_URL_REMOVED])
- [Coverage.py Documentation]([EXTERNAL_URL_REMOVED])
- [Testing Best Practices]([EXTERNAL_URL_REMOVED])