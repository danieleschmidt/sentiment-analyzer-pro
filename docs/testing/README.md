# Testing Guide

This document provides comprehensive guidance on testing the Sentiment Analyzer Pro application.

## Test Organization

Our test suite is organized into several categories:

### Test Structure
```
tests/
├── conftest.py           # Shared fixtures and configuration
├── pytest.ini           # Pytest configuration
├── fixtures/             # Test data files
│   └── test_data.csv
├── unit/                 # Unit tests for individual components
├── integration/          # Integration tests for component interactions  
├── e2e/                  # End-to-end tests for complete workflows
└── performance/          # Performance and benchmark tests
```

## Running Tests

### Quick Start
```bash
# Run all tests
make test

# Run with verbose output
make test-verbose

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m integration            # Integration tests only
pytest -m "not slow"             # Exclude slow tests
```

### Test Categories

#### Unit Tests (`-m unit`)
- Test individual functions and classes in isolation
- Fast execution (< 1 second per test)
- Mock external dependencies
- High coverage of edge cases

#### Integration Tests (`-m integration`)
- Test component interactions and workflows
- Verify data flow between modules
- Test serialization/deserialization
- Moderate execution time (1-10 seconds per test)

#### End-to-End Tests (`-m e2e`)
- Test complete user workflows
- API endpoint testing
- CLI command testing
- Docker container testing
- Slower execution (10+ seconds per test)

#### Performance Tests (`-m performance`)
- Benchmark key operations
- Memory usage validation
- Scalability testing
- Marked as slow by default

## Test Configuration

### Pytest Configuration
Key settings in `pytest.ini`:
- Minimum coverage threshold: 80%
- Automatic coverage reporting
- Strict marker enforcement
- Warning filters for clean output

### Environment Variables
Tests use these environment variables:
- `TESTING=true` - Automatically set during test runs
- `MODEL_PATH` - Path to test model files
- `LOG_LEVEL=DEBUG` - Enhanced logging for tests

### Fixtures

#### Data Fixtures
- `sample_data`: DataFrame with test sentiment data
- `sample_texts`: List of test text samples
- `temp_csv_file`: Temporary CSV file with test data
- `temp_model_file`: Temporary file for model serialization

#### Configuration Fixtures
- `test_config`: Test-specific configuration
- `mock_environment_vars`: Mock environment variables
- `test_data_dir`: Path to test data directory

## Writing Tests

### Unit Test Example
```python
import pytest
from src.preprocessing import preprocess_text

@pytest.mark.unit
def test_preprocess_text():
    """Test text preprocessing function."""
    text = "Hello World!"
    result = preprocess_text(text)
    
    assert isinstance(result, str)
    assert result.lower() == result  # Should be lowercase
```

### Integration Test Example
```python
@pytest.mark.integration
def test_training_pipeline(sample_data, temp_model_file):
    """Test the complete training pipeline."""
    # Train model
    model = train_model(sample_data)
    
    # Save and load
    joblib.dump(model, temp_model_file)
    loaded_model = joblib.load(temp_model_file)
    
    # Verify functionality
    predictions = predict_sentiment(loaded_model, ["test text"])
    assert len(predictions) == 1
```

### Performance Test Example
```python
@pytest.mark.performance
@pytest.mark.slow
def test_prediction_speed(trained_model):
    """Benchmark prediction performance."""
    texts = ["sample text"] * 1000
    
    start_time = time.time()
    predictions = predict_sentiment(trained_model, texts)
    duration = time.time() - start_time
    
    assert len(predictions) == 1000
    assert duration < 1.0  # Should complete in under 1 second
```

## Coverage Guidelines

### Target Coverage
- Overall project coverage: **≥ 80%**
- Critical modules (models, preprocessing): **≥ 90%**
- Utility modules: **≥ 70%**

### Coverage Reports
```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# View coverage in browser
open htmlcov/index.html

# Check coverage from command line
pytest --cov=src --cov-report=term-missing
```

### Coverage Exclusions
Add `# pragma: no cover` for:
- Debug code paths
- Platform-specific code
- Error handling for edge cases
- Abstract methods

## Continuous Integration

### Pre-commit Testing
All tests run automatically via pre-commit hooks:
```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Test Commands for CI
```bash
# Fast test suite (no slow tests)
pytest -m "not slow" --cov=src

# Full test suite with performance tests
pytest --cov=src

# Test specific environments
pytest tests/e2e/ -m "not docker"  # Skip Docker tests
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure package is installed in development mode
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

#### Missing Dependencies
```bash
# Install test dependencies
pip install -e .[dev]

# Check for optional dependencies
pytest tests/unit/  # Should work with minimal deps
pytest tests/e2e/   # May require full dependencies
```

#### Slow Tests
```bash
# Skip slow tests during development
pytest -m "not slow"

# Run only fast unit tests
pytest tests/unit/ -m "not slow"

# Profile slow tests
pytest --durations=10
```

### Test Data Issues
- Test data is stored in `tests/fixtures/`
- Temporary files are automatically cleaned up
- Use fixtures for consistent test data

### Mock and Patch Guidelines
- Mock external services and APIs
- Patch file I/O operations in unit tests
- Use `unittest.mock.patch` for temporary replacements
- Prefer dependency injection over patching when possible

## Best Practices

### Test Naming
- Use descriptive names: `test_preprocessing_handles_empty_string`
- Group related tests in classes: `TestModelTraining`
- Use `@pytest.mark` to categorize tests

### Test Organization
- One test file per source module
- Group tests by functionality
- Keep tests independent and isolated

### Assertions
- Use specific assertions: `assert len(result) == expected_length`
- Include helpful error messages: `assert result > 0, f"Expected positive result, got {result}"`
- Test both success and failure cases

### Performance Testing
- Set reasonable performance thresholds
- Test with realistic data sizes
- Monitor memory usage for large datasets
- Use `@pytest.mark.slow` for time-consuming tests

## Test Data Management

### Test Data Files
- Store in `tests/fixtures/`
- Use realistic but minimal datasets
- Include edge cases in test data
- Document data format and source

### Temporary Files
- Use pytest fixtures for cleanup
- Avoid hardcoded file paths
- Test both success and failure of file operations

### Mock Data
- Generate programmatically when possible
- Use factories for complex objects
- Ensure deterministic test results with random seeds