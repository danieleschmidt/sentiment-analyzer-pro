# Contributing

Thank you for your interest in contributing to Sentiment Analyzer Pro!

## Quick Start

The easiest way to get started is using our Makefile:

```bash
# Complete development environment setup
make setup

# Run all quality checks (linting + tests)
make check

# View all available commands
make help
```

## Development Setup

### Option 1: Using Makefile (Recommended)

1. **Complete setup** (installs all dependencies and configures environment):
   ```bash
   make setup
   ```

2. **Verify installation**:
   ```bash
   make test
   ```

3. **Optional: Install advanced features**:
   ```bash
   make install-ml    # For transformer models
   make install-web   # For web server functionality
   ```

### Option 2: Manual Setup

1. **Install core dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Install development dependencies**:
   ```bash
   pip install pytest pytest-cov autopep8 flake8 bandit safety pre-commit
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Development Workflow

### Available Make Commands

| Category | Command | Description |
|----------|---------|-------------|
| **Setup** | `make setup` | Complete development environment setup |
| | `make install` | Install package in development mode |
| | `make install-dev` | Install development dependencies |
| **Testing** | `make test` | Run test suite with coverage |
| | `make check` | Run all quality checks (lint + test) |
| | `make coverage` | Generate detailed coverage report |
| **Code Quality** | `make lint` | Run linting checks |
| | `make format` | Auto-format code |
| | `make security` | Run security scans |
| **Development** | `make dev` | Start development web server |
| | `make serve` | Start production web server |
| | `make train` | Train the baseline model |
| **Build** | `make build` | Build package for distribution |
| | `make docker-build` | Build Docker image |
| | `make clean` | Remove build artifacts |

### Before Committing

Always run the quality checks before committing:

```bash
# Run all checks (recommended)
make check

# Or run individual checks
make lint
make test
make security
```

### Development Server

Start the development server for testing web API changes:

```bash
make dev
# Server will be available at http://127.0.0.1:5000
```

### Running Tests

```bash
# Quick test run
make test

# Verbose test output
make test-verbose

# Generate coverage report
make coverage
# Report available at htmlcov/index.html
```

### Code Formatting

We use autopep8 for code formatting:

```bash
# Auto-format all code
make format

# Check formatting (without changes)
make lint
```

## Project Structure

```
sentiment-analyzer-pro/
├── src/                    # Source code
│   ├── sentiment_analyzer_pro/
│   ├── cli.py             # Command-line interface
│   ├── models.py          # ML models
│   ├── preprocessing.py   # Text preprocessing
│   └── webapp.py          # Web server
├── tests/                 # Test suite
├── data/                  # Sample datasets
├── docs/                  # Documentation
└── Makefile              # Development workflow
```

## Pull Request Guidelines

1. **Before starting work**:
   - Check existing issues and PRs to avoid duplication
   - For major changes, open an issue first to discuss the approach

2. **Development process**:
   ```bash
   # Create feature branch
   git checkout -b feature/your-feature-name
   
   # Make your changes
   # ...
   
   # Run quality checks
   make check
   
   # Commit with descriptive message
   git commit -m "feat: add new preprocessing feature"
   ```

3. **PR requirements**:
   - All tests must pass (`make test`)
   - Code must pass linting (`make lint`)
   - Security scans must pass (`make security`)
   - Include tests for new functionality
   - Update documentation if needed
   - Keep changes focused and atomic

4. **Testing your changes**:
   ```bash
   # Test the full development workflow
   make clean
   make setup
   make check
   
   # Test specific functionality
   make train         # Test training pipeline
   make dev          # Test web server
   ```

## Code Style

- Follow PEP 8 (enforced by `make lint`)
- Use descriptive variable and function names
- Add docstrings for public functions and classes
- Keep functions focused and under 50 lines when possible
- Use type hints where beneficial

## Security Guidelines

- Never commit secrets, API keys, or credentials
- Validate all user inputs in web endpoints
- Use the security scanning tools: `make security`
- Follow the principle of least privilege

## Getting Help

- Check the [documentation](docs/) for detailed guides
- Look at existing code for patterns and examples
- Run `make help` to see all available commands
- Open an issue for questions or bug reports

## Advanced Development

### Working with ML Models

```bash
# Install ML dependencies
make install-ml

# Train and compare models
make benchmark

# Test different model configurations
python -m src.model_comparison
```

### Docker Development

```bash
# Build and test Docker image
make docker-build
make docker-run
```

### Performance Testing

```bash
# Run benchmarks
make benchmark

# Profile specific components
python -m cProfile -s cumulative -m src.train
```

Thank you for contributing to Sentiment Analyzer Pro!
