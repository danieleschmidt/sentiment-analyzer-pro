# Development Guide

## Quick Setup

Complete development environment setup:
```bash
make setup && make test
```

## Requirements

- Python 3.8+
- pip or conda
- Git

## Development Workflow

1. **Clone and setup**:
   ```bash
   git clone <repository>
   cd sentiment-analyzer-pro
   make setup
   ```

2. **Run quality checks**:
   ```bash
   make check  # Runs lint + test + security
   ```

3. **Start development server**:
   ```bash
   make dev  # Available at http://127.0.0.1:5000
   ```

## Available Commands

Run `make help` for complete command reference.

Key commands:
- `make test` - Run test suite with coverage
- `make lint` - Code quality checks  
- `make format` - Auto-format code
- `make security` - Security scans

## Architecture

See [Getting Started](GETTING_STARTED.md) for project overview and architecture details.

For detailed contributing guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md).