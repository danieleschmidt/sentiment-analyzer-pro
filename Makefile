# Sentiment Analyzer Pro - Development Makefile
# Provides standardized commands for development workflow

# Configuration
PYTHON := python3
PIP := pip3
PYTEST := $(PYTHON) -m pytest
PACKAGE_NAME := sentiment_analyzer_pro
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

.PHONY: help setup install install-dev install-ml install-web clean test test-verbose lint format check security coverage dev serve build docker-build docker-run version venv setup-venv

# Default target
help: ## Show this help message
	@echo "Sentiment Analyzer Pro - Development Commands"
	@echo ""
	@echo "Setup Commands:"
	@echo "  setup         - Complete development environment setup"
	@echo "  setup-venv    - Setup with virtual environment (recommended)"
	@echo "  venv          - Create virtual environment only"
	@echo "  install       - Install package in development mode"
	@echo "  install-dev   - Install development dependencies"
	@echo "  install-ml    - Install ML dependencies (tensorflow, torch, etc.)"
	@echo "  install-web   - Install web server dependencies"
	@echo ""
	@echo "Development Commands:"
	@echo "  test          - Run test suite with coverage"
	@echo "  test-verbose  - Run tests with verbose output"
	@echo "  lint          - Run linting checks"
	@echo "  format        - Format code with autopep8"
	@echo "  check         - Run all quality checks (lint + test)"
	@echo "  security      - Run security scans"
	@echo "  coverage      - Generate test coverage report"
	@echo ""
	@echo "Application Commands:"
	@echo "  dev           - Start development web server"
	@echo "  serve         - Start production web server"
	@echo "  version       - Show package version"
	@echo ""
	@echo "Build Commands:"
	@echo "  build         - Build package for distribution"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-run    - Run application in Docker"
	@echo "  clean         - Remove build artifacts and cache files"
	@echo ""

# Setup and Installation
venv: ## Create virtual environment
	@echo "$(GREEN)Creating virtual environment...$(NC)"
	$(PYTHON) -m venv .venv
	@echo "$(GREEN)Virtual environment created in .venv/$(NC)"
	@echo "$(YELLOW)Activate with: source .venv/bin/activate$(NC)"

setup-venv: venv ## Setup with virtual environment (recommended)
	@echo "$(GREEN)Setting up development environment with virtual environment...$(NC)"
	@bash -c "source .venv/bin/activate && pip install -e . && pip install pytest pytest-cov autopep8 flake8 bandit safety pre-commit"
	@echo "$(GREEN)Development environment ready!$(NC)"
	@echo "$(YELLOW)Activate environment: source .venv/bin/activate$(NC)"
	@echo "$(YELLOW)Then run: make test$(NC)"

setup: ## Complete development environment setup
	@echo "$(GREEN)Setting up development environment...$(NC)"
	@$(MAKE) install-dev
	@$(MAKE) install
	@echo "$(GREEN)Development environment ready!$(NC)"
	@echo "$(YELLOW)Run 'make test' to verify installation$(NC)"

install: ## Install package in development mode
	@echo "$(GREEN)Installing package in development mode...$(NC)"
	$(PIP) install -e . --break-system-packages 2>/dev/null || \
	$(PIP) install -e . --user 2>/dev/null || \
	echo "$(YELLOW)Note: Package installation may require virtual environment$(NC)"

install-dev: ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	$(PIP) install pytest pytest-cov autopep8 flake8 bandit safety pre-commit --break-system-packages 2>/dev/null || \
	$(PIP) install pytest pytest-cov autopep8 flake8 bandit safety pre-commit --user 2>/dev/null || \
	echo "$(YELLOW)Note: Some dependencies may not be available. Consider using a virtual environment.$(NC)"
	@echo "$(YELLOW)Installing pre-commit hooks...$(NC)"
	pre-commit install 2>/dev/null || echo "$(YELLOW)Pre-commit not configured, skipping hooks$(NC)"

install-ml: ## Install ML dependencies (optional)
	@echo "$(GREEN)Installing ML dependencies...$(NC)"
	$(PIP) install -e .[ml] --break-system-packages 2>/dev/null || \
	$(PIP) install -e .[ml] --user 2>/dev/null || \
	echo "$(YELLOW)Note: ML dependencies may require virtual environment$(NC)"

install-web: ## Install web server dependencies (optional)
	@echo "$(GREEN)Installing web dependencies...$(NC)"
	$(PIP) install -e .[web] --break-system-packages 2>/dev/null || \
	$(PIP) install -e .[web] --user 2>/dev/null || \
	echo "$(YELLOW)Note: Web dependencies may require virtual environment$(NC)"

# Testing
test: ## Run test suite with coverage
	@echo "$(GREEN)Running test suite...$(NC)"
	@$(PYTEST) -v --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html || echo "$(RED)Tests failed or pytest not available$(NC)"

test-verbose: ## Run tests with verbose output
	@echo "$(GREEN)Running tests with verbose output...$(NC)"
	@$(PYTEST) -v --cov=$(SRC_DIR) --cov-report=term-missing || echo "$(RED)Tests failed or pytest not available$(NC)"

test-parallel: ## Run tests in parallel with pytest-xdist
	@echo "$(GREEN)Running tests in parallel...$(NC)"
	@$(PYTEST) -v --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html -n auto || echo "$(RED)Tests failed or pytest-xdist not available$(NC)"

test-fast: ## Run tests without coverage for quick feedback
	@echo "$(GREEN)Running fast tests...$(NC)"
	@$(PYTEST) -x -v -n auto || echo "$(RED)Tests failed or pytest not available$(NC)"

# Code Quality
lint: ## Run linting checks with ruff
	@echo "$(GREEN)Running linting checks...$(NC)"
	@ruff check $(SRC_DIR) $(TEST_DIR) 2>/dev/null || echo "$(YELLOW)ruff not available, skipping lint$(NC)"

format: ## Format code with autopep8
	@echo "$(GREEN)Formatting code...$(NC)"
	@autopep8 --in-place --recursive --max-line-length=88 $(SRC_DIR) $(TEST_DIR) 2>/dev/null || echo "$(YELLOW)autopep8 not available, skipping format$(NC)"

check: lint test ## Run all quality checks (lint + test)

security: ## Run security scans
	@echo "$(GREEN)Running security scans...$(NC)"
	@bandit -r $(SRC_DIR) -f json -o security-report.json 2>/dev/null || echo "$(YELLOW)bandit not available, skipping security scan$(NC)"
	@safety check 2>/dev/null || echo "$(YELLOW)safety not available, skipping dependency check$(NC)"

coverage: ## Generate test coverage report
	@echo "$(GREEN)Generating coverage report...$(NC)"
	@$(PYTEST) $(TEST_DIR)/ --cov=$(SRC_DIR) --cov-report=html --cov-report=term || echo "$(RED)Coverage generation failed$(NC)"
	@echo "$(GREEN)Coverage report generated in htmlcov/index.html$(NC)"

# Application Commands  
dev: ## Start development web server
	@echo "$(GREEN)Starting development server...$(NC)"
	@$(PYTHON) -m $(SRC_DIR).webapp || echo "$(RED)Failed to start dev server. Install web dependencies with 'make install-web'$(NC)"

serve: ## Start production web server
	@echo "$(GREEN)Starting production server...$(NC)"
	@sentiment-cli serve --host 127.0.0.1 --port 5000 || echo "$(RED)Failed to start server. Ensure package is installed with 'make install'$(NC)"

version: ## Show package version
	@$(PYTHON) -c "import sys; sys.path.insert(0, '$(SRC_DIR)'); from sentiment_analyzer_pro import __version__; print(__version__)" 2>/dev/null || echo "0.1.0"

# Build Commands
build: clean ## Build package for distribution
	@echo "$(GREEN)Building package...$(NC)"
	$(PYTHON) -m build

docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker build -t sentiment-analyzer-pro:latest .

docker-run: ## Run application in Docker
	@echo "$(GREEN)Running Docker container...$(NC)"
	docker run -p 5000:5000 sentiment-analyzer-pro:latest

# Cleanup
clean: ## Remove build artifacts and cache files
	@echo "$(GREEN)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf security-report.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "$(GREEN)Cleanup complete!$(NC)"

# Training and CLI shortcuts
train: ## Train the baseline model
	@echo "$(GREEN)Training baseline model...$(NC)"
	@$(PYTHON) -m $(SRC_DIR).train || echo "$(RED)Training failed. Ensure dependencies are installed$(NC)"

predict: ## Make predictions (requires model and data file)
	@echo "$(GREEN)Making predictions...$(NC)"
	@echo "$(YELLOW)Usage: make predict FILE=your_data.csv MODEL=your_model.joblib$(NC)"
	@test -n "$(FILE)" || (echo "$(RED)Error: FILE parameter required$(NC)" && exit 1)
	@sentiment-cli predict $(FILE) --model $(MODEL) || echo "$(RED)Prediction failed$(NC)"

# Advanced targets
benchmark: ## Run performance benchmarks
	@echo "$(GREEN)Running performance benchmarks...$(NC)"
	@$(PYTHON) -m $(SRC_DIR).model_comparison || echo "$(RED)Benchmarks failed$(NC)"

docs: ## Generate documentation
	@echo "$(GREEN)Documentation available in $(DOCS_DIR)/ directory$(NC)"
	@ls -la $(DOCS_DIR)/

# Debugging helpers
debug-env: ## Show environment information
	@echo "$(GREEN)Environment Information:$(NC)"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Pip: $$($(PIP) --version)"
	@echo "Working Directory: $$(pwd)"
	@echo "Package Directory: $(SRC_DIR)"
	@echo "Test Directory: $(TEST_DIR)"
	@$(PYTHON) -c "import sys; print('Python Path:', sys.path[:3])"