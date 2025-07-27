FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        build-essential \
        curl \
        git \
        vim \
        && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for development
RUN pip install --upgrade pip setuptools wheel

# Install pre-commit and common development tools
RUN pip install \
    pre-commit \
    black \
    isort \
    ruff \
    pytest \
    pytest-cov \
    mypy \
    bandit \
    safety

# Create workspace directory
WORKDIR /workspace

# Copy requirements for better caching
COPY requirements.txt pyproject.toml ./
RUN pip install -r requirements.txt

# Install the package in development mode
COPY . .
RUN pip install -e .[dev,ml,web]

# Set up git safe directory
RUN git config --global --add safe.directory /workspace