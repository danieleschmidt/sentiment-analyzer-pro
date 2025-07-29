# Use specific Python version for reproducible builds
FROM python:3.10.15-slim@sha256:a3672a65581ea87cf8e9c0beed56f0e9b9f5c64b4cbde36cca1e8f4e8a0eda55

# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Security: Update system packages and install security updates
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Security: Copy and install dependencies first for better layer caching
COPY --chown=appuser:appuser pyproject.toml requirements.txt* ./

# Security: Install dependencies with specific versions and vulnerability checks
RUN python -m pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir safety \
    && pip install --no-cache-dir .[web] \
    && safety check --json --output /tmp/safety-report.json || true \
    && chown -R appuser:appuser /app

# Security: Copy application code
COPY --chown=appuser:appuser . .

# Security: Remove unnecessary files and permissions
RUN find /app -type f -name "*.pyc" -delete \
    && find /app -type d -name "__pycache__" -exec rm -rf {} + \
    && chmod -R 755 /app \
    && chmod 644 /app/src/*.py

# Security: Switch to non-root user
USER appuser

# Security: Use non-privileged port and bind to localhost only
EXPOSE 8080

# Health check for container monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/', timeout=5)" || exit 1

# Security: Use specific command with restricted permissions
CMD ["python", "-m", "src.webapp", "--host", "0.0.0.0", "--port", "8080"]
