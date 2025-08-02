# Multi-stage Docker build for optimized production image

# Stage 1: Build stage
FROM python:3.10.15-slim@sha256:a3672a65581ea87cf8e9c0beed56f0e9b9f5c64b4cbde36cca1e8f4e8a0eda55 AS builder

# Install build dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy dependency files
COPY pyproject.toml requirements.txt* ./

# Install dependencies to local directory
RUN python -m pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --user .[web]

# Stage 2: Production stage
FROM python:3.10.15-slim@sha256:a3672a65581ea87cf8e9c0beed56f0e9b9f5c64b4cbde36cca1e8f4e8a0eda55 AS production

# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Security: Update system packages and install security updates
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Make sure scripts in .local are usable
ENV PATH=/home/appuser/.local/bin:$PATH

# Security: Copy application code
COPY --chown=appuser:appuser . .

# Security: Remove unnecessary files and set permissions
RUN find /app -type f -name "*.pyc" -delete \
    && find /app -type d -name "__pycache__" -exec rm -rf {} + \
    && chmod -R 755 /app \
    && chmod 644 /app/src/*.py \
    && chown -R appuser:appuser /app

# Security: Switch to non-root user
USER appuser

# Security: Use non-privileged port
EXPOSE 8080

# Health check for container monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/', timeout=5)" || exit 1

# Security: Use specific command with restricted permissions
CMD ["python", "-m", "src.webapp", "--host", "0.0.0.0", "--port", "8080"]
