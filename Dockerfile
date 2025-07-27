# Multi-stage build for optimized production image
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml ./
COPY src/sentiment_analyzer_pro/__init__.py src/sentiment_analyzer_pro/__init__.py

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python dependencies from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Install package
RUN pip install --no-cache-dir .[web]

# Create directories for models and logs
RUN mkdir -p /app/models /app/logs \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Add local bin to PATH for CLI commands
ENV PATH="/home/appuser/.local/bin:$PATH"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_ENV=production
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/sentiment_model.joblib

# Default command
CMD ["sentiment-web", "--host", "0.0.0.0", "--port", "5000"]
