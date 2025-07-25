# Sentiment Analyzer Pro - API Reference

## Overview
Sentiment Analyzer Pro provides both REST API endpoints and command-line interfaces for sentiment analysis using machine learning models.

## REST API Endpoints

### Base URL
When running locally: `http://localhost:5000`

### Authentication
No authentication required for local deployment.

---

## Prediction Endpoints

### POST /predict
Analyze sentiment of text input.

**Request:**
```http
POST /predict
Content-Type: application/json

{
  "text": "I love this product!"
}
```

**Request Schema:**
- `text` (string, required): Text to analyze (max 10,000 characters)
  - Automatically sanitized to remove potential XSS attacks
  - HTML tags and JavaScript code are stripped
  - Whitespace is normalized

**Response:**
```json
{
  "prediction": "positive",
  "confidence": 0.85
}
```

**Response Schema:**
- `prediction` (string): Sentiment classification (`positive`, `negative`, `neutral`)
- `confidence` (float): Model confidence score (0.0-1.0)

**Error Responses:**
```json
// 400 Bad Request - Invalid input
{
  "error": "Invalid request format",
  "details": ["Text field is required"]
}

// 413 Request Entity Too Large - Text too long
{
  "error": "Text too long",
  "max_length": 10000
}

// 429 Too Many Requests - Rate limit exceeded
{
  "error": "Rate limit exceeded"
}

// 500 Internal Server Error
{
  "error": "Internal server error during prediction"
}
```

---

## System Endpoints

### GET /
Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

### GET /version
Get application version.

**Response:**
```json
{
  "version": "0.1.0"
}
```

### GET /metrics
Prometheus-compatible metrics endpoint (returns plain text).

**Response:**
```
# Metrics in Prometheus format
sentiment_requests_total{method="POST",endpoint="/predict",status="success"} 42
sentiment_request_duration_seconds_bucket{method="POST",endpoint="/predict",le="0.1"} 35
# ... additional metrics
```

### GET /metrics/summary
JSON summary of service metrics.

**Response:**
```json
{
  "requests": 150,
  "predictions": 75,
  "prometheus_enabled": false,
  "fallback_metrics_count": 12,
  "latest_metrics": ["requests_GET_/metrics/summary_success", "predictions_sklearn_positive"]
}
```

---

## Rate Limiting

The API implements rate limiting to prevent abuse:
- **Default:** 100 requests per 60 seconds per IP address
- **Configurable:** Set via `RATE_LIMIT_MAX_REQUESTS` and `RATE_LIMIT_WINDOW` environment variables
- **Response:** HTTP 429 when limit exceeded

---

## Security Features

### Input Validation
- XSS protection: HTML tags and JavaScript are stripped from input
- Length validation: Text input limited to configurable maximum
- Content-Type validation: Only `application/json` accepted for POST requests

### Security Headers
All responses include security headers:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY` 
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`
- `Content-Security-Policy: default-src 'self'`

### File Upload Security
- Path traversal protection
- Configurable file size limits
- Allowed directory restrictions (configurable via `ALLOWED_TEMP_DIRS`)

---

## Configuration

The API can be configured via environment variables:

### Core Settings
- `MODEL_PATH`: Path to trained model file (default: `model.joblib`)
- `LOG_LEVEL`: Logging level (default: `INFO`)

### Security Settings
- `RATE_LIMIT_WINDOW`: Rate limit window in seconds (default: `60`)
- `RATE_LIMIT_MAX_REQUESTS`: Max requests per window (default: `100`)
- `MAX_FILE_SIZE_MB`: Maximum file size in MB (default: `100`)
- `MAX_DATASET_ROWS`: Maximum dataset rows (default: `1,000,000`)
- `ALLOWED_TEMP_DIRS`: Comma-separated list of allowed temp directories (default: `/tmp,/var/tmp`)

---

## Command Line Interface

### Training Models
```bash
# Train with default settings
python -m src.train

# Train with custom data and model output
python -m src.train --csv data/training.csv --model models/sentiment.joblib

# Train with specific columns
python -m src.train --text-column "review" --label-column "rating"
```

**Options:**
- `--csv`: Path to training CSV file (default: `data/sample_reviews.csv`)
- `--model`: Output path for trained model (default: `model.joblib`)
- `--text-column`: Name of text column (default: `text`)
- `--label-column`: Name of label column (default: `label`)

### Making Predictions
```bash
# Predict with default model
python -m src.predict --csv data/test.csv

# Predict with custom model
python -m src.predict --csv data/test.csv --model models/custom.joblib

# Output to specific file
python -m src.predict --csv data/test.csv --output predictions.csv
```

**Options:**
- `--csv`: Path to input CSV file (required)
- `--model`: Path to trained model file (default: from `MODEL_PATH` env var)
- `--output`: Output CSV file path (default: `predictions.csv`)
- `--text-column`: Name of text column (default: `text`)

### Data Preprocessing
```bash
# Basic preprocessing
python -m src.cli preprocess --csv data/raw.csv --output data/clean.csv

# With lemmatization
python -m src.cli preprocess --csv data/raw.csv --output data/clean.csv --lemmatize

# Specify text column
python -m src.cli preprocess --csv data/raw.csv --text-column "review_text"
```

### Model Evaluation
```bash
# Evaluate model performance
python -m src.cli evaluate --csv data/test.csv --model model.joblib

# Cross-validation
python -m src.cli crossval --csv data/training.csv --folds 5

# Custom scorer
python -m src.cli crossval --csv data/training.csv --scorer f1_macro
```

### Advanced Model Comparison
```bash
# Compare multiple model types
python -m src.cli compare --csv data/training.csv

# Include transformer models (requires transformers library)
python -m src.cli compare --csv data/training.csv --include-transformers

# Save detailed results
python -m src.cli compare --csv data/training.csv --output comparison_results.json
```

### Starting Web Server
```bash
# Start with default settings
python -m src.webapp

# Custom host and port
python -m src.webapp --host 0.0.0.0 --port 8080

# Debug mode
python -m src.webapp --debug

# Custom log level
python -m src.webapp --log-level DEBUG
```

---

## Model Types

### Supported Models

1. **Scikit-learn Models** (default):
   - Logistic Regression with TF-IDF vectorization
   - Fast training and prediction
   - Good baseline performance

2. **Naive Bayes Models**:
   - MultinomialNB with TF-IDF
   - Efficient for text classification
   - Handles large vocabularies well

3. **LSTM Models** (optional, requires TensorFlow):
   - Deep learning approach
   - Better context understanding
   - Longer training time

4. **Transformer Models** (optional, requires transformers library):
   - DistilBERT-based fine-tuning
   - State-of-the-art performance
   - Configurable hyperparameters

### Model Comparison Framework

The application includes a comprehensive model comparison system:

```python
from src.model_comparison import benchmark_models

# Compare all available models
results = benchmark_models(
    train_texts=train_texts,
    train_labels=train_labels,
    test_texts=test_texts,
    test_labels=test_labels,
    include_transformers=True,
    include_lstm=True
)

# Results include accuracy, F1, precision, recall, training time
print(results.get_summary_table())
```

---

## Monitoring and Metrics

### Available Metrics

The application exposes metrics for monitoring:

**Request Metrics:**
- `sentiment_requests_total`: Total API requests by method, endpoint, and status
- `sentiment_request_duration_seconds`: Request processing time

**Model Metrics:**
- `sentiment_predictions_total`: Total predictions by model type and result
- `sentiment_model_load_duration_seconds`: Model loading time

**Training Metrics:**
- `sentiment_training_duration_seconds`: Model training time
- `sentiment_training_accuracy`: Training accuracy by model type

**System Metrics:**
- `sentiment_active_connections`: Current active connections

### Structured Logging

All components use structured JSON logging:

```json
{
  "timestamp": "2025-01-24T05:30:00Z",
  "level": "INFO",
  "event_type": "prediction",
  "prediction_time_seconds": 0.045,
  "model_type": "sklearn",
  "client_ip": "127.0.0.1"
}
```

---

## Error Handling

### Common Error Codes

- **400 Bad Request**: Invalid input format, missing required fields
- **413 Payload Too Large**: Input text exceeds size limits
- **415 Unsupported Media Type**: Invalid Content-Type header
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Model loading or prediction errors

### Error Response Format

```json
{
  "error": "Brief error description",
  "details": ["Detailed error information"],
  "timestamp": "2025-01-24T05:30:00Z",
  "request_id": "abc123"
}
```

---

## Development and Testing

### Running Tests
```bash
# Full test suite
make test

# Specific test categories
pytest tests/test_webapp.py -v
pytest tests/test_models.py -v
pytest tests/test_security.py -v
```

### Code Quality
```bash
# Linting
make lint

# Security scanning
make security

# Type checking
make typecheck
```

### Development Server
```bash
# Start development server with auto-reload
make dev

# With debug logging
DEBUG=1 make dev
```

---

## Deployment

### Docker Deployment
```bash
# Build image
docker build -t sentiment-analyzer-pro .

# Run container
docker run -p 5000:5000 sentiment-analyzer-pro

# With environment variables
docker run -p 5000:5000 \
  -e MODEL_PATH=/app/models/custom.joblib \
  -e RATE_LIMIT_MAX_REQUESTS=200 \
  sentiment-analyzer-pro
```

### Production Configuration

For production deployment:

1. **Set secure environment variables:**
   ```bash
   export RATE_LIMIT_MAX_REQUESTS=100
   export MAX_FILE_SIZE_MB=50
   export LOG_LEVEL=WARNING
   ```

2. **Use a production WSGI server:**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 "src.webapp:app"
   ```

3. **Configure reverse proxy (nginx example):**
   ```nginx
   server {
       listen 80;
       location / {
           proxy_pass http://127.0.0.1:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
       }
   }
   ```

---

## Examples

### Python API Usage
```python
import requests

# Basic prediction
response = requests.post('http://localhost:5000/predict', 
                        json={'text': 'This movie is amazing!'})
result = response.json()
print(f"Sentiment: {result['prediction']}, Confidence: {result['confidence']}")

# Check service health
health = requests.get('http://localhost:5000/').json()
print(f"Service status: {health['status']}")

# Get metrics summary
metrics = requests.get('http://localhost:5000/metrics/summary').json()
print(f"Total predictions: {metrics['predictions']}")
```

### curl Examples
```bash
# Make prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'

# Health check
curl http://localhost:5000/

# Get version
curl http://localhost:5000/version

# Get metrics
curl http://localhost:5000/metrics

# Get metrics summary
curl http://localhost:5000/metrics/summary
```

---

## Troubleshooting

### Common Issues

1. **Model Loading Errors:**
   ```
   FileNotFoundError: [Errno 2] No such file or directory: 'model.joblib'
   ```
   - Ensure model file exists at specified path
   - Check `MODEL_PATH` environment variable
   - Train a model first using the CLI

2. **Rate Limiting Issues:**
   ```
   {"error": "Rate limit exceeded"}
   ```
   - Increase `RATE_LIMIT_MAX_REQUESTS` if needed
   - Check for IP address issues with proxies
   - Implement request spacing in client code

3. **Memory Issues with Large Models:**
   - Use smaller models for production
   - Increase container memory limits
   - Consider model quantization

4. **Performance Issues:**
   - Enable model caching (default enabled)
   - Use faster model types (sklearn vs transformers)
   - Check system resources

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python -m src.webapp --debug
```

This provides detailed information about:
- Request processing
- Model loading
- Prediction timing
- Security events
- Error contexts

---

## Support

For issues and feature requests:
- Check existing documentation
- Review test files for usage examples
- Enable debug logging for troubleshooting
- Monitor metrics for performance insights