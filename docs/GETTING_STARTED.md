# Getting Started with Sentiment Analyzer Pro

## Quick Start Guide

This guide will help you get up and running with Sentiment Analyzer Pro in less than 5 minutes.

## Prerequisites

- Python 3.9+ 
- pip package manager
- Git (for cloning the repository)

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd sentiment-analyzer-pro
```

### 2. Set Up Development Environment
```bash
# Install all dependencies and set up development environment
make setup

# Verify installation
make test
```

That's it! The `make setup` command handles:
- Installing Python dependencies
- Setting up the package in development mode
- Installing development tools (pytest, linting, security scanning)
- Configuring pre-commit hooks

## Your First Prediction

### Option 1: Web API (Recommended)

1. **Start the web server:**
   ```bash
   python -m src.webapp
   ```
   
   You should see:
   ```
   * Running on http://127.0.0.1:5000
   ```

2. **Make a prediction:**
   ```bash
   curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "I love this product!"}'
   ```
   
   Response:
   ```json
   {
     "prediction": "positive",
     "confidence": 0.85
   }
   ```

### Option 2: Command Line

1. **Create test data:**
   ```bash
   echo "text,label" > test_data.csv
   echo "I love this product!,positive" >> test_data.csv
   echo "This is terrible,negative" >> test_data.csv
   ```

2. **Train a model:**
   ```bash
   python -m src.train --csv test_data.csv
   ```

3. **Make predictions:**
   ```bash
   echo "text" > predict_data.csv
   echo "This is amazing!" >> predict_data.csv
   python -m src.predict --csv predict_data.csv
   ```

## Core Workflows

### 1. Training Custom Models

```bash
# Use the sample data
python -m src.train --csv data/sample_reviews.csv

# Train with your own data
python -m src.train --csv your_data.csv --text-column "review" --label-column "sentiment"

# Save to custom location
python -m src.train --csv your_data.csv --model models/custom_sentiment.joblib
```

### 2. Running the Web API

```bash
# Basic server
python -m src.webapp

# Production-ready server with gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 "src.webapp:app"

# With custom configuration
export MODEL_PATH="models/custom_sentiment.joblib"
export RATE_LIMIT_MAX_REQUESTS=200
python -m src.webapp --host 0.0.0.0 --port 8080
```

### 3. Batch Processing

```bash
# Process large datasets
python -m src.predict --csv large_dataset.csv --output results.csv

# Preprocess text data
python -m src.cli preprocess --csv raw_data.csv --output clean_data.csv --lemmatize
```

### 4. Model Evaluation

```bash
# Compare multiple models
python -m src.cli compare --csv training_data.csv

# Cross-validation
python -m src.cli crossval --csv training_data.csv --folds 5

# Detailed evaluation
python -m src.cli evaluate --csv test_data.csv --model model.joblib
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Model configuration
MODEL_PATH=models/production_model.joblib

# API configuration  
RATE_LIMIT_MAX_REQUESTS=100
RATE_LIMIT_WINDOW=60
MAX_FILE_SIZE_MB=100

# Security configuration
ALLOWED_TEMP_DIRS=/tmp,/var/tmp

# Logging
LOG_LEVEL=INFO
```

### Using Configuration

```bash
# Load from .env file
export $(cat .env | xargs)
python -m src.webapp

# Or set directly
MODEL_PATH=/path/to/model.joblib python -m src.webapp
```

## Development Workflow

### Running Tests
```bash
# Full test suite
make test

# With coverage report
make test-coverage

# Specific test file
pytest tests/test_webapp.py -v

# Watch mode for development
pytest --watch
```

### Code Quality
```bash
# Lint code
make lint

# Fix formatting issues
make format

# Security scan
make security

# Type checking
make typecheck

# All quality checks
make check
```

### Development Server
```bash
# Start with auto-reload
make dev

# Debug mode with detailed logging
DEBUG=1 make dev
```

## Advanced Features

### 1. Advanced Model Types

```bash
# Compare all model types (requires optional dependencies)
pip install tensorflow transformers torch
python -m src.cli compare --csv data.csv --include-transformers --include-lstm
```

### 2. Monitoring and Metrics

```bash
# Start server
python -m src.webapp

# View metrics (Prometheus format)
curl http://localhost:5000/metrics

# View metrics summary (JSON)
curl http://localhost:5000/metrics/summary

# Health check
curl http://localhost:5000/
```

### 3. Docker Deployment

```bash
# Build image
docker build -t sentiment-analyzer .

# Run container
docker run -p 5000:5000 sentiment-analyzer

# With environment variables
docker run -p 5000:5000 \
  -e MODEL_PATH=/app/models/custom.joblib \
  -e LOG_LEVEL=DEBUG \
  sentiment-analyzer
```

## Example Applications

### 1. Customer Review Analysis

```python
import requests
import pandas as pd

# Load your customer reviews
reviews = pd.read_csv('customer_reviews.csv')

# Analyze each review
results = []
for review in reviews['review_text']:
    response = requests.post('http://localhost:5000/predict',
                           json={'text': review})
    if response.status_code == 200:
        result = response.json()
        results.append({
            'review': review,
            'sentiment': result['prediction'],
            'confidence': result['confidence']
        })

# Create results DataFrame
sentiment_df = pd.DataFrame(results)
print(sentiment_df.groupby('sentiment').size())
```

### 2. Social Media Monitoring

```python
# Process tweets or social media posts
social_posts = [
    "Just tried the new product - absolutely love it! #amazing",
    "Customer service was disappointing today :(",
    "Neutral opinion about the latest update"
]

for post in social_posts:
    response = requests.post('http://localhost:5000/predict',
                           json={'text': post})
    result = response.json()
    print(f"'{post[:50]}...' -> {result['prediction']} ({result['confidence']:.2f})")
```

### 3. Real-time Analysis Dashboard

```python
import streamlit as st
import requests

st.title("Real-time Sentiment Analysis")

# Text input
user_text = st.text_area("Enter text to analyze:")

if st.button("Analyze Sentiment"):
    if user_text:
        # Call API
        response = requests.post('http://localhost:5000/predict',
                               json={'text': user_text})
        
        if response.status_code == 200:
            result = response.json()
            
            # Display results
            st.success(f"Sentiment: {result['prediction']}")
            st.info(f"Confidence: {result['confidence']:.2%}")
            
            # Confidence meter
            st.progress(result['confidence'])
        else:
            st.error("Analysis failed")
```

## Common Use Cases

### 1. E-commerce Product Reviews
- Analyze customer feedback automatically
- Identify dissatisfied customers for proactive support
- Track sentiment trends over time

### 2. Social Media Monitoring
- Monitor brand mentions across platforms
- Detect PR crises early
- Measure campaign effectiveness

### 3. Customer Support
- Prioritize tickets based on sentiment
- Route negative feedback to senior agents
- Measure support team performance

### 4. Content Moderation
- Identify potentially harmful content
- Flag posts requiring human review
- Maintain community standards

## Performance Tips

### 1. API Performance
- Use connection pooling for multiple requests
- Implement client-side caching for repeated text
- Consider batch processing for large datasets

### 2. Model Performance
- Use appropriate model complexity for your needs
- Consider model size vs accuracy tradeoffs
- Cache models in production environments

### 3. Scaling
- Use load balancers for high traffic
- Implement horizontal scaling with multiple instances  
- Monitor memory usage with large models

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure package is installed in development mode
pip install -e .

# Or use make setup
make setup
```

**2. Model Not Found**
```bash
# Train a model first
python -m src.train --csv data/sample_reviews.csv

# Or specify model path
MODEL_PATH=/path/to/model.joblib python -m src.webapp
```

**3. Port Already in Use**
```bash
# Use different port
python -m src.webapp --port 8080

# Or kill existing process
lsof -ti:5000 | xargs kill -9
```

**4. Rate Limiting Issues**
```bash
# Increase rate limits
export RATE_LIMIT_MAX_REQUESTS=500
python -m src.webapp
```

### Getting Help

1. **Check the logs:**
   ```bash
   LOG_LEVEL=DEBUG python -m src.webapp
   ```

2. **Review test files** for usage examples:
   ```bash
   ls tests/test_*.py
   ```

3. **Check configuration:**
   ```bash
   python -c "from src.config import Config; print(vars(Config))"
   ```

## Next Steps

- **Read the [API Reference](API_REFERENCE.md)** for detailed API documentation
- **Explore the [Model Architecture](MODEL_ARCHITECTURE.md)** to understand the ML pipeline
- **Check the [Security Guide](SECURITY.md)** for production deployment
- **Review [Contributing Guidelines](../CONTRIBUTING.md)** to contribute to the project

## Need Help?

- **Documentation**: Check the `docs/` directory for detailed guides
- **Examples**: Look at test files for usage examples
- **Issues**: Enable debug logging to troubleshoot problems
- **Performance**: Monitor the `/metrics` endpoint for insights

Happy analyzing! 🎉