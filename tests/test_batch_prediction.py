"""Tests for batch prediction functionality and performance optimizations."""

import pytest
import json
import joblib
from src.models import build_model


@pytest.fixture
def trained_model_path(tmp_path):
    """Create a trained model for testing."""
    model = build_model()
    model.fit(["I love this", "I hate this", "It's okay", "Amazing product"], 
              ["positive", "negative", "neutral", "positive"])
    model_file = tmp_path / 'batch_test_model.joblib'
    joblib.dump(model, model_file)
    return str(model_file)


def test_batch_prediction_endpoint(trained_model_path):
    """Test the batch prediction endpoint."""
    pytest.importorskip('flask')
    from src import webapp
    
    webapp.load_model.cache_clear()
    webapp.load_model._test_model_path = trained_model_path
    webapp._reset_counters()
    
    with webapp.app.test_client() as client:
        # Test batch prediction
        batch_data = {
            "texts": [
                "I love this product!",
                "This is terrible",
                "It's an okay product",
                "Amazing quality!"
            ]
        }
        
        resp = client.post('/predict/batch', 
                          json=batch_data,
                          content_type='application/json')
        
        assert resp.status_code == 200
        data = resp.get_json()
        
        # Check response structure
        assert 'predictions' in data
        assert 'batch_size' in data
        assert 'cache_hits' in data
        assert 'processing_time_seconds' in data
        
        # Check predictions array
        predictions = data['predictions']
        assert len(predictions) == 4
        assert data['batch_size'] == 4
        assert data['cache_hits'] == 0  # First time, no cache hits
        
        # Check each prediction has required fields
        for pred in predictions:
            assert 'prediction' in pred
            assert 'cached' in pred
            assert pred['cached'] is False  # First time, not cached
            assert pred['prediction'] in ['positive', 'negative', 'neutral']


def test_batch_prediction_caching(trained_model_path):
    """Test that batch predictions use caching effectively."""
    pytest.importorskip('flask')
    from src import webapp
    
    webapp.load_model.cache_clear()
    webapp.load_model._test_model_path = trained_model_path
    webapp._reset_counters()
    
    with webapp.app.test_client() as client:
        batch_data = {
            "texts": [
                "I love this product!",
                "This is terrible"
            ]
        }
        
        # First request - no cache hits
        resp1 = client.post('/predict/batch', 
                           json=batch_data,
                           content_type='application/json')
        data1 = resp1.get_json()
        assert data1['cache_hits'] == 0
        
        # Second request - should have cache hits
        resp2 = client.post('/predict/batch', 
                           json=batch_data,
                           content_type='application/json')
        data2 = resp2.get_json()
        assert data2['cache_hits'] == 2  # Both texts should be cached
        
        # Check that cached predictions match
        for i, pred in enumerate(data2['predictions']):
            assert pred['cached'] is True
            assert pred['prediction'] == data1['predictions'][i]['prediction']


def test_batch_prediction_mixed_cache(trained_model_path):
    """Test batch prediction with mix of cached and uncached texts."""
    pytest.importorskip('flask')
    from src import webapp
    
    webapp.load_model.cache_clear()
    webapp.load_model._test_model_path = trained_model_path
    webapp._reset_counters()
    
    with webapp.app.test_client() as client:
        # First request to populate cache
        client.post('/predict/batch', 
                   json={"texts": ["I love this product!"]},
                   content_type='application/json')
        
        # Second request with mix of cached and new texts
        batch_data = {
            "texts": [
                "I love this product!",  # Should be cached
                "Brand new text here"    # Should be new
            ]
        }
        
        resp = client.post('/predict/batch', 
                          json=batch_data,
                          content_type='application/json')
        data = resp.get_json()
        
        assert data['cache_hits'] == 1
        assert data['batch_size'] == 2
        
        # Check that first is cached, second is not
        assert data['predictions'][0]['cached'] is True
        assert data['predictions'][1]['cached'] is False


def test_batch_prediction_validation_errors(trained_model_path):
    """Test batch prediction validation and error handling."""
    pytest.importorskip('flask')
    from src import webapp
    
    webapp.load_model.cache_clear()
    webapp.load_model._test_model_path = trained_model_path
    
    with webapp.app.test_client() as client:
        # Test empty texts list
        resp = client.post('/predict/batch', 
                          json={"texts": []},
                          content_type='application/json')
        assert resp.status_code == 400
        
        # Test non-string in texts
        resp = client.post('/predict/batch', 
                          json={"texts": ["valid text", 123]},
                          content_type='application/json')
        assert resp.status_code == 400
        
        # Test too many texts (over limit)
        large_texts = ["text"] * 101  # Over the 100 limit
        resp = client.post('/predict/batch', 
                          json={"texts": large_texts},
                          content_type='application/json')
        assert resp.status_code == 400
        
        # Test missing texts field
        resp = client.post('/predict/batch', 
                          json={"wrong_field": ["text"]},
                          content_type='application/json')
        assert resp.status_code == 400


def test_batch_prediction_content_type_validation(trained_model_path):
    """Test that batch prediction requires JSON content type."""
    pytest.importorskip('flask')
    from src import webapp
    
    webapp.load_model.cache_clear()
    webapp.load_model._test_model_path = trained_model_path
    
    with webapp.app.test_client() as client:
        # Test without JSON content type
        resp = client.post('/predict/batch', 
                          data='{"texts": ["test"]}',
                          content_type='text/plain')
        assert resp.status_code == 400
        assert "Content-Type must be application/json" in resp.get_json()['error']


def test_single_prediction_caching(trained_model_path):
    """Test that single predictions also use caching."""
    pytest.importorskip('flask')
    from src import webapp
    
    webapp.load_model.cache_clear()
    webapp.load_model._test_model_path = trained_model_path
    webapp._reset_counters()
    
    with webapp.app.test_client() as client:
        # First request - no cache
        resp1 = client.post('/predict', 
                           json={"text": "I love this product!"},
                           content_type='application/json')
        data1 = resp1.get_json()
        assert data1['cached'] is False
        
        # Second request - should be cached
        resp2 = client.post('/predict', 
                           json={"text": "I love this product!"},
                           content_type='application/json')
        data2 = resp2.get_json()
        assert data2['cached'] is True
        assert data2['prediction'] == data1['prediction']


def test_enhanced_metrics_with_cache_info(trained_model_path):
    """Test that metrics endpoint includes cache information."""
    pytest.importorskip('flask')
    from src import webapp
    
    webapp.load_model.cache_clear()
    webapp.load_model._test_model_path = trained_model_path
    webapp._reset_counters()
    
    with webapp.app.test_client() as client:
        # Make some predictions to populate metrics
        client.post('/predict', 
                   json={"text": "test text"},
                   content_type='application/json')
        
        client.post('/predict/batch', 
                   json={"texts": ["batch text 1", "batch text 2"]},
                   content_type='application/json')
        
        # Check metrics summary
        resp = client.get('/metrics/summary')
        data = resp.get_json()
        
        assert 'cache_size' in data
        assert 'cache_hit_rate_percent' in data
        assert 'max_cache_size' in data
        assert data['predictions'] == 3  # 1 single + 2 batch
        assert data['max_cache_size'] == 1000  # Default cache size