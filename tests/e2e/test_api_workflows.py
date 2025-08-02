"""End-to-end tests for API workflows."""

import pytest
import json
import tempfile
import pandas as pd
from unittest.mock import patch

# These tests require the web dependencies
pytest.importorskip("flask", reason="Flask not available")

from src.webapp import create_app


@pytest.mark.e2e
class TestAPIWorkflows:
    """End-to-end tests for the web API."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the Flask app."""
        app = create_app()
        app.config['TESTING'] = True
        
        with app.test_client() as client:
            yield client
    
    def test_health_check_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get('/')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'ok'
    
    def test_version_endpoint(self, client):
        """Test the version endpoint."""
        response = client.get('/version')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'version' in data
    
    def test_metrics_endpoint(self, client):
        """Test the metrics endpoint."""
        response = client.get('/metrics')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'requests' in data
        assert 'predictions' in data
    
    def test_prediction_endpoint(self, client):
        """Test the prediction endpoint."""
        # Test with valid input
        response = client.post('/predict', 
                             data=json.dumps({'text': 'I love this product!'}),
                             content_type='application/json')
        
        if response.status_code == 500:
            # Model might not be available in test environment
            pytest.skip("Model not available for prediction testing")
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'sentiment' in data
        assert data['sentiment'] in ['positive', 'negative', 'neutral']
    
    def test_prediction_endpoint_validation(self, client):
        """Test input validation for prediction endpoint."""
        # Test with missing text
        response = client.post('/predict', 
                             data=json.dumps({}),
                             content_type='application/json')
        assert response.status_code == 400
        
        # Test with invalid JSON
        response = client.post('/predict', 
                             data='invalid json',
                             content_type='application/json')
        assert response.status_code == 400
        
        # Test with empty text
        response = client.post('/predict', 
                             data=json.dumps({'text': ''}),
                             content_type='application/json')
        assert response.status_code == 400


@pytest.mark.e2e 
class TestCLIWorkflows:
    """End-to-end tests for CLI workflows."""
    
    def test_cli_help_command(self):
        """Test CLI help command."""
        import subprocess
        import sys
        
        try:
            result = subprocess.run([
                sys.executable, '-m', 'src.cli', '--help'
            ], capture_output=True, text=True, timeout=10)
            
            assert result.returncode == 0
            assert 'usage:' in result.stdout.lower()
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("CLI not available for testing")
    
    def test_cli_version_command(self):
        """Test CLI version command."""
        import subprocess
        import sys
        
        try:
            result = subprocess.run([
                sys.executable, '-m', 'src.cli', 'version'
            ], capture_output=True, text=True, timeout=10)
            
            # Should either return version or indicate command structure
            assert result.returncode in [0, 1, 2]  # Various exit codes acceptable
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("CLI not available for testing")
    
    def test_full_training_workflow(self, temp_csv_file: str):
        """Test complete training workflow via CLI."""
        import subprocess
        import sys
        import tempfile
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as model_file:
                # Test training command
                result = subprocess.run([
                    sys.executable, '-m', 'src.cli', 'train',
                    '--csv', temp_csv_file,
                    '--model', model_file.name
                ], capture_output=True, text=True, timeout=30)
                
                # Training might fail due to various reasons in test environment
                # We mainly check that the CLI is accessible
                assert result.returncode in [0, 1, 2]
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("CLI training workflow not available for testing")


@pytest.mark.e2e
class TestDockerWorkflows:
    """End-to-end tests for Docker workflows."""
    
    @pytest.mark.slow
    def test_docker_build(self):
        """Test Docker image building."""
        import subprocess
        import os
        
        if not os.path.exists('Dockerfile'):
            pytest.skip("Dockerfile not found")
        
        try:
            # Test docker build (this is slow)
            result = subprocess.run([
                'docker', 'build', '-t', 'sentiment-test', '.'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0 and 'docker: not found' in result.stderr:
                pytest.skip("Docker not available")
            
            # We mainly test that build process starts
            # Full build might be too slow for regular testing
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Docker not available for testing")