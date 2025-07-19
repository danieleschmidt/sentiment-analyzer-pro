"""Security-focused tests for the sentiment analyzer."""

import pytest
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

from src.schemas import PredictRequest
from src.webapp import app, _check_rate_limit, RATE_LIMIT_REQUESTS
from src.cli import load_csv
import tempfile
import os


class TestInputValidation:
    """Test input validation and sanitization."""
    
    def test_predict_request_sanitizes_script_tags(self):
        """Test that script tags are removed from input."""
        malicious_text = "This is <script>alert('xss')</script> a test"
        req = PredictRequest(text=malicious_text)
        assert "<script>" not in req.text
        assert "alert" not in req.text
        assert "This is a test" == req.text
    
    def test_predict_request_sanitizes_javascript_urls(self):
        """Test that javascript: URLs are removed."""
        malicious_text = "Click javascript:alert('xss') here"
        req = PredictRequest(text=malicious_text)
        assert "javascript:" not in req.text
        assert "Click alert('xss') here" == req.text
    
    def test_predict_request_sanitizes_event_handlers(self):
        """Test that event handlers are removed."""
        malicious_text = "Text with onclick=alert('xss') handler"
        req = PredictRequest(text=malicious_text)
        assert "onclick=" not in req.text
        assert "Text with alert('xss') handler" == req.text
    
    def test_predict_request_validates_length(self):
        """Test that text length is validated."""
        # Test empty string
        with pytest.raises(ValidationError):
            PredictRequest(text="")
        
        # Test too long string
        long_text = "a" * 10001
        with pytest.raises(ValidationError):
            PredictRequest(text=long_text)
    
    def test_predict_request_validates_type(self):
        """Test that non-string input is rejected."""
        with pytest.raises(ValidationError):
            PredictRequest(text=123)
        
        with pytest.raises(ValidationError):
            PredictRequest(text=None)


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def test_rate_limit_allows_normal_requests(self):
        """Test that normal request rates are allowed."""
        # Clear any existing rate limit data
        RATE_LIMIT_REQUESTS.clear()
        
        assert _check_rate_limit("127.0.0.1") is True
    
    def test_rate_limit_blocks_excessive_requests(self):
        """Test that excessive requests are blocked."""
        # Clear any existing rate limit data
        RATE_LIMIT_REQUESTS.clear()
        
        # Make maximum allowed requests
        for _ in range(100):
            assert _check_rate_limit("192.168.1.1") is True
        
        # Next request should be blocked
        assert _check_rate_limit("192.168.1.1") is False
    
    def test_rate_limit_per_ip(self):
        """Test that rate limiting is per IP address."""
        # Clear any existing rate limit data
        RATE_LIMIT_REQUESTS.clear()
        
        # Max out one IP
        for _ in range(100):
            assert _check_rate_limit("192.168.1.1") is True
        
        # Different IP should still work
        assert _check_rate_limit("192.168.1.2") is True


class TestFileSecurity:
    """Test file operation security."""
    
    def test_load_csv_prevents_path_traversal(self):
        """Test that path traversal attacks are prevented."""
        with pytest.raises(SystemExit):
            load_csv("../../../etc/passwd")
        
        with pytest.raises(SystemExit):
            load_csv("/etc/passwd")
    
    def test_load_csv_validates_file_exists(self):
        """Test that non-existent files are handled."""
        with pytest.raises(SystemExit):
            load_csv("nonexistent_file.csv")
    
    def test_load_csv_validates_file_size(self):
        """Test that oversized files are rejected."""
        # Create a temporary large file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write header
            f.write("text,label\\n")
            # Write lots of data to make file > 100MB
            large_text = "a" * (1024 * 1024)  # 1MB string
            for _ in range(101):  # 101MB total
                f.write(f"\"{large_text}\",positive\\n")
            temp_path = f.name
        
        try:
            # Use only the filename, not full path
            filename = os.path.basename(temp_path)
            with patch('os.path.isfile', return_value=True), \
                 patch('os.path.getsize', return_value=101 * 1024 * 1024):
                with pytest.raises(SystemExit, match="File too large"):
                    load_csv(filename)
        finally:
            os.unlink(temp_path)


class TestWebAppSecurity:
    """Test web application security features."""
    
    def setUp(self):
        """Set up test client."""
        self.client = app.test_client()
        app.config['TESTING'] = True
    
    def test_security_headers_present(self):
        """Test that security headers are added to responses."""
        client = app.test_client()
        response = client.get('/')
        
        assert response.headers.get('X-Content-Type-Options') == 'nosniff'
        assert response.headers.get('X-Frame-Options') == 'DENY'
        assert response.headers.get('X-XSS-Protection') == '1; mode=block'
        assert 'Strict-Transport-Security' in response.headers
        assert 'Content-Security-Policy' in response.headers
    
    def test_predict_requires_json_content_type(self):
        """Test that predict endpoint requires JSON content type."""
        client = app.test_client()
        response = client.post('/predict', 
                             data='{"text": "test"}',
                             content_type='text/plain')
        
        assert response.status_code == 400
        assert "Content-Type must be application/json" in response.get_json()['error']
    
    def test_predict_handles_malformed_json(self):
        """Test that malformed JSON is handled gracefully."""
        client = app.test_client()
        response = client.post('/predict',
                             data='{"text": "incomplete',
                             content_type='application/json')
        
        assert response.status_code == 400
    
    @patch('src.webapp.load_model')
    def test_predict_handles_model_errors(self, mock_load_model):
        """Test that model errors are handled gracefully."""
        mock_load_model.side_effect = Exception("Model failed")
        
        client = app.test_client()
        response = client.post('/predict',
                             json={"text": "test text"},
                             content_type='application/json')
        
        assert response.status_code == 500
        assert response.get_json()['error'] == "Internal server error"