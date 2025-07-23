"""Additional tests to improve predict.py coverage."""

import pytest
import tempfile
import os
import pandas as pd
from unittest.mock import Mock, patch, mock_open
from src.predict import main


class TestPredictCoverageImprovements:
    """Test cases to cover missing lines in predict.py."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "test_data.csv")
        self.model_path = os.path.join(self.temp_dir, "test_model.joblib")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    @patch('pandas.read_csv')
    @patch('joblib.load')
    def test_empty_csv_data_error(self, mock_joblib_load, mock_read_csv):
        """Test handling of EmptyDataError during CSV reading."""
        mock_read_csv.side_effect = pd.errors.EmptyDataError("No columns to parse from file")
        
        with pytest.raises(SystemExit, match="Input CSV file is empty"):
            main(self.csv_path, self.model_path)
    
    @patch('pandas.read_csv')
    @patch('joblib.load')
    def test_csv_parser_error(self, mock_joblib_load, mock_read_csv):
        """Test handling of ParserError during CSV reading."""
        mock_read_csv.side_effect = pd.errors.ParserError("Error tokenizing data")
        
        with pytest.raises(SystemExit, match="Invalid CSV format"):
            main(self.csv_path, self.model_path)
    
    @patch('pandas.read_csv')
    @patch('joblib.load')
    def test_csv_permission_error(self, mock_joblib_load, mock_read_csv):
        """Test handling of PermissionError during CSV reading."""
        mock_read_csv.side_effect = PermissionError("Permission denied")
        
        with pytest.raises(SystemExit, match="Permission denied reading"):
            main(self.csv_path, self.model_path)
    
    @patch('joblib.load')
    def test_model_eof_error(self, mock_joblib_load):
        """Test handling of EOFError during model loading."""
        # Create valid CSV
        test_data = pd.DataFrame({"text": ["test text"]})
        test_data.to_csv(self.csv_path, index=False)
        
        mock_joblib_load.side_effect = EOFError("Unexpected end of file")
        
        with pytest.raises(SystemExit, match="Invalid or corrupted model file"):
            main(self.csv_path, self.model_path)
    
    @patch('joblib.load')
    def test_model_type_error(self, mock_joblib_load):
        """Test handling of TypeError during model loading."""
        # Create valid CSV
        test_data = pd.DataFrame({"text": ["test text"]})
        test_data.to_csv(self.csv_path, index=False)
        
        mock_joblib_load.side_effect = TypeError("Invalid model type")
        
        with pytest.raises(SystemExit, match="Invalid or corrupted model file"):
            main(self.csv_path, self.model_path)
    
    @patch('joblib.load')
    def test_model_permission_error(self, mock_joblib_load):
        """Test handling of PermissionError during model loading."""
        # Create valid CSV
        test_data = pd.DataFrame({"text": ["test text"]})
        test_data.to_csv(self.csv_path, index=False)
        
        mock_joblib_load.side_effect = PermissionError("Permission denied")
        
        with pytest.raises(SystemExit, match="Permission denied reading model file"):
            main(self.csv_path, self.model_path)
    
    @patch('joblib.load')
    def test_no_valid_text_data_warning(self, mock_joblib_load, caplog):
        """Test warning and early return when some text data is valid but some is null."""
        # Create CSV with mixed valid and null text values
        # This will pass the initial validation but trigger the warning later
        test_data = pd.DataFrame({"text": ["valid text", None, None]})
        test_data.to_csv(self.csv_path, index=False)
        
        mock_model = Mock()
        mock_model.predict.return_value = ["positive"]
        mock_joblib_load.return_value = mock_model
        
        import logging
        with caplog.at_level(logging.WARNING):
            result = main(self.csv_path, self.model_path)
        
        # Should complete successfully but log warning about skipped rows
        assert "Skipped 2 rows with missing text values" in caplog.text
    
    
    @patch('joblib.load')
    def test_unexpected_prediction_error(self, mock_joblib_load):
        """Test handling of unexpected errors during prediction."""
        # Create valid CSV
        test_data = pd.DataFrame({"text": ["test text"]})
        test_data.to_csv(self.csv_path, index=False)
        
        mock_model = Mock()
        mock_model.predict.side_effect = RuntimeError("Unexpected prediction error")
        mock_joblib_load.return_value = mock_model
        
        with pytest.raises(SystemExit, match="Unexpected error during prediction"):
            main(self.csv_path, self.model_path)
    
    @patch('joblib.load')
    def test_model_attribute_error_prediction(self, mock_joblib_load):
        """Test handling of AttributeError during prediction."""
        # Create valid CSV
        test_data = pd.DataFrame({"text": ["test text"]})
        test_data.to_csv(self.csv_path, index=False)
        
        mock_model = Mock()
        mock_model.predict.side_effect = AttributeError("Model has no predict method")
        mock_joblib_load.return_value = mock_model
        
        with pytest.raises(SystemExit, match="Model prediction failed"):
            main(self.csv_path, self.model_path)
    
    @patch('joblib.load')
    def test_model_value_error_prediction(self, mock_joblib_load):
        """Test handling of ValueError during prediction."""
        # Create valid CSV
        test_data = pd.DataFrame({"text": ["test text"]})
        test_data.to_csv(self.csv_path, index=False)
        
        mock_model = Mock()
        mock_model.predict.side_effect = ValueError("Invalid input shape")
        mock_joblib_load.return_value = mock_model
        
        with pytest.raises(SystemExit, match="Model prediction failed"):
            main(self.csv_path, self.model_path)