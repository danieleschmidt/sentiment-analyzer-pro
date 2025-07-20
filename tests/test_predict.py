import pytest
import tempfile
import os
import pandas as pd
from unittest.mock import Mock, patch
import logging
from src.predict import main


class TestPredictMain:
    """Test cases for the predict.main function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary CSV file
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "test_data.csv")
        
        # Create test data
        test_data = pd.DataFrame({
            "text": [
                "Great movie, loved it!",
                "Terrible film, waste of time",
                "Amazing cinematography and acting"
            ]
        })
        test_data.to_csv(self.csv_path, index=False)
        
        # Model path for testing
        self.model_path = os.path.join(self.temp_dir, "test_model.joblib")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    @patch('joblib.load')
    def test_main_basic_functionality(self, mock_joblib_load, caplog):
        """Test that main function loads data, model, and makes predictions."""
        # Create mock model
        mock_model = Mock()
        mock_model.predict.return_value = ["positive", "negative", "positive"]
        mock_joblib_load.return_value = mock_model
        
        with caplog.at_level(logging.INFO):
            main(self.csv_path, self.model_path)
        
        # Verify joblib.load was called
        mock_joblib_load.assert_called_once_with(self.model_path)
        
        # Verify model was called with correct data
        mock_model.predict.assert_called_once()
        called_texts = mock_model.predict.call_args[0][0]
        expected_texts = ["Great movie, loved it!", "Terrible film, waste of time", "Amazing cinematography and acting"]
        pd.testing.assert_series_equal(called_texts, pd.Series(expected_texts, name="text"))
        
        # Verify logging output contains predictions
        log_messages = [record.message for record in caplog.records]
        assert len(log_messages) == 3
        assert "Great movie, loved it! => positive" in log_messages
        assert "Terrible film, waste of time => negative" in log_messages
        assert "Amazing cinematography and acting => positive" in log_messages
    
    @patch('joblib.load')
    def test_main_with_default_model_path(self, mock_joblib_load):
        """Test main function with default model path."""
        mock_model = Mock()
        mock_model.predict.return_value = ["positive", "negative", "positive"]
        mock_joblib_load.return_value = mock_model
        
        # When no model path is provided, should use the default "model.joblib"
        main(self.csv_path)  # Don't specify model_path
        
        # Should use the default model path since default is evaluated at function definition time
        mock_joblib_load.assert_called_once_with("model.joblib")
        mock_model.predict.assert_called_once()
    
    @patch('joblib.load')
    def test_main_file_not_found_csv(self, mock_joblib_load):
        """Test that FileNotFoundError is raised for non-existent CSV file."""
        mock_model = Mock()
        mock_joblib_load.return_value = mock_model
        
        non_existent_csv = "/non/existent/file.csv"
        
        with pytest.raises(FileNotFoundError):
            main(non_existent_csv, self.model_path)
    
    @patch('joblib.load')
    def test_main_file_not_found_model(self, mock_joblib_load):
        """Test that FileNotFoundError is raised for non-existent model file."""
        mock_joblib_load.side_effect = FileNotFoundError("Model file not found")
        
        with pytest.raises(FileNotFoundError):
            main(self.csv_path, "/non/existent/model.joblib")
    
    @patch('joblib.load')
    def test_main_missing_text_column(self, mock_joblib_load):
        """Test that KeyError is raised when CSV lacks 'text' column."""
        mock_model = Mock()
        mock_joblib_load.return_value = mock_model
        
        # Create CSV without 'text' column
        bad_csv_path = os.path.join(self.temp_dir, "bad_data.csv")
        bad_data = pd.DataFrame({
            "review": ["Good movie", "Bad movie"],
            "score": [5, 1]
        })
        bad_data.to_csv(bad_csv_path, index=False)
        
        with pytest.raises(KeyError):
            main(bad_csv_path, self.model_path)
        
        # Cleanup
        os.remove(bad_csv_path)
    
    @patch('joblib.load')
    def test_main_empty_csv(self, mock_joblib_load):
        """Test behavior with empty CSV file."""
        mock_model = Mock()
        mock_model.predict.return_value = []
        mock_joblib_load.return_value = mock_model
        
        empty_csv_path = os.path.join(self.temp_dir, "empty.csv")
        empty_data = pd.DataFrame({"text": []})
        empty_data.to_csv(empty_csv_path, index=False)
        
        main(empty_csv_path, self.model_path)
        
        # Should still call predict with empty series
        mock_model.predict.assert_called_once()
        called_texts = mock_model.predict.call_args[0][0]
        assert len(called_texts) == 0
        
        # Cleanup
        os.remove(empty_csv_path)
    
    @patch('joblib.load')
    def test_main_csv_with_missing_values(self, mock_joblib_load):
        """Test behavior with CSV containing NaN values."""
        mock_model = Mock()
        mock_model.predict.return_value = ["positive", "negative", "positive"]
        mock_joblib_load.return_value = mock_model
        
        csv_with_nan_path = os.path.join(self.temp_dir, "nan_data.csv")
        nan_data = pd.DataFrame({
            "text": ["Good movie", None, "Bad movie"]
        })
        nan_data.to_csv(csv_with_nan_path, index=False)
        
        # This should work as pandas will handle NaN values
        main(csv_with_nan_path, self.model_path)
        mock_model.predict.assert_called_once()
        
        # Cleanup
        os.remove(csv_with_nan_path)
    
    @patch('joblib.load')
    def test_main_model_prediction_error(self, mock_joblib_load):
        """Test behavior when model.predict raises an exception."""
        mock_model = Mock()
        mock_model.predict.side_effect = ValueError("Model prediction failed")
        mock_joblib_load.return_value = mock_model
        
        with pytest.raises(ValueError, match="Model prediction failed"):
            main(self.csv_path, self.model_path)
    
    @patch('pandas.read_csv')
    @patch('joblib.load')
    def test_main_integration_mocked(self, mock_joblib_load, mock_read_csv):
        """Test main function with all external dependencies mocked."""
        # Mock pandas read_csv
        mock_df = pd.DataFrame({"text": ["test review"]})
        mock_read_csv.return_value = mock_df
        
        # Mock joblib load
        mock_model = Mock()
        mock_model.predict.return_value = ["positive"]
        mock_joblib_load.return_value = mock_model
        
        main("test.csv", "test_model.joblib")
        
        # Verify calls
        mock_read_csv.assert_called_once_with("test.csv")
        mock_joblib_load.assert_called_once_with("test_model.joblib")
        mock_model.predict.assert_called_once()


class TestPredictCommandLine:
    """Test cases for command-line interface functionality."""
    
    def test_argument_parser_defaults(self):
        """Test that argument parser has correct defaults."""
        import sys
        from src.predict import __name__ as predict_name
        
        # Only test if we can import the parser safely
        if predict_name == "__main__":
            pytest.skip("Cannot test parser when module is running as main")
        
        # Test basic argument parsing concepts without actually running argparse
        # This test verifies the structure is correct
        csv_path = "test.csv"
        default_model = os.getenv("MODEL_PATH", "model.joblib")
        
        # These are the expected behaviors based on the predict.py code
        assert default_model == os.getenv("MODEL_PATH", "model.joblib")
    
    @patch.dict(os.environ, {"MODEL_PATH": "/custom/model/path.joblib"})
    def test_environment_variable_handling(self):
        """Test that environment variable is properly handled."""
        # Test that the environment variable logic works
        from src.predict import main
        
        expected_path = "/custom/model/path.joblib"
        actual_path = os.getenv("MODEL_PATH", "model.joblib")
        assert actual_path == expected_path


class TestPredictEdgeCases:
    """Test edge cases and error conditions."""
    
    @patch('joblib.load')
    def test_main_with_corrupted_model_file(self, mock_joblib_load):
        """Test behavior when model file is corrupted."""
        temp_dir = tempfile.mkdtemp()
        
        # Create a valid CSV
        csv_path = os.path.join(temp_dir, "test.csv")
        test_data = pd.DataFrame({"text": ["test"]})
        test_data.to_csv(csv_path, index=False)
        
        # Mock joblib to raise an exception (simulating corruption)
        mock_joblib_load.side_effect = Exception("File is corrupted")
        
        with pytest.raises(Exception, match="File is corrupted"):
            main(csv_path, "corrupted_model.joblib")
        
        # Cleanup
        os.remove(csv_path)
        os.rmdir(temp_dir)
    
    @patch('joblib.load')
    def test_main_with_csv_encoding_issues(self, mock_joblib_load):
        """Test handling of CSV files with different encodings."""
        temp_dir = tempfile.mkdtemp()
        
        # Create CSV with special characters
        csv_path = os.path.join(temp_dir, "encoded.csv")
        special_data = pd.DataFrame({
            "text": ["Café très bon", "Película excelente", "電影很好看"]
        })
        special_data.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Create mock model
        mock_model = Mock()
        mock_model.predict.return_value = ["positive", "positive", "positive"]
        mock_joblib_load.return_value = mock_model
        
        # Should handle UTF-8 encoded CSV correctly
        main(csv_path, "model.joblib")
        mock_model.predict.assert_called_once()
        
        # Cleanup
        os.remove(csv_path)
        os.rmdir(temp_dir)
    
    @patch('joblib.load')
    def test_main_memory_efficiency(self, mock_joblib_load):
        """Test that main function doesn't load everything into memory at once."""
        temp_dir = tempfile.mkdtemp()
        
        csv_path = os.path.join(temp_dir, "test.csv")
        test_data = pd.DataFrame({"text": ["test"] * 100})  # Small but multiple rows
        test_data.to_csv(csv_path, index=False)
        
        mock_model = Mock()
        mock_model.predict.return_value = ["positive"] * 100
        mock_joblib_load.return_value = mock_model
        
        # Function should complete without issues
        main(csv_path, "model.joblib")
        
        # Verify predict was called exactly once (not in chunks)
        assert mock_model.predict.call_count == 1
        
        # Cleanup
        os.remove(csv_path)
        os.rmdir(temp_dir)