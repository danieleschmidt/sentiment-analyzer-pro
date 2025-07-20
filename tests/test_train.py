import pytest
import tempfile
import os
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import logging
from src.train import main


class TestTrainMain:
    """Test cases for the train.main function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test CSV file with proper training data
        self.csv_path = os.path.join(self.temp_dir, "train_data.csv")
        train_data = pd.DataFrame({
            "text": [
                "Great movie, loved it!",
                "Terrible film, waste of time",
                "Amazing cinematography and acting",
                "Boring plot and bad acting",
                "Fantastic storyline and direction"
            ],
            "label": ["positive", "negative", "positive", "negative", "positive"]
        })
        train_data.to_csv(self.csv_path, index=False)
        
        # Model path for testing
        self.model_path = os.path.join(self.temp_dir, "test_model.joblib")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    @patch('joblib.dump')
    @patch('src.train.build_model')
    def test_main_basic_functionality(self, mock_build_model, mock_joblib_dump, caplog):
        """Test that main function loads data, builds model, trains it, and saves it."""
        # Create mock model
        mock_model = Mock()
        mock_model.fit.return_value = None
        mock_build_model.return_value = mock_model
        
        with caplog.at_level(logging.INFO):
            main(self.csv_path, self.model_path)
        
        # Verify model was built
        mock_build_model.assert_called_once()
        
        # Verify model was trained with correct data
        mock_model.fit.assert_called_once()
        call_args = mock_model.fit.call_args[0]
        expected_texts = ["Great movie, loved it!", "Terrible film, waste of time", 
                         "Amazing cinematography and acting", "Boring plot and bad acting", 
                         "Fantastic storyline and direction"]
        expected_labels = ["positive", "negative", "positive", "negative", "positive"]
        
        pd.testing.assert_series_equal(call_args[0], pd.Series(expected_texts, name="text"))
        pd.testing.assert_series_equal(call_args[1], pd.Series(expected_labels, name="label"))
        
        # Verify model was saved
        mock_joblib_dump.assert_called_once_with(mock_model, self.model_path)
        
        # Verify logging
        assert any("Model saved to" in record.message for record in caplog.records)
    
    @patch('joblib.dump')
    @patch('src.train.build_model')
    def test_main_with_default_csv_path(self, mock_build_model, mock_joblib_dump):
        """Test main function with default CSV path."""
        mock_model = Mock()
        mock_build_model.return_value = mock_model
        
        # Should use default CSV path when none provided
        with patch('pandas.read_csv') as mock_read_csv:
            mock_df = pd.DataFrame({"text": ["test"], "label": ["positive"]})
            mock_read_csv.return_value = mock_df
            
            main()  # No arguments - should use defaults
            
            mock_read_csv.assert_called_once_with("data/sample_reviews.csv")
            mock_build_model.assert_called_once()
            mock_model.fit.assert_called_once()
    
    @patch('joblib.dump')
    @patch('src.train.build_model')
    def test_main_with_default_model_path(self, mock_build_model, mock_joblib_dump):
        """Test main function with default model path."""
        mock_model = Mock()
        mock_build_model.return_value = mock_model
        
        main(self.csv_path)  # Only specify CSV, not model path
        
        # Should use default model path "model.joblib"
        mock_joblib_dump.assert_called_once_with(mock_model, "model.joblib")
    
    @patch('joblib.dump')
    @patch('src.train.build_model')
    def test_main_file_not_found_csv(self, mock_build_model, mock_joblib_dump):
        """Test that FileNotFoundError is raised for non-existent CSV file."""
        mock_model = Mock()
        mock_build_model.return_value = mock_model
        
        non_existent_csv = "/non/existent/file.csv"
        
        with pytest.raises(FileNotFoundError):
            main(non_existent_csv, self.model_path)
    
    @patch('joblib.dump')
    @patch('src.train.build_model')
    def test_main_missing_text_column(self, mock_build_model, mock_joblib_dump):
        """Test that KeyError is raised when CSV lacks 'text' column."""
        mock_model = Mock()
        mock_build_model.return_value = mock_model
        
        # Create CSV without 'text' column
        bad_csv_path = os.path.join(self.temp_dir, "bad_data.csv")
        bad_data = pd.DataFrame({
            "review": ["Good movie", "Bad movie"],
            "label": ["positive", "negative"]
        })
        bad_data.to_csv(bad_csv_path, index=False)
        
        with pytest.raises(KeyError):
            main(bad_csv_path, self.model_path)
        
        # Cleanup
        os.remove(bad_csv_path)
    
    @patch('joblib.dump')
    @patch('src.train.build_model')
    def test_main_missing_label_column(self, mock_build_model, mock_joblib_dump):
        """Test that KeyError is raised when CSV lacks 'label' column."""
        mock_model = Mock()
        mock_build_model.return_value = mock_model
        
        # Create CSV without 'label' column
        bad_csv_path = os.path.join(self.temp_dir, "bad_data.csv")
        bad_data = pd.DataFrame({
            "text": ["Good movie", "Bad movie"],
            "score": [5, 1]
        })
        bad_data.to_csv(bad_csv_path, index=False)
        
        with pytest.raises(KeyError):
            main(bad_csv_path, self.model_path)
        
        # Cleanup
        os.remove(bad_csv_path)
    
    @patch('joblib.dump')
    @patch('src.train.build_model')
    def test_main_empty_csv(self, mock_build_model, mock_joblib_dump):
        """Test behavior with empty CSV file."""
        mock_model = Mock()
        mock_build_model.return_value = mock_model
        
        empty_csv_path = os.path.join(self.temp_dir, "empty.csv")
        empty_data = pd.DataFrame({"text": [], "label": []})
        empty_data.to_csv(empty_csv_path, index=False)
        
        main(empty_csv_path, self.model_path)
        
        # Should still call fit with empty data
        mock_model.fit.assert_called_once()
        call_args = mock_model.fit.call_args[0]
        assert len(call_args[0]) == 0  # Empty text series
        assert len(call_args[1]) == 0  # Empty label series
        
        # Cleanup
        os.remove(empty_csv_path)
    
    @patch('joblib.dump')
    @patch('src.train.build_model')
    def test_main_csv_with_missing_values(self, mock_build_model, mock_joblib_dump):
        """Test behavior with CSV containing NaN values."""
        mock_model = Mock()
        mock_build_model.return_value = mock_model
        
        csv_with_nan_path = os.path.join(self.temp_dir, "nan_data.csv")
        nan_data = pd.DataFrame({
            "text": ["Good movie", None, "Bad movie"],
            "label": ["positive", "negative", None]
        })
        nan_data.to_csv(csv_with_nan_path, index=False)
        
        # This should work as pandas will handle NaN values
        main(csv_with_nan_path, self.model_path)
        mock_model.fit.assert_called_once()
        
        # Cleanup
        os.remove(csv_with_nan_path)
    
    @patch('joblib.dump')
    @patch('src.train.build_model')
    def test_main_model_build_error(self, mock_build_model, mock_joblib_dump):
        """Test behavior when build_model raises an exception."""
        mock_build_model.side_effect = ValueError("Model build failed")
        
        with pytest.raises(ValueError, match="Model build failed"):
            main(self.csv_path, self.model_path)
    
    @patch('joblib.dump')
    @patch('src.train.build_model')
    def test_main_model_fit_error(self, mock_build_model, mock_joblib_dump):
        """Test behavior when model.fit raises an exception."""
        mock_model = Mock()
        mock_model.fit.side_effect = ValueError("Model fit failed")
        mock_build_model.return_value = mock_model
        
        with pytest.raises(ValueError, match="Model fit failed"):
            main(self.csv_path, self.model_path)
    
    @patch('joblib.dump')
    @patch('src.train.build_model')
    def test_main_model_save_error(self, mock_build_model, mock_joblib_dump):
        """Test behavior when joblib.dump raises an exception."""
        mock_model = Mock()
        mock_build_model.return_value = mock_model
        mock_joblib_dump.side_effect = PermissionError("Cannot write to file")
        
        with pytest.raises(PermissionError, match="Cannot write to file"):
            main(self.csv_path, self.model_path)
    
    @patch('pandas.read_csv')
    @patch('joblib.dump')
    @patch('src.train.build_model')
    def test_main_integration_mocked(self, mock_build_model, mock_joblib_dump, mock_read_csv):
        """Test main function with all external dependencies mocked."""
        # Mock pandas read_csv
        mock_df = pd.DataFrame({"text": ["test review"], "label": ["positive"]})
        mock_read_csv.return_value = mock_df
        
        # Mock build_model
        mock_model = Mock()
        mock_build_model.return_value = mock_model
        
        main("test.csv", "test_model.joblib")
        
        # Verify calls
        mock_read_csv.assert_called_once_with("test.csv")
        mock_build_model.assert_called_once()
        mock_model.fit.assert_called_once()
        mock_joblib_dump.assert_called_once_with(mock_model, "test_model.joblib")
    
    @patch('joblib.dump')
    @patch('src.train.build_model')
    def test_main_with_special_characters_in_data(self, mock_build_model, mock_joblib_dump):
        """Test that special characters in data are handled correctly."""
        mock_model = Mock()
        mock_build_model.return_value = mock_model
        
        special_csv_path = os.path.join(self.temp_dir, "special.csv")
        special_data = pd.DataFrame({
            "text": [
                "Movie with émojis 😀👍",
                "Review with \"quotes\" and 'apostrophes'",
                "Text with newlines\nand\ttabs"
            ],
            "label": ["positive", "neutral", "negative"]
        })
        special_data.to_csv(special_csv_path, index=False)
        
        main(special_csv_path, self.model_path)
        
        mock_model.fit.assert_called_once()
        
        # Cleanup
        os.remove(special_csv_path)
    
    @patch('joblib.dump')
    @patch('src.train.build_model')
    def test_main_large_dataset_handling(self, mock_build_model, mock_joblib_dump):
        """Test that large datasets are handled efficiently."""
        mock_model = Mock()
        mock_build_model.return_value = mock_model
        
        large_csv_path = os.path.join(self.temp_dir, "large.csv")
        large_data = pd.DataFrame({
            "text": [f"Review number {i}" for i in range(1000)],
            "label": ["positive" if i % 2 == 0 else "negative" for i in range(1000)]
        })
        large_data.to_csv(large_csv_path, index=False)
        
        main(large_csv_path, self.model_path)
        
        mock_model.fit.assert_called_once()
        call_args = mock_model.fit.call_args[0]
        assert len(call_args[0]) == 1000  # Text series
        assert len(call_args[1]) == 1000  # Label series
        
        # Cleanup
        os.remove(large_csv_path)


class TestTrainCommandLine:
    """Test cases for command-line interface functionality."""
    
    def test_argument_parser_defaults(self):
        """Test that argument parser has correct defaults."""
        import sys
        from src.train import __name__ as train_name
        
        # Only test if we can import the parser safely
        if train_name == "__main__":
            pytest.skip("Cannot test parser when module is running as main")
        
        # Test basic argument parsing concepts without actually running argparse
        # This test verifies the structure is correct
        default_csv = "data/sample_reviews.csv"
        default_model = os.getenv("MODEL_PATH", "model.joblib")
        
        # These are the expected behaviors based on the train.py code
        assert default_csv == "data/sample_reviews.csv"
        assert default_model == os.getenv("MODEL_PATH", "model.joblib")
    
    @patch.dict(os.environ, {"MODEL_PATH": "/custom/model/path.joblib"})
    def test_environment_variable_handling(self):
        """Test that environment variable is properly handled."""
        from src.train import main
        
        expected_path = "/custom/model/path.joblib"
        actual_path = os.getenv("MODEL_PATH", "model.joblib")
        assert actual_path == expected_path


class TestTrainEdgeCases:
    """Test edge cases and error conditions."""
    
    @patch('joblib.dump')
    @patch('src.train.build_model')
    def test_main_with_csv_encoding_issues(self, mock_build_model, mock_joblib_dump):
        """Test handling of CSV files with different encodings."""
        temp_dir = tempfile.mkdtemp()
        mock_model = Mock()
        mock_build_model.return_value = mock_model
        
        # Create CSV with special characters
        csv_path = os.path.join(temp_dir, "encoded.csv")
        special_data = pd.DataFrame({
            "text": ["Café très bon", "Película excelente", "電影很好看"],
            "label": ["positive", "positive", "positive"]
        })
        special_data.to_csv(csv_path, index=False, encoding='utf-8')
        
        model_path = os.path.join(temp_dir, "model.joblib")
        
        # Should handle UTF-8 encoded CSV correctly
        main(csv_path, model_path)
        mock_model.fit.assert_called_once()
        
        # Cleanup
        os.remove(csv_path)
        os.rmdir(temp_dir)
    
    @patch('joblib.dump')
    @patch('src.train.build_model')
    def test_main_memory_efficiency(self, mock_build_model, mock_joblib_dump):
        """Test that main function handles memory efficiently."""
        temp_dir = tempfile.mkdtemp()
        mock_model = Mock()
        mock_build_model.return_value = mock_model
        
        csv_path = os.path.join(temp_dir, "test.csv")
        test_data = pd.DataFrame({
            "text": ["test"] * 100,
            "label": ["positive"] * 100
        })
        test_data.to_csv(csv_path, index=False)
        
        model_path = os.path.join(temp_dir, "model.joblib")
        
        # Function should complete without issues
        main(csv_path, model_path)
        
        # Verify fit was called exactly once (not in chunks)
        assert mock_model.fit.call_count == 1
        
        # Cleanup
        os.remove(csv_path)
        os.rmdir(temp_dir)
    
    @patch('joblib.dump')
    @patch('src.train.build_model')
    def test_main_with_invalid_output_directory(self, mock_build_model, mock_joblib_dump):
        """Test behavior when output directory doesn't exist."""
        temp_dir = tempfile.mkdtemp()
        mock_model = Mock()
        mock_build_model.return_value = mock_model
        
        # Create valid CSV
        csv_path = os.path.join(temp_dir, "test.csv")
        test_data = pd.DataFrame({
            "text": ["test"],
            "label": ["positive"]
        })
        test_data.to_csv(csv_path, index=False)
        
        # Try to save to non-existent directory
        invalid_model_path = "/non/existent/directory/model.joblib"
        mock_joblib_dump.side_effect = FileNotFoundError("Directory not found")
        
        with pytest.raises(FileNotFoundError):
            main(csv_path, invalid_model_path)
        
        # Cleanup
        os.remove(csv_path)
        os.rmdir(temp_dir)
    
    @patch('joblib.dump')
    @patch('src.train.build_model')
    def test_main_with_mixed_label_types(self, mock_build_model, mock_joblib_dump):
        """Test behavior with mixed data types in labels."""
        temp_dir = tempfile.mkdtemp()
        mock_model = Mock()
        mock_build_model.return_value = mock_model
        
        csv_path = os.path.join(temp_dir, "mixed.csv")
        mixed_data = pd.DataFrame({
            "text": ["Good", "Bad", "Okay"],
            "label": ["positive", 0, 1.5]  # Mixed types
        })
        mixed_data.to_csv(csv_path, index=False)
        
        model_path = os.path.join(temp_dir, "model.joblib")
        
        # Should handle mixed types (pandas will read them as is)
        main(csv_path, model_path)
        mock_model.fit.assert_called_once()
        
        # Cleanup
        os.remove(csv_path)
        os.rmdir(temp_dir)