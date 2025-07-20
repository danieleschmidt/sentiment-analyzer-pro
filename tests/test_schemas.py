import pytest
from pydantic import ValidationError
from src.schemas import PredictRequest, validate_columns


class TestPredictRequest:
    """Test cases for PredictRequest model validation and sanitization."""
    
    def test_valid_text_input(self):
        """Test that valid text input is accepted."""
        request = PredictRequest(text="This is a valid movie review.")
        assert request.text == "This is a valid movie review."
    
    def test_text_sanitization_script_tags(self):
        """Test that script tags are removed from input."""
        malicious_input = "Good movie <script>alert('xss')</script> great acting"
        request = PredictRequest(text=malicious_input)
        assert "<script>" not in request.text
        assert "alert" not in request.text
        assert "Good movie great acting" == request.text
    
    def test_text_sanitization_javascript_protocol(self):
        """Test that javascript: protocol is removed."""
        malicious_input = "Click here javascript:alert('xss') for review"
        request = PredictRequest(text=malicious_input)
        assert "javascript:" not in request.text
        assert "Click here alert('xss') for review" == request.text
    
    def test_text_sanitization_event_handlers(self):
        """Test that event handlers are removed."""
        malicious_input = "Movie onclick=alert('xss') was great"
        request = PredictRequest(text=malicious_input)
        assert "onclick=" not in request.text
        assert "Movie alert('xss') was great" == request.text
    
    def test_text_sanitization_case_insensitive(self):
        """Test that sanitization works regardless of case."""
        malicious_input = "Review <SCRIPT>alert('xss')</SCRIPT> here"
        request = PredictRequest(text=malicious_input)
        assert "<SCRIPT>" not in request.text
        assert "Review here" == request.text
    
    def test_text_sanitization_multiline_script(self):
        """Test that multiline script tags are removed."""
        malicious_input = """Good movie <script>
        alert('xss');
        document.location='evil.com';
        </script> loved it"""
        request = PredictRequest(text=malicious_input)
        assert "<script>" not in request.text
        assert "alert" not in request.text
        assert "Good movie loved it" == request.text
    
    def test_whitespace_normalization(self):
        """Test that excessive whitespace is normalized."""
        text_with_whitespace = "This   has     excessive\n\nwhitespace   "
        request = PredictRequest(text=text_with_whitespace)
        assert request.text == "This has excessive whitespace"
    
    def test_empty_string_validation(self):
        """Test that empty strings are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PredictRequest(text="")
        assert "at least 1 character" in str(exc_info.value)
    
    def test_whitespace_only_string_validation(self):
        """Test that whitespace-only strings become empty after sanitization and are still accepted (since min_length applies after sanitization)."""
        # This test demonstrates current behavior - whitespace-only becomes empty but is still accepted
        request = PredictRequest(text="   \n\t   ")
        assert request.text == ""
    
    def test_non_string_input_validation(self):
        """Test that non-string inputs are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PredictRequest(text=123)
        assert "Input should be a valid string" in str(exc_info.value)
    
    def test_none_input_validation(self):
        """Test that None input is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PredictRequest(text=None)
        assert "Input should be a valid string" in str(exc_info.value)
    
    def test_max_length_validation(self):
        """Test that text exceeding max length is rejected."""
        long_text = "x" * 10001  # Exceeds 10000 character limit
        with pytest.raises(ValidationError) as exc_info:
            PredictRequest(text=long_text)
        assert "at most 10000 characters" in str(exc_info.value)
    
    def test_max_length_boundary(self):
        """Test that text at max length is accepted."""
        boundary_text = "x" * 10000  # Exactly 10000 characters
        request = PredictRequest(text=boundary_text)
        assert len(request.text) == 10000
    
    def test_unicode_text_handling(self):
        """Test that unicode characters are handled correctly."""
        unicode_text = "Great movie! 😀👍 Très bon film! 电影很好看"
        request = PredictRequest(text=unicode_text)
        assert request.text == unicode_text
    
    def test_special_characters_preservation(self):
        """Test that legitimate special characters are preserved."""
        special_text = "Rating: 5/5! Cost: $15.99. Time: 2h 30m. Grade: A++"
        request = PredictRequest(text=special_text)
        assert request.text == special_text
    
    def test_complex_sanitization_scenario(self):
        """Test complex input with multiple sanitization needs."""
        complex_input = """  <script>alert('xss')</script>  
        Good movie  onclick=hack()  javascript:evil()  
        Really   enjoyed   it!   """
        request = PredictRequest(text=complex_input)
        expected = "Good movie hack() evil() Really enjoyed it!"
        assert request.text == expected


class TestValidateColumns:
    """Test cases for validate_columns function."""
    
    def test_all_required_columns_present(self):
        """Test that no exception is raised when all required columns are present."""
        columns = ["text", "label", "id", "timestamp"]
        required = ["text", "label"]
        # Should not raise any exception
        validate_columns(columns, required)
    
    def test_missing_single_column(self):
        """Test that exception is raised for single missing column."""
        columns = ["text", "id", "timestamp"]
        required = ["text", "label"]
        with pytest.raises(ValueError) as exc_info:
            validate_columns(columns, required)
        assert "Missing required columns: label" in str(exc_info.value)
    
    def test_missing_multiple_columns(self):
        """Test that exception is raised for multiple missing columns."""
        columns = ["id", "timestamp"]
        required = ["text", "label", "score"]
        with pytest.raises(ValueError) as exc_info:
            validate_columns(columns, required)
        error_message = str(exc_info.value)
        assert "Missing required columns:" in error_message
        assert "text" in error_message
        assert "label" in error_message
        assert "score" in error_message
    
    def test_empty_required_columns(self):
        """Test that no exception is raised when no columns are required."""
        columns = ["text", "label"]
        required = []
        # Should not raise any exception
        validate_columns(columns, required)
    
    def test_empty_available_columns(self):
        """Test that exception is raised when no columns are available but some are required."""
        columns = []
        required = ["text", "label"]
        with pytest.raises(ValueError) as exc_info:
            validate_columns(columns, required)
        assert "Missing required columns: text, label" in str(exc_info.value)
    
    def test_exact_match_columns(self):
        """Test that no exception is raised when columns exactly match requirements."""
        columns = ["text", "label"]
        required = ["text", "label"]
        # Should not raise any exception
        validate_columns(columns, required)
    
    def test_case_sensitive_column_names(self):
        """Test that column validation is case-sensitive."""
        columns = ["Text", "Label"]  # Capital case
        required = ["text", "label"]  # Lower case
        with pytest.raises(ValueError) as exc_info:
            validate_columns(columns, required)
        assert "Missing required columns: text, label" in str(exc_info.value)
    
    def test_duplicate_columns_handling(self):
        """Test behavior with duplicate column names."""
        columns = ["text", "text", "label"]  # Duplicate 'text'
        required = ["text", "label"]
        # Should not raise any exception - duplicates don't matter for validation
        validate_columns(columns, required)
    
    def test_extra_columns_allowed(self):
        """Test that extra columns beyond requirements are allowed."""
        columns = ["text", "label", "id", "timestamp", "extra1", "extra2"]
        required = ["text", "label"]
        # Should not raise any exception
        validate_columns(columns, required)
    
    @pytest.mark.parametrize("columns,required", [
        (["text"], ["text"]),
        (["a", "b", "c"], ["b"]),
        (["col1", "col2", "col3"], ["col1", "col3"]),
    ])
    def test_parametrized_valid_cases(self, columns, required):
        """Test multiple valid column scenarios using parametrization."""
        # Should not raise any exception
        validate_columns(columns, required)
    
    @pytest.mark.parametrize("columns,required,missing", [
        ([], ["text"], "text"),
        (["label"], ["text", "label"], "text"),
        (["other"], ["text", "label"], "text, label"),
    ])
    def test_parametrized_invalid_cases(self, columns, required, missing):
        """Test multiple invalid column scenarios using parametrization."""
        with pytest.raises(ValueError) as exc_info:
            validate_columns(columns, required)
        assert f"Missing required columns: {missing}" in str(exc_info.value)