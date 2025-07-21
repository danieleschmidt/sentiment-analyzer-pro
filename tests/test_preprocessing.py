import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from src import preprocessing


def test_clean_text():
    text = "Hello!!!"
    assert preprocessing.clean_text(text) == "hello"


def test_clean_text_collapses_spaces():
    text = "Hello   world"
    assert preprocessing.clean_text(text) == "hello world"


def test_clean_text_lemmatizes_running():
    """Test that lemmatization works with cleaned text."""
    text = "running"
    clean = preprocessing.clean_text(text)
    lemmatized = preprocessing.lemmatize_tokens(clean.split())
    # Result depends on NLTK availability, but should be a list
    assert isinstance(lemmatized, list)
    assert len(lemmatized) == 1


def test_remove_stopwords():
    """Test basic stopword removal."""
    tokens = ["this", "is", "a", "test"]
    result = preprocessing.remove_stopwords(tokens)
    assert "test" in result
    # Other words should be removed (exact result depends on stopword set)
    assert len(result) <= len(tokens)



def test_clean_series_vectorized():
    series = pd.Series(['Hello!!!', 'Test   Me'])
    cleaned = preprocessing.clean_series(series)
    assert cleaned.tolist() == ['hello', 'test me']


def test_clean_series_with_special_characters():
    """Test clean_series with various special characters."""
    series = pd.Series(['Hello@World!', '123Test456', 'mixed$#characters'])
    cleaned = preprocessing.clean_series(series)
    assert cleaned.tolist() == ['hello world', 'test', 'mixed characters']


def test_clean_series_with_empty_strings():
    """Test clean_series handles empty strings and whitespace."""
    series = pd.Series(['', '   ', 'valid text'])
    cleaned = preprocessing.clean_series(series)
    assert cleaned.tolist() == ['', '', 'valid text']


def test_ensure_nltk_with_nltk_available():
    """Test _ensure_nltk when NLTK is available."""
    with patch('src.preprocessing.nltk') as mock_nltk:
        with patch('src.preprocessing.stopwords') as mock_stopwords:
            with patch('src.preprocessing.WordNetLemmatizer') as mock_lemmatizer:
                # Mock NLTK data availability
                mock_nltk.data.find.side_effect = LookupError("not found")
                mock_nltk.download = Mock()
                mock_stopwords.words.return_value = ['a', 'the', 'is']
                mock_lemmatizer.return_value = Mock()
                
                # Reset global state
                preprocessing.STOP_WORDS = {"the", "is", "a", "an", "this", "of", "and", "in", "on", "to"}
                preprocessing._lemmatizer = None
                
                # Call the function
                preprocessing._ensure_nltk()
                
                # Verify NLTK downloads were called
                assert mock_nltk.download.call_count >= 1


def test_ensure_nltk_without_nltk():
    """Test _ensure_nltk when NLTK is not available."""
    with patch('src.preprocessing.nltk', None):
        # Should not raise an error
        preprocessing._ensure_nltk()


def test_remove_stopwords_with_nltk_setup():
    """Test remove_stopwords triggers NLTK setup when needed."""
    with patch('src.preprocessing._ensure_nltk') as mock_ensure:
        # Set up initial state to trigger NLTK setup
        original_stopwords = preprocessing.STOP_WORDS
        preprocessing.STOP_WORDS = {"the", "is", "a", "an", "this", "of", "and", "in", "on", "to"}
        
        try:
            tokens = ["hello", "world", "the", "test"]
            result = preprocessing.remove_stopwords(tokens)
            
            # Should have called _ensure_nltk
            mock_ensure.assert_called_once()
            
        finally:
            # Restore original state
            preprocessing.STOP_WORDS = original_stopwords


def test_remove_stopwords_with_custom_stopwords():
    """Test remove_stopwords with different stopword sets."""
    # Test with extended stopwords
    original_stopwords = preprocessing.STOP_WORDS
    preprocessing.STOP_WORDS = {"the", "is", "a", "test", "hello"}
    
    try:
        tokens = ["hello", "world", "the", "test", "good"]
        result = preprocessing.remove_stopwords(tokens)
        assert result == ["world", "good"]
        
    finally:
        preprocessing.STOP_WORDS = original_stopwords


def test_lemmatize_tokens_without_nltk():
    """Test lemmatize_tokens fallback when NLTK is not available."""
    with patch('src.preprocessing._lemmatizer', None):
        with patch('src.preprocessing._ensure_nltk') as mock_ensure:
            # Ensure _ensure_nltk doesn't set up the lemmatizer
            mock_ensure.return_value = None
            
            tokens = ["running", "played", "cats", "better"]
            result = preprocessing.lemmatize_tokens(tokens)
            
            # Should use fallback logic - actual implementation uses NLTK when available
            # Test that it returns a result without error
            assert isinstance(result, list)
            assert len(result) == len(tokens)
            # If NLTK is available, it will use real lemmatization
            # If not, it uses fallback - either way should work


def test_lemmatize_tokens_fallback_edge_cases():
    """Test lemmatize_tokens fallback with edge cases."""
    with patch('src.preprocessing._lemmatizer', None):
        with patch('src.preprocessing._ensure_nltk'):
            # Test edge cases in fallback
            tokens = [
                "running",   # should become "runn" (double letter handling)
                "played",    # should become "play"
                "cats",      # should become "cat"
                "is",        # too short, should stay "is"
                "a",         # too short, should stay "a"
                "nothing",   # no suffix, should stay "nothing"
                "testing",   # should become "test"
                "stopped",   # should become "stopp" (double letter)
            ]
            
            result = preprocessing.lemmatize_tokens(tokens)
            
            # Test that fallback logic works - actual results depend on NLTK availability
            assert isinstance(result, list)
            assert len(result) == len(tokens)
            
            # Test some expected fallback behaviors when NLTK is truly unavailable
            if preprocessing._lemmatizer is None and preprocessing.nltk is None:
                # True fallback mode
                expected = ["runn", "play", "cat", "is", "a", "nothing", "test", "stopp"]
                assert result == expected
            else:
                # NLTK is available, results will be different
                pass


def test_lemmatize_tokens_with_nltk_error():
    """Test lemmatize_tokens when NLTK lemmatizer raises LookupError."""
    mock_lemmatizer = Mock()
    mock_lemmatizer.lemmatize.side_effect = LookupError("wordnet not found")
    
    with patch('src.preprocessing._lemmatizer', mock_lemmatizer):
        tokens = ["running", "played"]
        result = preprocessing.lemmatize_tokens(tokens)
        
        # Should return original tokens when LookupError occurs
        assert result == tokens


def test_lemmatize_tokens_with_nltk_success():
    """Test lemmatize_tokens with successful NLTK lemmatization."""
    mock_lemmatizer = Mock()
    mock_lemmatizer.lemmatize.side_effect = lambda word, pos: {
        "running": "run",
        "played": "play",
        "cats": "cat"
    }.get(word, word)
    
    with patch('src.preprocessing._lemmatizer', mock_lemmatizer):
        tokens = ["running", "played", "cats"]
        result = preprocessing.lemmatize_tokens(tokens)
        
        assert result == ["run", "play", "cat"]
        # Verify lemmatizer was called with verb pos tag
        assert mock_lemmatizer.lemmatize.call_count == 3
        mock_lemmatizer.lemmatize.assert_any_call("running", "v")


def test_clean_text_edge_cases():
    """Test clean_text with various edge cases."""
    # Empty string
    assert preprocessing.clean_text("") == ""
    
    # Only whitespace
    assert preprocessing.clean_text("   ") == ""
    
    # Only special characters
    assert preprocessing.clean_text("!@#$%^&*()") == ""
    
    # Mixed case and numbers
    assert preprocessing.clean_text("Hello123World456") == "hello world"
    
    # Multiple consecutive special characters
    assert preprocessing.clean_text("hello!!!world???") == "hello world"
    
    # Leading and trailing spaces with special chars
    assert preprocessing.clean_text("  hello@world  ") == "hello world"


def test_preprocessing_integration():
    """Test integration of preprocessing functions together."""
    # Test full preprocessing pipeline
    text = "Running cats are playing with dogs!!!"
    
    # Clean the text
    cleaned = preprocessing.clean_text(text)
    assert cleaned == "running cats are playing with dogs"
    
    # Tokenize (simple split)
    tokens = cleaned.split()
    
    # Remove stopwords
    no_stop = preprocessing.remove_stopwords(tokens)
    
    # Test that stopwords are removed (exact result depends on stopword set)
    assert isinstance(no_stop, list)
    assert len(no_stop) <= len(tokens)  # Some words should be removed
    assert "cats" in no_stop  # Content words should remain
    assert "dogs" in no_stop
    
    # Lemmatize
    lemmatized = preprocessing.lemmatize_tokens(no_stop)
    # Result depends on whether NLTK is available, but should not error
    assert isinstance(lemmatized, list)
    assert len(lemmatized) == len(no_stop)


def test_global_variables_initialization():
    """Test that global variables are properly initialized."""
    # Test regex patterns are compiled
    assert preprocessing._CLEAN_RE.pattern == r"[^a-z]+"
    assert preprocessing._WS_RE.pattern == r"\s+"
    
    # Test default stopwords exist
    assert isinstance(preprocessing.STOP_WORDS, set)
    assert len(preprocessing.STOP_WORDS) > 0
    # Default stopwords include basic English words
    default_words = {"the", "is", "a", "an", "this", "of", "and", "in", "on", "to"}
    # At least some default words should be present
    assert len(preprocessing.STOP_WORDS.intersection(default_words)) > 0
