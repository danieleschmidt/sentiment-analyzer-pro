import pytest
from src.preprocessing import basic_preprocess
import nltk

# Ensure NLTK 'punkt' tokenizer is available for tests
# This might be redundant if already handled in preprocessing.py,
# but good for test environment explicitness.
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)

def test_basic_preprocess_empty_string():
    """Test preprocessing an empty string."""
    assert basic_preprocess("") == []

def test_basic_preprocess_none_input():
    """Test preprocessing with None as input."""
    assert basic_preprocess(None) == []

def test_basic_preprocess_only_punctuation():
    """Test preprocessing a string with only punctuation."""
    assert basic_preprocess("!!! ??? ...") == []

def test_basic_preprocess_only_punctuation_and_spaces():
    """Test preprocessing a string with only punctuation and spaces."""
    assert basic_preprocess("  !!! ??? ...  ") == []

def test_basic_preprocess_lowercase_and_punctuation_removal():
    """Test lowercase conversion and punctuation removal."""
    text = "Hello, World! This is a Test."
    expected = ["hello", "world", "this", "is", "a", "test"]
    assert basic_preprocess(text) == expected

def test_basic_preprocess_no_punctuation():
    """Test preprocessing a string with no punctuation."""
    text = "This is a simple sentence"
    expected = ["this", "is", "a", "simple", "sentence"]
    assert basic_preprocess(text) == expected

def test_basic_preprocess_mixed_case():
    """Test preprocessing with mixed case text."""
    text = "MiXeD CaSe TeXt"
    expected = ["mixed", "case", "text"]
    assert basic_preprocess(text) == expected

def test_basic_preprocess_numbers_and_text():
    """Test preprocessing with numbers and text (numbers are preserved)."""
    text = "Product 123 is great, model 456 is not."
    expected = ["product", "123", "is", "great", "model", "456", "is", "not"]
    assert basic_preprocess(text) == expected

def test_basic_preprocess_leading_trailing_spaces():
    """Test preprocessing with leading/trailing spaces."""
    text = "  lots of spaces before and after  "
    expected = ["lots", "of", "spaces", "before", "and", "after"]
    assert basic_preprocess(text) == expected

def test_basic_preprocess_internal_multiple_spaces():
    """Test preprocessing with multiple internal spaces (NLTK tokenizer handles this)."""
    text = "Text   with    multiple     spaces"
    expected = ["text", "with", "multiple", "spaces"]
    assert basic_preprocess(text) == expected

def test_basic_preprocess_special_characters_within_words():
    """Test preprocessing with special characters that might be part of words (NLTK specific)."""
    # NLTK's default word_tokenize might split on hyphens depending on context
    # or keep them. This test documents current behavior.
    # string.punctuation includes hyphen, so it will be removed.
    text = "Well-being is important. It's a test-case."
    expected = ["wellbeing", "is", "important", "its", "a", "testcase"]
    assert basic_preprocess(text) == expected

def test_basic_preprocess_unicode_text():
    """Test preprocessing with simple unicode text (e.g., accented characters)."""
    # Lowercasing of unicode is locale-dependent. NLTK handles basic cases.
    # Punctuation removal is ASCII-based by default with string.punctuation.
    text = "Café crème is good. ¡Hola!"
    # Note: ¡ is in string.punctuation. Accented chars are lowercased.
    expected = ["café", "crème", "is", "good", "hola"]
    assert basic_preprocess(text) == expected

# To run these tests, navigate to the project root and run:
# python -m pytest
# (Assuming pytest is installed and src directory is importable)
