import nltk
import string

# Download nltk resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True) # Added quiet=True for cleaner output


def basic_preprocess(text):
    """Converts text to lowercase, removes punctuation, and tokenizes it using NLTK.

    This function performs basic text cleaning steps essential for many NLP tasks.
    Punctuation is removed based on `string.punctuation`. Tokenization is done
    using `nltk.word_tokenize`.

    Args:
        text (str): The input text string to preprocess.

    Returns:
        list[str]: A list of processed tokens. Returns an empty list if the input
                   is None or not a string, or if the text becomes empty after
                   preprocessing.

    Example:
        >>> basic_preprocess("Hello, World! This is a test.")
        ['hello', 'world', 'this', 'is', 'a', 'test']
        >>> basic_preprocess("SOME UPPERCASE TEXT!!!")
        ['some', 'uppercase', 'text']
        >>> basic_preprocess("")
        []
        >>> basic_preprocess(None) # Should ideally handle or document this
        [] # Current behavior based on implementation
    """
    if not isinstance(text, str): # Handle None or non-string inputs
        return []
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    if not text.strip(): # Handle cases where text becomes empty after cleaning
        return []
    tokens = nltk.word_tokenize(text)
    return tokens
