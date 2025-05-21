import nltk
import string
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag # For POS tagging in lemmatization

# --- NLTK Resource Management ---
NLTK_RESOURCES = {
    'tokenizers/punkt': 'punkt',
    'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger',
    'corpora/wordnet': 'wordnet',
    'corpora/stopwords': 'stopwords'
}

def download_nltk_resources(quiet=True):
    """Downloads required NLTK resources if they are not already present."""
    for resource_path, resource_id in NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource_path)
        except nltk.downloader.DownloadError:
            print(f"NLTK resource '{resource_id}' not found. Downloading...")
            nltk.download(resource_id, quiet=quiet)
            print(f"NLTK resource '{resource_id}' downloaded.")
        # else:
            # print(f"NLTK resource '{resource_id}' already available.")

# Call download function once when the module is loaded.
download_nltk_resources()

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts."""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": nltk.corpus.wordnet.ADJ,
                "N": nltk.corpus.wordnet.NOUN,
                "V": nltk.corpus.wordnet.VERB,
                "R": nltk.corpus.wordnet.ADV}
    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN) # Default to noun

def basic_preprocess(text):
    """Converts text to lowercase, removes punctuation, and tokenizes it using NLTK.
    (Existing function - ensure it's still here or its functionality is covered)
    Args:
        text (str): The input text string to preprocess.
    Returns:
        list[str]: A list of processed tokens.
    """
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    if not text.strip():
        return []
    tokens = nltk.word_tokenize(text)
    return tokens

def enhanced_preprocess(text, use_stemming=False, use_lemmatization=False, custom_stopwords_list=None):
    """Enhanced text preprocessing with stemming, lemmatization, and stopword control.

    Args:
        text (str): The input text.
        use_stemming (bool, optional): Apply Porter Stemmer. Defaults to False.
        use_lemmatization (bool, optional): Apply WordNet Lemmatizer.
                                           Overrides stemming if both are True. Defaults to False.
        custom_stopwords_list (list[str], optional): List of custom stopwords.
                                                If None, NLTK's English stopwords are used.
                                                If [], no stopwords are removed.
                                                Defaults to None.

    Returns:
        list[str]: A list of processed tokens.
    """
    if not isinstance(text, str):
        return []

    # 1. Lowercase
    text_lc = text.lower() # Renamed to avoid conflict with text parameter

    # 2. Remove punctuation
    text_punc = text_lc.translate(str.maketrans('', '', string.punctuation)) # Renamed

    # 3. Tokenize
    tokens = nltk.word_tokenize(text_punc) # Use text_punc
    
    # Handle empty tokens list after tokenization
    if not tokens:
        return []

    # 4. Stopword removal
    if custom_stopwords_list is None: # Use NLTK default
        # Ensure stopwords resource is available (should be by download_nltk_resources)
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
        except LookupError:
            print("Warning: NLTK 'english' stopwords not found. Skipping stopword removal.")
            # tokens remain as is
    elif isinstance(custom_stopwords_list, list) and len(custom_stopwords_list) > 0: # Use custom list
        stop_words = set(custom_stopwords_list)
        tokens = [token for token in tokens if token not in stop_words]
    # If custom_stopwords_list is an empty list [], no stopword removal is performed.
    
    # Handle empty tokens list after stopword removal
    if not tokens:
        return []

    # 5. Stemming or Lemmatization
    # Lemmatization takes precedence if both are True
    if use_lemmatization:
        # Ensure wordnet and averaged_perceptron_tagger are available
        try:
            nltk.corpus.wordnet.ensure_loaded()
            pos_tag(['test']) # Check if POS tagger is available
        except LookupError as e:
            print(f"Warning: Required NLTK resource for lemmatization ('{e}') not found. Skipping lemmatization.")
        else:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]
    elif use_stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
            
    # Remove any empty strings that might have resulted from preprocessing
    tokens = [token for token in tokens if token]

    return tokens
