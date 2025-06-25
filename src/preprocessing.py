"""Basic text preprocessing utilities."""

import re
from typing import List

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))
    _lemmatizer = WordNetLemmatizer()
except Exception:  # pragma: no cover - optional dependency
    nltk = None
    _lemmatizer = None
    STOP_WORDS = {"the", "is", "a", "an", "this", "of", "and", "in", "on", "to"}


def clean_text(text: str) -> str:
    """Lowercase, remove non-alphabetic characters, and trim."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove common English stop words from token list."""
    return [t for t in tokens if t not in STOP_WORDS]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Return lemmatized tokens."""
    if _lemmatizer is None:
        # simple fallback without nltk
        def _fallback(word: str) -> str:
            for suffix in ("ing", "ed", "s"):
                if word.endswith(suffix) and len(word) > len(suffix):
                    stem = word[: -len(suffix)]
                    if len(stem) >= 2 and stem[-1] == stem[-2]:
                        stem = stem[:-1]
                    return stem
            return word

        return [_fallback(t) for t in tokens]

    try:
        return [_lemmatizer.lemmatize(t, "v") for t in tokens]
    except LookupError:  # pragma: no cover - missing wordnet
        return tokens
