"""Basic text preprocessing utilities."""

import re
from typing import List

try:
    import nltk
    from nltk.corpus import stopwords

    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))
except Exception:  # pragma: no cover - optional dependency
    nltk = None
    STOP_WORDS = {"the", "is", "a", "an", "this", "of", "and", "in", "on", "to"}


def clean_text(text: str) -> str:
    """Lowercase, remove non-alphabetic characters, and trim."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove common English stop words from token list."""
    return [t for t in tokens if t not in STOP_WORDS]
