"""Basic text preprocessing utilities."""

import re
from typing import List

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Lowercase, remove non-alphabetic characters, and trim."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove common English stop words from token list."""
    return [t for t in tokens if t not in STOP_WORDS]
