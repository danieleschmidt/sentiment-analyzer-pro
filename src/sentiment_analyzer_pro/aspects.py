"""
AspectExtractor: Identifies aspects/entities in text and their associated sentiment.

Approach:
- Aspect vocabulary: domain-specific noun phrases (product, service, etc.)
- Proximity-based sentiment association: sentiment words near an aspect
  contribute to that aspect's score
- Handles contrastive connectors ("but", "however") to separate sentiment blocks

Example:
    "Battery life is great but camera is poor"
    → {battery_life: 0.7, camera: -0.65}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .lexicon import LexiconSentiment, LEXICON


# ---------------------------------------------------------------------------
# Aspect vocabulary: maps surface forms → canonical aspect names
# Covers consumer products, software, services, and general topics
# ---------------------------------------------------------------------------
ASPECT_VOCABULARY: Dict[str, str] = {
    # Hardware / devices
    "battery life": "battery_life", "battery": "battery", "charge": "charging",
    "charging": "charging", "screen": "display", "display": "display",
    "monitor": "display", "camera": "camera", "photo": "camera", "photos": "camera",
    "picture": "camera", "pictures": "camera", "keyboard": "keyboard",
    "trackpad": "trackpad", "touchpad": "trackpad", "speaker": "audio",
    "speakers": "audio", "audio": "audio", "sound": "audio", "microphone": "audio",
    "mic": "audio", "headphones": "audio", "earbuds": "audio",
    "processor": "performance", "cpu": "performance", "gpu": "performance",
    "ram": "performance", "memory": "performance", "storage": "storage",
    "ssd": "storage", "hard drive": "storage", "disk": "storage",
    "build quality": "build_quality", "build": "build_quality",
    "design": "design", "look": "design", "appearance": "design",
    "weight": "portability", "size": "portability", "portability": "portability",
    "port": "connectivity", "ports": "connectivity", "usb": "connectivity",
    "bluetooth": "connectivity", "wifi": "connectivity", "wi-fi": "connectivity",
    "connection": "connectivity", "network": "connectivity",
    # Software / apps
    "app": "app", "application": "app", "software": "software",
    "interface": "ui", "ui": "ui", "ux": "ui", "user interface": "ui",
    "user experience": "ui", "design": "design", "layout": "ui",
    "navigation": "ui", "menu": "ui", "settings": "settings",
    "performance": "performance", "speed": "performance", "loading": "performance",
    "startup": "performance", "boot": "performance", "response": "performance",
    "feature": "features", "features": "features", "functionality": "features",
    "function": "features", "update": "updates", "updates": "updates",
    "bug": "stability", "bugs": "stability", "crash": "stability",
    "crashes": "stability", "stability": "stability", "reliability": "reliability",
    "security": "security", "privacy": "security", "encryption": "security",
    "installation": "setup", "setup": "setup", "configuration": "setup",
    "documentation": "documentation", "docs": "documentation",
    "support": "support", "help": "support", "customer service": "customer_service",
    # Service / business
    "customer service": "customer_service", "service": "service",
    "staff": "staff", "employee": "staff", "team": "staff",
    "management": "management", "manager": "management",
    "shipping": "shipping", "delivery": "shipping", "packaging": "packaging",
    "price": "price", "pricing": "price", "cost": "price", "value": "value",
    "quality": "quality", "durability": "durability",
    "warranty": "warranty", "return": "returns", "returns": "returns",
    "refund": "returns", "checkout": "checkout", "payment": "payment",
    # Food / restaurant
    "food": "food", "meal": "food", "dish": "food", "taste": "taste",
    "flavor": "taste", "flavour": "taste", "portion": "portion",
    "menu": "menu", "drinks": "drinks", "drink": "drinks",
    "ambiance": "ambiance", "atmosphere": "ambiance", "decor": "ambiance",
    "noise": "ambiance", "location": "location", "parking": "parking",
    "cleanliness": "cleanliness", "clean": "cleanliness",
    # General
    "instructions": "instructions", "manual": "instructions",
    "content": "content", "information": "content", "data": "content",
}

# Contrastive connectors that split sentiment regions
CONTRASTIVE = {"but", "however", "although", "though", "yet", "except",
               "whereas", "while", "despite", "nevertheless", "nonetheless",
               "still", "on the other hand", "that said"}


@dataclass
class AspectSentiment:
    aspect: str
    score: float          # -1.0 to +1.0
    label: str            # positive / negative / neutral
    mentions: List[str]   # surface forms found
    evidence: List[str]   # nearby sentiment words


@dataclass
class AspectResult:
    aspects: Dict[str, AspectSentiment] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        return {k: v.score for k, v in self.aspects.items()}


class AspectExtractor:
    """
    Identifies aspects in text and scores their sentiment.

    Usage:
        extractor = AspectExtractor()
        result = extractor.extract("Battery life is great but camera is poor")
        print(result.to_dict())  # {'battery_life': 0.7, 'camera': -0.65}
    """

    def __init__(self, window: int = 6):
        """
        Args:
            window: Number of tokens on each side of an aspect to consider
                    for sentiment association.
        """
        self.window = window
        self._scorer = LexiconSentiment()
        self._aspects = ASPECT_VOCABULARY
        # Build multi-word aspect patterns sorted by length (longest first)
        self._mw_aspects = sorted(
            [(k, v) for k, v in self._aspects.items() if " " in k],
            key=lambda x: -len(x[0])
        )

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().strip())

    def _find_aspects(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Find aspect mentions in text. Returns list of (start_tok, end_tok, canonical).
        Works on token-level after normalizing.
        """
        import string as _string
        normalized = self._normalize(text)
        raw_tokens = normalized.split()
        # Strip punctuation from each token for matching, but keep original for position tracking
        tokens = [t.strip(_string.punctuation) for t in raw_tokens]
        found: List[Tuple[int, int, str]] = []
        covered: set = set()

        # First pass: multi-word aspects
        for surface, canonical in self._mw_aspects:
            words = surface.split()
            n = len(words)
            for i in range(len(tokens) - n + 1):
                if tokens[i:i + n] == words and not any(j in covered for j in range(i, i + n)):
                    found.append((i, i + n, canonical))
                    covered.update(range(i, i + n))

        # Second pass: single-word aspects
        for i, token in enumerate(tokens):
            if i not in covered and token in self._aspects:
                canonical = self._aspects[token]
                found.append((i, i + 1, canonical))
                covered.add(i)

        return found

    def _split_at_contrastives(self, tokens: List[str]) -> List[Tuple[int, int]]:
        """Return (start, end) index pairs for each semantic block."""
        splits = [0]
        for i, tok in enumerate(tokens):
            if tok in CONTRASTIVE:
                splits.append(i + 1)
        splits.append(len(tokens))
        return [(splits[i], splits[i + 1]) for i in range(len(splits) - 1)]

    def _score_window(self, tokens: List[str], center_start: int,
                      center_end: int, block_start: int, block_end: int) -> Tuple[float, List[str]]:
        """Score the sentiment of tokens around an aspect within its block."""
        import re as _re, string as _string
        left = max(block_start, center_start - self.window)
        right = min(block_end, center_end + self.window)

        context_tokens = tokens[left:center_start] + tokens[center_end:right]
        # Strip punctuation for lexicon lookup (tokens may have trailing commas etc.)
        clean_context = [t.strip(_string.punctuation) for t in context_tokens]
        sentiment_words = [t for t in clean_context if t in LEXICON]

        # Always score via the scorer (which handles its own tokenization),
        # but only return if there's actual sentiment signal or the window isn't empty.
        text_context = " ".join(tokens[left:right])
        if not text_context.strip():
            return 0.0, []

        result = self._scorer.score(text_context)
        return result.score, sentiment_words

    def extract(self, text: str) -> AspectResult:
        """Extract aspects and their sentiment from text."""
        normalized = self._normalize(text)
        tokens = normalized.split()
        blocks = self._split_at_contrastives(tokens)
        aspect_positions = self._find_aspects(text)

        aspect_data: Dict[str, Tuple[List[float], List[str], List[str]]] = {}

        for start, end, canonical in aspect_positions:
            # Find which block this aspect belongs to
            block = (0, len(tokens))
            for bs, be in blocks:
                if bs <= start < be:
                    block = (bs, be)
                    break

            score, evidence = self._score_window(tokens, start, end, *block)
            surface = " ".join(tokens[start:end])

            if canonical not in aspect_data:
                aspect_data[canonical] = ([], [surface], evidence)
            else:
                scores, surfaces, evs = aspect_data[canonical]
                if surface not in surfaces:
                    surfaces.append(surface)
                evs.extend(e for e in evidence if e not in evs)

            aspect_data[canonical][0].append(score)

        result = AspectResult()
        for canonical, (scores, surfaces, evidence) in aspect_data.items():
            avg_score = sum(scores) / len(scores) if scores else 0.0
            avg_score = round(avg_score, 4)
            if avg_score >= 0.05:
                label = "positive"
            elif avg_score <= -0.05:
                label = "negative"
            else:
                label = "neutral"

            result.aspects[canonical] = AspectSentiment(
                aspect=canonical,
                score=avg_score,
                label=label,
                mentions=surfaces,
                evidence=evidence,
            )

        return result

    def extract_batch(self, texts: List[str]) -> List[AspectResult]:
        return [self.extract(t) for t in texts]
