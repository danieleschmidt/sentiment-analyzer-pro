"""
SentimentAnalyzerPro: Orchestrates all analysis components into a unified result.

Combines:
- LexiconSentiment: overall polarity scoring
- AspectExtractor: per-aspect sentiment
- EmotionClassifier: emotion detection

Returns a structured SentimentResult dataclass.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .lexicon import LexiconSentiment, LexiconResult
from .aspects import AspectExtractor, AspectResult
from .emotions import EmotionClassifier, EmotionResult


@dataclass
class SentimentResult:
    """Structured output from SentimentAnalyzerPro."""

    # Overall sentiment
    text: str
    sentiment: str            # "positive", "negative", "neutral"
    score: float              # compound score: -1.0 to +1.0
    confidence: float         # 0.0 to 1.0

    # Lexicon details
    lexicon: LexiconResult

    # Aspect-level breakdown
    aspects: Dict[str, float]  # {aspect_name: score}
    aspect_labels: Dict[str, str]  # {aspect_name: label}

    # Emotion breakdown
    primary_emotion: Optional[str]
    secondary_emotion: Optional[str]
    emotion_valence: str
    emotion_scores: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary (JSON-serializable)."""
        return {
            "text": self.text,
            "sentiment": self.sentiment,
            "score": self.score,
            "confidence": self.confidence,
            "lexicon": {
                "score": self.lexicon.score,
                "positive": self.lexicon.positive,
                "negative": self.lexicon.negative,
                "neutral": self.lexicon.neutral,
                "label": self.lexicon.label,
            },
            "aspects": self.aspects,
            "aspect_labels": self.aspect_labels,
            "emotions": {
                "primary": self.primary_emotion,
                "secondary": self.secondary_emotion,
                "valence": self.emotion_valence,
                "scores": self.emotion_scores,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Human-readable one-liner summary."""
        parts = [f"[{self.sentiment.upper()}] score={self.score:+.3f}"]
        if self.primary_emotion:
            parts.append(f"emotion={self.primary_emotion}")
        if self.aspects:
            aspect_str = ", ".join(
                f"{k}={'↑' if v >= 0.05 else '↓' if v <= -0.05 else '~'}{abs(v):.2f}"
                for k, v in list(self.aspects.items())[:5]
            )
            parts.append(f"aspects=[{aspect_str}]")
        return " | ".join(parts)


class SentimentAnalyzerPro:
    """
    Multi-level sentiment analysis orchestrator.

    Usage:
        analyzer = SentimentAnalyzerPro()
        result = analyzer.analyze("The battery life is excellent but the camera is disappointing.")
        print(result.summary())
        print(result.to_json())
    """

    def __init__(
        self,
        lexicon_weight: float = 0.5,
        aspect_window: int = 6,
    ):
        """
        Args:
            lexicon_weight: Weight of lexicon score vs aspect scores in
                            computing confidence (0.0 to 1.0).
            aspect_window: Token window for aspect-sentiment association.
        """
        self._lexicon = LexiconSentiment()
        self._aspects = AspectExtractor(window=aspect_window)
        self._emotions = EmotionClassifier()
        self._lexicon_weight = lexicon_weight

    def analyze(self, text: str) -> SentimentResult:
        """
        Run full multi-level sentiment analysis on text.

        Args:
            text: Input text to analyze.

        Returns:
            SentimentResult with all analysis layers populated.
        """
        # --- Layer 1: Overall lexicon scoring ---
        lex_result = self._lexicon.score(text)

        # --- Layer 2: Aspect extraction ---
        aspect_result = self._aspects.extract(text)
        aspects_scores = {k: v.score for k, v in aspect_result.aspects.items()}
        aspects_labels = {k: v.label for k, v in aspect_result.aspects.items()}

        # --- Layer 3: Emotion classification ---
        emo_result = self._emotions.classify(text)

        # --- Synthesize overall sentiment ---
        # Primary signal: lexicon compound score
        # If aspects exist, adjust by their average
        if aspects_scores:
            aspect_avg = sum(aspects_scores.values()) / len(aspects_scores)
            combined_score = (
                self._lexicon_weight * lex_result.score
                + (1 - self._lexicon_weight) * aspect_avg
            )
        else:
            combined_score = lex_result.score

        combined_score = round(max(-1.0, min(1.0, combined_score)), 4)

        if combined_score >= 0.05:
            sentiment = "positive"
        elif combined_score <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        # Confidence: how strongly the score differs from 0, boosted by emotion signal
        lex_confidence = abs(combined_score)
        emo_confidence = 0.0
        if emo_result.primary:
            top_emo = emo_result.emotions[emo_result.primary].score
            emo_confidence = top_emo * 0.3  # emotion adds up to 30% confidence
        confidence = round(min(1.0, lex_confidence + emo_confidence), 4)

        return SentimentResult(
            text=text,
            sentiment=sentiment,
            score=combined_score,
            confidence=confidence,
            lexicon=lex_result,
            aspects=aspects_scores,
            aspect_labels=aspects_labels,
            primary_emotion=emo_result.primary,
            secondary_emotion=emo_result.secondary,
            emotion_valence=emo_result.valence,
            emotion_scores=emo_result.all_scores,
        )

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze a list of texts."""
        return [self.analyze(t) for t in texts]
