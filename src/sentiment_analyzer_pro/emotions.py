"""
EmotionClassifier: Maps text to 8 basic emotions using keyword patterns.

Based on Plutchik's Wheel of Emotions:
- Joy, Sadness, Anger, Fear, Surprise, Disgust, Trust, Anticipation

Each emotion has a set of trigger words/phrases. The classifier scores
each emotion by keyword density and returns the top emotions with scores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Emotion keyword vocabulary
# ---------------------------------------------------------------------------
EMOTION_PATTERNS: Dict[str, Set[str]] = {
    "joy": {
        "happy", "happiness", "joy", "joyful", "joyous", "glad", "pleased",
        "delighted", "ecstatic", "elated", "thrilled", "excited", "cheerful",
        "content", "satisfied", "love", "loved", "adore", "wonderful",
        "fantastic", "amazing", "great", "excellent", "awesome", "brilliant",
        "marvelous", "incredible", "smile", "smiling", "laugh", "laughing",
        "laughter", "fun", "enjoy", "enjoyed", "enjoying", "celebrate",
        "celebration", "cheer", "jubilant", "bliss", "blissful", "euphoric",
        "radiant", "beam", "glee", "gleeful", "merry", "jolly", "festive",
        "grateful", "thankful", "blessed", "fortunate",
    },
    "sadness": {
        "sad", "sadness", "unhappy", "unhappiness", "sorrow", "sorrowful",
        "grief", "grieve", "grieving", "grieved", "mourn", "mourning",
        "mourned", "depressed", "depression", "despair", "despairing",
        "heartbroken", "heartache", "tears", "cry", "crying", "cried",
        "weep", "weeping", "wept", "sob", "sobbing", "miserable",
        "misery", "gloomy", "gloom", "melancholy", "melancholic",
        "lonely", "loneliness", "alone", "isolated", "hopeless",
        "hopelessness", "devastated", "devastation", "loss", "lost",
        "regret", "regrets", "regretful", "disappointed", "disappointment",
        "lament", "lamenting", "wretched", "broken", "hurt", "pain",
        "anguish", "suffering", "suffer",
    },
    "anger": {
        "angry", "anger", "mad", "furious", "fury", "rage", "enraged",
        "irate", "outraged", "outrage", "livid", "infuriated", "infuriate",
        "hate", "hatred", "hated", "hating", "despise", "despised",
        "loathe", "loathing", "frustrate", "frustrated", "frustration",
        "irritate", "irritated", "irritating", "irritation", "annoy",
        "annoyed", "annoying", "annoyance", "hostile", "hostility",
        "aggression", "aggressive", "violent", "violence", "aggressive",
        "bitter", "bitterness", "resentful", "resentment", "resent",
        "indignant", "indignation", "wrathful", "wrath", "scold",
        "yell", "yelling", "shout", "shouting", "threaten", "threatening",
        "offensive", "offended", "offend",
    },
    "fear": {
        "fear", "afraid", "scared", "frightened", "fright", "terrified",
        "terror", "horrified", "horror", "panic", "panicking", "panicked",
        "dread", "dreading", "dreaded", "anxious", "anxiety", "nervous",
        "nervousness", "worry", "worried", "worrying", "apprehensive",
        "apprehension", "uneasy", "unease", "tense", "tension", "phobia",
        "paranoid", "paranoia", "distressed", "distress", "alarmed",
        "alarm", "shocked", "shock", "startled", "startle", "timid",
        "timidity", "hesitant", "hesitation", "insecure", "vulnerability",
        "vulnerable", "threatened", "threat", "dangerous", "danger",
        "risk", "risky", "unsafe", "uncertain", "uncertainty",
    },
    "surprise": {
        "surprise", "surprised", "surprising", "astonished", "astonishment",
        "astonishing", "amazed", "amazement", "astound", "astounded",
        "astounding", "shocked", "shock", "unexpected", "unexpectedly",
        "suddenly", "sudden", "abruptly", "abrupt", "startled", "startle",
        "unbelievable", "unbelievably", "incredible", "incredibly",
        "remarkable", "remarkably", "extraordinary", "wow", "whoa",
        "whoah", "unimaginable", "stunning", "stun", "stunned",
        "flabbergasted", "speechless", "jaw-dropping", "mind-blowing",
        "unforeseen", "unpredictable", "revelation", "revelation",
    },
    "disgust": {
        "disgust", "disgusted", "disgusting", "revolted", "revolting",
        "revolt", "repulsed", "repulsive", "repulse", "nauseated",
        "nausea", "nauseous", "nauseating", "sick", "sickening",
        "gross", "grotesque", "vile", "nasty", "filthy", "filth",
        "dirty", "foul", "awful", "dreadful", "horrible", "horrid",
        "repugnant", "abhorrent", "abhor", "loathe", "loathing",
        "despise", "contempt", "contemptible", "distasteful",
        "offensive", "obscene", "appalling", "appalled", "atrocious",
        "putrid", "rotten", "stink", "stinking", "stench",
    },
    "trust": {
        "trust", "trusting", "trusted", "trustworthy", "reliable",
        "reliability", "honest", "honesty", "sincere", "sincerity",
        "genuine", "authentic", "authenticity", "faithful", "faithful",
        "loyal", "loyalty", "dependable", "dependability", "credible",
        "credibility", "transparent", "transparency", "fair", "fairness",
        "integrity", "confident", "confidence", "assured", "assurance",
        "secure", "security", "safe", "safety", "believe", "believed",
        "believing", "faith", "committed", "commitment", "dedicated",
        "dedication", "consistent", "consistency", "stable", "stability",
        "responsible", "responsibility", "accountable", "accountability",
        "verify", "verified", "proven", "legitimate", "legitimacy",
    },
    "anticipation": {
        "anticipate", "anticipation", "anticipating", "expect", "expected",
        "expecting", "expectation", "hope", "hoping", "hoped", "hopeful",
        "eagerly", "eager", "eagerness", "excited", "excitement", "await",
        "awaiting", "awaited", "look forward", "looking forward", "plan",
        "planning", "planned", "prepare", "preparing", "prepared",
        "ready", "readiness", "upcoming", "soon", "future", "coming",
        "next", "eventually", "impatient", "impatience", "desire",
        "craving", "crave", "wanting", "want", "wish", "wishing",
        "longing", "yearn", "yearning", "aspire", "aspiration",
        "goal", "target", "objective", "prospect", "potential",
    },
}

# Intensity modifiers
INTENSIFIERS = {"very", "really", "extremely", "absolutely", "completely",
                "totally", "utterly", "incredibly", "so", "highly", "super"}
DIMINISHERS = {"somewhat", "slightly", "a bit", "a little", "kind of",
               "sort of", "rather", "fairly", "mostly", "almost"}


@dataclass
class EmotionScore:
    emotion: str
    score: float       # 0.0 to 1.0
    keywords: List[str]  # matched keywords

    def __repr__(self) -> str:
        return f"EmotionScore({self.emotion}, {self.score:.3f})"


@dataclass
class EmotionResult:
    emotions: Dict[str, EmotionScore]
    primary: Optional[str]           # highest-scoring emotion
    secondary: Optional[str]         # second-highest
    valence: str                     # positive / negative / mixed / neutral
    all_scores: Dict[str, float] = field(default_factory=dict)

    def top(self, n: int = 3) -> List[EmotionScore]:
        return sorted(self.emotions.values(), key=lambda x: -x.score)[:n]


# Emotion valence groupings
_POSITIVE_EMOTIONS = {"joy", "trust", "anticipation", "surprise"}
_NEGATIVE_EMOTIONS = {"sadness", "anger", "fear", "disgust"}


class EmotionClassifier:
    """
    Maps text to 8 basic Plutchik emotions using keyword pattern matching.

    Usage:
        clf = EmotionClassifier()
        result = clf.classify("I'm so excited and can't wait for the results!")
        print(result.primary)  # 'anticipation' or 'joy'
        print(result.top(3))
    """

    def __init__(self):
        self._patterns = EMOTION_PATTERNS
        # Pre-build token sets per emotion
        self._token_sets: Dict[str, Set[str]] = {
            e: set(kws) for e, kws in self._patterns.items()
        }

    def _tokenize(self, text: str) -> List[str]:
        return re.sub(r"[^\w\s'-]", " ", text.lower()).split()

    def classify(self, text: str) -> EmotionResult:
        """Classify text into 8 basic emotions."""
        tokens = self._tokenize(text)
        token_set = set(tokens)

        # Score each emotion
        raw: Dict[str, Tuple[float, List[str]]] = {}
        for emotion, keywords in self._token_sets.items():
            matched = list(keywords & token_set)
            if not matched:
                raw[emotion] = (0.0, [])
                continue

            # Base score: fraction of text tokens that are emotion keywords
            # + bonus for multiple distinct matches
            base = len(matched) / max(len(tokens), 1)
            bonus = min(len(matched) * 0.05, 0.3)

            # Check for intensifiers near emotion words
            intensity = 1.0
            for i, tok in enumerate(tokens):
                if tok in INTENSIFIERS:
                    # Look ahead 2 tokens for emotion keywords
                    window = tokens[i + 1: i + 3]
                    if any(w in keywords for w in window):
                        intensity = max(intensity, 1.3)
                elif tok in DIMINISHERS:
                    window = tokens[i + 1: i + 3]
                    if any(w in keywords for w in window):
                        intensity = min(intensity, 0.7)

            score = min(1.0, (base + bonus) * intensity)
            raw[emotion] = (score, matched)

        # Normalize to max 1.0 across all emotions if needed
        max_score = max((v[0] for v in raw.values()), default=0.0)
        if max_score > 0:
            # Relative scaling so scores are interpretable
            scale = 1.0 / max(max_score, 0.1)
            scale = min(scale, 5.0)  # don't inflate tiny signals too much
        else:
            scale = 1.0

        emotions: Dict[str, EmotionScore] = {}
        all_scores: Dict[str, float] = {}
        for emotion, (score, keywords) in raw.items():
            final = min(1.0, score * scale) if max_score > 0 else 0.0
            emotions[emotion] = EmotionScore(
                emotion=emotion,
                score=round(final, 4),
                keywords=keywords,
            )
            all_scores[emotion] = round(final, 4)

        # Rank
        ranked = sorted(emotions.values(), key=lambda x: -x.score)
        primary = ranked[0].emotion if ranked[0].score > 0 else None
        secondary = ranked[1].emotion if len(ranked) > 1 and ranked[1].score > 0 else None

        # Valence
        if primary is None:
            valence = "neutral"
        else:
            pos_score = sum(emotions[e].score for e in _POSITIVE_EMOTIONS)
            neg_score = sum(emotions[e].score for e in _NEGATIVE_EMOTIONS)
            if pos_score > 0 and neg_score > 0 and min(pos_score, neg_score) / max(pos_score, neg_score) > 0.4:
                valence = "mixed"
            elif pos_score >= neg_score:
                valence = "positive"
            else:
                valence = "negative"

        return EmotionResult(
            emotions=emotions,
            primary=primary,
            secondary=secondary,
            valence=valence,
            all_scores=all_scores,
        )

    def classify_batch(self, texts: List[str]) -> List[EmotionResult]:
        return [self.classify(t) for t in texts]
