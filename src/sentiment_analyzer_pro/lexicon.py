"""
LexiconSentiment: VADER-style lexicon-based sentiment scorer.

Scores text using a bundled ~200-word sentiment lexicon with:
- Word polarity scores (-1.0 to +1.0)
- Valence shifters (negation, intensifiers, diminishers)
- Punctuation and capitalization boosters
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Bundled sentiment lexicon: (word → score)
# Scores range from -1.0 (very negative) to +1.0 (very positive)
# ---------------------------------------------------------------------------
LEXICON: Dict[str, float] = {
    # --- Strong positive ---
    "excellent": 0.9, "outstanding": 0.9, "superb": 0.9, "magnificent": 0.9,
    "phenomenal": 0.9, "exceptional": 0.9, "extraordinary": 0.9, "brilliant": 0.85,
    "fantastic": 0.85, "wonderful": 0.85, "amazing": 0.85, "awesome": 0.85,
    "spectacular": 0.85, "marvelous": 0.85, "terrific": 0.8, "incredible": 0.8,
    "perfect": 0.8, "flawless": 0.8, "delightful": 0.8, "splendid": 0.8,
    # --- Moderate positive ---
    "great": 0.75, "good": 0.7, "nice": 0.65, "pleasant": 0.65, "fine": 0.6,
    "happy": 0.7, "pleased": 0.65, "satisfied": 0.65, "glad": 0.65,
    "love": 0.8, "loved": 0.8, "enjoy": 0.7, "enjoyed": 0.7, "like": 0.55,
    "liked": 0.55, "appreciate": 0.65, "helpful": 0.65, "useful": 0.6,
    "positive": 0.6, "beautiful": 0.75, "clean": 0.5, "clear": 0.45,
    "easy": 0.55, "fun": 0.65, "exciting": 0.75, "impressive": 0.7,
    "reliable": 0.65, "efficient": 0.65, "elegant": 0.7, "smooth": 0.6,
    "fast": 0.55, "quick": 0.5, "smart": 0.6, "clever": 0.6, "simple": 0.45,
    "comfortable": 0.65, "convenient": 0.6, "friendly": 0.65, "kind": 0.65,
    "warm": 0.6, "welcoming": 0.65, "generous": 0.7, "honest": 0.65,
    "trustworthy": 0.7, "professional": 0.6, "polite": 0.6, "responsive": 0.55,
    "innovative": 0.7, "creative": 0.65, "intuitive": 0.65, "robust": 0.6,
    "powerful": 0.65, "solid": 0.6, "stable": 0.55, "secure": 0.6,
    "quality": 0.65, "premium": 0.65, "value": 0.55, "affordable": 0.6,
    "recommend": 0.7, "recommended": 0.7, "worth": 0.55, "worthwhile": 0.65,
    "beneficial": 0.65, "effective": 0.65, "capable": 0.55, "competent": 0.6,
    "accurate": 0.6, "precise": 0.6, "correct": 0.5, "right": 0.45,
    "better": 0.55, "best": 0.8, "improved": 0.55, "improvement": 0.55,
    "upgrade": 0.5, "progress": 0.55, "success": 0.7, "successful": 0.7,
    "win": 0.7, "won": 0.7, "victory": 0.75, "accomplish": 0.65,
    "achieve": 0.65, "achievement": 0.7, "thrive": 0.75, "flourish": 0.75,
    "healthy": 0.65, "fresh": 0.55, "vibrant": 0.65, "energetic": 0.65,
    "alive": 0.55, "bright": 0.6, "hopeful": 0.65, "optimistic": 0.65,
    "confident": 0.65, "proud": 0.65, "grateful": 0.7, "thankful": 0.7,
    "blessed": 0.7, "fortunate": 0.7, "lucky": 0.65,
    # --- Mild positive ---
    "ok": 0.3, "okay": 0.3, "decent": 0.4, "acceptable": 0.35, "adequate": 0.35,
    "fair": 0.35, "reasonable": 0.4, "alright": 0.3, "passable": 0.3,
    # --- Strong negative ---
    "terrible": -0.9, "horrible": -0.9, "awful": -0.9, "dreadful": -0.9,
    "appalling": -0.9, "atrocious": -0.9, "disgusting": -0.85, "revolting": -0.85,
    "abysmal": -0.9, "catastrophic": -0.9, "disastrous": -0.85, "devastating": -0.85,
    "horrendous": -0.9, "vile": -0.85, "wretched": -0.8, "deplorable": -0.85,
    "pathetic": -0.8, "useless": -0.8, "worthless": -0.8, "garbage": -0.8,
    "trash": -0.75, "junk": -0.7, "waste": -0.65, "failure": -0.8,
    "fail": -0.75, "failed": -0.75, "broken": -0.75, "corrupt": -0.8,
    # --- Moderate negative ---
    "bad": -0.7, "poor": -0.65, "worst": -0.85, "worse": -0.6,
    "hate": -0.8, "hated": -0.8, "dislike": -0.6, "disliked": -0.6,
    "disappoint": -0.65, "disappointed": -0.7, "disappointing": -0.7,
    "frustrat": -0.7, "frustrated": -0.7, "frustrating": -0.7,
    "annoying": -0.65, "annoyed": -0.65, "irritating": -0.65, "irritated": -0.65,
    "angry": -0.7, "anger": -0.7, "furious": -0.8, "rage": -0.8,
    "upset": -0.65, "unhappy": -0.7, "sad": -0.65, "depressed": -0.75,
    "miserable": -0.8, "suffer": -0.75, "suffering": -0.75, "pain": -0.65,
    "hurt": -0.65, "harm": -0.7, "damage": -0.65, "problem": -0.55,
    "issue": -0.45, "error": -0.6, "bug": -0.55, "glitch": -0.55,
    "crash": -0.7, "slow": -0.5, "laggy": -0.6, "clunky": -0.55,
    "confusing": -0.6, "confused": -0.55, "complicated": -0.5, "complex": -0.3,
    "difficult": -0.5, "hard": -0.4, "tedious": -0.55, "boring": -0.55,
    "dull": -0.5, "bland": -0.45, "mediocre": -0.5, "average": -0.2,
    "expensive": -0.55, "overpriced": -0.65, "cheap": -0.4,
    "unreliable": -0.7, "unstable": -0.65, "insecure": -0.65,
    "rude": -0.7, "unfriendly": -0.65, "hostile": -0.75, "mean": -0.65,
    "dishonest": -0.75, "deceptive": -0.75, "misleading": -0.7,
    "unprofessional": -0.65, "incompetent": -0.7, "inefficient": -0.6,
    "regret": -0.65, "regretted": -0.65, "mistake": -0.6, "wrong": -0.55,
    "false": -0.6, "incorrect": -0.55, "inaccurate": -0.55,
    "lacking": -0.5, "missing": -0.45, "absent": -0.35, "limited": -0.4,
    "weak": -0.5, "flawed": -0.6, "defective": -0.7, "faulty": -0.65,
    # --- Mild negative ---
    "meh": -0.2, "subpar": -0.45, "underwhelming": -0.5, "overrated": -0.55,
    "concern": -0.35, "worried": -0.5, "worry": -0.5, "doubt": -0.4,
    "skeptical": -0.35, "questionable": -0.45,
}

# Negation words that flip sentiment direction
NEGATIONS = {
    "not", "no", "never", "nothing", "neither", "nobody", "nowhere",
    "cannot", "can't", "won't", "don't", "doesn't", "didn't", "isn't",
    "aren't", "wasn't", "weren't", "haven't", "hadn't", "shouldn't",
    "wouldn't", "couldn't", "hardly", "barely", "scarcely",
}

# Intensifiers boost the absolute score
INTENSIFIERS = {
    "very": 1.3, "really": 1.25, "extremely": 1.5, "absolutely": 1.5,
    "completely": 1.4, "totally": 1.4, "utterly": 1.5, "incredibly": 1.4,
    "unbelievably": 1.5, "exceptionally": 1.4, "particularly": 1.2,
    "especially": 1.25, "quite": 1.15, "highly": 1.3, "super": 1.3,
    "so": 1.2, "too": 1.1,
}

# Diminishers reduce the absolute score
DIMINISHERS = {
    "somewhat": 0.6, "slightly": 0.5, "a bit": 0.6, "a little": 0.55,
    "kind of": 0.65, "sort of": 0.65, "rather": 0.8, "fairly": 0.8,
    "mostly": 0.85, "almost": 0.8, "nearly": 0.85,
}


@dataclass
class LexiconResult:
    score: float          # compound score: -1.0 to +1.0
    positive: float       # positive component (0.0 to 1.0)
    negative: float       # negative component (0.0 to 1.0)
    neutral: float        # neutral component (0.0 to 1.0)
    label: str            # "positive", "negative", or "neutral"
    token_scores: List[Tuple[str, float]] = field(default_factory=list)


class LexiconSentiment:
    """
    VADER-style lexicon-based sentiment scorer.

    Usage:
        scorer = LexiconSentiment()
        result = scorer.score("This product is absolutely amazing!")
        print(result.score, result.label)
    """

    def __init__(self, extra_lexicon: Dict[str, float] | None = None):
        self.lexicon = dict(LEXICON)
        if extra_lexicon:
            self.lexicon.update(extra_lexicon)

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase, strip punctuation, split on whitespace."""
        return text.lower().translate(str.maketrans("", "", string.punctuation)).split()

    def _cap_boost(self, text: str) -> float:
        """Boost for ALL-CAPS words in sentiment context (like VADER)."""
        words = text.split()
        cap_count = sum(1 for w in words if w.isupper() and len(w) > 1)
        return min(cap_count * 0.05, 0.25)

    def _punct_boost(self, text: str) -> float:
        """Boost for exclamation/question marks."""
        excl = min(text.count("!"), 3)
        return excl * 0.04

    def score(self, text: str) -> LexiconResult:
        """Score a piece of text. Returns LexiconResult."""
        tokens = self._tokenize(text)
        raw_scores: List[float] = []
        token_scores: List[Tuple[str, float]] = []

        negated = False
        multiplier = 1.0
        window_reset = 0

        for i, token in enumerate(tokens):
            # Reset negation window after 3 tokens
            if i - window_reset > 3:
                negated = False
                multiplier = 1.0

            if token in NEGATIONS:
                negated = True
                window_reset = i
                continue

            if token in INTENSIFIERS:
                multiplier *= INTENSIFIERS[token]
                window_reset = i
                continue

            # Check diminishers (single-word form)
            if token in {k.split()[0] for k in DIMINISHERS if " " not in k}:
                for dim, val in DIMINISHERS.items():
                    if token == dim:
                        multiplier *= val
                        window_reset = i
                        break
                continue

            if token in self.lexicon:
                s = self.lexicon[token] * multiplier
                if negated:
                    s = -s * 0.75  # negation doesn't fully flip, dampened
                    negated = False
                    multiplier = 1.0
                raw_scores.append(s)
                token_scores.append((token, s))
            else:
                multiplier = 1.0  # reset if not a modifier

        if not raw_scores:
            return LexiconResult(
                score=0.0, positive=0.0, negative=0.0, neutral=1.0,
                label="neutral", token_scores=[]
            )

        # Aggregate: sum with diminishing returns (VADER-style normalization)
        total = sum(raw_scores)
        n = len(raw_scores)
        norm_alpha = 10.0  # normalization constant
        compound = total / (abs(total) + norm_alpha)

        # Boosts
        compound += self._cap_boost(text) * (1 if compound >= 0 else -1)
        compound += self._punct_boost(text) * (1 if compound >= 0 else -1)
        compound = max(-1.0, min(1.0, compound))

        # Positive/negative/neutral proportions
        pos_sum = sum(s for s in raw_scores if s > 0) or 0.0
        neg_sum = sum(abs(s) for s in raw_scores if s < 0) or 0.0
        total_abs = pos_sum + neg_sum + 1e-9

        # Adjust by sentence length
        positive = pos_sum / total_abs
        negative = neg_sum / total_abs
        neutral = max(0.0, 1.0 - (pos_sum + neg_sum) / (n * 0.5 + 1e-9))
        total_dims = positive + negative + neutral
        positive /= total_dims
        negative /= total_dims
        neutral /= total_dims

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return LexiconResult(
            score=round(compound, 4),
            positive=round(positive, 4),
            negative=round(negative, 4),
            neutral=round(neutral, 4),
            label=label,
            token_scores=token_scores,
        )

    def score_batch(self, texts: List[str]) -> List[LexiconResult]:
        return [self.score(t) for t in texts]
