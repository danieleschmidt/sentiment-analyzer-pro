"""
QNP Multilingual Support System
Advanced multilingual capabilities for global QNP deployment.

Features:
- Real-time translation with sentiment preservation
- Cross-lingual sentiment analysis
- Cultural context adaptation
- Language-specific model optimization
- Multilingual compliance integration
"""

import json
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupportedLanguage(Enum):
    """Supported languages for QNP processing"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HINDI = "hi"
    KOREAN = "ko"

@dataclass
class LanguageProfile:
    """Language-specific processing profile"""
    language: SupportedLanguage
    name: str
    sentiment_model: str
    cultural_adjustments: Dict[str, float] = field(default_factory=dict)
    preprocessing_rules: List[str] = field(default_factory=list)
    postprocessing_rules: List[str] = field(default_factory=list)
    translation_quality_threshold: float = 0.8
    
class CulturalContextAdapter:
    """Adapts sentiment analysis based on cultural context"""
    
    def __init__(self):
        self.cultural_mappings = self._load_cultural_mappings()
        self.expression_patterns = self._load_expression_patterns()
    
    def _load_cultural_mappings(self) -> Dict[str, Dict[str, float]]:
        """Load cultural adjustment mappings for different languages"""
        return {
            "ja": {  # Japanese - more reserved expression
                "positive_adjustment": 0.8,  # Reduce positive intensity
                "negative_adjustment": 1.2,  # Increase negative sensitivity
                "neutral_threshold": 0.6     # Higher threshold for neutral
            },
            "de": {  # German - direct expression
                "positive_adjustment": 1.1,
                "negative_adjustment": 1.1,
                "neutral_threshold": 0.4
            },
            "es": {  # Spanish - expressive
                "positive_adjustment": 1.2,
                "negative_adjustment": 1.2,
                "neutral_threshold": 0.3
            },
            "ar": {  # Arabic - context-dependent
                "positive_adjustment": 0.9,
                "negative_adjustment": 0.9,
                "formal_context_modifier": 1.3
            },
            "zh": {  # Chinese - implicit expression
                "positive_adjustment": 0.7,
                "negative_adjustment": 0.8,
                "context_sensitivity": 1.5
            }
        }
    
    def _load_expression_patterns(self) -> Dict[str, List[Dict]]:
        """Load language-specific expression patterns"""
        return {
            "en": [
                {"pattern": r"\bawesome\b", "sentiment": "very_positive"},
                {"pattern": r"\bterrible\b", "sentiment": "very_negative"},
            ],
            "es": [
                {"pattern": r"\bincreíble\b", "sentiment": "very_positive"},
                {"pattern": r"\bhorrible\b", "sentiment": "very_negative"},
                {"pattern": r"\bregular\b", "sentiment": "neutral"},
            ],
            "fr": [
                {"pattern": r"\bexcellent\b", "sentiment": "very_positive"},
                {"pattern": r"\baffreux\b", "sentiment": "very_negative"},
            ],
            "de": [
                {"pattern": r"\bausgezeichnet\b", "sentiment": "very_positive"},
                {"pattern": r"\bschrecklich\b", "sentiment": "very_negative"},
            ],
            "ja": [
                {"pattern": r"素晴らしい", "sentiment": "positive"},  # subarashii
                {"pattern": r"最悪", "sentiment": "negative"},        # saiaku
                {"pattern": r"普通", "sentiment": "neutral"},         # futsuu
            ],
            "zh": [
                {"pattern": r"很好", "sentiment": "positive"},
                {"pattern": r"很糟", "sentiment": "negative"},
                {"pattern": r"一般", "sentiment": "neutral"},
            ]
        }
    
    def adapt_sentiment(self, sentiment_result: Dict[str, Any], 
                       language: SupportedLanguage,
                       text: str) -> Dict[str, Any]:
        """Adapt sentiment results based on cultural context"""
        
        lang_code = language.value
        if lang_code not in self.cultural_mappings:
            return sentiment_result  # No adaptation needed
        
        adjustments = self.cultural_mappings[lang_code]
        adapted_result = sentiment_result.copy()
        
        # Apply cultural adjustments
        if "positive_adjustment" in adjustments:
            adapted_result["positive"] *= adjustments["positive_adjustment"]
        
        if "negative_adjustment" in adjustments:
            adapted_result["negative"] *= adjustments["negative_adjustment"]
        
        # Renormalize probabilities
        total = adapted_result["positive"] + adapted_result["negative"] + adapted_result["neutral"]
        if total > 0:
            adapted_result["positive"] /= total
            adapted_result["negative"] /= total
            adapted_result["neutral"] /= total
        
        # Apply pattern-based adjustments
        if lang_code in self.expression_patterns:
            for pattern_info in self.expression_patterns[lang_code]:
                if re.search(pattern_info["pattern"], text, re.IGNORECASE):
                    sentiment_type = pattern_info["sentiment"]
                    
                    if sentiment_type == "very_positive":
                        adapted_result["positive"] = min(1.0, adapted_result["positive"] * 1.3)
                    elif sentiment_type == "very_negative":
                        adapted_result["negative"] = min(1.0, adapted_result["negative"] * 1.3)
                    elif sentiment_type == "neutral":
                        adapted_result["neutral"] = min(1.0, adapted_result["neutral"] * 1.2)
                    
                    # Renormalize again
                    total = sum([adapted_result["positive"], adapted_result["negative"], adapted_result["neutral"]])
                    if total > 0:
                        adapted_result["positive"] /= total
                        adapted_result["negative"] /= total
                        adapted_result["neutral"] /= total
        
        adapted_result["cultural_adaptation"] = True
        adapted_result["source_language"] = lang_code
        
        return adapted_result

class SimpleTranslator:
    """Simple translation service for demonstration purposes"""
    
    def __init__(self):
        # In production, this would use actual translation services
        self.translation_cache: Dict[str, Dict[str, str]] = {}
        self.sample_translations = self._load_sample_translations()
    
    def _load_sample_translations(self) -> Dict[str, Dict[str, str]]:
        """Load sample translations for demonstration"""
        return {
            "Hello, this is great!": {
                "es": "¡Hola, esto es genial!",
                "fr": "Bonjour, c'est génial!",
                "de": "Hallo, das ist großartig!",
                "ja": "こんにちは、これは素晴らしいです！",
                "zh": "你好，这很棒！"
            },
            "This is terrible quality.": {
                "es": "Esta es una calidad terrible.",
                "fr": "C'est une qualité terrible.",
                "de": "Das ist schreckliche Qualität.",
                "ja": "これはひどい品質です。",
                "zh": "这质量很糟糕。"
            },
            "The service was okay.": {
                "es": "El servicio estaba bien.",
                "fr": "Le service était correct.",
                "de": "Der Service war okay.",
                "ja": "サービスは普通でした。",
                "zh": "服务还可以。"
            }
        }
    
    def translate(self, text: str, target_language: SupportedLanguage,
                 source_language: SupportedLanguage = SupportedLanguage.ENGLISH) -> Dict[str, Any]:
        """Translate text to target language"""
        
        # Check cache first
        cache_key = f"{source_language.value}->{target_language.value}:{text}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        # Check sample translations
        if text in self.sample_translations and target_language.value in self.sample_translations[text]:
            translated_text = self.sample_translations[text][target_language.value]
            confidence = 0.95
        else:
            # Fallback - simple word substitution for demonstration
            translated_text = self._simple_translate(text, target_language)
            confidence = 0.6  # Lower confidence for simple translation
        
        result = {
            "translated_text": translated_text,
            "source_language": source_language.value,
            "target_language": target_language.value,
            "translation_confidence": confidence,
            "translation_time_ms": 50  # Simulated
        }
        
        # Cache result
        self.translation_cache[cache_key] = result
        
        return result
    
    def _simple_translate(self, text: str, target_language: SupportedLanguage) -> str:
        """Simple demonstration translation"""
        # This is a very basic demonstration - real implementation would use proper translation APIs
        word_mappings = {
            SupportedLanguage.SPANISH: {
                "great": "genial", "good": "bueno", "bad": "malo", "terrible": "terrible",
                "excellent": "excelente", "okay": "bien", "service": "servicio"
            },
            SupportedLanguage.FRENCH: {
                "great": "génial", "good": "bon", "bad": "mauvais", "terrible": "terrible",
                "excellent": "excellent", "okay": "correct", "service": "service"
            },
            SupportedLanguage.GERMAN: {
                "great": "großartig", "good": "gut", "bad": "schlecht", "terrible": "schrecklich",
                "excellent": "ausgezeichnet", "okay": "okay", "service": "Service"
            }
        }
        
        if target_language in word_mappings:
            translated = text.lower()
            for en_word, translated_word in word_mappings[target_language].items():
                translated = translated.replace(en_word, translated_word)
            return translated
        
        return f"[{target_language.value.upper()}] {text}"  # Fallback

class QNPMultilingualProcessor:
    """Main multilingual processing system for QNP"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Initialize components
        self.cultural_adapter = CulturalContextAdapter()
        self.translator = SimpleTranslator()
        
        # Language profiles
        self.language_profiles = self._initialize_language_profiles()
        
        # Load translations for UI/messages
        self.ui_translations = self._load_ui_translations()
        
        # Processing statistics
        self.processing_stats = {
            "total_requests": 0,
            "by_language": {lang.value: 0 for lang in SupportedLanguage},
            "translations_performed": 0,
            "cultural_adaptations": 0
        }
    
    def _initialize_language_profiles(self) -> Dict[SupportedLanguage, LanguageProfile]:
        """Initialize language-specific processing profiles"""
        profiles = {}
        
        # English profile (baseline)
        profiles[SupportedLanguage.ENGLISH] = LanguageProfile(
            language=SupportedLanguage.ENGLISH,
            name="English",
            sentiment_model="base_english",
            cultural_adjustments={},
            preprocessing_rules=["normalize_contractions", "handle_slang"]
        )
        
        # Spanish profile
        profiles[SupportedLanguage.SPANISH] = LanguageProfile(
            language=SupportedLanguage.SPANISH,
            name="Español",
            sentiment_model="spanish_sentiment",
            cultural_adjustments={"expressiveness": 1.2},
            preprocessing_rules=["handle_accents", "normalize_regional_variants"]
        )
        
        # French profile
        profiles[SupportedLanguage.FRENCH] = LanguageProfile(
            language=SupportedLanguage.FRENCH,
            name="Français",
            sentiment_model="french_sentiment",
            cultural_adjustments={"formality": 1.1},
            preprocessing_rules=["handle_accents", "formal_informal_distinction"]
        )
        
        # German profile
        profiles[SupportedLanguage.GERMAN] = LanguageProfile(
            language=SupportedLanguage.GERMAN,
            name="Deutsch",
            sentiment_model="german_sentiment",
            cultural_adjustments={"directness": 1.15},
            preprocessing_rules=["compound_word_splitting", "case_normalization"]
        )
        
        # Japanese profile
        profiles[SupportedLanguage.JAPANESE] = LanguageProfile(
            language=SupportedLanguage.JAPANESE,
            name="日本語",
            sentiment_model="japanese_sentiment",
            cultural_adjustments={"indirectness": 0.8, "politeness": 1.3},
            preprocessing_rules=["handle_kanji_variants", "politeness_level_detection"]
        )
        
        # Chinese profile
        profiles[SupportedLanguage.CHINESE_SIMPLIFIED] = LanguageProfile(
            language=SupportedLanguage.CHINESE_SIMPLIFIED,
            name="中文",
            sentiment_model="chinese_sentiment",
            cultural_adjustments={"context_dependency": 1.4},
            preprocessing_rules=["traditional_simplified_conversion", "word_segmentation"]
        )
        
        return profiles
    
    def _load_ui_translations(self) -> Dict[str, Dict[str, str]]:
        """Load UI translations for different languages"""
        return {
            "sentiment_positive": {
                "en": "Positive",
                "es": "Positivo",
                "fr": "Positif",
                "de": "Positiv",
                "ja": "ポジティブ",
                "zh": "积极"
            },
            "sentiment_negative": {
                "en": "Negative",
                "es": "Negativo",
                "fr": "Négatif",
                "de": "Negativ",
                "ja": "ネガティブ",
                "zh": "消极"
            },
            "sentiment_neutral": {
                "en": "Neutral",
                "es": "Neutral",
                "fr": "Neutre",
                "de": "Neutral",
                "ja": "中立",
                "zh": "中性"
            },
            "processing_complete": {
                "en": "Processing complete",
                "es": "Procesamiento completo",
                "fr": "Traitement terminé",
                "de": "Verarbeitung abgeschlossen",
                "ja": "処理完了",
                "zh": "处理完成"
            },
            "high_confidence": {
                "en": "High confidence",
                "es": "Alta confianza",
                "fr": "Haute confiance",
                "de": "Hohes Vertrauen",
                "ja": "高い信頼性",
                "zh": "高置信度"
            }
        }
    
    def detect_language(self, text: str) -> Tuple[SupportedLanguage, float]:
        """Detect language of input text"""
        
        # Simple language detection based on character patterns
        # In production, this would use proper language detection libraries
        
        # Check for specific character sets
        if re.search(r'[ひらがなカタカナ漢字]', text):
            return SupportedLanguage.JAPANESE, 0.9
        
        if re.search(r'[一-龯]', text) and not re.search(r'[ひらがなカタカナ]', text):
            return SupportedLanguage.CHINESE_SIMPLIFIED, 0.85
        
        if re.search(r'[ñáéíóúü¿¡]', text):
            return SupportedLanguage.SPANISH, 0.8
        
        if re.search(r'[àâäçéèêëîïôöùûüÿ]', text):
            return SupportedLanguage.FRENCH, 0.8
        
        if re.search(r'[äöüß]', text):
            return SupportedLanguage.GERMAN, 0.8
        
        # Default to English
        return SupportedLanguage.ENGLISH, 0.7
    
    def process_multilingual_sentiment(self, text: str, 
                                     source_language: Optional[SupportedLanguage] = None,
                                     target_language: Optional[SupportedLanguage] = None) -> Dict[str, Any]:
        """Process sentiment analysis with multilingual support"""
        
        start_time = time.time()
        
        # Detect source language if not provided
        if source_language is None:
            detected_lang, confidence = self.detect_language(text)
            source_language = detected_lang
            language_detection = {
                "detected_language": detected_lang.value,
                "detection_confidence": confidence
            }
        else:
            language_detection = {
                "specified_language": source_language.value,
                "detection_confidence": 1.0
            }
        
        # Update processing statistics
        self.processing_stats["total_requests"] += 1
        self.processing_stats["by_language"][source_language.value] += 1
        
        processing_text = text
        translation_info = None
        
        # Translate if needed
        if target_language and target_language != source_language:
            translation_result = self.translator.translate(
                text, target_language, source_language
            )
            processing_text = translation_result["translated_text"]
            translation_info = translation_result
            self.processing_stats["translations_performed"] += 1
            
            # Use target language for processing
            processing_language = target_language
        else:
            processing_language = source_language
        
        # Get language profile
        profile = self.language_profiles.get(processing_language, 
                                           self.language_profiles[SupportedLanguage.ENGLISH])
        
        # Simulate sentiment analysis (in production, this would use actual models)
        base_sentiment = self._simulate_sentiment_analysis(processing_text, profile)
        
        # Apply cultural adaptations
        adapted_sentiment = self.cultural_adapter.adapt_sentiment(
            base_sentiment, processing_language, processing_text
        )
        
        if adapted_sentiment.get("cultural_adaptation", False):
            self.processing_stats["cultural_adaptations"] += 1
        
        # Prepare final result
        processing_time = time.time() - start_time
        
        result = {
            **adapted_sentiment,
            "multilingual_info": {
                "source_language": source_language.value,
                "processing_language": processing_language.value,
                "language_detection": language_detection,
                "translation": translation_info,
                "cultural_adaptation_applied": adapted_sentiment.get("cultural_adaptation", False),
                "language_profile": profile.name
            },
            "processing_time_ms": processing_time * 1000,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def _simulate_sentiment_analysis(self, text: str, profile: LanguageProfile) -> Dict[str, Any]:
        """Simulate language-specific sentiment analysis"""
        
        # Simple rule-based sentiment for demonstration
        text_lower = text.lower()
        
        positive_indicators = [
            "good", "great", "excellent", "amazing", "wonderful", "fantastic", "love",
            "bueno", "genial", "excelente", "increíble",  # Spanish
            "bon", "génial", "excellent", "merveilleux",  # French
            "gut", "großartig", "ausgezeichnet", "wunderbar",  # German
            "良い", "素晴らしい", "すごい",  # Japanese (simplified)
            "好", "很好", "太棒了"  # Chinese (simplified)
        ]
        
        negative_indicators = [
            "bad", "terrible", "awful", "horrible", "hate", "disappointing",
            "malo", "terrible", "horrible", "odio",  # Spanish
            "mauvais", "terrible", "affreux", "déteste",  # French
            "schlecht", "schrecklich", "furchtbar", "hasse",  # German
            "悪い", "ひどい", "最悪",  # Japanese (simplified)
            "坏", "糟糕", "很糟"  # Chinese (simplified)
        ]
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in text_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in text_lower)
        
        total_indicators = positive_count + negative_count
        
        if total_indicators == 0:
            # No clear indicators - neutral
            sentiment_result = {
                "positive": 0.25,
                "negative": 0.25,
                "neutral": 0.50,
                "confidence": 0.3
            }
        else:
            positive_prob = positive_count / total_indicators
            negative_prob = negative_count / total_indicators
            neutral_prob = 1.0 - positive_prob - negative_prob
            
            sentiment_result = {
                "positive": positive_prob,
                "negative": negative_prob,
                "neutral": max(0, neutral_prob),
                "confidence": min(0.9, total_indicators / 5.0)  # Higher confidence with more indicators
            }
        
        # Apply language profile adjustments
        for adjustment, factor in profile.cultural_adjustments.items():
            if adjustment == "expressiveness":
                # More expressive languages show stronger sentiment
                if sentiment_result["positive"] > sentiment_result["negative"]:
                    sentiment_result["positive"] *= factor
                else:
                    sentiment_result["negative"] *= factor
        
        # Renormalize
        total = sum([sentiment_result["positive"], sentiment_result["negative"], sentiment_result["neutral"]])
        if total > 0:
            sentiment_result["positive"] /= total
            sentiment_result["negative"] /= total
            sentiment_result["neutral"] /= total
        
        return sentiment_result
    
    def get_ui_translation(self, message_key: str, language: SupportedLanguage) -> str:
        """Get UI message translation"""
        
        if message_key in self.ui_translations:
            return self.ui_translations[message_key].get(
                language.value, 
                self.ui_translations[message_key]["en"]  # Fallback to English
            )
        
        return message_key  # Return key if translation not found
    
    def format_result_for_language(self, result: Dict[str, Any], 
                                  language: SupportedLanguage) -> Dict[str, Any]:
        """Format result with language-specific UI elements"""
        
        formatted_result = result.copy()
        
        # Determine dominant sentiment
        sentiments = ["positive", "negative", "neutral"]
        dominant_sentiment = max(sentiments, key=lambda s: result.get(s, 0))
        
        # Add localized labels
        formatted_result["localized_labels"] = {
            "dominant_sentiment": self.get_ui_translation(f"sentiment_{dominant_sentiment}", language),
            "status_message": self.get_ui_translation("processing_complete", language),
            "confidence_level": self.get_ui_translation("high_confidence", language) if result.get("confidence", 0) > 0.7 else "Medium confidence"
        }
        
        # Format numbers according to language conventions
        if language in [SupportedLanguage.GERMAN, SupportedLanguage.FRENCH]:
            # European number format (comma as decimal separator)
            formatted_result["formatted_scores"] = {
                "positive": f"{result.get('positive', 0):.2f}".replace(".", ","),
                "negative": f"{result.get('negative', 0):.2f}".replace(".", ","),
                "neutral": f"{result.get('neutral', 0):.2f}".replace(".", ",")
            }
        else:
            # Standard format
            formatted_result["formatted_scores"] = {
                "positive": f"{result.get('positive', 0):.2f}",
                "negative": f"{result.get('negative', 0):.2f}",
                "neutral": f"{result.get('neutral', 0):.2f}"
            }
        
        return formatted_result
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get multilingual processing statistics"""
        return {
            "processing_statistics": self.processing_stats,
            "supported_languages": [
                {
                    "code": lang.value,
                    "name": profile.name,
                    "cultural_adaptations": len(profile.cultural_adjustments) > 0
                }
                for lang, profile in self.language_profiles.items()
            ],
            "capabilities": {
                "language_detection": True,
                "translation": True,
                "cultural_adaptation": True,
                "ui_localization": True
            }
        }

# Factory function
def create_multilingual_processor(config: Optional[Dict] = None) -> QNPMultilingualProcessor:
    """Create multilingual processor"""
    return QNPMultilingualProcessor(config)

# CLI interface for multilingual processing
def main():
    """Demo multilingual processing system"""
    print("🌍 QNP Multilingual Support System Demo")
    print("=" * 50)
    
    # Create processor
    processor = create_multilingual_processor()
    
    # Test texts in different languages
    test_texts = [
        ("Hello, this is great!", SupportedLanguage.ENGLISH),
        ("¡Hola, esto es genial!", SupportedLanguage.SPANISH),
        ("Bonjour, c'est génial!", SupportedLanguage.FRENCH),
        ("こんにちは、これは素晴らしいです！", SupportedLanguage.JAPANESE),
        ("This is terrible quality.", SupportedLanguage.ENGLISH)
    ]
    
    print("🔍 Processing multilingual sentiment analysis...")
    
    for text, expected_lang in test_texts:
        print(f"\n📝 Text: {text}")
        
        # Process with language detection
        result = processor.process_multilingual_sentiment(text)
        
        # Format for the detected language
        detected_lang = SupportedLanguage(result["multilingual_info"]["source_language"])
        formatted = processor.format_result_for_language(result, detected_lang)
        
        print(f"   Language: {result['multilingual_info']['language_profile']}")
        print(f"   Sentiment: {formatted['localized_labels']['dominant_sentiment']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Cultural adaptation: {result['multilingual_info']['cultural_adaptation_applied']}")
    
    # Test translation
    print(f"\n🔄 Testing translation...")
    translation_result = processor.process_multilingual_sentiment(
        "This product is excellent!",
        source_language=SupportedLanguage.ENGLISH,
        target_language=SupportedLanguage.SPANISH
    )
    
    if translation_result["multilingual_info"]["translation"]:
        print(f"   Original: This product is excellent!")
        print(f"   Spanish: {translation_result['multilingual_info']['translation']['translated_text']}")
        print(f"   Sentiment: {translation_result['positive']:.2f} positive")
    
    # Show statistics
    stats = processor.get_processing_statistics()
    print(f"\n📊 Processing Statistics:")
    print(f"   Total requests: {stats['processing_statistics']['total_requests']}")
    print(f"   Languages supported: {len(stats['supported_languages'])}")
    print(f"   Translations performed: {stats['processing_statistics']['translations_performed']}")
    print(f"   Cultural adaptations: {stats['processing_statistics']['cultural_adaptations']}")

if __name__ == "__main__":
    main()