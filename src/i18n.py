"""Internationalization support for global deployment."""

import json
import os
from typing import Dict, Any, Optional
from enum import Enum

class SupportedLanguages(Enum):
    """Supported languages for the sentiment analyzer."""
    EN = "en"
    ES = "es"
    FR = "fr"
    DE = "de"
    JA = "ja"
    ZH = "zh"

class I18nManager:
    """Manages internationalization for the sentiment analyzer."""
    
    def __init__(self, default_language: str = "en"):
        self.default_language = default_language
        self.current_language = default_language
        self.translations: Dict[str, Dict[str, str]] = {}
        self._load_translations()
    
    def _load_translations(self):
        """Load translation files."""
        translations_dir = os.path.join(os.path.dirname(__file__), "translations")
        
        if not os.path.exists(translations_dir):
            os.makedirs(translations_dir, exist_ok=True)
            
        for lang in SupportedLanguages:
            translation_file = os.path.join(translations_dir, f"{lang.value}.json")
            try:
                if os.path.exists(translation_file):
                    with open(translation_file, 'r', encoding='utf-8') as f:
                        self.translations[lang.value] = json.load(f)
                else:
                    self.translations[lang.value] = self._get_default_translations(lang.value)
                    self._save_translation_file(lang.value)
            except Exception as e:
                print(f"Warning: Could not load translations for {lang.value}: {e}")
                self.translations[lang.value] = self._get_default_translations(lang.value)
    
    def _get_default_translations(self, language: str) -> Dict[str, str]:
        """Get default translations for a language."""
        translations = {
            "en": {
                "model_training": "Training model",
                "model_trained": "Model training completed",
                "prediction_started": "Starting prediction",
                "prediction_completed": "Prediction completed",
                "positive_sentiment": "Positive",
                "negative_sentiment": "Negative",
                "neutral_sentiment": "Neutral",
                "error_occurred": "An error occurred",
                "processing": "Processing...",
                "validation_error": "Validation error",
                "file_not_found": "File not found",
                "invalid_input": "Invalid input"
            },
            "es": {
                "model_training": "Entrenando modelo",
                "model_trained": "Entrenamiento del modelo completado",
                "prediction_started": "Iniciando predicción",
                "prediction_completed": "Predicción completada",
                "positive_sentiment": "Positivo",
                "negative_sentiment": "Negativo",
                "neutral_sentiment": "Neutral",
                "error_occurred": "Ocurrió un error",
                "processing": "Procesando...",
                "validation_error": "Error de validación",
                "file_not_found": "Archivo no encontrado",
                "invalid_input": "Entrada inválida"
            },
            "fr": {
                "model_training": "Entraînement du modèle",
                "model_trained": "Entraînement du modèle terminé",
                "prediction_started": "Démarrage de la prédiction",
                "prediction_completed": "Prédiction terminée",
                "positive_sentiment": "Positif",
                "negative_sentiment": "Négatif",
                "neutral_sentiment": "Neutre",
                "error_occurred": "Une erreur s'est produite",
                "processing": "Traitement...",
                "validation_error": "Erreur de validation",
                "file_not_found": "Fichier non trouvé",
                "invalid_input": "Entrée invalide"
            },
            "de": {
                "model_training": "Modell-Training",
                "model_trained": "Modell-Training abgeschlossen",
                "prediction_started": "Vorhersage gestartet",
                "prediction_completed": "Vorhersage abgeschlossen",
                "positive_sentiment": "Positiv",
                "negative_sentiment": "Negativ",
                "neutral_sentiment": "Neutral",
                "error_occurred": "Ein Fehler ist aufgetreten",
                "processing": "Verarbeitung...",
                "validation_error": "Validierungsfehler",
                "file_not_found": "Datei nicht gefunden",
                "invalid_input": "Ungültige Eingabe"
            },
            "ja": {
                "model_training": "モデル訓練",
                "model_trained": "モデル訓練完了",
                "prediction_started": "予測開始",
                "prediction_completed": "予測完了",
                "positive_sentiment": "ポジティブ",
                "negative_sentiment": "ネガティブ",
                "neutral_sentiment": "ニュートラル",
                "error_occurred": "エラーが発生しました",
                "processing": "処理中...",
                "validation_error": "検証エラー",
                "file_not_found": "ファイルが見つかりません",
                "invalid_input": "無効な入力"
            },
            "zh": {
                "model_training": "模型训练",
                "model_trained": "模型训练完成",
                "prediction_started": "开始预测",
                "prediction_completed": "预测完成",
                "positive_sentiment": "积极",
                "negative_sentiment": "消极",
                "neutral_sentiment": "中性",
                "error_occurred": "发生错误",
                "processing": "处理中...",
                "validation_error": "验证错误",
                "file_not_found": "文件未找到",
                "invalid_input": "无效输入"
            }
        }
        return translations.get(language, translations["en"])
    
    def _save_translation_file(self, language: str):
        """Save translation file."""
        translations_dir = os.path.join(os.path.dirname(__file__), "translations")
        translation_file = os.path.join(translations_dir, f"{language}.json")
        
        try:
            with open(translation_file, 'w', encoding='utf-8') as f:
                json.dump(self.translations[language], f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Could not save translations for {language}: {e}")
    
    def set_language(self, language: str):
        """Set the current language."""
        if language in [lang.value for lang in SupportedLanguages]:
            self.current_language = language
        else:
            print(f"Warning: Unsupported language {language}, using {self.default_language}")
            self.current_language = self.default_language
    
    def t(self, key: str, **kwargs) -> str:
        """Translate a key to the current language."""
        translation = self.translations.get(self.current_language, {}).get(
            key, self.translations.get(self.default_language, {}).get(key, key)
        )
        
        if kwargs:
            try:
                return translation.format(**kwargs)
            except Exception:
                return translation
        
        return translation
    
    def get_supported_languages(self) -> list:
        """Get list of supported language codes."""
        return [lang.value for lang in SupportedLanguages]

_i18n_manager = I18nManager()

def t(key: str, **kwargs) -> str:
    """Global translation function."""
    return _i18n_manager.t(key, **kwargs)

def set_language(language: str):
    """Set the global language."""
    _i18n_manager.set_language(language)

def get_supported_languages() -> list:
    """Get supported languages."""
    return _i18n_manager.get_supported_languages()