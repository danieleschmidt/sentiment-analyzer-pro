
import json
import os
from pathlib import Path
from typing import Dict, Optional, Any

class InternationalizationManager:
    """Comprehensive internationalization management system."""
    
    def __init__(self, default_language: str = "en"):
        self.default_language = default_language
        self.current_language = default_language
        self.translations: Dict[str, Dict[str, str]] = {}
        self.supported_languages = []
        
        # Load all translations
        self.load_translations()
    
    def load_translations(self):
        """Load all translation files."""
        translations_dir = Path(__file__).parent / "translations"
        
        if not translations_dir.exists():
            return
        
        for lang_file in translations_dir.glob("*.json"):
            lang_code = lang_file.stem
            
            try:
                with open(lang_file, 'r', encoding='utf-8') as f:
                    self.translations[lang_code] = json.load(f)
                    self.supported_languages.append(lang_code)
            except Exception as e:
                print(f"Warning: Could not load translations for {lang_code}: {e}")
    
    def set_language(self, language: str) -> bool:
        """Set the current language."""
        if language in self.supported_languages:
            self.current_language = language
            return True
        return False
    
    def get_text(self, key: str, language: Optional[str] = None) -> str:
        """Get translated text for a key."""
        target_language = language or self.current_language
        
        # Try target language
        if target_language in self.translations:
            if key in self.translations[target_language]:
                return self.translations[target_language][key]
        
        # Fallback to default language
        if self.default_language in self.translations:
            if key in self.translations[self.default_language]:
                return self.translations[self.default_language][key]
        
        # Return key if no translation found
        return key
    
    def get_language_name(self, language_code: str) -> str:
        """Get display name for language code."""
        language_names = {
            "en": "English",
            "es": "Español",
            "fr": "Français", 
            "de": "Deutsch",
            "ja": "日本語",
            "zh": "中文"
        }
        return language_names.get(language_code, language_code)
    
    def detect_browser_language(self, accept_language_header: str) -> str:
        """Detect preferred language from browser Accept-Language header."""
        if not accept_language_header:
            return self.default_language
        
        # Parse Accept-Language header
        languages = []
        for lang in accept_language_header.split(','):
            lang = lang.strip()
            if ';' in lang:
                lang_code, quality = lang.split(';', 1)
                try:
                    quality = float(quality.split('=')[1])
                except:
                    quality = 1.0
            else:
                lang_code = lang
                quality = 1.0
            
            # Extract primary language code
            primary_lang = lang_code.split('-')[0].lower()
            languages.append((primary_lang, quality))
        
        # Sort by quality and find supported language
        languages.sort(key=lambda x: x[1], reverse=True)
        
        for lang_code, _ in languages:
            if lang_code in self.supported_languages:
                return lang_code
        
        return self.default_language
    
    def get_all_translations(self, language: Optional[str] = None) -> Dict[str, str]:
        """Get all translations for a language."""
        target_language = language or self.current_language
        return self.translations.get(target_language, {})
    
    def format_message(self, key: str, **kwargs) -> str:
        """Get translated text with formatting."""
        template = self.get_text(key)
        try:
            return template.format(**kwargs)
        except:
            return template

# Global i18n manager instance
i18n = InternationalizationManager()
