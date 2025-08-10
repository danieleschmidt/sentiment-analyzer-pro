"""
Data validation and integrity framework
Generation 2: Make It Robust - Comprehensive data validation
"""
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import re
from pathlib import Path
import hashlib
import time

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Result of a validation check"""
    field: str
    severity: ValidationSeverity
    message: str
    value: Any = None
    expected: Any = None
    rule: str = ""

@dataclass
class DataQualityReport:
    """Comprehensive data quality report"""
    dataset_name: str
    timestamp: float
    total_records: int
    validation_results: List[ValidationResult]
    quality_score: float
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "total_records": self.total_records,
            "validation_results": [
                {
                    "field": vr.field,
                    "severity": vr.severity.value,
                    "message": vr.message,
                    "value": str(vr.value)[:100] if vr.value is not None else None,
                    "expected": str(vr.expected)[:100] if vr.expected is not None else None,
                    "rule": vr.rule
                }
                for vr in self.validation_results
            ],
            "quality_score": self.quality_score,
            "recommendations": self.recommendations
        }

class DataValidator:
    """Comprehensive data validation framework"""
    
    def __init__(self):
        self.validation_rules = {}
        self.custom_validators = {}
        self._register_default_validators()
    
    def _register_default_validators(self):
        """Register default validation functions"""
        self.custom_validators.update({
            'not_null': self._validate_not_null,
            'data_type': self._validate_data_type,
            'range': self._validate_range,
            'length': self._validate_length,
            'pattern': self._validate_pattern,
            'unique': self._validate_unique,
            'enum': self._validate_enum,
            'sentiment_label': self._validate_sentiment_label,
            'text_quality': self._validate_text_quality,
            'encoding': self._validate_encoding,
            'duplicates': self._validate_duplicates,
            'missing_ratio': self._validate_missing_ratio,
            'statistical_outliers': self._validate_statistical_outliers
        })
    
    def register_validator(self, name: str, validator_func: Callable):
        """Register custom validator"""
        self.custom_validators[name] = validator_func
    
    def add_validation_rule(self, field: str, rule_type: str, **params):
        """Add validation rule for a field"""
        if field not in self.validation_rules:
            self.validation_rules[field] = []
        
        self.validation_rules[field].append({
            'type': rule_type,
            'params': params
        })
    
    def validate_dataframe(self, df: pd.DataFrame, 
                          dataset_name: str = "dataset") -> DataQualityReport:
        """Validate entire DataFrame"""
        validation_results = []
        
        # Basic DataFrame checks
        if df.empty:
            validation_results.append(ValidationResult(
                field="dataset",
                severity=ValidationSeverity.ERROR,
                message="Dataset is empty",
                rule="not_empty"
            ))
        
        # Validate each field with registered rules
        for field, rules in self.validation_rules.items():
            if field in df.columns:
                for rule in rules:
                    validator = self.custom_validators.get(rule['type'])
                    if validator:
                        try:
                            results = validator(df[field], **rule['params'])
                            if isinstance(results, list):
                                validation_results.extend(results)
                            else:
                                validation_results.append(results)
                        except Exception as e:
                            validation_results.append(ValidationResult(
                                field=field,
                                severity=ValidationSeverity.ERROR,
                                message=f"Validation failed: {str(e)}",
                                rule=rule['type']
                            ))
            else:
                validation_results.append(ValidationResult(
                    field=field,
                    severity=ValidationSeverity.WARNING,
                    message=f"Field '{field}' not found in dataset",
                    rule="field_exists"
                ))
        
        # Generate overall quality score and recommendations
        quality_score = self._calculate_quality_score(validation_results)
        recommendations = self._generate_recommendations(validation_results, df)
        
        return DataQualityReport(
            dataset_name=dataset_name,
            timestamp=time.time(),
            total_records=len(df),
            validation_results=validation_results,
            quality_score=quality_score,
            recommendations=recommendations
        )
    
    def _validate_not_null(self, series: pd.Series, **params) -> ValidationResult:
        """Validate that series has no null values"""
        null_count = series.isnull().sum()
        if null_count > 0:
            return ValidationResult(
                field=series.name,
                severity=ValidationSeverity.WARNING,
                message=f"{null_count} null values found",
                value=null_count,
                rule="not_null"
            )
        return ValidationResult(
            field=series.name,
            severity=ValidationSeverity.INFO,
            message="No null values",
            rule="not_null"
        )
    
    def _validate_data_type(self, series: pd.Series, expected_type: str, **params) -> ValidationResult:
        """Validate data type"""
        actual_type = str(series.dtype)
        if expected_type not in actual_type:
            return ValidationResult(
                field=series.name,
                severity=ValidationSeverity.ERROR,
                message=f"Expected type {expected_type}, got {actual_type}",
                value=actual_type,
                expected=expected_type,
                rule="data_type"
            )
        return ValidationResult(
            field=series.name,
            severity=ValidationSeverity.INFO,
            message=f"Correct data type: {actual_type}",
            rule="data_type"
        )
    
    def _validate_range(self, series: pd.Series, min_val=None, max_val=None, **params) -> ValidationResult:
        """Validate numeric range"""
        if not pd.api.types.is_numeric_dtype(series):
            return ValidationResult(
                field=series.name,
                severity=ValidationSeverity.ERROR,
                message="Cannot validate range on non-numeric data",
                rule="range"
            )
        
        violations = []
        if min_val is not None:
            below_min = (series < min_val).sum()
            if below_min > 0:
                violations.append(f"{below_min} values below minimum {min_val}")
        
        if max_val is not None:
            above_max = (series > max_val).sum()
            if above_max > 0:
                violations.append(f"{above_max} values above maximum {max_val}")
        
        if violations:
            return ValidationResult(
                field=series.name,
                severity=ValidationSeverity.WARNING,
                message="; ".join(violations),
                rule="range"
            )
        
        return ValidationResult(
            field=series.name,
            severity=ValidationSeverity.INFO,
            message="All values within range",
            rule="range"
        )
    
    def _validate_length(self, series: pd.Series, min_len=None, max_len=None, **params) -> ValidationResult:
        """Validate string length"""
        if series.dtype != 'object':
            return ValidationResult(
                field=series.name,
                severity=ValidationSeverity.ERROR,
                message="Cannot validate length on non-string data",
                rule="length"
            )
        
        lengths = series.str.len()
        violations = []
        
        if min_len is not None:
            too_short = (lengths < min_len).sum()
            if too_short > 0:
                violations.append(f"{too_short} values shorter than {min_len}")
        
        if max_len is not None:
            too_long = (lengths > max_len).sum()
            if too_long > 0:
                violations.append(f"{too_long} values longer than {max_len}")
        
        if violations:
            return ValidationResult(
                field=series.name,
                severity=ValidationSeverity.WARNING,
                message="; ".join(violations),
                rule="length"
            )
        
        return ValidationResult(
            field=series.name,
            severity=ValidationSeverity.INFO,
            message="All lengths within range",
            rule="length"
        )
    
    def _validate_pattern(self, series: pd.Series, pattern: str, **params) -> ValidationResult:
        """Validate regex pattern"""
        if series.dtype != 'object':
            return ValidationResult(
                field=series.name,
                severity=ValidationSeverity.ERROR,
                message="Cannot validate pattern on non-string data",
                rule="pattern"
            )
        
        try:
            matches = series.str.contains(pattern, regex=True, na=False)
            non_matches = (~matches).sum()
            
            if non_matches > 0:
                return ValidationResult(
                    field=series.name,
                    severity=ValidationSeverity.WARNING,
                    message=f"{non_matches} values don't match pattern",
                    value=non_matches,
                    expected=pattern,
                    rule="pattern"
                )
            
            return ValidationResult(
                field=series.name,
                severity=ValidationSeverity.INFO,
                message="All values match pattern",
                rule="pattern"
            )
        except Exception as e:
            return ValidationResult(
                field=series.name,
                severity=ValidationSeverity.ERROR,
                message=f"Pattern validation failed: {str(e)}",
                rule="pattern"
            )
    
    def _validate_unique(self, series: pd.Series, **params) -> ValidationResult:
        """Validate uniqueness"""
        duplicates = series.duplicated().sum()
        if duplicates > 0:
            return ValidationResult(
                field=series.name,
                severity=ValidationSeverity.WARNING,
                message=f"{duplicates} duplicate values found",
                value=duplicates,
                rule="unique"
            )
        
        return ValidationResult(
            field=series.name,
            severity=ValidationSeverity.INFO,
            message="All values are unique",
            rule="unique"
        )
    
    def _validate_enum(self, series: pd.Series, allowed_values: List[Any], **params) -> ValidationResult:
        """Validate against allowed values"""
        invalid_values = ~series.isin(allowed_values)
        invalid_count = invalid_values.sum()
        
        if invalid_count > 0:
            return ValidationResult(
                field=series.name,
                severity=ValidationSeverity.ERROR,
                message=f"{invalid_count} invalid values found",
                value=invalid_count,
                expected=allowed_values,
                rule="enum"
            )
        
        return ValidationResult(
            field=series.name,
            severity=ValidationSeverity.INFO,
            message="All values are valid",
            rule="enum"
        )
    
    def _validate_sentiment_label(self, series: pd.Series, **params) -> ValidationResult:
        """Validate sentiment labels"""
        valid_labels = ['positive', 'negative', 'neutral']
        return self._validate_enum(series, valid_labels)
    
    def _validate_text_quality(self, series: pd.Series, **params) -> List[ValidationResult]:
        """Validate text quality metrics"""
        results = []
        
        if series.dtype != 'object':
            return [ValidationResult(
                field=series.name,
                severity=ValidationSeverity.ERROR,
                message="Cannot validate text quality on non-string data",
                rule="text_quality"
            )]
        
        # Check for empty strings
        empty_count = (series.str.strip() == '').sum()
        if empty_count > 0:
            results.append(ValidationResult(
                field=series.name,
                severity=ValidationSeverity.WARNING,
                message=f"{empty_count} empty strings found",
                value=empty_count,
                rule="text_quality"
            ))
        
        # Check for very short texts (less than 5 characters)
        too_short = (series.str.len() < 5).sum()
        if too_short > 0:
            results.append(ValidationResult(
                field=series.name,
                severity=ValidationSeverity.WARNING,
                message=f"{too_short} texts are very short (< 5 chars)",
                value=too_short,
                rule="text_quality"
            ))
        
        # Check for very long texts (more than 5000 characters)
        too_long = (series.str.len() > 5000).sum()
        if too_long > 0:
            results.append(ValidationResult(
                field=series.name,
                severity=ValidationSeverity.WARNING,
                message=f"{too_long} texts are very long (> 5000 chars)",
                value=too_long,
                rule="text_quality"
            ))
        
        # Check for suspicious patterns
        suspicious_patterns = [
            (r'^(.)\1{10,}', 'repeated_character'),  # Same character repeated 10+ times
            (r'\d{10,}', 'long_number_sequence'),    # 10+ consecutive digits
            (r'[^\w\s]{20,}', 'special_char_sequence')  # 20+ special characters
        ]
        
        for pattern, description in suspicious_patterns:
            suspicious_count = series.str.contains(pattern, regex=True, na=False).sum()
            if suspicious_count > 0:
                results.append(ValidationResult(
                    field=series.name,
                    severity=ValidationSeverity.WARNING,
                    message=f"{suspicious_count} texts with {description}",
                    value=suspicious_count,
                    rule="text_quality"
                ))
        
        if not results:
            results.append(ValidationResult(
                field=series.name,
                severity=ValidationSeverity.INFO,
                message="Text quality checks passed",
                rule="text_quality"
            ))
        
        return results
    
    def _validate_encoding(self, series: pd.Series, **params) -> ValidationResult:
        """Validate text encoding"""
        if series.dtype != 'object':
            return ValidationResult(
                field=series.name,
                severity=ValidationSeverity.ERROR,
                message="Cannot validate encoding on non-string data",
                rule="encoding"
            )
        
        encoding_issues = 0
        for text in series.dropna():
            try:
                text.encode('utf-8').decode('utf-8')
            except UnicodeError:
                encoding_issues += 1
        
        if encoding_issues > 0:
            return ValidationResult(
                field=series.name,
                severity=ValidationSeverity.WARNING,
                message=f"{encoding_issues} texts with encoding issues",
                value=encoding_issues,
                rule="encoding"
            )
        
        return ValidationResult(
            field=series.name,
            severity=ValidationSeverity.INFO,
            message="No encoding issues found",
            rule="encoding"
        )
    
    def _validate_duplicates(self, series: pd.Series, **params) -> ValidationResult:
        """Check for duplicate values"""
        return self._validate_unique(series)
    
    def _validate_missing_ratio(self, series: pd.Series, max_missing_ratio: float = 0.1, **params) -> ValidationResult:
        """Validate missing value ratio"""
        missing_ratio = series.isnull().sum() / len(series)
        
        if missing_ratio > max_missing_ratio:
            return ValidationResult(
                field=series.name,
                severity=ValidationSeverity.WARNING,
                message=f"Missing ratio {missing_ratio:.2%} exceeds threshold {max_missing_ratio:.2%}",
                value=missing_ratio,
                expected=max_missing_ratio,
                rule="missing_ratio"
            )
        
        return ValidationResult(
            field=series.name,
            severity=ValidationSeverity.INFO,
            message=f"Missing ratio {missing_ratio:.2%} within threshold",
            rule="missing_ratio"
        )
    
    def _validate_statistical_outliers(self, series: pd.Series, **params) -> ValidationResult:
        """Detect statistical outliers using IQR method"""
        if not pd.api.types.is_numeric_dtype(series):
            return ValidationResult(
                field=series.name,
                severity=ValidationSeverity.INFO,
                message="Cannot detect outliers on non-numeric data",
                rule="statistical_outliers"
            )
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((series < lower_bound) | (series > upper_bound)).sum()
        
        if outliers > 0:
            outlier_ratio = outliers / len(series)
            severity = ValidationSeverity.WARNING if outlier_ratio < 0.05 else ValidationSeverity.ERROR
            
            return ValidationResult(
                field=series.name,
                severity=severity,
                message=f"{outliers} statistical outliers found ({outlier_ratio:.2%})",
                value=outliers,
                rule="statistical_outliers"
            )
        
        return ValidationResult(
            field=series.name,
            severity=ValidationSeverity.INFO,
            message="No statistical outliers found",
            rule="statistical_outliers"
        )
    
    def _calculate_quality_score(self, results: List[ValidationResult]) -> float:
        """Calculate overall quality score (0-100)"""
        if not results:
            return 100.0
        
        severity_weights = {
            ValidationSeverity.INFO: 0,
            ValidationSeverity.WARNING: -5,
            ValidationSeverity.ERROR: -20,
            ValidationSeverity.CRITICAL: -50
        }
        
        total_penalty = sum(severity_weights.get(result.severity, 0) for result in results)
        base_score = 100.0
        
        # Apply penalties
        quality_score = max(0.0, base_score + total_penalty)
        
        return quality_score
    
    def _generate_recommendations(self, results: List[ValidationResult], 
                                 df: pd.DataFrame) -> List[str]:
        """Generate data quality recommendations"""
        recommendations = []
        
        # Count issues by severity
        severity_counts = {}
        for result in results:
            severity_counts[result.severity] = severity_counts.get(result.severity, 0) + 1
        
        # Generate specific recommendations
        if severity_counts.get(ValidationSeverity.CRITICAL, 0) > 0:
            recommendations.append("Address critical data quality issues immediately")
        
        if severity_counts.get(ValidationSeverity.ERROR, 0) > 0:
            recommendations.append("Fix data type and validation errors")
        
        if severity_counts.get(ValidationSeverity.WARNING, 0) > 5:
            recommendations.append("Review and clean data to reduce warnings")
        
        # Dataset size recommendations
        if len(df) < 100:
            recommendations.append("Consider collecting more data for better model performance")
        
        # Missing data recommendations
        missing_data = df.isnull().sum().sum()
        if missing_data > 0:
            missing_ratio = missing_data / (len(df) * len(df.columns))
            if missing_ratio > 0.1:
                recommendations.append("Implement missing data imputation strategies")
        
        # Text-specific recommendations
        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            avg_length = df[col].str.len().mean()
            if avg_length < 10:
                recommendations.append(f"Consider enriching short text data in column '{col}'")
        
        if not recommendations:
            recommendations.append("Data quality looks good! Continue monitoring regularly")
        
        return recommendations

def setup_sentiment_data_validation() -> DataValidator:
    """Setup validation rules for sentiment analysis datasets"""
    validator = DataValidator()
    
    # Text column validation
    validator.add_validation_rule('text', 'not_null')
    validator.add_validation_rule('text', 'data_type', expected_type='object')
    validator.add_validation_rule('text', 'length', min_len=1, max_len=10000)
    validator.add_validation_rule('text', 'text_quality')
    validator.add_validation_rule('text', 'encoding')
    
    # Label column validation
    validator.add_validation_rule('label', 'not_null')
    validator.add_validation_rule('label', 'sentiment_label')
    
    # Optional: confidence/score validation
    validator.add_validation_rule('confidence', 'data_type', expected_type='float')
    validator.add_validation_rule('confidence', 'range', min_val=0.0, max_val=1.0)
    
    return validator

if __name__ == "__main__":
    # Test data validation
    validator = setup_sentiment_data_validation()
    
    # Create test data
    test_data = pd.DataFrame({
        'text': ['This is great!', 'I hate this', '', 'A' * 100, None],
        'label': ['positive', 'negative', 'neutral', 'invalid', 'positive'],
        'confidence': [0.9, 0.8, 0.5, 1.2, 0.7]
    })
    
    # Validate data
    report = validator.validate_dataframe(test_data, "test_dataset")
    
    print(f"Quality Score: {report.quality_score:.1f}")
    print(f"Total Issues: {len(report.validation_results)}")
    
    for result in report.validation_results:
        print(f"  {result.severity.value.upper()}: {result.field} - {result.message}")
    
    print("\nRecommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")