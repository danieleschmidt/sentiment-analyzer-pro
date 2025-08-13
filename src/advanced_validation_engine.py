"""Advanced Validation Engine - Comprehensive data and system validation."""

import asyncio
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from pathlib import Path
import hashlib
import threading

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError, validator
import joblib

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationType(Enum):
    """Types of validations."""

    DATA_QUALITY = "data_quality"
    SCHEMA = "schema"
    BUSINESS_RULE = "business_rule"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MODEL = "model"
    CONFIG = "config"
    SYSTEM = "system"


@dataclass
class ValidationResult:
    """Result of a validation check."""

    validator_name: str
    validation_type: ValidationType
    severity: ValidationSeverity
    passed: bool
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    execution_time_ms: float
    context: Dict[str, Any]


@dataclass
class ValidationSummary:
    """Summary of all validation results."""

    total_checks: int
    passed: int
    failed: int
    warnings: int
    errors: int
    critical: int
    execution_time_ms: float
    success_rate: float
    timestamp: datetime
    results: List[ValidationResult]


class BaseValidator(ABC):
    """Base class for all validators."""

    def __init__(
        self,
        name: str,
        validation_type: ValidationType,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ):
        self.name = name
        self.validation_type = validation_type
        self.severity = severity
        self.enabled = True

    @abstractmethod
    def validate(self, data: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Perform validation and return result."""
        pass

    def _create_result(
        self,
        passed: bool,
        message: str,
        details: Dict[str, Any],
        execution_time: float,
        context: Dict[str, Any] = None,
    ) -> ValidationResult:
        """Create validation result."""
        return ValidationResult(
            validator_name=self.name,
            validation_type=self.validation_type,
            severity=self.severity,
            passed=passed,
            message=message,
            details=details or {},
            timestamp=datetime.now(),
            execution_time_ms=execution_time,
            context=context or {},
        )


class DataQualityValidator(BaseValidator):
    """Validates data quality metrics."""

    def __init__(
        self,
        name: str = "data_quality",
        min_completeness: float = 0.95,
        max_duplicates: float = 0.05,
    ):
        super().__init__(name, ValidationType.DATA_QUALITY, ValidationSeverity.WARNING)
        self.min_completeness = min_completeness
        self.max_duplicates = max_duplicates

    def validate(self, data: Any, context: Dict[str, Any] = None) -> ValidationResult:
        start_time = time.time()

        try:
            if not isinstance(data, pd.DataFrame):
                if hasattr(data, "to_pandas"):
                    df = data.to_pandas()
                else:
                    return self._create_result(
                        False,
                        "Data is not a DataFrame or convertible to DataFrame",
                        {"data_type": str(type(data))},
                        (time.time() - start_time) * 1000,
                        context,
                    )
            else:
                df = data

            if df.empty:
                return self._create_result(
                    False,
                    "Dataset is empty",
                    {"row_count": 0},
                    (time.time() - start_time) * 1000,
                    context,
                )

            # Calculate metrics
            total_cells = df.size
            missing_cells = df.isnull().sum().sum()
            completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 0

            duplicate_rows = df.duplicated().sum()
            duplicate_rate = duplicate_rows / len(df) if len(df) > 0 else 0

            # Check quality thresholds
            issues = []
            if completeness < self.min_completeness:
                issues.append(
                    f"Completeness {completeness:.2%} below threshold {self.min_completeness:.2%}"
                )

            if duplicate_rate > self.max_duplicates:
                issues.append(
                    f"Duplicate rate {duplicate_rate:.2%} above threshold {self.max_duplicates:.2%}"
                )

            details = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "completeness": completeness,
                "missing_cells": int(missing_cells),
                "duplicate_rows": int(duplicate_rows),
                "duplicate_rate": duplicate_rate,
                "data_types": df.dtypes.to_dict(),
            }

            passed = len(issues) == 0
            message = "Data quality checks passed" if passed else "; ".join(issues)

            return self._create_result(
                passed, message, details, (time.time() - start_time) * 1000, context
            )

        except Exception as e:
            return self._create_result(
                False,
                f"Data quality validation failed: {str(e)}",
                {"error": str(e)},
                (time.time() - start_time) * 1000,
                context,
            )


class SchemaValidator(BaseValidator):
    """Validates data against expected schema."""

    def __init__(self, name: str = "schema", expected_schema: Dict[str, Any] = None):
        super().__init__(name, ValidationType.SCHEMA, ValidationSeverity.ERROR)
        self.expected_schema = expected_schema or {}

    def validate(self, data: Any, context: Dict[str, Any] = None) -> ValidationResult:
        start_time = time.time()

        try:
            if not isinstance(data, pd.DataFrame):
                return self._create_result(
                    False,
                    "Schema validation requires DataFrame",
                    {"data_type": str(type(data))},
                    (time.time() - start_time) * 1000,
                    context,
                )

            df = data
            issues = []

            # Check required columns
            expected_columns = set(self.expected_schema.get("columns", []))
            actual_columns = set(df.columns)

            missing_columns = expected_columns - actual_columns
            extra_columns = actual_columns - expected_columns

            if missing_columns:
                issues.append(f"Missing columns: {list(missing_columns)}")

            # Check data types
            expected_types = self.expected_schema.get("types", {})
            type_mismatches = []

            for col, expected_type in expected_types.items():
                if col in df.columns:
                    actual_type = str(df[col].dtype)
                    if not self._types_compatible(actual_type, expected_type):
                        type_mismatches.append(
                            f"{col}: expected {expected_type}, got {actual_type}"
                        )

            if type_mismatches:
                issues.append(f"Type mismatches: {type_mismatches}")

            # Check constraints
            constraints = self.expected_schema.get("constraints", {})
            constraint_violations = []

            for col, constraint in constraints.items():
                if col in df.columns:
                    violations = self._check_constraints(df[col], constraint)
                    if violations:
                        constraint_violations.extend(
                            [f"{col}: {v}" for v in violations]
                        )

            if constraint_violations:
                issues.append(f"Constraint violations: {constraint_violations}")

            details = {
                "expected_columns": list(expected_columns),
                "actual_columns": list(actual_columns),
                "missing_columns": list(missing_columns),
                "extra_columns": list(extra_columns),
                "type_mismatches": type_mismatches,
                "constraint_violations": constraint_violations,
            }

            passed = len(issues) == 0
            message = "Schema validation passed" if passed else "; ".join(issues)

            return self._create_result(
                passed, message, details, (time.time() - start_time) * 1000, context
            )

        except Exception as e:
            return self._create_result(
                False,
                f"Schema validation failed: {str(e)}",
                {"error": str(e)},
                (time.time() - start_time) * 1000,
                context,
            )

    def _types_compatible(self, actual: str, expected: str) -> bool:
        """Check if data types are compatible."""
        type_mappings = {
            "int": ["int64", "int32", "int"],
            "float": ["float64", "float32", "float"],
            "string": ["object", "string"],
            "bool": ["bool"],
            "datetime": ["datetime64", "datetime"],
        }

        for exp_type, compatible_types in type_mappings.items():
            if expected.lower() == exp_type:
                return any(comp in actual.lower() for comp in compatible_types)

        return actual.lower() == expected.lower()

    def _check_constraints(
        self, series: pd.Series, constraints: Dict[str, Any]
    ) -> List[str]:
        """Check column constraints."""
        violations = []

        if "min" in constraints:
            if series.min() < constraints["min"]:
                violations.append(
                    f"minimum value {series.min()} below {constraints['min']}"
                )

        if "max" in constraints:
            if series.max() > constraints["max"]:
                violations.append(
                    f"maximum value {series.max()} above {constraints['max']}"
                )

        if "allowed_values" in constraints:
            invalid_values = set(series.unique()) - set(constraints["allowed_values"])
            if invalid_values:
                violations.append(f"invalid values: {list(invalid_values)}")

        if "pattern" in constraints and series.dtype == "object":
            pattern = re.compile(constraints["pattern"])
            invalid_count = sum(
                1 for val in series.dropna() if not pattern.match(str(val))
            )
            if invalid_count > 0:
                violations.append(f"{invalid_count} values don't match pattern")

        return violations


class SecurityValidator(BaseValidator):
    """Validates security aspects of data and inputs."""

    def __init__(self, name: str = "security"):
        super().__init__(name, ValidationType.SECURITY, ValidationSeverity.CRITICAL)
        self.suspicious_patterns = [
            r"<script[^>]*>.*?</script>",  # XSS
            r"union\s+select",  # SQL injection
            r"javascript:",  # JavaScript execution
            r"\.\./.*",  # Path traversal
            r"eval\s*\(",  # Code execution
        ]

    def validate(self, data: Any, context: Dict[str, Any] = None) -> ValidationResult:
        start_time = time.time()

        try:
            security_issues = []

            # Convert data to strings for pattern matching
            if isinstance(data, pd.DataFrame):
                text_data = (
                    data.select_dtypes(include=["object"]).astype(str).values.flatten()
                )
            elif isinstance(data, (list, tuple)):
                text_data = [str(item) for item in data]
            elif isinstance(data, dict):
                text_data = [str(value) for value in data.values()]
            else:
                text_data = [str(data)]

            # Check for suspicious patterns
            for text in text_data:
                if pd.isna(text) or text == "nan":
                    continue

                for pattern in self.suspicious_patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        security_issues.append(
                            f"Suspicious pattern detected: {pattern}"
                        )
                        break

            # Check for potential data exfiltration
            large_text_count = sum(1 for text in text_data if len(str(text)) > 10000)
            if large_text_count > 0:
                security_issues.append(
                    f"{large_text_count} unusually large text fields detected"
                )

            # Check for potential PII
            pii_patterns = {
                "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
                "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            }

            pii_found = []
            for text in text_data:
                if pd.isna(text) or text == "nan":
                    continue

                for pii_type, pattern in pii_patterns.items():
                    if re.search(pattern, str(text)):
                        pii_found.append(pii_type)

            details = {
                "security_issues": security_issues,
                "pii_detected": list(set(pii_found)),
                "text_fields_checked": len(text_data),
                "large_text_count": large_text_count,
            }

            # Security validation passes if no critical issues found
            critical_issues = len(security_issues)
            passed = critical_issues == 0

            if not passed:
                message = (
                    f"Security validation failed: {len(security_issues)} issues found"
                )
            elif pii_found:
                message = f"Security validation passed with warnings: PII detected ({', '.join(set(pii_found))})"
            else:
                message = "Security validation passed"

            return self._create_result(
                passed, message, details, (time.time() - start_time) * 1000, context
            )

        except Exception as e:
            return self._create_result(
                False,
                f"Security validation failed: {str(e)}",
                {"error": str(e)},
                (time.time() - start_time) * 1000,
                context,
            )


class ModelValidator(BaseValidator):
    """Validates model performance and characteristics."""

    def __init__(
        self,
        name: str = "model",
        min_accuracy: float = 0.8,
        max_latency_ms: float = 100,
    ):
        super().__init__(name, ValidationType.MODEL, ValidationSeverity.ERROR)
        self.min_accuracy = min_accuracy
        self.max_latency_ms = max_latency_ms

    def validate(self, data: Any, context: Dict[str, Any] = None) -> ValidationResult:
        start_time = time.time()

        try:
            if not hasattr(data, "predict"):
                return self._create_result(
                    False,
                    "Object is not a valid model (no predict method)",
                    {"has_predict": False},
                    (time.time() - start_time) * 1000,
                    context,
                )

            model = data
            issues = []

            # Test model with sample data
            test_texts = [
                "This is a positive review",
                "This is a negative review",
                "This is a neutral statement",
            ]

            # Test prediction latency
            prediction_times = []
            predictions = []

            for text in test_texts:
                pred_start = time.time()
                try:
                    pred = model.predict([text])[0]
                    pred_time = (time.time() - pred_start) * 1000
                    prediction_times.append(pred_time)
                    predictions.append(pred)
                except Exception as e:
                    issues.append(f"Prediction failed for '{text}': {str(e)}")

            avg_latency = (
                np.mean(prediction_times) if prediction_times else float("inf")
            )

            # Check latency threshold
            if avg_latency > self.max_latency_ms:
                issues.append(
                    f"Average prediction latency {avg_latency:.2f}ms exceeds threshold {self.max_latency_ms}ms"
                )

            # Check prediction diversity
            unique_predictions = len(set(predictions))
            if unique_predictions == 1 and len(predictions) > 1:
                issues.append(
                    "Model produces identical predictions for different inputs"
                )

            # Test model consistency
            consistency_test = "Test message for consistency"
            consistency_predictions = []
            for _ in range(3):
                try:
                    pred = model.predict([consistency_test])[0]
                    consistency_predictions.append(pred)
                except Exception as e:
                    issues.append(f"Consistency test failed: {str(e)}")

            if len(set(consistency_predictions)) > 1:
                issues.append("Model predictions are inconsistent for identical inputs")

            details = {
                "prediction_times_ms": prediction_times,
                "average_latency_ms": avg_latency,
                "predictions": predictions,
                "unique_predictions": unique_predictions,
                "consistency_predictions": consistency_predictions,
                "has_predict_method": hasattr(model, "predict"),
                "model_type": str(type(model).__name__),
            }

            passed = len(issues) == 0
            message = "Model validation passed" if passed else "; ".join(issues)

            return self._create_result(
                passed, message, details, (time.time() - start_time) * 1000, context
            )

        except Exception as e:
            return self._create_result(
                False,
                f"Model validation failed: {str(e)}",
                {"error": str(e)},
                (time.time() - start_time) * 1000,
                context,
            )


class ConfigValidator(BaseValidator):
    """Validates configuration files and settings."""

    def __init__(self, name: str = "config", required_keys: List[str] = None):
        super().__init__(name, ValidationType.CONFIG, ValidationSeverity.ERROR)
        self.required_keys = required_keys or []

    def validate(self, data: Any, context: Dict[str, Any] = None) -> ValidationResult:
        start_time = time.time()

        try:
            if isinstance(data, (str, Path)):
                # Load configuration file
                config_path = Path(data)
                if not config_path.exists():
                    return self._create_result(
                        False,
                        f"Configuration file not found: {config_path}",
                        {"file_exists": False},
                        (time.time() - start_time) * 1000,
                        context,
                    )

                try:
                    with open(config_path, "r") as f:
                        if config_path.suffix.lower() == ".json":
                            config_data = json.load(f)
                        else:
                            # Assume it's a text file with key=value pairs
                            config_data = {}
                            for line in f:
                                if "=" in line and not line.strip().startswith("#"):
                                    key, value = line.strip().split("=", 1)
                                    config_data[key.strip()] = value.strip()
                except Exception as e:
                    return self._create_result(
                        False,
                        f"Failed to parse configuration file: {str(e)}",
                        {"parse_error": str(e)},
                        (time.time() - start_time) * 1000,
                        context,
                    )
            elif isinstance(data, dict):
                config_data = data
            else:
                return self._create_result(
                    False,
                    "Configuration must be a file path or dictionary",
                    {"data_type": str(type(data))},
                    (time.time() - start_time) * 1000,
                    context,
                )

            issues = []

            # Check required keys
            missing_keys = [key for key in self.required_keys if key not in config_data]
            if missing_keys:
                issues.append(f"Missing required keys: {missing_keys}")

            # Check for invalid values
            invalid_values = []
            for key, value in config_data.items():
                if value is None or (isinstance(value, str) and value.strip() == ""):
                    invalid_values.append(key)

            if invalid_values:
                issues.append(f"Keys with invalid/empty values: {invalid_values}")

            # Check for security issues in config
            security_keys = ["password", "secret", "key", "token", "api_key"]
            exposed_secrets = []

            for key in config_data:
                if any(sec_key in key.lower() for sec_key in security_keys):
                    value = str(config_data[key])
                    if len(value) < 8:  # Very short secrets are suspicious
                        exposed_secrets.append(f"{key}: suspiciously short")
                    elif value.lower() in ["password", "secret", "key", "123456"]:
                        exposed_secrets.append(f"{key}: using default/weak value")

            if exposed_secrets:
                issues.append(f"Security concerns: {exposed_secrets}")

            details = {
                "config_keys": list(config_data.keys()),
                "required_keys": self.required_keys,
                "missing_keys": missing_keys,
                "invalid_values": invalid_values,
                "security_issues": exposed_secrets,
                "config_size": len(config_data),
            }

            passed = len(issues) == 0
            message = "Configuration validation passed" if passed else "; ".join(issues)

            return self._create_result(
                passed, message, details, (time.time() - start_time) * 1000, context
            )

        except Exception as e:
            return self._create_result(
                False,
                f"Configuration validation failed: {str(e)}",
                {"error": str(e)},
                (time.time() - start_time) * 1000,
                context,
            )


class AdvancedValidationEngine:
    """Main validation engine that orchestrates all validators."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.validators: Dict[str, BaseValidator] = {}
        self.validation_history: List[ValidationSummary] = []
        self.auto_fix_enabled = self.config.get("auto_fix_enabled", False)

        # Register default validators
        self._register_default_validators()

    def _register_default_validators(self):
        """Register default validators."""
        self.validators.update(
            {
                "data_quality": DataQualityValidator(),
                "schema": SchemaValidator(),
                "security": SecurityValidator(),
                "model": ModelValidator(),
                "config": ConfigValidator(),
            }
        )

    def register_validator(self, validator: BaseValidator):
        """Register a custom validator."""
        self.validators[validator.name] = validator
        logger.info(f"Registered validator: {validator.name}")

    def unregister_validator(self, validator_name: str):
        """Unregister a validator."""
        if validator_name in self.validators:
            del self.validators[validator_name]
            logger.info(f"Unregistered validator: {validator_name}")

    def validate(
        self,
        data: Any,
        validator_names: Optional[List[str]] = None,
        context: Dict[str, Any] = None,
        parallel: bool = True,
    ) -> ValidationSummary:
        """Run validation checks."""
        start_time = time.time()

        # Determine which validators to run
        if validator_names is None:
            validators_to_run = list(self.validators.values())
        else:
            validators_to_run = [
                self.validators[name]
                for name in validator_names
                if name in self.validators
            ]

        # Filter enabled validators
        validators_to_run = [v for v in validators_to_run if v.enabled]

        results = []

        if parallel and len(validators_to_run) > 1:
            # Run validators in parallel
            results = self._run_validators_parallel(validators_to_run, data, context)
        else:
            # Run validators sequentially
            for validator in validators_to_run:
                try:
                    result = validator.validate(data, context)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Validator {validator.name} failed: {e}")
                    # Create error result
                    error_result = ValidationResult(
                        validator_name=validator.name,
                        validation_type=validator.validation_type,
                        severity=ValidationSeverity.ERROR,
                        passed=False,
                        message=f"Validator execution failed: {str(e)}",
                        details={"error": str(e)},
                        timestamp=datetime.now(),
                        execution_time_ms=0.0,
                        context=context or {},
                    )
                    results.append(error_result)

        # Create summary
        summary = self._create_summary(results, time.time() - start_time)

        # Store in history
        self.validation_history.append(summary)

        # Auto-fix if enabled and there are fixable issues
        if self.auto_fix_enabled and not summary.passed:
            self._attempt_auto_fix(data, results)

        return summary

    def _run_validators_parallel(
        self, validators: List[BaseValidator], data: Any, context: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Run validators in parallel using threading."""
        results = []
        threads = []
        result_lock = threading.Lock()

        def run_validator(validator):
            try:
                result = validator.validate(data, context)
                with result_lock:
                    results.append(result)
            except Exception as e:
                logger.error(f"Validator {validator.name} failed: {e}")
                error_result = ValidationResult(
                    validator_name=validator.name,
                    validation_type=validator.validation_type,
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Validator execution failed: {str(e)}",
                    details={"error": str(e)},
                    timestamp=datetime.now(),
                    execution_time_ms=0.0,
                    context=context or {},
                )
                with result_lock:
                    results.append(error_result)

        # Start threads
        for validator in validators:
            thread = threading.Thread(target=run_validator, args=(validator,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout per validator

        return results

    def _create_summary(
        self, results: List[ValidationResult], execution_time: float
    ) -> ValidationSummary:
        """Create validation summary from results."""
        total_checks = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total_checks - passed

        warnings = sum(
            1
            for r in results
            if not r.passed and r.severity == ValidationSeverity.WARNING
        )
        errors = sum(
            1
            for r in results
            if not r.passed and r.severity == ValidationSeverity.ERROR
        )
        critical = sum(
            1
            for r in results
            if not r.passed and r.severity == ValidationSeverity.CRITICAL
        )

        success_rate = (passed / total_checks) * 100 if total_checks > 0 else 0

        return ValidationSummary(
            total_checks=total_checks,
            passed=passed,
            failed=failed,
            warnings=warnings,
            errors=errors,
            critical=critical,
            execution_time_ms=execution_time * 1000,
            success_rate=success_rate,
            timestamp=datetime.now(),
            results=results,
        )

    def _attempt_auto_fix(self, data: Any, results: List[ValidationResult]):
        """Attempt to automatically fix validation issues."""
        logger.info("Attempting auto-fix for validation issues")

        for result in results:
            if not result.passed and result.severity != ValidationSeverity.CRITICAL:
                try:
                    self._auto_fix_issue(result, data)
                except Exception as e:
                    logger.warning(f"Auto-fix failed for {result.validator_name}: {e}")

    def _auto_fix_issue(self, result: ValidationResult, data: Any):
        """Auto-fix a specific validation issue."""
        # This is a placeholder for auto-fix logic
        # In practice, you would implement specific fixes for each validator type
        logger.info(
            f"Would attempt to fix issue from {result.validator_name}: {result.message}"
        )

    def validate_async(
        self,
        data: Any,
        validator_names: Optional[List[str]] = None,
        context: Dict[str, Any] = None,
    ) -> asyncio.Future:
        """Run validation asynchronously."""
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(
            None, self.validate, data, validator_names, context, True
        )

    def get_validation_history(
        self, limit: Optional[int] = None
    ) -> List[ValidationSummary]:
        """Get validation history."""
        if limit is None:
            return self.validation_history.copy()
        else:
            return self.validation_history[-limit:]

    def get_validator_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for each validator."""
        validator_stats = {}

        for summary in self.validation_history:
            for result in summary.results:
                validator_name = result.validator_name

                if validator_name not in validator_stats:
                    validator_stats[validator_name] = {
                        "total_runs": 0,
                        "total_time_ms": 0.0,
                        "success_count": 0,
                        "failure_count": 0,
                        "avg_time_ms": 0.0,
                        "success_rate": 0.0,
                    }

                stats = validator_stats[validator_name]
                stats["total_runs"] += 1
                stats["total_time_ms"] += result.execution_time_ms

                if result.passed:
                    stats["success_count"] += 1
                else:
                    stats["failure_count"] += 1

                stats["avg_time_ms"] = stats["total_time_ms"] / stats["total_runs"]
                stats["success_rate"] = (
                    stats["success_count"] / stats["total_runs"]
                ) * 100

        return validator_stats

    def enable_validator(self, validator_name: str):
        """Enable a validator."""
        if validator_name in self.validators:
            self.validators[validator_name].enabled = True
            logger.info(f"Enabled validator: {validator_name}")

    def disable_validator(self, validator_name: str):
        """Disable a validator."""
        if validator_name in self.validators:
            self.validators[validator_name].enabled = False
            logger.info(f"Disabled validator: {validator_name}")

    def get_enabled_validators(self) -> List[str]:
        """Get list of enabled validators."""
        return [
            name for name, validator in self.validators.items() if validator.enabled
        ]

    def export_results(self, summary: ValidationSummary, filepath: str):
        """Export validation results to file."""
        try:
            export_data = {
                "summary": asdict(summary),
                "results": [asdict(result) for result in summary.results],
            }

            # Handle datetime serialization
            def serialize_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2, default=serialize_datetime)

            logger.info(f"Validation results exported to {filepath}")

        except Exception as e:
            logger.error(f"Failed to export results: {e}")

    def clear_history(self):
        """Clear validation history."""
        self.validation_history.clear()
        logger.info("Validation history cleared")


# Global validation engine instance
_validation_engine = None


def get_validation_engine() -> AdvancedValidationEngine:
    """Get global validation engine instance."""
    global _validation_engine
    if _validation_engine is None:
        _validation_engine = AdvancedValidationEngine()
    return _validation_engine


def validate_data(
    data: Any,
    validator_names: Optional[List[str]] = None,
    context: Dict[str, Any] = None,
) -> ValidationSummary:
    """Validate data using the global validation engine."""
    engine = get_validation_engine()
    return engine.validate(data, validator_names, context)


def register_custom_validator(validator: BaseValidator):
    """Register a custom validator."""
    engine = get_validation_engine()
    engine.register_validator(validator)
