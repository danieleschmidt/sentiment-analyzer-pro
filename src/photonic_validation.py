"""
Photonic-MLIR Bridge - Advanced Validation Framework

This module provides comprehensive validation for photonic circuits, components,
and synthesis operations with multi-level validation strategies.
"""

import re
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation levels for different use cases."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


class ValidationResult(Enum):
    """Validation result status."""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    level: ValidationResult
    category: str
    message: str
    component_id: Optional[str] = None
    suggestion: Optional[str] = None
    auto_fixable: bool = False


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    timestamp: float
    validation_level: ValidationLevel
    overall_result: ValidationResult
    issues: List[ValidationIssue]
    statistics: Dict[str, Any]
    recommendations: List[str]
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return self.overall_result in (ValidationResult.PASS, ValidationResult.WARNING)
    
    @property
    def has_critical_issues(self) -> bool:
        """Check for critical validation issues."""
        return any(issue.level == ValidationResult.CRITICAL for issue in self.issues)


class ValidationRule(ABC):
    """Abstract base class for validation rules."""
    
    def __init__(self, name: str, category: str, severity: ValidationResult = ValidationResult.FAIL):
        self.name = name
        self.category = category
        self.severity = severity
    
    @abstractmethod
    def validate(self, target: Any) -> List[ValidationIssue]:
        """Validate target and return issues."""
        pass


class PhotonicValidator:
    """Advanced photonic circuit validator with comprehensive rules."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.rules: Dict[str, List[ValidationRule]] = {
            "component": [],
            "circuit": [],
            "connection": [],
            "synthesis": []
        }
        self.custom_rules: List[ValidationRule] = []
        
        self._initialize_validation_rules()
    
    def _initialize_validation_rules(self):
        """Initialize comprehensive validation rules."""
        
        # Component validation rules
        self.rules["component"].extend([
            ComponentTypeValidationRule(),
            ComponentParameterValidationRule(),
            ComponentPositionValidationRule(),
            ComponentWavelengthValidationRule(),
            ComponentPhysicalConstraintsRule()
        ])
        
        # Circuit validation rules
        self.rules["circuit"].extend([
            CircuitNameValidationRule(),
            CircuitTopologyValidationRule(),
            CircuitConnectivityRule(),
            CircuitPowerBudgetRule(),
            CircuitScalabilityRule()
        ])
        
        # Connection validation rules
        self.rules["connection"].extend([
            ConnectionValidityRule(),
            ConnectionLossValidationRule(),
            ConnectionDelayValidationRule(),
            ConnectionPortValidationRule()
        ])
        
        # Synthesis validation rules
        self.rules["synthesis"].extend([
            SynthesisComplexityRule(),
            SynthesisConstraintsRule(),
            SynthesisFeasibilityRule()
        ])
        
        logger.info(f"Initialized validator with {sum(len(rules) for rules in self.rules.values())} rules")
    
    def validate_component(self, component) -> ValidationReport:
        """Validate a single photonic component."""
        import time
        
        timestamp = time.time()
        issues = []
        
        # Apply component validation rules
        for rule in self.rules["component"]:
            try:
                rule_issues = rule.validate(component)
                issues.extend(rule_issues)
            except Exception as e:
                logger.warning(f"Validation rule {rule.name} failed: {e}")
                issues.append(ValidationIssue(
                    level=ValidationResult.WARNING,
                    category="validation_error",
                    message=f"Rule {rule.name} failed: {e}",
                    component_id=component.id if hasattr(component, 'id') else None
                ))
        
        # Determine overall result
        overall_result = self._determine_overall_result(issues)
        
        # Generate statistics
        statistics = self._generate_statistics(issues)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, "component")
        
        return ValidationReport(
            timestamp=timestamp,
            validation_level=self.validation_level,
            overall_result=overall_result,
            issues=issues,
            statistics=statistics,
            recommendations=recommendations
        )
    
    def validate_circuit(self, circuit) -> ValidationReport:
        """Validate a complete photonic circuit."""
        import time
        
        timestamp = time.time()
        issues = []
        
        # Validate individual components first
        for component in circuit.components:
            component_report = self.validate_component(component)
            # Add component issues with context
            for issue in component_report.issues:
                issue.component_id = component.id
                issues.append(issue)
        
        # Validate connections
        for connection in circuit.connections:
            connection_issues = self._validate_connection(connection, circuit)
            issues.extend(connection_issues)
        
        # Apply circuit-level validation rules
        for rule in self.rules["circuit"]:
            try:
                rule_issues = rule.validate(circuit)
                issues.extend(rule_issues)
            except Exception as e:
                logger.warning(f"Circuit validation rule {rule.name} failed: {e}")
                issues.append(ValidationIssue(
                    level=ValidationResult.WARNING,
                    category="validation_error",
                    message=f"Circuit rule {rule.name} failed: {e}"
                ))
        
        # Determine overall result
        overall_result = self._determine_overall_result(issues)
        
        # Generate statistics
        statistics = self._generate_statistics(issues)
        statistics.update({
            "component_count": len(circuit.components),
            "connection_count": len(circuit.connections),
            "circuit_name": circuit.name
        })
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, "circuit")
        
        return ValidationReport(
            timestamp=timestamp,
            validation_level=self.validation_level,
            overall_result=overall_result,
            issues=issues,
            statistics=statistics,
            recommendations=recommendations
        )
    
    def validate_synthesis_input(self, circuit, synthesis_params: Dict[str, Any] = None) -> ValidationReport:
        """Validate inputs before synthesis."""
        import time
        
        timestamp = time.time()
        issues = []
        
        # First validate the circuit
        circuit_report = self.validate_circuit(circuit)
        issues.extend(circuit_report.issues)
        
        # Apply synthesis validation rules
        synthesis_context = {
            "circuit": circuit,
            "parameters": synthesis_params or {}
        }
        
        for rule in self.rules["synthesis"]:
            try:
                rule_issues = rule.validate(synthesis_context)
                issues.extend(rule_issues)
            except Exception as e:
                logger.warning(f"Synthesis validation rule {rule.name} failed: {e}")
                issues.append(ValidationIssue(
                    level=ValidationResult.WARNING,
                    category="validation_error",
                    message=f"Synthesis rule {rule.name} failed: {e}"
                ))
        
        # Determine overall result
        overall_result = self._determine_overall_result(issues)
        
        # Generate statistics
        statistics = self._generate_statistics(issues)
        statistics.update({
            "synthesis_ready": overall_result in (ValidationResult.PASS, ValidationResult.WARNING),
            "estimated_complexity": self._estimate_synthesis_complexity(circuit)
        })
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, "synthesis")
        
        return ValidationReport(
            timestamp=timestamp,
            validation_level=self.validation_level,
            overall_result=overall_result,
            issues=issues,
            statistics=statistics,
            recommendations=recommendations
        )
    
    def _validate_connection(self, connection, circuit) -> List[ValidationIssue]:
        """Validate a single connection."""
        issues = []
        
        for rule in self.rules["connection"]:
            try:
                context = {"connection": connection, "circuit": circuit}
                rule_issues = rule.validate(context)
                issues.extend(rule_issues)
            except Exception as e:
                logger.warning(f"Connection validation rule {rule.name} failed: {e}")
                issues.append(ValidationIssue(
                    level=ValidationResult.WARNING,
                    category="validation_error",
                    message=f"Connection rule {rule.name} failed: {e}"
                ))
        
        return issues
    
    def _determine_overall_result(self, issues: List[ValidationIssue]) -> ValidationResult:
        """Determine overall validation result from issues."""
        if not issues:
            return ValidationResult.PASS
        
        # Check for critical issues
        if any(issue.level == ValidationResult.CRITICAL for issue in issues):
            return ValidationResult.CRITICAL
        
        # Check for failures
        if any(issue.level == ValidationResult.FAIL for issue in issues):
            return ValidationResult.FAIL
        
        # Check for warnings
        if any(issue.level == ValidationResult.WARNING for issue in issues):
            return ValidationResult.WARNING
        
        return ValidationResult.PASS
    
    def _generate_statistics(self, issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Generate validation statistics."""
        stats = {
            "total_issues": len(issues),
            "by_level": {},
            "by_category": {},
            "auto_fixable": sum(1 for issue in issues if issue.auto_fixable)
        }
        
        for issue in issues:
            # By level
            level = issue.level.value
            stats["by_level"][level] = stats["by_level"].get(level, 0) + 1
            
            # By category
            category = issue.category
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
        
        return stats
    
    def _generate_recommendations(self, issues: List[ValidationIssue], context: str) -> List[str]:
        """Generate validation recommendations."""
        recommendations = []
        
        # Critical issues
        critical_issues = [issue for issue in issues if issue.level == ValidationResult.CRITICAL]
        if critical_issues:
            recommendations.append(f"🚨 Address {len(critical_issues)} critical issue(s) before proceeding")
        
        # Auto-fixable issues
        auto_fixable = [issue for issue in issues if issue.auto_fixable]
        if auto_fixable:
            recommendations.append(f"🔧 {len(auto_fixable)} issue(s) can be automatically fixed")
        
        # Context-specific recommendations
        if context == "component":
            recommendations.extend([
                "Verify component parameters are within physical limits",
                "Check component positioning for manufacturing constraints"
            ])
        elif context == "circuit":
            recommendations.extend([
                "Validate circuit topology for optical feasibility",
                "Review power budget and loss calculations"
            ])
        elif context == "synthesis":
            recommendations.extend([
                "Optimize circuit complexity for synthesis performance",
                "Consider manufacturing constraints in design"
            ])
        
        # Category-specific recommendations
        categories = set(issue.category for issue in issues)
        if "security" in categories:
            recommendations.append("🛡️ Review security validation issues")
        if "performance" in categories:
            recommendations.append("⚡ Address performance validation concerns")
        
        return recommendations
    
    def _estimate_synthesis_complexity(self, circuit) -> str:
        """Estimate synthesis complexity."""
        component_count = len(circuit.components)
        connection_count = len(circuit.connections)
        
        complexity_score = component_count + (connection_count * 0.5)
        
        if complexity_score < 10:
            return "low"
        elif complexity_score < 50:
            return "medium"
        elif complexity_score < 100:
            return "high"
        else:
            return "very_high"


# Specific validation rule implementations
class ComponentTypeValidationRule(ValidationRule):
    """Validate component type."""
    
    def __init__(self):
        super().__init__("component_type", "component_structure")
    
    def validate(self, component) -> List[ValidationIssue]:
        issues = []
        
        if not hasattr(component, 'component_type'):
            issues.append(ValidationIssue(
                level=ValidationResult.CRITICAL,
                category=self.category,
                message="Component missing type specification",
                suggestion="Add component_type attribute"
            ))
        
        return issues


class ComponentParameterValidationRule(ValidationRule):
    """Validate component parameters."""
    
    def __init__(self):
        super().__init__("component_parameters", "component_parameters")
    
    def validate(self, component) -> List[ValidationIssue]:
        issues = []
        
        if not hasattr(component, 'parameters') or not component.parameters:
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                category=self.category,
                message="Component has no parameters defined",
                suggestion="Add relevant physical parameters"
            ))
        else:
            # Validate parameter values
            for param, value in component.parameters.items():
                if isinstance(value, (int, float)):
                    if value < 0 and param in ['length', 'width', 'radius', 'power']:
                        issues.append(ValidationIssue(
                            level=ValidationResult.FAIL,
                            category=self.category,
                            message=f"Parameter '{param}' cannot be negative: {value}",
                            suggestion=f"Set {param} to positive value",
                            auto_fixable=True
                        ))
        
        return issues


class ComponentPositionValidationRule(ValidationRule):
    """Validate component position."""
    
    def __init__(self):
        super().__init__("component_position", "component_layout")
    
    def validate(self, component) -> List[ValidationIssue]:
        issues = []
        
        if not hasattr(component, 'position'):
            issues.append(ValidationIssue(
                level=ValidationResult.FAIL,
                category=self.category,
                message="Component missing position information",
                suggestion="Add position attribute as (x, y) tuple"
            ))
        elif len(component.position) != 2:
            issues.append(ValidationIssue(
                level=ValidationResult.FAIL,
                category=self.category,
                message="Component position must be 2D coordinate",
                suggestion="Use (x, y) tuple for position"
            ))
        
        return issues


class ComponentWavelengthValidationRule(ValidationRule):
    """Validate component wavelength band."""
    
    def __init__(self):
        super().__init__("component_wavelength", "optical_properties")
    
    def validate(self, component) -> List[ValidationIssue]:
        issues = []
        
        if hasattr(component, 'wavelength_band'):
            # Check if wavelength parameters are consistent
            if 'wavelength' in component.parameters:
                wavelength = component.parameters['wavelength']
                if wavelength < 1000 or wavelength > 2000:
                    issues.append(ValidationIssue(
                        level=ValidationResult.WARNING,
                        category=self.category,
                        message=f"Wavelength {wavelength}nm outside typical range (1000-2000nm)",
                        suggestion="Verify wavelength specification"
                    ))
        
        return issues


class ComponentPhysicalConstraintsRule(ValidationRule):
    """Validate physical manufacturing constraints."""
    
    def __init__(self):
        super().__init__("physical_constraints", "manufacturing")
    
    def validate(self, component) -> List[ValidationIssue]:
        issues = []
        
        # Check minimum feature sizes
        if 'width' in component.parameters:
            width = component.parameters['width']
            if width < 0.1:  # 100nm minimum
                issues.append(ValidationIssue(
                    level=ValidationResult.WARNING,
                    category=self.category,
                    message=f"Width {width}μm below manufacturing limit (0.1μm)",
                    suggestion="Increase width to meet manufacturing constraints"
                ))
        
        return issues


class CircuitNameValidationRule(ValidationRule):
    """Validate circuit name."""
    
    def __init__(self):
        super().__init__("circuit_name", "circuit_metadata")
    
    def validate(self, circuit) -> List[ValidationIssue]:
        issues = []
        
        if not circuit.name or circuit.name == "untitled_circuit":
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                category=self.category,
                message="Circuit has default or empty name",
                suggestion="Provide descriptive circuit name"
            ))
        
        return issues


class CircuitTopologyValidationRule(ValidationRule):
    """Validate circuit topology."""
    
    def __init__(self):
        super().__init__("circuit_topology", "circuit_structure")
    
    def validate(self, circuit) -> List[ValidationIssue]:
        issues = []
        
        if len(circuit.components) == 0:
            issues.append(ValidationIssue(
                level=ValidationResult.CRITICAL,
                category=self.category,
                message="Circuit contains no components",
                suggestion="Add photonic components to circuit"
            ))
        
        if len(circuit.connections) == 0 and len(circuit.components) > 1:
            issues.append(ValidationIssue(
                level=ValidationResult.FAIL,
                category=self.category,
                message="Multi-component circuit has no connections",
                suggestion="Add connections between components"
            ))
        
        return issues


class CircuitConnectivityRule(ValidationRule):
    """Validate circuit connectivity."""
    
    def __init__(self):
        super().__init__("circuit_connectivity", "circuit_structure")
    
    def validate(self, circuit) -> List[ValidationIssue]:
        issues = []
        
        # Check for isolated components
        component_ids = {c.id for c in circuit.components}
        connected_ids = set()
        
        for conn in circuit.connections:
            connected_ids.add(conn.source_component)
            connected_ids.add(conn.target_component)
        
        isolated = component_ids - connected_ids
        if isolated and len(circuit.components) > 1:
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                category=self.category,
                message=f"Found {len(isolated)} isolated component(s)",
                suggestion="Connect all components or remove unused ones"
            ))
        
        return issues


class CircuitPowerBudgetRule(ValidationRule):
    """Validate optical power budget."""
    
    def __init__(self):
        super().__init__("power_budget", "optical_performance")
    
    def validate(self, circuit) -> List[ValidationIssue]:
        issues = []
        
        # Calculate total loss
        total_loss = sum(conn.loss_db for conn in circuit.connections)
        
        if total_loss > 20:  # 20dB threshold
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                category=self.category,
                message=f"High total loss: {total_loss:.1f}dB",
                suggestion="Review component selection and optimize for lower loss"
            ))
        
        return issues


class CircuitScalabilityRule(ValidationRule):
    """Validate circuit scalability."""
    
    def __init__(self):
        super().__init__("scalability", "performance")
    
    def validate(self, circuit) -> List[ValidationIssue]:
        issues = []
        
        component_count = len(circuit.components)
        if component_count > 1000:
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                category=self.category,
                message=f"Large circuit with {component_count} components",
                suggestion="Consider hierarchical design or optimization"
            ))
        
        return issues


class ConnectionValidityRule(ValidationRule):
    """Validate connection validity."""
    
    def __init__(self):
        super().__init__("connection_validity", "connection_structure")
    
    def validate(self, context) -> List[ValidationIssue]:
        issues = []
        connection = context["connection"]
        circuit = context["circuit"]
        
        # Check if source and target components exist
        component_ids = {c.id for c in circuit.components}
        
        if connection.source_component not in component_ids:
            issues.append(ValidationIssue(
                level=ValidationResult.CRITICAL,
                category=self.category,
                message=f"Source component {connection.source_component} not found",
                suggestion="Verify component ID exists in circuit"
            ))
        
        if connection.target_component not in component_ids:
            issues.append(ValidationIssue(
                level=ValidationResult.CRITICAL,
                category=self.category,
                message=f"Target component {connection.target_component} not found",
                suggestion="Verify component ID exists in circuit"
            ))
        
        return issues


class ConnectionLossValidationRule(ValidationRule):
    """Validate connection loss values."""
    
    def __init__(self):
        super().__init__("connection_loss", "optical_performance")
    
    def validate(self, context) -> List[ValidationIssue]:
        issues = []
        connection = context["connection"]
        
        if connection.loss_db < 0:
            issues.append(ValidationIssue(
                level=ValidationResult.FAIL,
                category=self.category,
                message=f"Negative loss value: {connection.loss_db}dB",
                suggestion="Loss values must be positive",
                auto_fixable=True
            ))
        elif connection.loss_db > 10:
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                category=self.category,
                message=f"High loss value: {connection.loss_db}dB",
                suggestion="Review connection design for excessive loss"
            ))
        
        return issues


class ConnectionDelayValidationRule(ValidationRule):
    """Validate connection delay values."""
    
    def __init__(self):
        super().__init__("connection_delay", "timing")
    
    def validate(self, context) -> List[ValidationIssue]:
        issues = []
        connection = context["connection"]
        
        if connection.delay_ps < 0:
            issues.append(ValidationIssue(
                level=ValidationResult.FAIL,
                category=self.category,
                message=f"Negative delay value: {connection.delay_ps}ps",
                suggestion="Delay values must be positive",
                auto_fixable=True
            ))
        
        return issues


class ConnectionPortValidationRule(ValidationRule):
    """Validate connection port assignments."""
    
    def __init__(self):
        super().__init__("connection_ports", "connection_structure")
    
    def validate(self, context) -> List[ValidationIssue]:
        issues = []
        connection = context["connection"]
        
        if connection.source_port < 0 or connection.target_port < 0:
            issues.append(ValidationIssue(
                level=ValidationResult.FAIL,
                category=self.category,
                message="Port numbers must be non-negative",
                suggestion="Use valid port numbers (0, 1, 2, ...)"
            ))
        
        return issues


class SynthesisComplexityRule(ValidationRule):
    """Validate synthesis complexity."""
    
    def __init__(self):
        super().__init__("synthesis_complexity", "synthesis_performance")
    
    def validate(self, context) -> List[ValidationIssue]:
        issues = []
        circuit = context["circuit"]
        
        complexity_score = len(circuit.components) + len(circuit.connections) * 0.5
        
        if complexity_score > 500:
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                category=self.category,
                message=f"High synthesis complexity (score: {complexity_score:.1f})",
                suggestion="Consider circuit optimization or decomposition"
            ))
        
        return issues


class SynthesisConstraintsRule(ValidationRule):
    """Validate synthesis constraints."""
    
    def __init__(self):
        super().__init__("synthesis_constraints", "synthesis_feasibility")
    
    def validate(self, context) -> List[ValidationIssue]:
        issues = []
        circuit = context["circuit"]
        
        # Check for unsupported component types (in a real implementation)
        supported_types = {"waveguide", "beam_splitter", "phase_shifter", "mach_zehnder"}
        
        for component in circuit.components:
            if hasattr(component, 'component_type') and component.component_type.value not in supported_types:
                issues.append(ValidationIssue(
                    level=ValidationResult.WARNING,
                    category=self.category,
                    message=f"Component type {component.component_type.value} may have limited synthesis support",
                    suggestion="Verify MLIR dialect support for this component type"
                ))
        
        return issues


class SynthesisFeasibilityRule(ValidationRule):
    """Validate synthesis feasibility."""
    
    def __init__(self):
        super().__init__("synthesis_feasibility", "synthesis_feasibility")
    
    def validate(self, context) -> List[ValidationIssue]:
        issues = []
        circuit = context["circuit"]
        
        # Check for manufacturability
        if len(circuit.components) > 10000:
            issues.append(ValidationIssue(
                level=ValidationResult.CRITICAL,
                category=self.category,
                message="Circuit exceeds practical manufacturing limits",
                suggestion="Reduce circuit complexity or use hierarchical design"
            ))
        
        return issues


# Global validator instance
_validator = PhotonicValidator()


def validate_photonic_component(component, validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationReport:
    """Validate a photonic component."""
    validator = PhotonicValidator(validation_level)
    return validator.validate_component(component)


def validate_photonic_circuit(circuit, validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationReport:
    """Validate a photonic circuit."""
    validator = PhotonicValidator(validation_level)
    return validator.validate_circuit(circuit)


def validate_synthesis_readiness(circuit, synthesis_params: Dict[str, Any] = None,
                                validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationReport:
    """Validate circuit readiness for synthesis."""
    validator = PhotonicValidator(validation_level)
    return validator.validate_synthesis_input(circuit, synthesis_params)


if __name__ == "__main__":
    # Demo validation capabilities
    print("🔍 Photonic-MLIR Bridge - Advanced Validation Demo")
    print("=" * 60)
    
    # Test with simple circuit
    from .photonic_mlir_bridge import create_simple_mzi_circuit
    
    circuit = create_simple_mzi_circuit()
    
    # Validate circuit
    report = validate_photonic_circuit(circuit, ValidationLevel.STRICT)
    
    print(f"\nValidation Report:")
    print(f"Overall Result: {report.overall_result.value.upper()}")
    print(f"Total Issues: {report.statistics['total_issues']}")
    print(f"Is Valid: {report.is_valid}")
    print(f"Has Critical Issues: {report.has_critical_issues}")
    
    if report.issues:
        print(f"\nTop Issues:")
        for issue in report.issues[:5]:
            print(f"  {issue.level.value.upper()}: {issue.message}")
    
    if report.recommendations:
        print(f"\nRecommendations:")
        for rec in report.recommendations[:3]:
            print(f"  • {rec}")
    
    print("\n✅ Validation system operational!")