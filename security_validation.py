#!/usr/bin/env python3
"""
Enhanced Security Validation and Compliance Testing
Generation 2: Security and Compliance Robustness
"""

import re
import json
import hashlib
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityThreat:
    threat_type: str
    level: ThreatLevel
    description: str
    pattern: str
    mitigation: str

class AdvancedSecurityValidator:
    """Advanced security validation with threat detection."""
    
    def __init__(self):
        self.threat_patterns = [
            SecurityThreat(
                threat_type="script_injection",
                level=ThreatLevel.HIGH,
                description="Script tag injection attempt",
                pattern=r'<script[^>]*>.*?</script>',
                mitigation="Strip script tags and content"
            ),
            SecurityThreat(
                threat_type="sql_injection",
                level=ThreatLevel.HIGH,
                description="SQL injection pattern",
                pattern=r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|CREATE|ALTER)\b.*[;\'"--]',
                mitigation="Parameterized queries and input sanitization"
            ),
            SecurityThreat(
                threat_type="command_injection",
                level=ThreatLevel.CRITICAL,
                description="Command injection attempt",
                pattern=r'[;&|`$(){}[\]<>]',
                mitigation="Whitelist allowed characters"
            ),
            SecurityThreat(
                threat_type="path_traversal",
                level=ThreatLevel.HIGH,
                description="Path traversal attempt",
                pattern=r'\.\.[/\\]',
                mitigation="Normalize and validate file paths"
            ),
            SecurityThreat(
                threat_type="xss_attempt",
                level=ThreatLevel.MEDIUM,
                description="Cross-site scripting attempt",
                pattern=r'(javascript:|vbscript:|on\w+\s*=)',
                mitigation="HTML entity encoding"
            )
        ]
        
        self.threat_log = []
    
    def validate_input(self, input_text: str, context: str = "general") -> Dict[str, Any]:
        """Comprehensive security validation of input."""
        result = {
            "is_safe": True,
            "threats_detected": [],
            "sanitized_input": input_text,
            "risk_score": 0,
            "validation_context": context
        }
        
        if not isinstance(input_text, str):
            result["is_safe"] = False
            result["threats_detected"].append({
                "type": "invalid_type",
                "level": "medium",
                "description": "Input is not a string type"
            })
            return result
        
        # Check against threat patterns
        for threat in self.threat_patterns:
            if re.search(threat.pattern, input_text, re.IGNORECASE | re.DOTALL):
                threat_info = {
                    "type": threat.threat_type,
                    "level": threat.level.value,
                    "description": threat.description,
                    "mitigation": threat.mitigation,
                    "pattern_matched": threat.pattern
                }
                result["threats_detected"].append(threat_info)
                
                # Increase risk score based on threat level
                risk_increment = {
                    ThreatLevel.LOW: 10,
                    ThreatLevel.MEDIUM: 25,
                    ThreatLevel.HIGH: 50,
                    ThreatLevel.CRITICAL: 100
                }
                result["risk_score"] += risk_increment[threat.level]
                
                # Log the threat
                self.threat_log.append({
                    "timestamp": time.time(),
                    "threat": threat_info,
                    "input_sample": input_text[:100],
                    "context": context
                })
        
        # Determine if input is safe
        if result["threats_detected"]:
            result["is_safe"] = False
            result["sanitized_input"] = self._sanitize_input(input_text)
        
        return result
    
    def _sanitize_input(self, input_text: str) -> str:
        """Sanitize potentially dangerous input."""
        sanitized = input_text
        
        # Remove script tags
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove dangerous protocols
        sanitized = re.sub(r'(javascript:|vbscript:)', '', sanitized, flags=re.IGNORECASE)
        
        # Remove potential command injection characters
        sanitized = re.sub(r'[;&|`$<>]', '', sanitized)
        
        # Normalize path traversal attempts
        sanitized = re.sub(r'\.\.[/\\]', '', sanitized)
        
        return sanitized.strip()
    
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get summary of detected threats."""
        if not self.threat_log:
            return {"total_threats": 0, "threat_types": {}, "risk_levels": {}}
        
        threat_types = {}
        risk_levels = {}
        
        for log_entry in self.threat_log:
            threat_type = log_entry["threat"]["type"]
            risk_level = log_entry["threat"]["level"]
            
            threat_types[threat_type] = threat_types.get(threat_type, 0) + 1
            risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
        
        return {
            "total_threats": len(self.threat_log),
            "threat_types": threat_types,
            "risk_levels": risk_levels,
            "recent_threats": self.threat_log[-5:]  # Last 5 threats
        }

class ComplianceValidator:
    """GDPR, CCPA, and PDPA compliance validation."""
    
    def __init__(self):
        self.personal_data_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',  # Credit card
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone number
            r'\b\d{1,5}\s\w+\s(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b'  # Address
        ]
        
        self.processing_log = []
    
    def scan_for_personal_data(self, text: str, purpose: str = "sentiment_analysis") -> Dict[str, Any]:
        """Scan text for potential personal data."""
        result = {
            "contains_personal_data": False,
            "data_types_found": [],
            "compliance_status": "compliant",
            "processing_purpose": purpose,
            "recommendations": []
        }
        
        data_type_names = ["SSN", "Email", "Credit Card", "Phone", "Address"]
        
        for i, pattern in enumerate(self.personal_data_patterns):
            if re.search(pattern, text):
                result["contains_personal_data"] = True
                result["data_types_found"].append(data_type_names[i])
        
        if result["contains_personal_data"]:
            result["compliance_status"] = "requires_review"
            result["recommendations"].extend([
                "Ensure user consent is obtained for data processing",
                "Implement data minimization principles",
                "Provide clear privacy notice to users",
                "Enable data subject rights (access, rectification, erasure)"
            ])
            
            # Log the processing
            self.processing_log.append({
                "timestamp": time.time(),
                "purpose": purpose,
                "data_types": result["data_types_found"],
                "text_hash": hashlib.sha256(text.encode()).hexdigest()[:16]
            })
        
        return result
    
    def generate_privacy_report(self) -> Dict[str, Any]:
        """Generate privacy compliance report."""
        if not self.processing_log:
            return {
                "status": "no_personal_data_processed",
                "total_processing_events": 0
            }
        
        data_types_processed = {}
        purposes = {}
        
        for log_entry in self.processing_log:
            purpose = log_entry["purpose"]
            purposes[purpose] = purposes.get(purpose, 0) + 1
            
            for data_type in log_entry["data_types"]:
                data_types_processed[data_type] = data_types_processed.get(data_type, 0) + 1
        
        return {
            "status": "personal_data_processed",
            "total_processing_events": len(self.processing_log),
            "data_types_processed": data_types_processed,
            "processing_purposes": purposes,
            "compliance_recommendations": [
                "Regular privacy impact assessments",
                "Data retention policy implementation",
                "User consent management system",
                "Data breach notification procedures"
            ]
        }

def test_security_and_compliance():
    """Test security and compliance validation."""
    print("Testing Security and Compliance Validation...")
    
    # Initialize validators
    security_validator = AdvancedSecurityValidator()
    compliance_validator = ComplianceValidator()
    
    # Test security validation
    print("\n1. Security Validation Tests:")
    
    test_inputs = [
        "This is a normal sentiment analysis text",
        "<script>alert('xss')</script>",
        "SELECT * FROM users WHERE id = 1; DROP TABLE users;",
        "../../etc/passwd",
        "javascript:alert('xss')",
        "Normal text with no threats"
    ]
    
    for i, test_input in enumerate(test_inputs):
        result = security_validator.validate_input(test_input, context="sentiment_analysis")
        print(f"   Test {i+1}: Safe={result['is_safe']}, Risk Score={result['risk_score']}, Threats={len(result['threats_detected'])}")
    
    # Test compliance validation
    print("\n2. Compliance Validation Tests:")
    
    compliance_inputs = [
        "I love this product! My email is john@example.com",
        "Call me at 555-123-4567 for more details",
        "My credit card 1234-5678-9012-3456 was charged",
        "Just normal sentiment text with no personal data",
        "SSN: 123-45-6789 needs to be protected"
    ]
    
    for i, test_input in enumerate(compliance_inputs):
        result = compliance_validator.scan_for_personal_data(test_input)
        print(f"   Test {i+1}: Personal Data={result['contains_personal_data']}, Types={result['data_types_found']}")
    
    # Generate reports
    print("\n3. Security and Compliance Reports:")
    
    threat_summary = security_validator.get_threat_summary()
    print(f"   Total Security Threats Detected: {threat_summary['total_threats']}")
    print(f"   Threat Types: {threat_summary['threat_types']}")
    
    privacy_report = compliance_validator.generate_privacy_report()
    print(f"   Privacy Status: {privacy_report['status']}")
    if 'total_processing_events' in privacy_report:
        print(f"   Personal Data Processing Events: {privacy_report['total_processing_events']}")
    
    print("\n✓ Security and compliance validation completed!")

if __name__ == "__main__":
    test_security_and_compliance()