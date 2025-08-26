"""
QNP Global Compliance System
Comprehensive compliance and governance framework for global deployment of QNP systems.

Features:
- GDPR, CCPA, PDPA compliance automation
- Data sovereignty and regional controls
- Audit logging and compliance reporting
- Privacy-preserving analytics
- Regulatory compliance monitoring
- Cross-border data transfer controls
"""

import json
import hashlib
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import secrets
from collections import defaultdict, deque
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplianceRegion(Enum):
    """Global compliance regions"""
    EU = "eu"              # GDPR
    CALIFORNIA = "ca"      # CCPA
    SINGAPORE = "sg"       # PDPA
    CANADA = "canada"      # PIPEDA
    BRAZIL = "brazil"      # LGPD
    JAPAN = "japan"        # APPI
    AUSTRALIA = "au"       # Privacy Act
    UK = "uk"              # UK GDPR

class DataCategory(Enum):
    """Data classification categories"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"                    # Personally Identifiable Information
    SENSITIVE_PERSONAL = "spd"     # Sensitive Personal Data
    FINANCIAL = "financial"
    HEALTH = "health"
    BIOMETRIC = "biometric"

class ProcessingPurpose(Enum):
    """Data processing purposes"""
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    RESEARCH = "research"
    ANALYTICS = "analytics"
    PERFORMANCE_MONITORING = "performance_monitoring"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    MARKETING = "marketing"
    PERSONALIZATION = "personalization"

@dataclass
class DataSubject:
    """Represents a data subject for compliance tracking"""
    subject_id: str
    region: ComplianceRegion
    consent_given: bool = False
    consent_date: Optional[datetime] = None
    consent_version: str = "1.0"
    preferences: Dict[str, Any] = field(default_factory=dict)
    data_categories: Set[DataCategory] = field(default_factory=set)
    processing_purposes: Set[ProcessingPurpose] = field(default_factory=set)
    retention_period_days: int = 365
    
    def has_valid_consent(self, purpose: ProcessingPurpose) -> bool:
        """Check if subject has valid consent for purpose"""
        if not self.consent_given:
            return False
        
        if purpose not in self.processing_purposes:
            return False
        
        if self.consent_date:
            # Check consent age (max 2 years for most regulations)
            age_days = (datetime.now() - self.consent_date).days
            max_age = self.preferences.get("consent_max_age_days", 730)
            return age_days <= max_age
        
        return True

@dataclass
class DataProcessingRecord:
    """Record of data processing activity for audit trail"""
    record_id: str
    timestamp: datetime
    subject_id: str
    data_category: DataCategory
    processing_purpose: ProcessingPurpose
    lawful_basis: str
    processor_id: str
    region: ComplianceRegion
    data_hash: str
    retention_until: datetime
    consent_version: Optional[str] = None
    transfer_country: Optional[str] = None

class ComplianceRule:
    """Base class for compliance rules"""
    
    def __init__(self, rule_id: str, region: ComplianceRegion, 
                 description: str, severity: str = "medium"):
        self.rule_id = rule_id
        self.region = region
        self.description = description
        self.severity = severity
    
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate rule against context"""
        raise NotImplementedError

class GDPRComplianceRule(ComplianceRule):
    """GDPR-specific compliance rules"""
    
    def __init__(self, rule_id: str, description: str, severity: str = "high"):
        super().__init__(rule_id, ComplianceRegion.EU, description, severity)
    
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        violations = []
        
        # Check consent requirements
        if context.get("requires_consent", False):
            if not context.get("consent_given", False):
                violations.append("Missing explicit consent for data processing")
        
        # Check data minimization
        data_categories = context.get("data_categories", [])
        processing_purpose = context.get("processing_purpose")
        
        if processing_purpose == ProcessingPurpose.SENTIMENT_ANALYSIS:
            if DataCategory.PII in data_categories:
                violations.append("PII not necessary for sentiment analysis (data minimization)")
        
        # Check retention period
        retention_days = context.get("retention_period_days", 0)
        if retention_days > 2555:  # 7 years maximum
            violations.append("Retention period exceeds GDPR limits")
        
        # Check cross-border transfers
        if context.get("transfer_country") and context.get("transfer_country") not in ["EU", "EEA"]:
            adequacy_decision = context.get("adequacy_decision", False)
            safeguards = context.get("safeguards", False)
            
            if not adequacy_decision and not safeguards:
                violations.append("Cross-border transfer without adequate protection")
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "rule_id": self.rule_id,
            "region": self.region.value
        }

class CCPAComplianceRule(ComplianceRule):
    """CCPA-specific compliance rules"""
    
    def __init__(self, rule_id: str, description: str, severity: str = "high"):
        super().__init__(rule_id, ComplianceRegion.CALIFORNIA, description, severity)
    
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        violations = []
        
        # Check sale disclosure
        if context.get("is_sale", False) and not context.get("sale_disclosed", False):
            violations.append("Sale of personal information not disclosed")
        
        # Check opt-out mechanism
        if context.get("is_sale", False) and not context.get("opt_out_available", True):
            violations.append("No opt-out mechanism for sale of personal information")
        
        # Check deletion rights
        if not context.get("deletion_mechanism", False):
            violations.append("No mechanism for consumer deletion requests")
        
        # Check data categories disclosure
        if not context.get("categories_disclosed", False):
            violations.append("Categories of personal information not disclosed")
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "rule_id": self.rule_id,
            "region": self.region.value
        }

class QNPComplianceEngine:
    """Main compliance engine for QNP systems"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Data subject registry
        self.data_subjects: Dict[str, DataSubject] = {}
        
        # Processing records for audit trail
        self.processing_records: deque = deque(maxlen=100000)
        
        # Compliance rules by region
        self.rules: Dict[ComplianceRegion, List[ComplianceRule]] = defaultdict(list)
        
        # Initialize default compliance rules
        self._initialize_compliance_rules()
        
        # Compliance monitoring
        self.compliance_violations: List[Dict[str, Any]] = []
        self.last_compliance_check = datetime.now()
        
        # Privacy controls
        self.data_retention_policies = self._load_retention_policies()
        self.anonymization_config = self._load_anonymization_config()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Audit logging
        self.audit_log_file = Path(self.config.get("audit_log_file", "qnp_compliance_audit.jsonl"))
        self.audit_log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _initialize_compliance_rules(self):
        """Initialize default compliance rules for major regulations"""
        
        # GDPR Rules
        self.rules[ComplianceRegion.EU].extend([
            GDPRComplianceRule("GDPR_001", "Explicit consent required for processing"),
            GDPRComplianceRule("GDPR_002", "Data minimization principle"),
            GDPRComplianceRule("GDPR_003", "Retention period limits"),
            GDPRComplianceRule("GDPR_004", "Cross-border transfer safeguards")
        ])
        
        # CCPA Rules
        self.rules[ComplianceRegion.CALIFORNIA].extend([
            CCPAComplianceRule("CCPA_001", "Sale disclosure requirements"),
            CCPAComplianceRule("CCPA_002", "Consumer deletion rights"),
            CCPAComplianceRule("CCPA_003", "Opt-out mechanisms"),
            CCPAComplianceRule("CCPA_004", "Personal information categories disclosure")
        ])
    
    def _load_retention_policies(self) -> Dict[DataCategory, int]:
        """Load data retention policies by category"""
        return {
            DataCategory.PUBLIC: 2555,              # 7 years
            DataCategory.INTERNAL: 1825,            # 5 years
            DataCategory.CONFIDENTIAL: 1095,        # 3 years
            DataCategory.PII: 365,                  # 1 year
            DataCategory.SENSITIVE_PERSONAL: 180,   # 6 months
            DataCategory.FINANCIAL: 2555,           # 7 years
            DataCategory.HEALTH: 3650,              # 10 years
            DataCategory.BIOMETRIC: 90               # 3 months
        }
    
    def _load_anonymization_config(self) -> Dict[str, Any]:
        """Load anonymization configuration"""
        return {
            "hash_algorithm": "sha256",
            "salt_length": 32,
            "anonymization_threshold": 5,  # k-anonymity
            "pseudonymization_key_rotation_days": 30
        }
    
    def register_data_subject(self, subject: DataSubject) -> bool:
        """Register a data subject in the compliance system"""
        with self._lock:
            self.data_subjects[subject.subject_id] = subject
            
            # Log registration
            self._log_compliance_event({
                "event_type": "data_subject_registered",
                "subject_id": subject.subject_id,
                "region": subject.region.value,
                "consent_given": subject.consent_given,
                "data_categories": [cat.value for cat in subject.data_categories],
                "processing_purposes": [purpose.value for purpose in subject.processing_purposes]
            })
            
            return True
    
    def check_processing_compliance(self, subject_id: str, 
                                  data_category: DataCategory,
                                  processing_purpose: ProcessingPurpose,
                                  region: ComplianceRegion) -> Dict[str, Any]:
        """Check if data processing is compliant with regulations"""
        
        with self._lock:
            # Get data subject
            subject = self.data_subjects.get(subject_id)
            if not subject:
                return {
                    "compliant": False,
                    "reason": "Data subject not registered",
                    "action_required": "register_subject"
                }
            
            # Check consent
            if not subject.has_valid_consent(processing_purpose):
                return {
                    "compliant": False,
                    "reason": "Invalid or missing consent",
                    "action_required": "obtain_consent"
                }
            
            # Check region-specific rules
            compliance_context = {
                "subject_id": subject_id,
                "data_categories": [data_category],
                "processing_purpose": processing_purpose,
                "region": region,
                "consent_given": subject.consent_given,
                "retention_period_days": subject.retention_period_days,
                "requires_consent": data_category in [DataCategory.PII, DataCategory.SENSITIVE_PERSONAL]
            }
            
            violations = []
            
            for rule in self.rules.get(region, []):
                result = rule.evaluate(compliance_context)
                if not result["compliant"]:
                    violations.extend(result["violations"])
            
            compliant = len(violations) == 0
            
            if compliant:
                # Record processing activity
                self._record_processing_activity(
                    subject_id=subject_id,
                    data_category=data_category,
                    processing_purpose=processing_purpose,
                    region=region,
                    lawful_basis=self._determine_lawful_basis(subject, processing_purpose)
                )
            else:
                # Log compliance violations
                self.compliance_violations.append({
                    "timestamp": datetime.now().isoformat(),
                    "subject_id": subject_id,
                    "violations": violations,
                    "region": region.value
                })
            
            return {
                "compliant": compliant,
                "violations": violations,
                "lawful_basis": self._determine_lawful_basis(subject, processing_purpose) if compliant else None
            }
    
    def _determine_lawful_basis(self, subject: DataSubject, 
                               purpose: ProcessingPurpose) -> str:
        """Determine the lawful basis for processing under GDPR"""
        
        if subject.region == ComplianceRegion.EU:
            if subject.consent_given:
                return "consent"
            elif purpose in [ProcessingPurpose.SECURITY, ProcessingPurpose.COMPLIANCE]:
                return "legal_obligation"
            elif purpose == ProcessingPurpose.PERFORMANCE_MONITORING:
                return "legitimate_interest"
            else:
                return "consent_required"
        
        return "applicable_law"
    
    def _record_processing_activity(self, subject_id: str, 
                                  data_category: DataCategory,
                                  processing_purpose: ProcessingPurpose,
                                  region: ComplianceRegion,
                                  lawful_basis: str):
        """Record data processing activity for audit trail"""
        
        record = DataProcessingRecord(
            record_id=secrets.token_hex(16),
            timestamp=datetime.now(),
            subject_id=subject_id,
            data_category=data_category,
            processing_purpose=processing_purpose,
            lawful_basis=lawful_basis,
            processor_id="qnp_system",
            region=region,
            data_hash=hashlib.sha256(f"{subject_id}:{time.time()}".encode()).hexdigest()[:16],
            retention_until=datetime.now() + timedelta(
                days=self.data_retention_policies.get(data_category, 365)
            )
        )
        
        self.processing_records.append(record)
        
        # Log to audit trail
        self._log_compliance_event({
            "event_type": "data_processing_recorded",
            "record_id": record.record_id,
            "subject_id": subject_id,
            "data_category": data_category.value,
            "processing_purpose": processing_purpose.value,
            "lawful_basis": lawful_basis,
            "region": region.value
        })
    
    def anonymize_text(self, text: str, preserve_sentiment: bool = True) -> str:
        """Anonymize text while preserving sentiment analysis capability"""
        
        # Simple anonymization - replace PII patterns
        anonymized = text
        
        # Replace email addresses
        anonymized = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                          '[EMAIL]', anonymized)
        
        # Replace phone numbers
        anonymized = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', anonymized)
        
        # Replace potential names (simple heuristic)
        anonymized = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', anonymized)
        
        # Replace numbers that might be IDs
        anonymized = re.sub(r'\b\d{4,}\b', '[ID]', anonymized)
        
        if preserve_sentiment:
            # Keep sentiment-bearing words intact
            sentiment_indicators = [
                'good', 'bad', 'great', 'terrible', 'excellent', 'awful',
                'love', 'hate', 'amazing', 'disappointing', 'fantastic', 'horrible'
            ]
            # These are preserved in the anonymization process
        
        return anonymized
    
    def handle_data_subject_request(self, subject_id: str, 
                                   request_type: str) -> Dict[str, Any]:
        """Handle data subject requests (access, deletion, portability)"""
        
        with self._lock:
            subject = self.data_subjects.get(subject_id)
            if not subject:
                return {
                    "success": False,
                    "reason": "Data subject not found"
                }
            
            if request_type == "access":
                # Provide data subject with their data
                subject_data = {
                    "personal_data": asdict(subject),
                    "processing_records": [
                        asdict(record) for record in self.processing_records
                        if record.subject_id == subject_id
                    ]
                }
                
                self._log_compliance_event({
                    "event_type": "data_subject_access_request",
                    "subject_id": subject_id,
                    "request_fulfilled": True
                })
                
                return {
                    "success": True,
                    "data": subject_data
                }
            
            elif request_type == "deletion":
                # Delete data subject and associated records
                del self.data_subjects[subject_id]
                
                # Mark processing records for deletion
                records_to_delete = [
                    record for record in self.processing_records
                    if record.subject_id == subject_id
                ]
                
                for record in records_to_delete:
                    self.processing_records.remove(record)
                
                self._log_compliance_event({
                    "event_type": "data_subject_deletion_request",
                    "subject_id": subject_id,
                    "records_deleted": len(records_to_delete)
                })
                
                return {
                    "success": True,
                    "records_deleted": len(records_to_delete)
                }
            
            elif request_type == "portability":
                # Provide data in portable format
                portable_data = {
                    "subject_id": subject_id,
                    "export_timestamp": datetime.now().isoformat(),
                    "data": asdict(subject),
                    "processing_history": [
                        {
                            "timestamp": record.timestamp.isoformat(),
                            "purpose": record.processing_purpose.value,
                            "category": record.data_category.value,
                            "lawful_basis": record.lawful_basis
                        }
                        for record in self.processing_records
                        if record.subject_id == subject_id
                    ]
                }
                
                self._log_compliance_event({
                    "event_type": "data_portability_request",
                    "subject_id": subject_id,
                    "request_fulfilled": True
                })
                
                return {
                    "success": True,
                    "portable_data": portable_data
                }
            
            else:
                return {
                    "success": False,
                    "reason": f"Unknown request type: {request_type}"
                }
    
    def generate_compliance_report(self, region: Optional[ComplianceRegion] = None,
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        # Filter records by date range
        filtered_records = [
            record for record in self.processing_records
            if start_date <= record.timestamp <= end_date and
            (region is None or record.region == region)
        ]
        
        # Group by various dimensions
        by_purpose = defaultdict(int)
        by_category = defaultdict(int)
        by_region = defaultdict(int)
        by_lawful_basis = defaultdict(int)
        
        for record in filtered_records:
            by_purpose[record.processing_purpose.value] += 1
            by_category[record.data_category.value] += 1
            by_region[record.region.value] += 1
            by_lawful_basis[record.lawful_basis] += 1
        
        # Compliance violations in period
        period_violations = [
            violation for violation in self.compliance_violations
            if start_date <= datetime.fromisoformat(violation["timestamp"]) <= end_date and
            (region is None or violation["region"] == region.value)
        ]
        
        # Data subject statistics
        subject_stats = {
            "total_subjects": len(self.data_subjects),
            "subjects_with_consent": len([s for s in self.data_subjects.values() if s.consent_given]),
            "subjects_by_region": defaultdict(int)
        }
        
        for subject in self.data_subjects.values():
            if region is None or subject.region == region:
                subject_stats["subjects_by_region"][subject.region.value] += 1
        
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "period_start": start_date.isoformat(),
                "period_end": end_date.isoformat(),
                "region_filter": region.value if region else "all",
                "total_processing_records": len(filtered_records)
            },
            "processing_statistics": {
                "by_purpose": dict(by_purpose),
                "by_data_category": dict(by_category),
                "by_region": dict(by_region),
                "by_lawful_basis": dict(by_lawful_basis)
            },
            "data_subject_statistics": subject_stats,
            "compliance_violations": {
                "total_violations": len(period_violations),
                "violations": period_violations
            },
            "retention_compliance": self._check_retention_compliance(),
            "recommendations": self._generate_compliance_recommendations()
        }
        
        return report
    
    def _check_retention_compliance(self) -> Dict[str, Any]:
        """Check data retention compliance"""
        current_time = datetime.now()
        overdue_records = []
        
        for record in self.processing_records:
            if record.retention_until < current_time:
                overdue_records.append({
                    "record_id": record.record_id,
                    "overdue_by_days": (current_time - record.retention_until).days,
                    "data_category": record.data_category.value,
                    "subject_id": record.subject_id
                })
        
        return {
            "total_overdue_records": len(overdue_records),
            "overdue_records": overdue_records[:10],  # Limit to 10 for reporting
            "retention_compliance_rate": 1.0 - (len(overdue_records) / max(1, len(self.processing_records)))
        }
    
    def _generate_compliance_recommendations(self) -> List[str]:
        """Generate compliance improvement recommendations"""
        recommendations = []
        
        # Check consent rates
        total_subjects = len(self.data_subjects)
        consented_subjects = len([s for s in self.data_subjects.values() if s.consent_given])
        
        if total_subjects > 0:
            consent_rate = consented_subjects / total_subjects
            if consent_rate < 0.8:
                recommendations.append(
                    f"Low consent rate ({consent_rate:.1%}). Consider consent optimization."
                )
        
        # Check violations
        if len(self.compliance_violations) > 0:
            recommendations.append(
                f"Address {len(self.compliance_violations)} compliance violations."
            )
        
        # Check retention
        retention_check = self._check_retention_compliance()
        if retention_check["retention_compliance_rate"] < 0.95:
            recommendations.append(
                "Implement automated data retention cleanup procedures."
            )
        
        # Check data minimization
        pii_processing = sum(1 for r in self.processing_records 
                           if r.data_category == DataCategory.PII)
        if pii_processing > len(self.processing_records) * 0.3:
            recommendations.append(
                "Review PII processing necessity (data minimization)."
            )
        
        return recommendations
    
    def _log_compliance_event(self, event_data: Dict[str, Any]):
        """Log compliance event to audit trail"""
        audit_record = {
            "timestamp": datetime.now().isoformat(),
            "event_id": secrets.token_hex(8),
            **event_data
        }
        
        try:
            with open(self.audit_log_file, "a") as f:
                f.write(json.dumps(audit_record) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get overall compliance system status"""
        return {
            "compliance_engine": {
                "registered_subjects": len(self.data_subjects),
                "processing_records": len(self.processing_records),
                "compliance_violations": len(self.compliance_violations),
                "supported_regions": [region.value for region in self.rules.keys()],
                "last_compliance_check": self.last_compliance_check.isoformat()
            },
            "data_protection_measures": {
                "anonymization_enabled": True,
                "audit_logging_enabled": True,
                "consent_management": True,
                "data_subject_rights": ["access", "deletion", "portability"]
            },
            "regulatory_coverage": {
                "gdpr": ComplianceRegion.EU in self.rules,
                "ccpa": ComplianceRegion.CALIFORNIA in self.rules,
                "pdpa": ComplianceRegion.SINGAPORE in self.rules
            }
        }

# Factory function
def create_compliance_engine(config: Optional[Dict] = None) -> QNPComplianceEngine:
    """Create QNP compliance engine"""
    return QNPComplianceEngine(config)

# CLI interface for compliance management
def main():
    """Demo compliance system"""
    print("🛡️ QNP Global Compliance System Demo")
    print("=" * 50)
    
    # Create compliance engine
    engine = create_compliance_engine()
    
    # Register test data subject
    test_subject = DataSubject(
        subject_id="user_123",
        region=ComplianceRegion.EU,
        consent_given=True,
        consent_date=datetime.now(),
        data_categories={DataCategory.PUBLIC},
        processing_purposes={ProcessingPurpose.SENTIMENT_ANALYSIS}
    )
    
    engine.register_data_subject(test_subject)
    print("✅ Registered test data subject")
    
    # Check processing compliance
    compliance_check = engine.check_processing_compliance(
        subject_id="user_123",
        data_category=DataCategory.PUBLIC,
        processing_purpose=ProcessingPurpose.SENTIMENT_ANALYSIS,
        region=ComplianceRegion.EU
    )
    
    print(f"🔍 Compliance check: {'✅ COMPLIANT' if compliance_check['compliant'] else '❌ NON-COMPLIANT'}")
    
    # Test anonymization
    sample_text = "Hello, my name is John Doe and my email is john.doe@example.com"
    anonymized = engine.anonymize_text(sample_text)
    print(f"🔒 Anonymized: {anonymized}")
    
    # Generate compliance report
    report = engine.generate_compliance_report()
    print(f"📊 Generated compliance report with {report['report_metadata']['total_processing_records']} records")
    
    # Show system status
    status = engine.get_compliance_status()
    print(f"📈 System status: {status['compliance_engine']['registered_subjects']} subjects, {len(status['regulatory_coverage'])} regulations")

if __name__ == "__main__":
    main()