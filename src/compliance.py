"""Global compliance and data protection support for GDPR, CCPA, PDPA."""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

class ComplianceRegion(Enum):
    """Supported compliance regions."""
    EU_GDPR = "eu_gdpr"
    US_CCPA = "us_ccpa"
    SG_PDPA = "sg_pdpa"
    GLOBAL = "global"

class DataProcessingPurpose(Enum):
    """Data processing purposes for compliance."""
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    MODEL_TRAINING = "model_training"
    PERFORMANCE_ANALYTICS = "performance_analytics"
    QUALITY_IMPROVEMENT = "quality_improvement"

@dataclass
class ConsentRecord:
    """Record of user consent for data processing."""
    user_id: str
    purpose: DataProcessingPurpose
    granted: bool
    timestamp: datetime
    region: ComplianceRegion
    expiry: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['expiry'] = self.expiry.isoformat() if self.expiry else None
        data['purpose'] = self.purpose.value
        data['region'] = self.region.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConsentRecord':
        """Create from dictionary."""
        return cls(
            user_id=data['user_id'],
            purpose=DataProcessingPurpose(data['purpose']),
            granted=data['granted'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            region=ComplianceRegion(data['region']),
            expiry=datetime.fromisoformat(data['expiry']) if data['expiry'] else None
        )

@dataclass
class DataProcessingRecord:
    """Record of data processing activity."""
    processing_id: str
    user_id: str
    data_type: str
    purpose: DataProcessingPurpose
    timestamp: datetime
    region: ComplianceRegion
    retention_until: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['retention_until'] = self.retention_until.isoformat() if self.retention_until else None
        data['purpose'] = self.purpose.value
        data['region'] = self.region.value
        return data

class ComplianceManager:
    """Manages data protection compliance across regions."""
    
    def __init__(self, region: ComplianceRegion = ComplianceRegion.GLOBAL):
        self.region = region
        self.consent_records: Dict[str, List[ConsentRecord]] = {}
        self.processing_records: List[DataProcessingRecord] = []
        self.retention_policies = self._get_retention_policies()
    
    def _get_retention_policies(self) -> Dict[ComplianceRegion, Dict[str, int]]:
        """Get data retention policies by region (days)."""
        return {
            ComplianceRegion.EU_GDPR: {
                "sentiment_analysis": 365,
                "model_training": 1095,
                "analytics": 730
            },
            ComplianceRegion.US_CCPA: {
                "sentiment_analysis": 365,
                "model_training": 1095,
                "analytics": 730
            },
            ComplianceRegion.SG_PDPA: {
                "sentiment_analysis": 365,
                "model_training": 1095,
                "analytics": 730
            },
            ComplianceRegion.GLOBAL: {
                "sentiment_analysis": 365,
                "model_training": 730,
                "analytics": 365
            }
        }
    
    def record_consent(
        self, 
        user_id: str, 
        purpose: DataProcessingPurpose, 
        granted: bool,
        region: Optional[ComplianceRegion] = None
    ) -> ConsentRecord:
        """Record user consent for data processing."""
        if region is None:
            region = self.region
        
        expiry = None
        if granted and region == ComplianceRegion.EU_GDPR:
            expiry = datetime.now() + timedelta(days=730)
        
        consent = ConsentRecord(
            user_id=user_id,
            purpose=purpose,
            granted=granted,
            timestamp=datetime.now(),
            region=region,
            expiry=expiry
        )
        
        if user_id not in self.consent_records:
            self.consent_records[user_id] = []
        
        self.consent_records[user_id].append(consent)
        
        logger.info(f"Consent recorded for user {user_id}: {purpose.value} = {granted}")
        return consent
    
    def check_consent(
        self, 
        user_id: str, 
        purpose: DataProcessingPurpose
    ) -> bool:
        """Check if user has granted consent for specific purpose."""
        if user_id not in self.consent_records:
            return False
        
        user_consents = self.consent_records[user_id]
        latest_consent = None
        
        for consent in reversed(user_consents):
            if consent.purpose == purpose:
                latest_consent = consent
                break
        
        if latest_consent is None:
            return False
        
        if not latest_consent.granted:
            return False
        
        if latest_consent.expiry and datetime.now() > latest_consent.expiry:
            logger.warning(f"Consent expired for user {user_id}: {purpose.value}")
            return False
        
        return True
    
    def process_data(
        self, 
        user_id: str, 
        data_type: str, 
        purpose: DataProcessingPurpose,
        check_consent: bool = True
    ) -> Optional[str]:
        """Process data with compliance checks."""
        if check_consent and not self.check_consent(user_id, purpose):
            logger.error(f"Data processing denied: No valid consent for {purpose.value}")
            return None
        
        processing_id = str(uuid.uuid4())
        retention_days = self.retention_policies[self.region].get(
            purpose.value.split('_')[0], 365
        )
        
        record = DataProcessingRecord(
            processing_id=processing_id,
            user_id=user_id,
            data_type=data_type,
            purpose=purpose,
            timestamp=datetime.now(),
            region=self.region,
            retention_until=datetime.now() + timedelta(days=retention_days)
        )
        
        self.processing_records.append(record)
        
        logger.info(f"Data processed: {processing_id} for user {user_id}")
        return processing_id
    
    def anonymize_data(self, data: str, user_id: str) -> str:
        """Anonymize data for compliance."""
        hash_input = f"{user_id}:{data}".encode()
        return hashlib.sha256(hash_input).hexdigest()
    
    def handle_deletion_request(self, user_id: str) -> bool:
        """Handle user data deletion request (Right to be forgotten)."""
        try:
            if user_id in self.consent_records:
                del self.consent_records[user_id]
            
            self.processing_records = [
                record for record in self.processing_records 
                if record.user_id != user_id
            ]
            
            logger.info(f"Data deletion completed for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Data deletion failed for user {user_id}: {e}")
            return False
    
    def get_user_data(self, user_id: str) -> Dict[str, Any]:
        """Get all data for a user (Data portability)."""
        user_consents = self.consent_records.get(user_id, [])
        user_processing = [
            record for record in self.processing_records 
            if record.user_id == user_id
        ]
        
        return {
            "user_id": user_id,
            "consents": [consent.to_dict() for consent in user_consents],
            "processing_records": [record.to_dict() for record in user_processing],
            "export_timestamp": datetime.now().isoformat()
        }
    
    def cleanup_expired_data(self) -> int:
        """Clean up expired data based on retention policies."""
        now = datetime.now()
        initial_count = len(self.processing_records)
        
        self.processing_records = [
            record for record in self.processing_records
            if not record.retention_until or record.retention_until > now
        ]
        
        cleaned_count = initial_count - len(self.processing_records)
        logger.info(f"Cleaned up {cleaned_count} expired data records")
        return cleaned_count
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report."""
        now = datetime.now()
        
        consent_summary = {}
        for purpose in DataProcessingPurpose:
            granted = sum(
                1 for consents in self.consent_records.values()
                for consent in consents
                if consent.purpose == purpose and consent.granted
            )
            consent_summary[purpose.value] = granted
        
        return {
            "region": self.region.value,
            "report_timestamp": now.isoformat(),
            "total_users": len(self.consent_records),
            "total_processing_records": len(self.processing_records),
            "consent_summary": consent_summary,
            "retention_policies": self.retention_policies[self.region]
        }

_global_compliance_manager = ComplianceManager()

def set_compliance_region(region: ComplianceRegion):
    """Set global compliance region."""
    global _global_compliance_manager
    _global_compliance_manager.region = region

def get_compliance_manager() -> ComplianceManager:
    """Get global compliance manager."""
    return _global_compliance_manager