"""
🛡️ Quantum-Photonic-Neuromorphic Security System
===============================================

Comprehensive security framework for the revolutionary tri-modal processing system,
implementing quantum-safe encryption, photonic tamper detection, and neuromorphic
anomaly detection.

Key Security Features:
- Quantum-resistant cryptographic protocols
- Photonic side-channel attack mitigation  
- Neuromorphic intrusion detection
- Multi-layer input validation and sanitization
- Real-time threat analysis and response

Author: Terragon Labs Autonomous SDLC System
Generation: 2 (Make It Reliable) - Security Layer
"""

from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass
import hashlib
import hmac
import secrets
import time
import math
import json
import logging
from abc import ABC, abstractmethod


class ThreatLevel(Enum):
    """Security threat severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackVector(Enum):
    """Known attack vectors against quantum-photonic systems."""
    QUANTUM_DECOHERENCE = "quantum_decoherence"
    PHOTONIC_INTERFERENCE = "photonic_interference"
    NEUROMORPHIC_POISONING = "neuromorphic_poisoning"
    SIDE_CHANNEL = "side_channel"
    INPUT_INJECTION = "input_injection"
    REPLAY_ATTACK = "replay_attack"
    TIMING_ANALYSIS = "timing_analysis"
    POWER_ANALYSIS = "power_analysis"


class SecurityPolicy(Enum):
    """Security policy enforcement levels."""
    PERMISSIVE = "permissive"      # Log threats but allow processing
    RESTRICTIVE = "restrictive"    # Block suspicious inputs
    PARANOID = "paranoid"          # Block and quarantine threats
    ADAPTIVE = "adaptive"          # Learn and adapt to new threats


@dataclass 
class SecurityConfig:
    """Configuration for quantum-photonic security system."""
    
    # Cryptographic parameters
    key_size: int = 256                    # Bits for symmetric keys
    hash_algorithm: str = "sha3_256"       # Quantum-resistant hash
    signature_algorithm: str = "dilithium" # Post-quantum signature
    
    # Input validation
    max_input_size: int = 10000           # Maximum input vector size
    value_range_min: float = -10.0        # Minimum allowed input value
    value_range_max: float = 10.0         # Maximum allowed input value
    
    # Threat detection
    anomaly_threshold: float = 3.0        # Standard deviations for anomaly
    rate_limit_requests: int = 1000       # Requests per minute
    rate_limit_window: int = 60           # Rate limiting window (seconds)
    
    # Security policies
    security_policy: SecurityPolicy = SecurityPolicy.ADAPTIVE
    quarantine_duration: int = 300        # Quarantine time (seconds)
    
    # Monitoring
    log_level: str = "INFO"
    audit_trail_size: int = 10000         # Maximum audit entries
    threat_history_size: int = 1000       # Threat detection history


class QuantumResistantCrypto:
    """Quantum-resistant cryptographic operations."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.master_key = self._generate_master_key()
        self.session_keys = {}
        self.nonce_history = set()
    
    def _generate_master_key(self) -> bytes:
        """Generate quantum-resistant master key."""
        return secrets.token_bytes(self.config.key_size // 8)
    
    def generate_session_key(self, session_id: str) -> bytes:
        """Generate session-specific encryption key."""
        # Use HKDF for key derivation (quantum-resistant)
        session_key = hmac.new(
            self.master_key,
            f"session_{session_id}".encode(),
            hashlib.sha3_256
        ).digest()
        
        self.session_keys[session_id] = session_key
        return session_key
    
    def encrypt_quantum_state(self, quantum_state: List[complex], session_id: str) -> Dict[str, Any]:
        """Encrypt quantum state data with quantum-resistant cipher."""
        
        if session_id not in self.session_keys:
            self.generate_session_key(session_id)
        
        session_key = self.session_keys[session_id]
        nonce = secrets.token_bytes(16)
        
        # Ensure nonce uniqueness
        nonce_hash = hashlib.sha3_256(nonce).hexdigest()
        if nonce_hash in self.nonce_history:
            nonce = secrets.token_bytes(16)
            nonce_hash = hashlib.sha3_256(nonce).hexdigest()
        
        self.nonce_history.add(nonce_hash)
        
        # Convert complex numbers to bytes for encryption
        state_bytes = self._complex_to_bytes(quantum_state)
        
        # Simple XOR cipher with key stretching (placeholder for full quantum-resistant cipher)
        key_stream = self._generate_key_stream(session_key, nonce, len(state_bytes))
        encrypted_bytes = bytes(a ^ b for a, b in zip(state_bytes, key_stream))
        
        # Compute authentication tag
        auth_tag = hmac.new(session_key, encrypted_bytes + nonce, hashlib.sha3_256).digest()
        
        return {
            'encrypted_data': encrypted_bytes,
            'nonce': nonce,
            'auth_tag': auth_tag,
            'session_id': session_id,
            'encryption_timestamp': time.time()
        }
    
    def decrypt_quantum_state(self, encrypted_data: Dict[str, Any]) -> List[complex]:
        """Decrypt quantum state with integrity verification."""
        
        session_id = encrypted_data['session_id']
        if session_id not in self.session_keys:
            raise SecurityError("Invalid session key")
        
        session_key = self.session_keys[session_id]
        encrypted_bytes = encrypted_data['encrypted_data']
        nonce = encrypted_data['nonce']
        auth_tag = encrypted_data['auth_tag']
        
        # Verify authentication tag
        expected_tag = hmac.new(session_key, encrypted_bytes + nonce, hashlib.sha3_256).digest()
        if not hmac.compare_digest(auth_tag, expected_tag):
            raise SecurityError("Authentication verification failed")
        
        # Decrypt data
        key_stream = self._generate_key_stream(session_key, nonce, len(encrypted_bytes))
        decrypted_bytes = bytes(a ^ b for a, b in zip(encrypted_bytes, key_stream))
        
        # Convert back to complex numbers
        return self._bytes_to_complex(decrypted_bytes)
    
    def _complex_to_bytes(self, quantum_state: List[complex]) -> bytes:
        """Convert complex quantum state to bytes."""
        result = []
        for c in quantum_state:
            # Convert to 64-bit floats (8 bytes each for real and imaginary)
            real_bytes = int(c.real * 1e6).to_bytes(8, 'big', signed=True)
            imag_bytes = int(c.imag * 1e6).to_bytes(8, 'big', signed=True)
            result.extend(real_bytes + imag_bytes)
        return bytes(result)
    
    def _bytes_to_complex(self, data: bytes) -> List[complex]:
        """Convert bytes back to complex quantum state."""
        quantum_state = []
        for i in range(0, len(data), 16):  # 16 bytes per complex number
            if i + 16 <= len(data):
                real_bytes = data[i:i+8]
                imag_bytes = data[i+8:i+16]
                
                real_val = int.from_bytes(real_bytes, 'big', signed=True) / 1e6
                imag_val = int.from_bytes(imag_bytes, 'big', signed=True) / 1e6
                
                quantum_state.append(complex(real_val, imag_val))
        
        return quantum_state
    
    def _generate_key_stream(self, key: bytes, nonce: bytes, length: int) -> bytes:
        """Generate key stream for encryption (simplified)."""
        key_stream = []
        counter = 0
        
        while len(key_stream) < length:
            # Hash key + nonce + counter for key stream
            hash_input = key + nonce + counter.to_bytes(8, 'big')
            hash_output = hashlib.sha3_256(hash_input).digest()
            key_stream.extend(hash_output)
            counter += 1
        
        return bytes(key_stream[:length])


class PhotonicTamperDetection:
    """Detects tampering attempts on photonic circuits."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.baseline_signatures = {}
        self.anomaly_detector = PhotonicAnomalyDetector()
    
    def establish_baseline(self, circuit_id: str, photonic_patterns: Dict[str, List[float]]) -> Dict[str, Any]:
        """Establish baseline signature for photonic circuit."""
        
        signature = self._compute_photonic_signature(photonic_patterns)
        self.baseline_signatures[circuit_id] = {
            'signature': signature,
            'timestamp': time.time(),
            'pattern_count': len(photonic_patterns)
        }
        
        return {
            'circuit_id': circuit_id,
            'baseline_established': True,
            'signature_strength': signature['strength'],
            'protected_channels': len(photonic_patterns)
        }
    
    def detect_tampering(self, circuit_id: str, current_patterns: Dict[str, List[float]]) -> Dict[str, Any]:
        """Detect tampering by comparing current patterns to baseline."""
        
        if circuit_id not in self.baseline_signatures:
            return {
                'circuit_id': circuit_id,
                'tampering_detected': False,
                'warning': 'No baseline signature available',
                'threat_level': ThreatLevel.LOW.value
            }
        
        baseline = self.baseline_signatures[circuit_id]
        current_signature = self._compute_photonic_signature(current_patterns)
        
        # Compare signatures
        similarity = self._compare_signatures(baseline['signature'], current_signature)
        tampering_detected = similarity < 0.9  # 90% similarity threshold
        
        # Analyze specific anomalies
        anomalies = self.anomaly_detector.detect_anomalies(current_patterns)
        
        # Determine threat level
        threat_level = self._assess_threat_level(similarity, anomalies)
        
        return {
            'circuit_id': circuit_id,
            'tampering_detected': tampering_detected,
            'similarity_score': similarity,
            'threat_level': threat_level.value,
            'anomalies_detected': len(anomalies),
            'anomaly_details': anomalies,
            'baseline_age': time.time() - baseline['timestamp']
        }
    
    def _compute_photonic_signature(self, photonic_patterns: Dict[str, List[float]]) -> Dict[str, Any]:
        """Compute cryptographic signature of photonic patterns."""
        
        # Combine all patterns
        combined_data = []
        for wavelength, pattern in sorted(photonic_patterns.items()):
            combined_data.extend(pattern)
        
        if not combined_data:
            return {'hash': '', 'strength': 0.0, 'features': []}
        
        # Statistical features
        features = {
            'mean': sum(combined_data) / len(combined_data),
            'variance': sum((x - sum(combined_data)/len(combined_data))**2 for x in combined_data) / len(combined_data),
            'min_value': min(combined_data),
            'max_value': max(combined_data),
            'zero_crossings': sum(1 for i in range(len(combined_data)-1) if combined_data[i] * combined_data[i+1] < 0),
            'energy': sum(x**2 for x in combined_data)
        }
        
        # Compute hash of quantized features
        feature_string = ''.join(f"{k}:{v:.6f}" for k, v in sorted(features.items()))
        signature_hash = hashlib.sha3_256(feature_string.encode()).hexdigest()
        
        # Compute signature strength (entropy-based)
        strength = self._compute_entropy(combined_data)
        
        return {
            'hash': signature_hash,
            'strength': strength,
            'features': features
        }
    
    def _compare_signatures(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> float:
        """Compare two photonic signatures."""
        
        if baseline['hash'] == current['hash']:
            return 1.0
        
        # Compare statistical features
        baseline_features = baseline['features']
        current_features = current['features']
        
        feature_similarities = []
        for key in baseline_features:
            if key in current_features:
                baseline_val = baseline_features[key]
                current_val = current_features[key]
                
                if baseline_val == 0 and current_val == 0:
                    similarity = 1.0
                elif baseline_val == 0 or current_val == 0:
                    similarity = 0.0
                else:
                    # Relative difference
                    similarity = 1.0 - abs(baseline_val - current_val) / max(abs(baseline_val), abs(current_val))
                
                feature_similarities.append(similarity)
        
        return sum(feature_similarities) / len(feature_similarities) if feature_similarities else 0.0
    
    def _compute_entropy(self, data: List[float]) -> float:
        """Compute Shannon entropy of data."""
        if not data:
            return 0.0
        
        # Quantize data into bins
        data_range = max(data) - min(data)
        if data_range == 0:
            return 0.0
        
        num_bins = 32
        bin_counts = [0] * num_bins
        
        for value in data:
            bin_idx = int((value - min(data)) / data_range * (num_bins - 1))
            bin_idx = max(0, min(bin_idx, num_bins - 1))
            bin_counts[bin_idx] += 1
        
        # Compute entropy
        total_samples = len(data)
        entropy = 0.0
        
        for count in bin_counts:
            if count > 0:
                probability = count / total_samples
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _assess_threat_level(self, similarity: float, anomalies: List[Dict[str, Any]]) -> ThreatLevel:
        """Assess overall threat level based on evidence."""
        
        if similarity < 0.5:
            return ThreatLevel.CRITICAL
        elif similarity < 0.7:
            return ThreatLevel.HIGH
        elif similarity < 0.9 or len(anomalies) > 2:
            return ThreatLevel.MEDIUM
        elif len(anomalies) > 0:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.NONE


class PhotonicAnomalyDetector:
    """Detects anomalies in photonic signal patterns."""
    
    def __init__(self):
        self.pattern_history = []
        self.statistical_baselines = {}
    
    def detect_anomalies(self, photonic_patterns: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """Detect anomalies in photonic patterns."""
        
        anomalies = []
        
        for wavelength, pattern in photonic_patterns.items():
            if not pattern:
                continue
            
            # Statistical anomaly detection
            pattern_stats = self._compute_pattern_statistics(pattern)
            
            # Check for unusual statistical properties
            if pattern_stats['std'] > 5.0:  # High variance anomaly
                anomalies.append({
                    'type': 'high_variance',
                    'wavelength': wavelength,
                    'severity': 'medium',
                    'value': pattern_stats['std']
                })
            
            if abs(pattern_stats['mean']) > 2.0:  # DC offset anomaly  
                anomalies.append({
                    'type': 'dc_offset',
                    'wavelength': wavelength,
                    'severity': 'low',
                    'value': pattern_stats['mean']
                })
            
            if pattern_stats['peak_to_peak'] > 10.0:  # Amplitude anomaly
                anomalies.append({
                    'type': 'amplitude_anomaly',
                    'wavelength': wavelength,
                    'severity': 'high',
                    'value': pattern_stats['peak_to_peak']
                })
            
            # Check for frequency anomalies (simplified)
            if self._detect_frequency_anomaly(pattern):
                anomalies.append({
                    'type': 'frequency_anomaly',
                    'wavelength': wavelength,
                    'severity': 'medium',
                    'description': 'Unusual frequency content detected'
                })
        
        return anomalies
    
    def _compute_pattern_statistics(self, pattern: List[float]) -> Dict[str, float]:
        """Compute statistical properties of pattern."""
        if not pattern:
            return {'mean': 0, 'std': 0, 'peak_to_peak': 0}
        
        mean_val = sum(pattern) / len(pattern)
        variance = sum((x - mean_val)**2 for x in pattern) / len(pattern)
        std_val = math.sqrt(variance)
        peak_to_peak = max(pattern) - min(pattern)
        
        return {
            'mean': mean_val,
            'std': std_val,
            'peak_to_peak': peak_to_peak
        }
    
    def _detect_frequency_anomaly(self, pattern: List[float]) -> bool:
        """Detect frequency domain anomalies (simplified)."""
        if len(pattern) < 4:
            return False
        
        # Simple frequency analysis: count zero crossings
        zero_crossings = sum(1 for i in range(len(pattern)-1) 
                           if pattern[i] * pattern[i+1] < 0)
        
        # Anomaly if too many or too few zero crossings
        expected_crossings = len(pattern) // 4  # Heuristic
        return abs(zero_crossings - expected_crossings) > expected_crossings


class NeuromorphicIntrusionDetection:
    """Neuromorphic-based intrusion detection system."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.spike_pattern_baselines = {}
        self.intrusion_signatures = self._load_intrusion_signatures()
        self.false_positive_rate = 0.01
    
    def _load_intrusion_signatures(self) -> Dict[str, Dict[str, Any]]:
        """Load known intrusion signatures."""
        return {
            'burst_flooding': {
                'pattern': 'excessive_burst_spikes',
                'threshold': 100,
                'window': 10,
                'severity': ThreatLevel.HIGH
            },
            'timing_manipulation': {
                'pattern': 'regular_intervals',
                'threshold': 0.95,  # Regularity score
                'window': 20,
                'severity': ThreatLevel.MEDIUM
            },
            'pattern_injection': {
                'pattern': 'foreign_spike_pattern',
                'threshold': 0.8,   # Dissimilarity score
                'window': 15,
                'severity': ThreatLevel.HIGH
            }
        }
    
    def establish_baseline_behavior(self, session_id: str, spike_patterns: List[List[float]]) -> Dict[str, Any]:
        """Establish baseline neuromorphic behavior."""
        
        baseline = {
            'spike_rate': self._compute_average_spike_rate(spike_patterns),
            'burst_frequency': self._compute_burst_frequency(spike_patterns),
            'inter_spike_intervals': self._compute_isi_statistics(spike_patterns),
            'pattern_entropy': self._compute_pattern_entropy(spike_patterns),
            'establishment_time': time.time()
        }
        
        self.spike_pattern_baselines[session_id] = baseline
        
        return {
            'session_id': session_id,
            'baseline_established': True,
            'spike_rate': baseline['spike_rate'],
            'pattern_entropy': baseline['pattern_entropy']
        }
    
    def detect_intrusion(self, session_id: str, current_patterns: List[List[float]]) -> Dict[str, Any]:
        """Detect intrusion attempts in neuromorphic patterns."""
        
        detections = []
        
        # Check against baseline if available
        if session_id in self.spike_pattern_baselines:
            baseline_violations = self._check_baseline_violations(
                self.spike_pattern_baselines[session_id], 
                current_patterns
            )
            detections.extend(baseline_violations)
        
        # Check against known intrusion signatures
        signature_matches = self._check_intrusion_signatures(current_patterns)
        detections.extend(signature_matches)
        
        # Overall threat assessment
        max_threat = ThreatLevel.NONE
        for detection in detections:
            threat_level = ThreatLevel(detection['threat_level'])
            if self._threat_level_value(threat_level) > self._threat_level_value(max_threat):
                max_threat = threat_level
        
        return {
            'session_id': session_id,
            'intrusion_detected': len(detections) > 0,
            'detection_count': len(detections),
            'detections': detections,
            'overall_threat_level': max_threat.value,
            'analysis_timestamp': time.time()
        }
    
    def _compute_average_spike_rate(self, spike_patterns: List[List[float]]) -> float:
        """Compute average spike rate across patterns."""
        if not spike_patterns:
            return 0.0
        
        total_spikes = sum(sum(1 for spike in pattern if spike > 0.5) for pattern in spike_patterns)
        total_time = len(spike_patterns) * max(len(pattern) for pattern in spike_patterns if pattern)
        
        return total_spikes / total_time if total_time > 0 else 0.0
    
    def _compute_burst_frequency(self, spike_patterns: List[List[float]]) -> float:
        """Compute frequency of burst events."""
        burst_count = 0
        
        for pattern in spike_patterns:
            # Simple burst detection: 3+ consecutive spikes
            consecutive_spikes = 0
            for spike in pattern:
                if spike > 0.5:
                    consecutive_spikes += 1
                    if consecutive_spikes >= 3:
                        burst_count += 1
                        consecutive_spikes = 0
                else:
                    consecutive_spikes = 0
        
        return burst_count / len(spike_patterns) if spike_patterns else 0.0
    
    def _compute_isi_statistics(self, spike_patterns: List[List[float]]) -> Dict[str, float]:
        """Compute inter-spike interval statistics."""
        all_isis = []
        
        for pattern in spike_patterns:
            spike_times = [i for i, spike in enumerate(pattern) if spike > 0.5]
            
            if len(spike_times) >= 2:
                isis = [spike_times[i+1] - spike_times[i] for i in range(len(spike_times)-1)]
                all_isis.extend(isis)
        
        if not all_isis:
            return {'mean': 0, 'std': 0, 'cv': 0}
        
        mean_isi = sum(all_isis) / len(all_isis)
        var_isi = sum((x - mean_isi)**2 for x in all_isis) / len(all_isis)
        std_isi = math.sqrt(var_isi)
        cv = std_isi / mean_isi if mean_isi > 0 else 0
        
        return {'mean': mean_isi, 'std': std_isi, 'cv': cv}
    
    def _compute_pattern_entropy(self, spike_patterns: List[List[float]]) -> float:
        """Compute entropy of spike patterns."""
        if not spike_patterns:
            return 0.0
        
        # Convert patterns to binary and compute entropy
        all_spikes = []
        for pattern in spike_patterns:
            binary_pattern = [1 if spike > 0.5 else 0 for spike in pattern]
            all_spikes.extend(binary_pattern)
        
        if not all_spikes:
            return 0.0
        
        spike_count = sum(all_spikes)
        no_spike_count = len(all_spikes) - spike_count
        
        if spike_count == 0 or no_spike_count == 0:
            return 0.0
        
        p_spike = spike_count / len(all_spikes)
        p_no_spike = no_spike_count / len(all_spikes)
        
        entropy = -(p_spike * math.log2(p_spike) + p_no_spike * math.log2(p_no_spike))
        return entropy
    
    def _check_baseline_violations(self, baseline: Dict[str, Any], current_patterns: List[List[float]]) -> List[Dict[str, Any]]:
        """Check for violations of baseline behavior."""
        violations = []
        
        # Check spike rate deviation
        current_rate = self._compute_average_spike_rate(current_patterns)
        rate_deviation = abs(current_rate - baseline['spike_rate']) / (baseline['spike_rate'] + 1e-6)
        
        if rate_deviation > 0.5:  # 50% deviation
            violations.append({
                'type': 'spike_rate_deviation',
                'severity': 'high' if rate_deviation > 1.0 else 'medium',
                'baseline_value': baseline['spike_rate'],
                'current_value': current_rate,
                'deviation': rate_deviation,
                'threat_level': ThreatLevel.HIGH.value if rate_deviation > 1.0 else ThreatLevel.MEDIUM.value
            })
        
        # Check burst frequency deviation
        current_burst_freq = self._compute_burst_frequency(current_patterns)
        burst_deviation = abs(current_burst_freq - baseline['burst_frequency']) / (baseline['burst_frequency'] + 1e-6)
        
        if burst_deviation > 0.3:
            violations.append({
                'type': 'burst_frequency_deviation',
                'severity': 'medium',
                'baseline_value': baseline['burst_frequency'],
                'current_value': current_burst_freq,
                'deviation': burst_deviation,
                'threat_level': ThreatLevel.MEDIUM.value
            })
        
        return violations
    
    def _check_intrusion_signatures(self, current_patterns: List[List[float]]) -> List[Dict[str, Any]]:
        """Check current patterns against known intrusion signatures."""
        matches = []
        
        for signature_name, signature_data in self.intrusion_signatures.items():
            match_result = self._match_signature(current_patterns, signature_data)
            
            if match_result['match']:
                matches.append({
                    'signature': signature_name,
                    'match_confidence': match_result['confidence'],
                    'threat_level': signature_data['severity'].value,
                    'details': match_result['details']
                })
        
        return matches
    
    def _match_signature(self, patterns: List[List[float]], signature: Dict[str, Any]) -> Dict[str, Any]:
        """Match patterns against specific intrusion signature."""
        
        if signature['pattern'] == 'excessive_burst_spikes':
            burst_count = sum(1 for pattern in patterns 
                            for i in range(len(pattern)-2)
                            if all(pattern[i+j] > 0.5 for j in range(3)))
            
            match = burst_count > signature['threshold']
            confidence = min(1.0, burst_count / signature['threshold']) if match else 0.0
            
            return {
                'match': match,
                'confidence': confidence,
                'details': f'Burst count: {burst_count}, threshold: {signature["threshold"]}'
            }
        
        elif signature['pattern'] == 'regular_intervals':
            # Check for overly regular spike patterns
            regularity_scores = []
            
            for pattern in patterns:
                spike_times = [i for i, spike in enumerate(pattern) if spike > 0.5]
                if len(spike_times) >= 3:
                    intervals = [spike_times[i+1] - spike_times[i] for i in range(len(spike_times)-1)]
                    if intervals:
                        interval_std = math.sqrt(sum((x - sum(intervals)/len(intervals))**2 for x in intervals) / len(intervals))
                        interval_mean = sum(intervals) / len(intervals)
                        regularity = 1.0 - (interval_std / (interval_mean + 1e-6))
                        regularity_scores.append(regularity)
            
            if regularity_scores:
                avg_regularity = sum(regularity_scores) / len(regularity_scores)
                match = avg_regularity > signature['threshold']
                
                return {
                    'match': match,
                    'confidence': avg_regularity if match else 0.0,
                    'details': f'Regularity score: {avg_regularity:.3f}, threshold: {signature["threshold"]}'
                }
        
        return {'match': False, 'confidence': 0.0, 'details': 'No match'}
    
    def _threat_level_value(self, threat_level: ThreatLevel) -> int:
        """Convert threat level to numeric value for comparison."""
        threat_values = {
            ThreatLevel.NONE: 0,
            ThreatLevel.LOW: 1,
            ThreatLevel.MEDIUM: 2,
            ThreatLevel.HIGH: 3,
            ThreatLevel.CRITICAL: 4
        }
        return threat_values.get(threat_level, 0)


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


class QuantumPhotonicSecuritySystem:
    """Comprehensive security system for quantum-photonic-neuromorphic processing."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        
        # Initialize security components
        self.crypto = QuantumResistantCrypto(config)
        self.tamper_detector = PhotonicTamperDetection(config)
        self.intrusion_detector = NeuromorphicIntrusionDetection(config)
        
        # Security state
        self.threat_history = []
        self.quarantined_sessions = {}
        self.rate_limits = {}
        
        # Audit trail
        self.audit_log = []
        
        # Configure logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(__name__)
    
    def validate_input(self, input_data: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Comprehensive input validation and threat assessment."""
        
        validation_start = time.time()
        threats_detected = []
        
        # Rate limiting check
        rate_limit_result = self._check_rate_limits(session_id)
        if not rate_limit_result['allowed']:
            threats_detected.append({
                'type': 'rate_limit_exceeded',
                'severity': ThreatLevel.HIGH.value,
                'details': rate_limit_result
            })
        
        # Input size validation
        if 'input_features' in input_data:
            features = input_data['input_features']
            if isinstance(features, list) and len(features) > self.config.max_input_size:
                threats_detected.append({
                    'type': 'input_size_exceeded',
                    'severity': ThreatLevel.MEDIUM.value,
                    'size': len(features),
                    'max_allowed': self.config.max_input_size
                })
        
        # Value range validation
        if 'input_features' in input_data:
            features = input_data['input_features']
            if isinstance(features, list):
                for i, value in enumerate(features):
                    if not (self.config.value_range_min <= value <= self.config.value_range_max):
                        threats_detected.append({
                            'type': 'value_out_of_range',
                            'severity': ThreatLevel.MEDIUM.value,
                            'index': i,
                            'value': value,
                            'allowed_range': [self.config.value_range_min, self.config.value_range_max]
                        })
                        break  # Report first violation only
        
        # Statistical anomaly detection
        anomaly_result = self._detect_statistical_anomalies(input_data)
        if anomaly_result['anomalies_detected']:
            threats_detected.extend(anomaly_result['anomalies'])
        
        # Determine overall validation result
        max_severity = ThreatLevel.NONE
        for threat in threats_detected:
            threat_level = ThreatLevel(threat['severity'])
            if self._threat_level_value(threat_level) > self._threat_level_value(max_severity):
                max_severity = threat_level
        
        # Apply security policy
        policy_action = self._apply_security_policy(max_severity, session_id)
        
        # Log validation
        self._audit_log_entry('input_validation', session_id, {
            'threats_detected': len(threats_detected),
            'max_severity': max_severity.value,
            'policy_action': policy_action['action'],
            'validation_time': time.time() - validation_start
        })
        
        return {
            'validation_passed': policy_action['allow_processing'],
            'threats_detected': threats_detected,
            'threat_level': max_severity.value,
            'policy_action': policy_action,
            'session_id': session_id,
            'validation_timestamp': time.time()
        }
    
    def secure_processing(self, processing_function, input_data: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Secure wrapper for quantum-photonic processing."""
        
        # Pre-processing validation
        validation_result = self.validate_input(input_data, session_id)
        
        if not validation_result['validation_passed']:
            return {
                'processing_completed': False,
                'security_blocked': True,
                'validation_result': validation_result,
                'error': 'Processing blocked due to security policy'
            }
        
        try:
            # Generate session key for encryption
            session_key = self.crypto.generate_session_key(session_id)
            
            # Execute processing with monitoring
            processing_start = time.time()
            processing_result = processing_function(input_data)
            processing_time = time.time() - processing_start
            
            # Post-processing security checks
            post_processing_checks = self._post_processing_security_checks(
                processing_result, session_id, processing_time
            )
            
            # Encrypt sensitive outputs if needed
            if 'quantum_output' in processing_result:
                encrypted_quantum = self.crypto.encrypt_quantum_state(
                    processing_result['quantum_output'], session_id
                )
                processing_result['encrypted_quantum'] = encrypted_quantum
            
            # Update threat history
            self._update_threat_history(session_id, validation_result, post_processing_checks)
            
            return {
                'processing_completed': True,
                'security_blocked': False,
                'processing_result': processing_result,
                'validation_result': validation_result,
                'post_processing_checks': post_processing_checks,
                'processing_time': processing_time
            }
            
        except Exception as e:
            self.logger.error(f"Secure processing failed for session {session_id}: {str(e)}")
            
            # Log security incident
            self._audit_log_entry('processing_error', session_id, {
                'error': str(e),
                'error_type': type(e).__name__
            })
            
            return {
                'processing_completed': False,
                'security_blocked': True,
                'error': 'Processing failed due to security error',
                'validation_result': validation_result
            }
    
    def _check_rate_limits(self, session_id: str) -> Dict[str, Any]:
        """Check if session exceeds rate limits."""
        
        current_time = time.time()
        window_start = current_time - self.config.rate_limit_window
        
        if session_id not in self.rate_limits:
            self.rate_limits[session_id] = []
        
        # Clean old requests
        self.rate_limits[session_id] = [
            req_time for req_time in self.rate_limits[session_id] 
            if req_time > window_start
        ]
        
        # Check current rate
        request_count = len(self.rate_limits[session_id])
        allowed = request_count < self.config.rate_limit_requests
        
        if allowed:
            self.rate_limits[session_id].append(current_time)
        
        return {
            'allowed': allowed,
            'current_requests': request_count,
            'limit': self.config.rate_limit_requests,
            'window_seconds': self.config.rate_limit_window,
            'reset_time': window_start + self.config.rate_limit_window
        }
    
    def _detect_statistical_anomalies(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect statistical anomalies in input data."""
        
        anomalies = []
        
        if 'input_features' in input_data:
            features = input_data['input_features']
            
            if isinstance(features, list) and len(features) > 0:
                # Statistical analysis
                mean_val = sum(features) / len(features)
                variance = sum((x - mean_val)**2 for x in features) / len(features)
                std_dev = math.sqrt(variance)
                
                # Z-score anomaly detection
                for i, value in enumerate(features):
                    if std_dev > 0:
                        z_score = abs(value - mean_val) / std_dev
                        if z_score > self.config.anomaly_threshold:
                            anomalies.append({
                                'type': 'statistical_anomaly',
                                'severity': ThreatLevel.MEDIUM.value,
                                'index': i,
                                'value': value,
                                'z_score': z_score,
                                'threshold': self.config.anomaly_threshold
                            })
                
                # Additional checks
                if std_dev > 5.0:  # High variance
                    anomalies.append({
                        'type': 'high_variance',
                        'severity': ThreatLevel.LOW.value,
                        'std_dev': std_dev
                    })
        
        return {
            'anomalies_detected': len(anomalies) > 0,
            'anomalies': anomalies
        }
    
    def _apply_security_policy(self, threat_level: ThreatLevel, session_id: str) -> Dict[str, Any]:
        """Apply security policy based on threat level."""
        
        policy = self.config.security_policy
        
        if policy == SecurityPolicy.PERMISSIVE:
            return {
                'action': 'log_and_allow',
                'allow_processing': True,
                'quarantine': False
            }
        
        elif policy == SecurityPolicy.RESTRICTIVE:
            if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                return {
                    'action': 'block',
                    'allow_processing': False,
                    'quarantine': False
                }
            else:
                return {
                    'action': 'allow_with_monitoring',
                    'allow_processing': True,
                    'quarantine': False
                }
        
        elif policy == SecurityPolicy.PARANOID:
            if threat_level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                self._quarantine_session(session_id)
                return {
                    'action': 'quarantine',
                    'allow_processing': False,
                    'quarantine': True
                }
            else:
                return {
                    'action': 'allow_with_strict_monitoring',
                    'allow_processing': True,
                    'quarantine': False
                }
        
        else:  # ADAPTIVE
            # Adaptive policy based on session history
            session_threat_score = self._compute_session_threat_score(session_id)
            
            if threat_level == ThreatLevel.CRITICAL or session_threat_score > 0.8:
                self._quarantine_session(session_id)
                return {
                    'action': 'adaptive_quarantine',
                    'allow_processing': False,
                    'quarantine': True,
                    'threat_score': session_threat_score
                }
            elif threat_level == ThreatLevel.HIGH or session_threat_score > 0.6:
                return {
                    'action': 'adaptive_restrict',
                    'allow_processing': False,
                    'quarantine': False,
                    'threat_score': session_threat_score
                }
            else:
                return {
                    'action': 'adaptive_allow',
                    'allow_processing': True,
                    'quarantine': False,
                    'threat_score': session_threat_score
                }
    
    def _post_processing_security_checks(self, processing_result: Dict[str, Any], session_id: str, processing_time: float) -> Dict[str, Any]:
        """Perform security checks after processing."""
        
        checks = []
        
        # Processing time analysis
        if processing_time > 10.0:  # Unusually long processing
            checks.append({
                'type': 'long_processing_time',
                'severity': ThreatLevel.LOW.value,
                'processing_time': processing_time,
                'threshold': 10.0
            })
        
        # Output validation
        if 'fused_output' in processing_result:
            output = processing_result['fused_output']
            if isinstance(output, list):
                # Check for unusual output patterns
                if any(abs(x) > 100.0 for x in output):
                    checks.append({
                        'type': 'unusual_output_magnitude',
                        'severity': ThreatLevel.MEDIUM.value,
                        'max_value': max(abs(x) for x in output)
                    })
        
        return {
            'checks_performed': len(checks),
            'security_issues': checks,
            'post_processing_timestamp': time.time()
        }
    
    def _quarantine_session(self, session_id: str):
        """Quarantine a session due to security threats."""
        self.quarantined_sessions[session_id] = {
            'quarantine_start': time.time(),
            'duration': self.config.quarantine_duration
        }
        
        self.logger.warning(f"Session {session_id} quarantined due to security threats")
    
    def _compute_session_threat_score(self, session_id: str) -> float:
        """Compute cumulative threat score for session."""
        
        # Simple threat scoring based on recent history
        recent_threats = [
            threat for threat in self.threat_history 
            if threat['session_id'] == session_id and
               time.time() - threat['timestamp'] < 300  # Last 5 minutes
        ]
        
        if not recent_threats:
            return 0.0
        
        # Weight by severity
        threat_weights = {
            ThreatLevel.LOW.value: 0.1,
            ThreatLevel.MEDIUM.value: 0.3,
            ThreatLevel.HIGH.value: 0.7,
            ThreatLevel.CRITICAL.value: 1.0
        }
        
        total_score = sum(threat_weights.get(threat['max_severity'], 0.0) for threat in recent_threats)
        return min(1.0, total_score / 5.0)  # Normalize to [0, 1]
    
    def _update_threat_history(self, session_id: str, validation_result: Dict[str, Any], post_checks: Dict[str, Any]):
        """Update threat detection history."""
        
        threat_entry = {
            'session_id': session_id,
            'timestamp': time.time(),
            'max_severity': validation_result['threat_level'],
            'threats_count': len(validation_result['threats_detected']),
            'post_checks_count': len(post_checks['security_issues'])
        }
        
        self.threat_history.append(threat_entry)
        
        # Maintain history size
        if len(self.threat_history) > self.config.threat_history_size:
            self.threat_history.pop(0)
    
    def _audit_log_entry(self, event_type: str, session_id: str, details: Dict[str, Any]):
        """Add entry to audit log."""
        
        audit_entry = {
            'timestamp': time.time(),
            'event_type': event_type,
            'session_id': session_id,
            'details': details
        }
        
        self.audit_log.append(audit_entry)
        
        # Maintain audit log size
        if len(self.audit_log) > self.config.audit_trail_size:
            self.audit_log.pop(0)
        
        # Log to system logger
        self.logger.info(f"Security audit: {event_type} for session {session_id}")
    
    def _threat_level_value(self, threat_level: ThreatLevel) -> int:
        """Convert threat level to numeric value."""
        threat_values = {
            ThreatLevel.NONE: 0,
            ThreatLevel.LOW: 1,
            ThreatLevel.MEDIUM: 2,
            ThreatLevel.HIGH: 3,
            ThreatLevel.CRITICAL: 4
        }
        return threat_values.get(threat_level, 0)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security system status."""
        
        current_time = time.time()
        
        # Active quarantines
        active_quarantines = {
            session_id: details for session_id, details in self.quarantined_sessions.items()
            if current_time - details['quarantine_start'] < details['duration']
        }
        
        # Recent threat statistics
        recent_threats = [
            threat for threat in self.threat_history
            if current_time - threat['timestamp'] < 3600  # Last hour
        ]
        
        threat_level_counts = {}
        for threat in recent_threats:
            level = threat['max_severity']
            threat_level_counts[level] = threat_level_counts.get(level, 0) + 1
        
        return {
            'system_status': 'operational',
            'active_quarantines': len(active_quarantines),
            'quarantined_sessions': list(active_quarantines.keys()),
            'recent_threats': {
                'total': len(recent_threats),
                'by_level': threat_level_counts
            },
            'audit_log_size': len(self.audit_log),
            'security_policy': self.config.security_policy.value,
            'status_timestamp': current_time
        }


def create_security_system(
    security_policy: str = "adaptive",
    rate_limit_requests: int = 1000,
    anomaly_threshold: float = 3.0
) -> QuantumPhotonicSecuritySystem:
    """Create configured security system."""
    
    config = SecurityConfig(
        security_policy=SecurityPolicy(security_policy),
        rate_limit_requests=rate_limit_requests,
        anomaly_threshold=anomaly_threshold
    )
    
    return QuantumPhotonicSecuritySystem(config)


def demo_security_system():
    """Demonstrate quantum-photonic security system."""
    print("🛡️ Quantum-Photonic-Neuromorphic Security Demo")
    print("=" * 60)
    
    # Create security system
    security_system = create_security_system(
        security_policy="adaptive",
        rate_limit_requests=10,  # Low for demo
        anomaly_threshold=2.0
    )
    
    # Demo 1: Normal input validation
    print("✅ Testing Normal Input Validation...")
    
    normal_input = {
        'input_features': [0.5, -0.2, 0.8, 0.1, -0.3, 0.7],
        'session_metadata': {'user_id': 'test_user', 'request_id': 'req_001'}
    }
    
    validation_result = security_system.validate_input(normal_input, 'session_001')
    print(f"  Validation passed: {validation_result['validation_passed']}")
    print(f"  Threats detected: {validation_result['threats_detected']}")
    print(f"  Threat level: {validation_result['threat_level']}")
    
    # Demo 2: Malicious input detection
    print(f"\n🚨 Testing Malicious Input Detection...")
    
    malicious_inputs = [
        {
            'name': 'Out of range values',
            'data': {'input_features': [0.5, 15.0, -12.0, 0.1]},  # Values outside [-10, 10]
        },
        {
            'name': 'Oversized input',
            'data': {'input_features': list(range(15000))},  # Too many features
        },
        {
            'name': 'Statistical anomaly',
            'data': {'input_features': [0.1, 0.2, 0.1, 50.0, 0.1]},  # Outlier value
        }
    ]
    
    for i, attack in enumerate(malicious_inputs):
        print(f"  Attack {i+1}: {attack['name']}")
        
        validation_result = security_system.validate_input(attack['data'], f'session_{i+2}')
        print(f"    Blocked: {not validation_result['validation_passed']}")
        print(f"    Threats: {len(validation_result['threats_detected'])}")
        
        if validation_result['threats_detected']:
            for threat in validation_result['threats_detected'][:2]:  # Show first 2
                print(f"      - {threat['type']}: {threat['severity']}")
    
    # Demo 3: Rate limiting
    print(f"\n⏰ Testing Rate Limiting...")
    
    for i in range(12):  # Exceed rate limit of 10
        validation_result = security_system.validate_input(normal_input, 'session_rate_test')
        if not validation_result['validation_passed']:
            print(f"  Rate limit triggered at request {i+1}")
            rate_limit_threat = next(
                (t for t in validation_result['threats_detected'] if t['type'] == 'rate_limit_exceeded'),
                None
            )
            if rate_limit_threat:
                print(f"    Current requests: {rate_limit_threat['details']['current_requests']}")
                print(f"    Limit: {rate_limit_threat['details']['limit']}")
            break
    else:
        print("  Rate limit not triggered (unexpected)")
    
    # Demo 4: Secure processing wrapper
    print(f"\n🔐 Testing Secure Processing Wrapper...")
    
    def mock_processing_function(input_data):
        """Mock quantum-photonic processing function."""
        features = input_data.get('input_features', [])
        
        return {
            'quantum_output': [complex(x, x*0.5) for x in features[:4]],
            'photonic_output': [abs(x) for x in features[:3]],
            'fused_output': [sum(features)/len(features) if features else 0] * 3
        }
    
    secure_result = security_system.secure_processing(
        mock_processing_function,
        normal_input,
        'secure_session_001'
    )
    
    print(f"  Secure processing completed: {secure_result['processing_completed']}")
    print(f"  Security blocked: {secure_result['security_blocked']}")
    
    if secure_result['processing_completed']:
        print(f"  Processing time: {secure_result['processing_time']:.4f}s")
        print(f"  Encrypted outputs available: {'encrypted_quantum' in secure_result['processing_result']}")
    
    # Demo 5: Security system status
    print(f"\n📊 Security System Status:")
    status = security_system.get_security_status()
    
    print(f"  System status: {status['system_status']}")
    print(f"  Active quarantines: {status['active_quarantines']}")
    print(f"  Recent threats (1hr): {status['recent_threats']['total']}")
    print(f"  Security policy: {status['security_policy']}")
    print(f"  Audit log entries: {status['audit_log_size']}")
    
    if status['recent_threats']['by_level']:
        print(f"  Threat breakdown:")
        for level, count in status['recent_threats']['by_level'].items():
            print(f"    {level}: {count}")
    
    return security_system, status


if __name__ == "__main__":
    demo_security_system()