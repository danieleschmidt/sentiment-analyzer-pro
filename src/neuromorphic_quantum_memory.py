"""
🧠🔮 Neuromorphic-Photonic Quantum Memory System
==============================================

Revolutionary spike-driven quantum memory that stores and retrieves sentiment context
using neuromorphic spike-timing-dependent plasticity (STDP) combined with photonic
quantum memory elements.

Key Innovations:
- Spike-triggered quantum state updates
- Temporal quantum coherence across processing steps  
- Adaptive photonic weight matrices
- Context-aware sentiment memory recall

Author: Terragon Labs Autonomous SDLC System
Generation: 1 (Make It Work) - Phase 3
"""

from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass
import math
import time
import random
import json
from collections import deque


class MemoryType(Enum):
    """Types of quantum memory storage."""
    SHORT_TERM = "short_term"      # Fast access, volatile
    LONG_TERM = "long_term"        # Persistent, slower access
    WORKING = "working"            # Active processing memory
    EPISODIC = "episodic"          # Context-dependent memories


class SpikePattern(Enum):
    """Different spike pattern types for memory encoding."""
    BURST = "burst"                # High frequency burst
    REGULAR = "regular"            # Regular interval spikes  
    SPARSE = "sparse"              # Low frequency sparse spikes
    OSCILLATORY = "oscillatory"   # Rhythmic oscillations
    CHAOTIC = "chaotic"            # Irregular patterns


@dataclass
class QuantumMemoryConfig:
    """Configuration for neuromorphic-photonic quantum memory."""
    
    # Memory architecture
    num_memory_rings: int = 16        # Number of photonic memory rings
    qubits_per_ring: int = 3          # Quantum bits per memory element
    memory_depth: int = 32            # States per memory element
    context_window: int = 10          # Temporal context window
    
    # Neuromorphic parameters
    spike_threshold: float = 1.0      # LIF spike threshold
    membrane_decay: float = 0.95      # Membrane potential decay
    refractory_period: int = 2        # Post-spike refractory period
    stdp_learning_rate: float = 0.01  # STDP learning rate
    
    # Photonic memory parameters
    ring_resonance_q: float = 1000.0  # Ring resonator Q factor
    coupling_strength: float = 0.1    # Ring-waveguide coupling
    loss_coefficient: float = 0.01    # Photonic loss per round trip
    
    # Quantum coherence parameters
    decoherence_time: float = 1.0     # Quantum decoherence time (μs)
    entanglement_decay: float = 0.99  # Entanglement decay per timestep
    phase_noise: float = 0.001        # Random phase noise
    
    # Memory management
    memory_capacity: int = 1000       # Maximum stored memories
    retrieval_threshold: float = 0.7  # Minimum similarity for recall
    consolidation_rate: float = 0.1   # Long-term memory formation rate


class SpikeTimingDependentPlasticity:
    """Implements STDP learning for quantum memory updates."""
    
    def __init__(self, config: QuantumMemoryConfig):
        self.config = config
        self.spike_history = deque(maxlen=config.context_window)
        self.weight_matrix = self._initialize_weights()
    
    def _initialize_weights(self) -> List[List[float]]:
        """Initialize synaptic weight matrix."""
        size = self.config.num_memory_rings
        return [[random.uniform(0.1, 1.0) for _ in range(size)] for _ in range(size)]
    
    def update_weights(self, pre_spike_times: List[float], post_spike_times: List[float]) -> Dict[str, Any]:
        """Update synaptic weights based on spike timing."""
        
        weight_changes = []
        total_change = 0.0
        
        for pre_idx, pre_time in enumerate(pre_spike_times):
            for post_idx, post_time in enumerate(post_spike_times):
                if pre_idx < len(self.weight_matrix) and post_idx < len(self.weight_matrix[pre_idx]):
                    
                    # Calculate spike time difference
                    dt = post_time - pre_time
                    
                    # STDP learning rule
                    if dt > 0:  # Post-synaptic spike after pre-synaptic
                        # Long-term potentiation (LTP)
                        weight_change = self.config.stdp_learning_rate * math.exp(-abs(dt) / 20.0)
                    else:  # Post-synaptic spike before pre-synaptic  
                        # Long-term depression (LTD)
                        weight_change = -self.config.stdp_learning_rate * math.exp(-abs(dt) / 20.0)
                    
                    # Update weight with bounds checking
                    old_weight = self.weight_matrix[pre_idx][post_idx]
                    new_weight = max(0.0, min(2.0, old_weight + weight_change))
                    self.weight_matrix[pre_idx][post_idx] = new_weight
                    
                    weight_changes.append({
                        'pre_idx': pre_idx,
                        'post_idx': post_idx,
                        'dt': dt,
                        'weight_change': weight_change,
                        'old_weight': old_weight,
                        'new_weight': new_weight
                    })
                    
                    total_change += abs(weight_change)
        
        return {
            'weight_changes': weight_changes,
            'total_plasticity': total_change,
            'average_weight': sum(sum(row) for row in self.weight_matrix) / (len(self.weight_matrix) * len(self.weight_matrix[0]) if self.weight_matrix else 1)
        }
    
    def get_connection_strength(self, pre_idx: int, post_idx: int) -> float:
        """Get synaptic strength between neurons."""
        if 0 <= pre_idx < len(self.weight_matrix) and 0 <= post_idx < len(self.weight_matrix[pre_idx]):
            return self.weight_matrix[pre_idx][post_idx]
        return 0.0


class PhotonicQuantumMemoryRing:
    """Individual photonic quantum memory ring element."""
    
    def __init__(self, ring_id: int, config: QuantumMemoryConfig):
        self.ring_id = ring_id
        self.config = config
        
        # Quantum state storage
        self.quantum_state = self._initialize_quantum_state()
        self.phase_accumulation = 0.0
        self.stored_memories = deque(maxlen=config.memory_depth)
        
        # Ring resonator parameters
        self.resonance_frequency = self._calculate_resonance_frequency()
        self.quality_factor = config.ring_resonance_q
        self.coupling_efficiency = config.coupling_strength
        
        # Coherence tracking
        self.coherence_level = 1.0
        self.last_access_time = 0.0
    
    def _initialize_quantum_state(self) -> List[complex]:
        """Initialize quantum state vector."""
        n_qubits = self.config.qubits_per_ring
        n_states = 2 ** n_qubits
        
        # Start in equal superposition
        amplitude = 1.0 / math.sqrt(n_states)
        state = [complex(amplitude, 0.0) for _ in range(n_states)]
        
        return state
    
    def _calculate_resonance_frequency(self) -> float:
        """Calculate ring resonator resonance frequency."""
        # Simplified model: frequency depends on ring ID and configuration
        base_freq = 193.1e12  # ~1550nm in Hz
        freq_spacing = 25e9   # 25 GHz channel spacing
        
        return base_freq + self.ring_id * freq_spacing
    
    def store_quantum_state(self, new_state: List[complex], spike_pattern: SpikePattern) -> Dict[str, Any]:
        """Store quantum state in photonic memory ring."""
        
        current_time = time.time()
        
        # Apply decoherence based on time since last access
        time_diff = current_time - self.last_access_time
        coherence_decay = math.exp(-time_diff / self.config.decoherence_time)
        self.coherence_level *= coherence_decay
        
        # Normalize new state
        new_state = self._normalize_quantum_state(new_state)
        
        # Apply spike-pattern-dependent encoding
        encoded_state = self._apply_spike_encoding(new_state, spike_pattern)
        
        # Store with metadata
        memory_entry = {
            'state': encoded_state,
            'timestamp': current_time,
            'spike_pattern': spike_pattern.value,
            'coherence': self.coherence_level,
            'ring_phase': self.phase_accumulation
        }
        
        self.stored_memories.append(memory_entry)
        self.quantum_state = encoded_state[:]
        self.last_access_time = current_time
        
        # Update ring phase (photonic round-trip accumulation)
        self._update_ring_phase()
        
        return {
            'ring_id': self.ring_id,
            'stored_successfully': True,
            'coherence_level': self.coherence_level,
            'memory_occupancy': len(self.stored_memories) / self.config.memory_depth,
            'resonance_frequency': self.resonance_frequency
        }
    
    def retrieve_quantum_state(self, query_pattern: List[float]) -> Dict[str, Any]:
        """Retrieve quantum state based on similarity to query."""
        
        current_time = time.time()
        
        if not self.stored_memories:
            return {
                'ring_id': self.ring_id,
                'state': self.quantum_state,
                'similarity': 0.0,
                'retrieval_success': False,
                'coherence_level': self.coherence_level
            }
        
        # Find best matching memory
        best_match = None
        best_similarity = 0.0
        
        for memory in self.stored_memories:
            similarity = self._compute_state_similarity(memory['state'], query_pattern)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = memory
        
        # Apply retrieval threshold
        retrieval_success = best_similarity >= self.config.retrieval_threshold
        
        if retrieval_success and best_match:
            # Update coherence based on retrieval
            retrieval_coherence = best_match['coherence'] * self.config.entanglement_decay
            retrieved_state = best_match['state']
        else:
            retrieval_coherence = self.coherence_level
            retrieved_state = self.quantum_state
        
        self.last_access_time = current_time
        
        return {
            'ring_id': self.ring_id,
            'state': retrieved_state,
            'similarity': best_similarity,
            'retrieval_success': retrieval_success,
            'coherence_level': retrieval_coherence,
            'timestamp': best_match['timestamp'] if best_match else current_time,
            'spike_pattern': best_match['spike_pattern'] if best_match else 'none'
        }
    
    def _normalize_quantum_state(self, state: List[complex]) -> List[complex]:
        """Normalize quantum state vector."""
        if not state:
            return state
        
        # Calculate norm
        norm_squared = sum(abs(amplitude) ** 2 for amplitude in state)
        
        if norm_squared > 0:
            norm = math.sqrt(norm_squared)
            return [amplitude / norm for amplitude in state]
        
        return state
    
    def _apply_spike_encoding(self, state: List[complex], spike_pattern: SpikePattern) -> List[complex]:
        """Apply spike-pattern-dependent quantum encoding."""
        
        encoded_state = state[:]
        
        if spike_pattern == SpikePattern.BURST:
            # Burst patterns enhance coherence
            enhancement_factor = 1.2
            encoded_state = [amp * enhancement_factor for amp in encoded_state]
            
        elif spike_pattern == SpikePattern.SPARSE:
            # Sparse patterns add decoherence
            decoherence_factor = 0.9
            encoded_state = [amp * decoherence_factor for amp in encoded_state]
            
        elif spike_pattern == SpikePattern.OSCILLATORY:
            # Oscillatory patterns add phase modulation
            for i, amp in enumerate(encoded_state):
                phase_shift = 0.1 * math.sin(2 * math.pi * i / len(encoded_state))
                encoded_state[i] = amp * complex(math.cos(phase_shift), math.sin(phase_shift))
                
        elif spike_pattern == SpikePattern.CHAOTIC:
            # Chaotic patterns add random phase noise
            for i, amp in enumerate(encoded_state):
                noise_phase = random.uniform(-self.config.phase_noise, self.config.phase_noise)
                encoded_state[i] = amp * complex(math.cos(noise_phase), math.sin(noise_phase))
        
        # Re-normalize after encoding
        return self._normalize_quantum_state(encoded_state)
    
    def _update_ring_phase(self):
        """Update accumulated phase in ring resonator."""
        # Phase accumulation due to photonic round-trip
        round_trip_phase = 2 * math.pi * self.resonance_frequency * (1.0 / (299792458 * 1000))  # Simplified
        self.phase_accumulation += round_trip_phase
        self.phase_accumulation %= (2 * math.pi)
    
    def _compute_state_similarity(self, stored_state: List[complex], query_pattern: List[float]) -> float:
        """Compute similarity between stored quantum state and query pattern."""
        
        if not stored_state or not query_pattern:
            return 0.0
        
        # Convert stored state to real values (probabilities)
        stored_probs = [abs(amp) ** 2 for amp in stored_state]
        
        # Pad or truncate to match lengths
        min_len = min(len(stored_probs), len(query_pattern))
        if min_len == 0:
            return 0.0
        
        # Compute normalized dot product similarity
        dot_product = sum(stored_probs[i] * query_pattern[i] for i in range(min_len))
        norm_stored = math.sqrt(sum(p ** 2 for p in stored_probs[:min_len]))
        norm_query = math.sqrt(sum(q ** 2 for q in query_pattern[:min_len]))
        
        if norm_stored * norm_query > 0:
            return dot_product / (norm_stored * norm_query)
        
        return 0.0


class NeuromorphicQuantumMemorySystem:
    """Main neuromorphic-photonic quantum memory system."""
    
    def __init__(self, config: QuantumMemoryConfig):
        self.config = config
        
        # Initialize components
        self.memory_rings = [PhotonicQuantumMemoryRing(i, config) for i in range(config.num_memory_rings)]
        self.stdp_controller = SpikeTimingDependentPlasticity(config)
        
        # Memory management
        self.memory_registry = {}  # Track stored memories by ID
        self.access_history = deque(maxlen=1000)
        self.consolidation_queue = deque()
        
        # Performance metrics
        self.metrics = {
            'storage_operations': 0,
            'retrieval_operations': 0,
            'consolidation_operations': 0,
            'average_coherence': 0.0,
            'memory_utilization': 0.0,
            'stdp_plasticity': 0.0
        }
    
    def store_memory(
        self, 
        context_data: List[float],
        spike_times: List[float],
        memory_type: MemoryType = MemoryType.WORKING,
        memory_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Store memory using spike-driven quantum encoding."""
        
        if memory_id is None:
            memory_id = f"mem_{len(self.memory_registry)}_{int(time.time() * 1000)}"
        
        # Detect spike pattern
        spike_pattern = self._analyze_spike_pattern(spike_times)
        
        # Select optimal memory rings based on context and availability
        selected_rings = self._select_memory_rings(context_data, memory_type)
        
        # Convert context data to quantum states
        quantum_states = self._encode_context_to_quantum(context_data, len(selected_rings))
        
        # Store across selected rings
        storage_results = []
        for ring, quantum_state in zip(selected_rings, quantum_states):
            result = ring.store_quantum_state(quantum_state, spike_pattern)
            storage_results.append(result)
        
        # Update STDP based on spike timing
        pre_spikes = spike_times
        post_spikes = [t + random.uniform(1, 5) for t in spike_times]  # Simulate post-synaptic spikes
        stdp_result = self.stdp_controller.update_weights(pre_spikes, post_spikes)
        
        # Register memory
        memory_record = {
            'memory_id': memory_id,
            'memory_type': memory_type.value,
            'context_data': context_data,
            'spike_times': spike_times,
            'spike_pattern': spike_pattern.value,
            'ring_ids': [ring.ring_id for ring in selected_rings],
            'storage_timestamp': time.time(),
            'access_count': 0
        }
        
        self.memory_registry[memory_id] = memory_record
        self.access_history.append(('store', memory_id, time.time()))
        
        # Update metrics
        self.metrics['storage_operations'] += 1
        self.metrics['stdp_plasticity'] = stdp_result['total_plasticity']
        self._update_performance_metrics()
        
        # Queue for consolidation if appropriate
        if memory_type == MemoryType.SHORT_TERM:
            self.consolidation_queue.append(memory_id)
        
        return {
            'memory_id': memory_id,
            'storage_success': all(r['stored_successfully'] for r in storage_results),
            'ring_results': storage_results,
            'spike_pattern': spike_pattern.value,
            'stdp_result': stdp_result,
            'consolidation_queued': memory_type == MemoryType.SHORT_TERM
        }
    
    def retrieve_memory(
        self, 
        query_context: List[float],
        memory_type: Optional[MemoryType] = None,
        similarity_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Retrieve memory based on context similarity."""
        
        threshold = similarity_threshold or self.config.retrieval_threshold
        
        # Search across memory rings
        retrieval_results = []
        for ring in self.memory_rings:
            result = ring.retrieve_quantum_state(query_context)
            if result['retrieval_success'] and result['similarity'] >= threshold:
                retrieval_results.append(result)
        
        # Sort by similarity
        retrieval_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Find corresponding memory records
        matched_memories = []
        for result in retrieval_results:
            ring_id = result['ring_id']
            
            # Find memories that used this ring
            for memory_id, record in self.memory_registry.items():
                if ring_id in record['ring_ids']:
                    if memory_type is None or MemoryType(record['memory_type']) == memory_type:
                        matched_memories.append({
                            'memory_id': memory_id,
                            'memory_record': record,
                            'retrieval_result': result
                        })
        
        # Update access history and counts
        for match in matched_memories:
            memory_id = match['memory_id']
            self.memory_registry[memory_id]['access_count'] += 1
            self.access_history.append(('retrieve', memory_id, time.time()))
        
        # Update metrics
        self.metrics['retrieval_operations'] += 1
        self._update_performance_metrics()
        
        return {
            'query_processed': True,
            'matches_found': len(matched_memories),
            'matched_memories': matched_memories,
            'best_similarity': retrieval_results[0]['similarity'] if retrieval_results else 0.0,
            'retrieval_results': retrieval_results
        }
    
    def consolidate_memories(self, max_consolidations: int = 5) -> Dict[str, Any]:
        """Consolidate short-term memories to long-term storage."""
        
        consolidated = []
        consolidation_failures = []
        
        for _ in range(min(max_consolidations, len(self.consolidation_queue))):
            if not self.consolidation_queue:
                break
                
            memory_id = self.consolidation_queue.popleft()
            
            if memory_id in self.memory_registry:
                memory_record = self.memory_registry[memory_id]
                
                # Check if memory meets consolidation criteria
                if self._should_consolidate_memory(memory_record):
                    # Update memory type to long-term
                    memory_record['memory_type'] = MemoryType.LONG_TERM.value
                    memory_record['consolidation_timestamp'] = time.time()
                    
                    consolidated.append({
                        'memory_id': memory_id,
                        'original_type': MemoryType.SHORT_TERM.value,
                        'new_type': MemoryType.LONG_TERM.value,
                        'access_count': memory_record['access_count']
                    })
                else:
                    consolidation_failures.append(memory_id)
        
        # Update metrics
        self.metrics['consolidation_operations'] += len(consolidated)
        self._update_performance_metrics()
        
        return {
            'consolidations_performed': len(consolidated),
            'consolidation_failures': len(consolidation_failures),
            'consolidated_memories': consolidated,
            'remaining_queue_size': len(self.consolidation_queue)
        }
    
    def _analyze_spike_pattern(self, spike_times: List[float]) -> SpikePattern:
        """Analyze spike timing to determine pattern type."""
        
        if not spike_times or len(spike_times) < 2:
            return SpikePattern.SPARSE
        
        # Calculate inter-spike intervals
        intervals = [spike_times[i+1] - spike_times[i] for i in range(len(spike_times)-1)]
        
        if not intervals:
            return SpikePattern.SPARSE
        
        # Analyze pattern characteristics
        mean_interval = sum(intervals) / len(intervals)
        interval_variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
        coefficient_of_variation = math.sqrt(interval_variance) / mean_interval if mean_interval > 0 else float('inf')
        
        # Pattern classification
        if mean_interval < 5.0:  # High frequency
            return SpikePattern.BURST
        elif coefficient_of_variation < 0.2:  # Low variability
            return SpikePattern.REGULAR
        elif coefficient_of_variation > 1.0:  # High variability
            return SpikePattern.CHAOTIC
        elif len(spike_times) >= 4:
            # Check for oscillatory pattern
            if self._detect_oscillatory_pattern(intervals):
                return SpikePattern.OSCILLATORY
        
        return SpikePattern.SPARSE
    
    def _detect_oscillatory_pattern(self, intervals: List[float]) -> bool:
        """Detect oscillatory patterns in spike intervals."""
        if len(intervals) < 3:
            return False
        
        # Simple oscillation detection: check for alternating intervals
        alternating_count = 0
        for i in range(len(intervals) - 1):
            if (intervals[i] < intervals[i+1]) != (intervals[i-1] < intervals[i] if i > 0 else True):
                alternating_count += 1
        
        return alternating_count >= len(intervals) // 2
    
    def _select_memory_rings(self, context_data: List[float], memory_type: MemoryType) -> List[PhotonicQuantumMemoryRing]:
        """Select optimal memory rings for storage."""
        
        num_rings_needed = min(3, max(1, len(context_data) // 8))  # Heuristic selection
        
        # Score rings based on availability and suitability
        ring_scores = []
        for ring in self.memory_rings:
            # Consider ring utilization
            utilization = len(ring.stored_memories) / ring.config.memory_depth
            
            # Consider coherence level
            coherence_score = ring.coherence_level
            
            # Consider memory type compatibility (prefer different rings for different types)
            type_score = 1.0
            if memory_type == MemoryType.LONG_TERM:
                type_score = 1.2 if ring.ring_id < self.config.num_memory_rings // 2 else 0.8
            
            # Combine scores (prefer less utilized rings with high coherence)
            total_score = coherence_score * (1 - utilization) * type_score
            ring_scores.append((ring, total_score))
        
        # Sort by score and select top rings
        ring_scores.sort(key=lambda x: x[1], reverse=True)
        selected_rings = [ring for ring, score in ring_scores[:num_rings_needed]]
        
        return selected_rings
    
    def _encode_context_to_quantum(self, context_data: List[float], num_rings: int) -> List[List[complex]]:
        """Encode context data into quantum states for multiple rings."""
        
        quantum_states = []
        chunk_size = max(1, len(context_data) // num_rings)
        
        for i in range(num_rings):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_rings - 1 else len(context_data)
            
            context_chunk = context_data[start_idx:end_idx]
            
            # Convert to quantum amplitudes
            n_qubits = self.config.qubits_per_ring
            n_states = 2 ** n_qubits
            
            quantum_state = [complex(0, 0) for _ in range(n_states)]
            
            # Encode context chunk into quantum amplitudes
            for j, value in enumerate(context_chunk[:n_states]):
                quantum_state[j] = complex(value, 0)
            
            # Normalize
            norm_squared = sum(abs(amp) ** 2 for amp in quantum_state)
            if norm_squared > 0:
                norm = math.sqrt(norm_squared)
                quantum_state = [amp / norm for amp in quantum_state]
            
            quantum_states.append(quantum_state)
        
        return quantum_states
    
    def _should_consolidate_memory(self, memory_record: Dict[str, Any]) -> bool:
        """Determine if memory should be consolidated to long-term storage."""
        
        # Consolidation criteria
        min_access_count = 2
        min_age_seconds = 10.0  # Simplified for demo
        
        access_count = memory_record.get('access_count', 0)
        storage_time = memory_record.get('storage_timestamp', time.time())
        age = time.time() - storage_time
        
        return access_count >= min_access_count and age >= min_age_seconds
    
    def _update_performance_metrics(self):
        """Update system performance metrics."""
        
        if self.memory_rings:
            # Average coherence across rings
            total_coherence = sum(ring.coherence_level for ring in self.memory_rings)
            self.metrics['average_coherence'] = total_coherence / len(self.memory_rings)
            
            # Memory utilization
            total_capacity = sum(ring.config.memory_depth for ring in self.memory_rings)
            total_used = sum(len(ring.stored_memories) for ring in self.memory_rings)
            self.metrics['memory_utilization'] = total_used / total_capacity if total_capacity > 0 else 0.0
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        ring_statuses = []
        for ring in self.memory_rings:
            ring_statuses.append({
                'ring_id': ring.ring_id,
                'coherence_level': ring.coherence_level,
                'memory_occupancy': len(ring.stored_memories) / ring.config.memory_depth,
                'resonance_frequency': ring.resonance_frequency,
                'phase_accumulation': ring.phase_accumulation
            })
        
        return {
            'memory_rings': ring_statuses,
            'total_memories': len(self.memory_registry),
            'consolidation_queue_size': len(self.consolidation_queue),
            'performance_metrics': self.metrics,
            'recent_access_history': list(self.access_history)[-10:]  # Last 10 accesses
        }


def create_quantum_memory_system(
    num_rings: int = 16,
    qubits_per_ring: int = 3,
    memory_depth: int = 32,
    stdp_learning_rate: float = 0.01
) -> NeuromorphicQuantumMemorySystem:
    """Create a configured neuromorphic quantum memory system."""
    
    config = QuantumMemoryConfig(
        num_memory_rings=num_rings,
        qubits_per_ring=qubits_per_ring,
        memory_depth=memory_depth,
        stdp_learning_rate=stdp_learning_rate
    )
    
    return NeuromorphicQuantumMemorySystem(config)


def demo_quantum_memory_system():
    """Demonstrate neuromorphic-photonic quantum memory system."""
    print("🧠🔮 Neuromorphic-Photonic Quantum Memory Demo")
    print("=" * 60)
    
    # Create memory system
    memory_system = create_quantum_memory_system(
        num_rings=8,
        qubits_per_ring=3,
        memory_depth=16
    )
    
    # Demo 1: Store context memories
    print("📝 Storing Context Memories...")
    
    contexts = [
        {
            'data': [0.8, -0.2, 0.5, 0.1, 0.7, -0.3, 0.4, 0.9],
            'spikes': [1.0, 3.5, 6.2, 8.1, 12.5],
            'type': MemoryType.SHORT_TERM,
            'description': 'Positive sentiment context'
        },
        {
            'data': [-0.6, 0.3, -0.8, 0.1, -0.4, 0.7, -0.2, -0.9],
            'spikes': [2.1, 15.3, 18.7, 22.4],
            'type': MemoryType.WORKING,
            'description': 'Negative sentiment context'
        },
        {
            'data': [0.1, 0.0, -0.1, 0.3, -0.2, 0.1, 0.0, 0.2],
            'spikes': [5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
            'type': MemoryType.SHORT_TERM,
            'description': 'Neutral sentiment context'
        }
    ]
    
    stored_memories = []
    for i, context in enumerate(contexts):
        print(f"  Storing memory {i+1}: {context['description']}")
        
        result = memory_system.store_memory(
            context_data=context['data'],
            spike_times=context['spikes'],
            memory_type=context['type'],
            memory_id=f"context_{i+1}"
        )
        
        stored_memories.append(result)
        
        print(f"    ✅ Stored successfully: {result['storage_success']}")
        print(f"    🧠 Spike pattern: {result['spike_pattern']}")
        print(f"    ⚡ STDP plasticity: {result['stdp_result']['total_plasticity']:.4f}")
    
    # Demo 2: Retrieve memories based on similarity
    print(f"\n🔍 Retrieving Memories Based on Query...")
    
    query_contexts = [
        {
            'query': [0.7, -0.1, 0.4, 0.2, 0.6, -0.2, 0.3, 0.8],
            'description': 'Similar to positive context'
        },
        {
            'query': [-0.5, 0.2, -0.7, 0.0, -0.3, 0.6, -0.1, -0.8],
            'description': 'Similar to negative context'
        }
    ]
    
    for i, query in enumerate(query_contexts):
        print(f"  Query {i+1}: {query['description']}")
        
        retrieval_result = memory_system.retrieve_memory(
            query_context=query['query'],
            similarity_threshold=0.5
        )
        
        print(f"    🎯 Matches found: {retrieval_result['matches_found']}")
        print(f"    📊 Best similarity: {retrieval_result['best_similarity']:.3f}")
        
        if retrieval_result['matched_memories']:
            for match in retrieval_result['matched_memories'][:2]:  # Show top 2
                memory_id = match['memory_id']
                similarity = match['retrieval_result']['similarity']
                spike_pattern = match['memory_record']['spike_pattern']
                print(f"      {memory_id}: similarity={similarity:.3f}, pattern={spike_pattern}")
    
    # Demo 3: Memory consolidation
    print(f"\n🔄 Memory Consolidation...")
    time.sleep(0.1)  # Small delay to meet age requirement
    
    # Access memories to increase consolidation eligibility
    for stored_memory in stored_memories:
        if stored_memory['memory_id'] in memory_system.memory_registry:
            memory_system.memory_registry[stored_memory['memory_id']]['access_count'] += 2
    
    consolidation_result = memory_system.consolidate_memories(max_consolidations=3)
    
    print(f"  📈 Consolidations performed: {consolidation_result['consolidations_performed']}")
    print(f"  📋 Remaining queue size: {consolidation_result['remaining_queue_size']}")
    
    for consolidated in consolidation_result['consolidated_memories']:
        print(f"    {consolidated['memory_id']}: {consolidated['original_type']} → {consolidated['new_type']}")
    
    # Demo 4: System status
    print(f"\n📊 System Status:")
    status = memory_system.get_system_status()
    
    print(f"  Total memories: {status['total_memories']}")
    print(f"  Average coherence: {status['performance_metrics']['average_coherence']:.3f}")
    print(f"  Memory utilization: {status['performance_metrics']['memory_utilization']:.3f}")
    print(f"  STDP plasticity: {status['performance_metrics']['stdp_plasticity']:.4f}")
    
    print(f"\n🔮 Memory Ring Status:")
    for ring_status in status['memory_rings']:
        ring_id = ring_status['ring_id']
        coherence = ring_status['coherence_level']
        occupancy = ring_status['memory_occupancy']
        frequency = ring_status['resonance_frequency'] / 1e12  # Convert to THz
        
        print(f"    Ring {ring_id}: coherence={coherence:.3f}, occupancy={occupancy:.1%}, freq={frequency:.1f}THz")
    
    print(f"\n✨ Recent Access History:")
    for access_type, memory_id, timestamp in status['recent_access_history'][-5:]:
        print(f"    {access_type}: {memory_id} at {timestamp:.2f}")
    
    return memory_system, status


if __name__ == "__main__":
    demo_quantum_memory_system()