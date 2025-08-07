"""
🌈 WDM Quantum Multimodal Processing Engine
==========================================

Revolutionary wavelength-division multiplexed quantum-photonic processing
where different optical wavelengths carry quantum-encoded sentiment information
from multiple input modalities (text, audio, visual, temporal).

Key Innovations:
- Wavelength-specific quantum state encoding
- Multi-modal information multiplexing
- Photonic quantum interference processing
- Dynamic wavelength allocation optimization

Author: Terragon Labs Autonomous SDLC System  
Generation: 1 (Make It Work) - Phase 2
"""

from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass
import math
import time
import random
import json


class InputModality(Enum):
    """Different input modalities for multimodal processing."""
    TEXT = "text"
    AUDIO = "audio"
    VISUAL = "visual"
    TEMPORAL = "temporal"
    CONTEXT = "context"


class WavelengthAllocation(Enum):
    """Wavelength allocation strategies."""
    STATIC = "static"           # Fixed wavelength assignment
    DYNAMIC = "dynamic"         # Adaptive based on content
    PRIORITY = "priority"       # Priority-based allocation
    BALANCED = "balanced"       # Equal resource allocation


@dataclass
class WDMQuantumConfig:
    """Configuration for WDM quantum multimodal processing."""
    
    # Wavelength configuration
    base_wavelength: float = 1550.0  # nm
    channel_spacing: float = 0.8     # nm (dense WDM)
    num_channels: int = 8            # Total WDM channels
    channel_bandwidth: float = 50.0  # GHz
    
    # Quantum parameters per channel
    qubits_per_channel: int = 4
    quantum_layers: int = 2
    entanglement_strength: float = 0.5
    
    # Modality mapping
    text_channels: int = 3
    audio_channels: int = 2  
    visual_channels: int = 2
    temporal_channels: int = 1
    
    # Processing parameters
    interference_strength: float = 0.3
    crosstalk_suppression: float = 0.9
    wavelength_allocation: WavelengthAllocation = WavelengthAllocation.DYNAMIC
    
    # Output configuration
    output_classes: int = 3
    fusion_strategy: str = "spectral_fusion"


class QuantumWavelengthEncoder:
    """Encodes quantum states into specific wavelength channels."""
    
    def __init__(self, config: WDMQuantumConfig):
        self.config = config
        self.wavelengths = self._generate_wavelength_grid()
        self.quantum_params = self._initialize_quantum_parameters()
    
    def _generate_wavelength_grid(self) -> List[float]:
        """Generate wavelength grid for WDM channels."""
        wavelengths = []
        for i in range(self.config.num_channels):
            wl = self.config.base_wavelength + i * self.config.channel_spacing
            wavelengths.append(wl)
        return wavelengths
    
    def _initialize_quantum_parameters(self) -> Dict[float, List[float]]:
        """Initialize quantum circuit parameters for each wavelength."""
        params = {}
        for wl in self.wavelengths:
            # Random initialization of quantum parameters
            num_params = self.config.qubits_per_channel * self.config.quantum_layers * 3  # RX, RY, RZ gates
            params[wl] = [random.uniform(0, 2*math.pi) for _ in range(num_params)]
        return params
    
    def encode_modality_to_wavelength(
        self, 
        modality: InputModality,
        data: List[float],
        target_wavelength: Optional[float] = None
    ) -> Dict[str, Any]:
        """Encode modality data into quantum states at specific wavelength."""
        
        # Select target wavelength
        if target_wavelength is None:
            target_wavelength = self._select_optimal_wavelength(modality, data)
        
        # Quantum amplitude encoding
        n_qubits = self.config.qubits_per_channel
        n_amplitudes = 2 ** n_qubits
        
        # Normalize data to quantum amplitudes
        amplitudes = [0.0] * n_amplitudes
        for i, value in enumerate(data[:n_amplitudes]):
            amplitudes[i] = value
        
        # Normalize quantum state
        norm = math.sqrt(sum(amp ** 2 for amp in amplitudes))
        if norm > 0:
            amplitudes = [amp / norm for amp in amplitudes]
        
        # Apply variational quantum circuit
        quantum_state = self._apply_variational_circuit(
            amplitudes, 
            self.quantum_params[target_wavelength],
            modality
        )
        
        return {
            'wavelength': target_wavelength,
            'modality': modality.value,
            'quantum_state': quantum_state,
            'encoding_fidelity': self._compute_encoding_fidelity(data, quantum_state),
            'channel_utilization': len(data) / n_amplitudes
        }
    
    def _select_optimal_wavelength(self, modality: InputModality, data: List[float]) -> float:
        """Select optimal wavelength channel for modality and data characteristics."""
        if self.config.wavelength_allocation == WavelengthAllocation.STATIC:
            # Static assignment based on modality
            modality_map = {
                InputModality.TEXT: 0,
                InputModality.AUDIO: 3,
                InputModality.VISUAL: 5,
                InputModality.TEMPORAL: 7
            }
            idx = modality_map.get(modality, 0)
            return self.wavelengths[idx % len(self.wavelengths)]
        
        elif self.config.wavelength_allocation == WavelengthAllocation.DYNAMIC:
            # Dynamic assignment based on data characteristics
            data_energy = sum(x ** 2 for x in data) if data else 0
            data_variance = self._compute_variance(data)
            
            # Select wavelength based on data properties
            optimal_idx = int((data_energy + data_variance) * len(self.wavelengths)) % len(self.wavelengths)
            return self.wavelengths[optimal_idx]
        
        elif self.config.wavelength_allocation == WavelengthAllocation.PRIORITY:
            # Priority-based assignment (text gets shortest wavelengths)
            priority_map = {
                InputModality.TEXT: 0,
                InputModality.TEMPORAL: 1, 
                InputModality.AUDIO: 2,
                InputModality.VISUAL: 3
            }
            base_idx = priority_map.get(modality, 0)
            return self.wavelengths[base_idx % len(self.wavelengths)]
        
        else:  # BALANCED
            # Balanced allocation across all channels
            modality_hash = hash(modality.value)
            idx = modality_hash % len(self.wavelengths)
            return self.wavelengths[idx]
    
    def _apply_variational_circuit(
        self, 
        amplitudes: List[float], 
        params: List[float],
        modality: InputModality
    ) -> List[float]:
        """Apply modality-specific variational quantum circuit."""
        
        n_qubits = self.config.qubits_per_channel
        state = amplitudes[:]
        
        # Apply layered quantum gates
        param_idx = 0
        for layer in range(self.config.quantum_layers):
            
            # Single-qubit rotations (modality-dependent)
            for qubit in range(n_qubits):
                if param_idx + 2 < len(params):
                    theta_x = params[param_idx]
                    theta_y = params[param_idx + 1] 
                    theta_z = params[param_idx + 2]
                    param_idx += 3
                    
                    # Apply rotation gates (simplified simulation)
                    state = self._apply_rotation_gates(state, qubit, theta_x, theta_y, theta_z, modality)
            
            # Entangling gates based on modality
            if layer < self.config.quantum_layers - 1:
                state = self._apply_entangling_gates(state, modality)
        
        return state
    
    def _apply_rotation_gates(
        self, 
        state: List[float], 
        qubit: int, 
        theta_x: float, 
        theta_y: float,
        theta_z: float,
        modality: InputModality
    ) -> List[float]:
        """Apply single-qubit rotation gates with modality weighting."""
        
        # Modality-specific gate weighting
        modality_weights = {
            InputModality.TEXT: [1.0, 0.8, 0.6],      # Emphasize X rotations for text
            InputModality.AUDIO: [0.6, 1.0, 0.8],     # Emphasize Y rotations for audio
            InputModality.VISUAL: [0.8, 0.6, 1.0],    # Emphasize Z rotations for visual
            InputModality.TEMPORAL: [0.9, 0.9, 0.9]   # Balanced for temporal
        }
        
        weights = modality_weights.get(modality, [1.0, 1.0, 1.0])
        
        # Apply weighted rotations (simplified)
        new_state = state[:]
        for i in range(0, len(state), 2):
            if i + 1 < len(state):
                # Simplified rotation effect
                cos_half = math.cos((theta_x * weights[0] + theta_y * weights[1] + theta_z * weights[2]) / 2)
                sin_half = math.sin((theta_x * weights[0] + theta_y * weights[1] + theta_z * weights[2]) / 2)
                
                amp0, amp1 = state[i], state[i + 1]
                new_state[i] = cos_half * amp0 - sin_half * amp1
                new_state[i + 1] = sin_half * amp0 + cos_half * amp1
        
        return new_state
    
    def _apply_entangling_gates(self, state: List[float], modality: InputModality) -> List[float]:
        """Apply modality-specific entangling gates."""
        
        # Different entanglement patterns for different modalities
        if modality == InputModality.TEXT:
            # Linear entanglement for sequential text processing
            return self._linear_entanglement(state)
        elif modality == InputModality.AUDIO:
            # Circular entanglement for temporal audio patterns  
            return self._circular_entanglement(state)
        elif modality == InputModality.VISUAL:
            # Grid entanglement for spatial visual patterns
            return self._grid_entanglement(state)
        else:  # TEMPORAL or CONTEXT
            # All-to-all entanglement for global dependencies
            return self._all_to_all_entanglement(state)
    
    def _linear_entanglement(self, state: List[float]) -> List[float]:
        """Linear chain entanglement pattern."""
        new_state = state[:]
        n_qubits = self.config.qubits_per_channel
        
        for i in range(0, len(state), 2**(n_qubits-1)):
            chunk_size = min(2**(n_qubits-1), len(state) - i)
            if chunk_size >= 2:
                # Simple two-qubit gate effect
                mid = chunk_size // 2
                for j in range(mid):
                    if i + j + mid < len(new_state):
                        entangle_strength = self.config.entanglement_strength
                        amp1, amp2 = state[i + j], state[i + j + mid]
                        new_state[i + j] = math.cos(entangle_strength) * amp1 - math.sin(entangle_strength) * amp2
                        new_state[i + j + mid] = math.sin(entangle_strength) * amp1 + math.cos(entangle_strength) * amp2
        
        return new_state
    
    def _circular_entanglement(self, state: List[float]) -> List[float]:
        """Circular entanglement pattern for temporal processing.""" 
        new_state = state[:]
        n = len(state)
        
        for i in range(n):
            j = (i + 1) % n
            entangle_strength = self.config.entanglement_strength * 0.7  # Weaker circular coupling
            amp1, amp2 = state[i], state[j]
            new_state[i] = math.cos(entangle_strength) * amp1 + math.sin(entangle_strength) * amp2
        
        return new_state
    
    def _grid_entanglement(self, state: List[float]) -> List[float]:
        """Grid-like entanglement pattern for spatial processing."""
        new_state = state[:]
        grid_size = int(math.sqrt(len(state)))
        
        if grid_size * grid_size == len(state):
            # Apply nearest-neighbor entanglement in 2D grid
            for i in range(grid_size):
                for j in range(grid_size):
                    idx = i * grid_size + j
                    
                    # Entangle with right neighbor
                    if j + 1 < grid_size:
                        right_idx = i * grid_size + (j + 1)
                        amp1, amp2 = state[idx], state[right_idx]
                        entangle_strength = self.config.entanglement_strength * 0.5
                        new_state[idx] = math.cos(entangle_strength) * amp1 - math.sin(entangle_strength) * amp2
                    
                    # Entangle with bottom neighbor  
                    if i + 1 < grid_size:
                        bottom_idx = (i + 1) * grid_size + j
                        amp1, amp2 = state[idx], state[bottom_idx]
                        entangle_strength = self.config.entanglement_strength * 0.5
                        new_state[bottom_idx] = math.sin(entangle_strength) * amp1 + math.cos(entangle_strength) * amp2
        
        return new_state
    
    def _all_to_all_entanglement(self, state: List[float]) -> List[float]:
        """All-to-all entanglement for global correlations."""
        new_state = state[:]
        n = len(state)
        
        # Global mixing with reduced strength
        global_strength = self.config.entanglement_strength * 0.3
        global_sum = sum(state)
        
        for i in range(n):
            new_state[i] = (1 - global_strength) * state[i] + global_strength * (global_sum - state[i]) / (n - 1)
        
        return new_state
    
    def _compute_variance(self, data: List[float]) -> float:
        """Compute variance of data."""
        if not data:
            return 0.0
        
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        return variance
    
    def _compute_encoding_fidelity(self, original_data: List[float], quantum_state: List[float]) -> float:
        """Compute fidelity of quantum encoding."""
        if not original_data or not quantum_state:
            return 0.0
        
        # Simplified fidelity based on state overlap
        min_len = min(len(original_data), len(quantum_state))
        if min_len == 0:
            return 0.0
        
        overlap = sum(original_data[i] * quantum_state[i] for i in range(min_len))
        norm_orig = math.sqrt(sum(x ** 2 for x in original_data[:min_len]))
        norm_quantum = math.sqrt(sum(x ** 2 for x in quantum_state[:min_len]))
        
        if norm_orig * norm_quantum > 0:
            return abs(overlap) / (norm_orig * norm_quantum)
        return 0.0


class PhotonicQuantumInterferometer:
    """Implements quantum interference effects in photonic domain."""
    
    def __init__(self, config: WDMQuantumConfig):
        self.config = config
        self.interference_matrix = self._build_interference_matrix()
    
    def _build_interference_matrix(self) -> List[List[float]]:
        """Build interference matrix for wavelength interactions."""
        n = self.config.num_channels
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0  # Self-interference
                else:
                    # Wavelength-dependent interference
                    wl_i = self.config.base_wavelength + i * self.config.channel_spacing
                    wl_j = self.config.base_wavelength + j * self.config.channel_spacing
                    
                    # Interference strength inversely related to wavelength separation
                    separation = abs(wl_i - wl_j)
                    interference = self.config.interference_strength * math.exp(-separation / 10.0)
                    matrix[i][j] = interference
        
        return matrix
    
    def apply_quantum_interference(self, wavelength_states: Dict[float, List[float]]) -> Dict[float, List[float]]:
        """Apply quantum interference effects between wavelength channels."""
        
        wavelengths = sorted(wavelength_states.keys())
        states = [wavelength_states[wl] for wl in wavelengths]
        
        # Ensure all states have same dimension
        max_dim = max(len(state) for state in states) if states else 0
        normalized_states = []
        
        for state in states:
            normalized_state = state[:] + [0.0] * (max_dim - len(state))
            normalized_states.append(normalized_state)
        
        # Apply interference matrix
        interfered_states = []
        for i, wl in enumerate(wavelengths):
            interfered_state = [0.0] * max_dim
            
            for j, source_state in enumerate(normalized_states):
                interference_coeff = self.interference_matrix[i][j] if i < len(self.interference_matrix) and j < len(self.interference_matrix[i]) else 0.0
                
                for k in range(max_dim):
                    interfered_state[k] += interference_coeff * source_state[k]
            
            # Apply crosstalk suppression
            suppression = self.config.crosstalk_suppression
            for k in range(max_dim):
                if i < len(normalized_states):
                    interfered_state[k] = suppression * interfered_state[k] + (1 - suppression) * normalized_states[i][k]
            
            interfered_states.append(interfered_state)
        
        # Return interfered states
        result = {}
        for i, wl in enumerate(wavelengths):
            result[wl] = interfered_states[i]
        
        return result


class WDMQuantumMultimodalProcessor:
    """Main processor for WDM quantum multimodal processing."""
    
    def __init__(self, config: WDMQuantumConfig):
        self.config = config
        self.encoder = QuantumWavelengthEncoder(config)
        self.interferometer = PhotonicQuantumInterferometer(config)
        
        # Performance tracking
        self.metrics = {
            'encoding_time': 0.0,
            'interference_time': 0.0, 
            'fusion_time': 0.0,
            'total_processing_time': 0.0,
            'channel_utilization': {},
            'encoding_fidelities': {}
        }
    
    def process_multimodal_input(
        self, 
        multimodal_data: Dict[InputModality, List[float]]
    ) -> Dict[str, Any]:
        """Process multimodal input through WDM quantum pipeline."""
        
        total_start = time.time()
        
        # Stage 1: Encode each modality to wavelength channels
        encoding_start = time.time()
        wavelength_states = {}
        encoding_results = {}
        
        for modality, data in multimodal_data.items():
            if data:  # Only process non-empty data
                encoding_result = self.encoder.encode_modality_to_wavelength(modality, data)
                wavelength = encoding_result['wavelength']
                
                wavelength_states[wavelength] = encoding_result['quantum_state']
                encoding_results[modality.value] = encoding_result
                
                # Track metrics
                self.metrics['channel_utilization'][f"λ{wavelength:.1f}"] = encoding_result['channel_utilization']
                self.metrics['encoding_fidelities'][modality.value] = encoding_result['encoding_fidelity']
        
        self.metrics['encoding_time'] = time.time() - encoding_start
        
        # Stage 2: Apply quantum interference between channels
        interference_start = time.time()
        interfered_states = self.interferometer.apply_quantum_interference(wavelength_states)
        self.metrics['interference_time'] = time.time() - interference_start
        
        # Stage 3: Spectral fusion and classification
        fusion_start = time.time()
        fused_result = self._apply_spectral_fusion(interfered_states, encoding_results)
        self.metrics['fusion_time'] = time.time() - fusion_start
        
        self.metrics['total_processing_time'] = time.time() - total_start
        
        return {
            'wavelength_states': wavelength_states,
            'interfered_states': interfered_states,
            'encoding_results': encoding_results,
            'fused_output': fused_result,
            'wavelength_allocation': self._get_wavelength_allocation_summary(),
            'performance_metrics': self.get_performance_metrics()
        }
    
    def _apply_spectral_fusion(
        self, 
        interfered_states: Dict[float, List[float]],
        encoding_results: Dict[str, Any]
    ) -> List[float]:
        """Apply spectral fusion to combine wavelength channels."""
        
        if self.config.fusion_strategy == "spectral_fusion":
            return self._spectral_weighted_fusion(interfered_states, encoding_results)
        elif self.config.fusion_strategy == "max_pooling":
            return self._spectral_max_pooling(interfered_states)
        elif self.config.fusion_strategy == "attention_fusion":
            return self._spectral_attention_fusion(interfered_states, encoding_results)
        else:
            return self._simple_spectral_fusion(interfered_states)
    
    def _spectral_weighted_fusion(
        self, 
        interfered_states: Dict[float, List[float]],
        encoding_results: Dict[str, Any]
    ) -> List[float]:
        """Weighted fusion based on encoding fidelity and channel utilization."""
        
        if not interfered_states:
            return [0.0] * self.config.output_classes
        
        # Determine output dimension
        max_dim = max(len(state) for state in interfered_states.values())
        output_dim = min(max_dim, self.config.output_classes)
        
        weighted_sum = [0.0] * output_dim
        total_weight = 0.0
        
        for wavelength, state in interfered_states.items():
            # Compute weight based on encoding quality
            weight = 1.0
            
            # Find corresponding encoding result
            for modality, result in encoding_results.items():
                if result['wavelength'] == wavelength:
                    fidelity = result['encoding_fidelity']
                    utilization = result['channel_utilization']
                    weight = fidelity * utilization
                    break
            
            # Accumulate weighted sum
            for i in range(output_dim):
                if i < len(state):
                    weighted_sum[i] += weight * state[i]
            
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            weighted_sum = [x / total_weight for x in weighted_sum]
        
        return weighted_sum
    
    def _spectral_max_pooling(self, interfered_states: Dict[float, List[float]]) -> List[float]:
        """Max pooling across spectral channels."""
        if not interfered_states:
            return [0.0] * self.config.output_classes
        
        max_dim = max(len(state) for state in interfered_states.values())
        output_dim = min(max_dim, self.config.output_classes)
        
        max_values = [-float('inf')] * output_dim
        
        for state in interfered_states.values():
            for i in range(min(len(state), output_dim)):
                max_values[i] = max(max_values[i], state[i])
        
        # Handle -inf values
        max_values = [max(0.0, x) if x != -float('inf') else 0.0 for x in max_values]
        
        return max_values
    
    def _spectral_attention_fusion(
        self, 
        interfered_states: Dict[float, List[float]],
        encoding_results: Dict[str, Any]
    ) -> List[float]:
        """Attention-based spectral fusion."""
        if not interfered_states:
            return [0.0] * self.config.output_classes
        
        wavelengths = list(interfered_states.keys())
        states = list(interfered_states.values())
        
        if not states:
            return [0.0] * self.config.output_classes
        
        max_dim = max(len(state) for state in states)
        output_dim = min(max_dim, self.config.output_classes)
        
        # Compute attention weights
        attention_weights = []
        for i, wavelength in enumerate(wavelengths):
            # Attention based on state energy and encoding quality
            state_energy = sum(x ** 2 for x in states[i])
            
            encoding_quality = 1.0
            for result in encoding_results.values():
                if result['wavelength'] == wavelength:
                    encoding_quality = result['encoding_fidelity']
                    break
            
            attention = state_energy * encoding_quality
            attention_weights.append(attention)
        
        # Normalize attention weights
        total_attention = sum(attention_weights)
        if total_attention > 0:
            attention_weights = [w / total_attention for w in attention_weights]
        else:
            attention_weights = [1.0 / len(attention_weights)] * len(attention_weights)
        
        # Apply attention-weighted fusion
        fused_output = [0.0] * output_dim
        for i, (state, weight) in enumerate(zip(states, attention_weights)):
            for j in range(min(len(state), output_dim)):
                fused_output[j] += weight * state[j]
        
        return fused_output
    
    def _simple_spectral_fusion(self, interfered_states: Dict[float, List[float]]) -> List[float]:
        """Simple averaging fusion."""
        if not interfered_states:
            return [0.0] * self.config.output_classes
        
        states = list(interfered_states.values())
        max_dim = max(len(state) for state in states)
        output_dim = min(max_dim, self.config.output_classes)
        
        averaged_output = [0.0] * output_dim
        
        for state in states:
            for i in range(min(len(state), output_dim)):
                averaged_output[i] += state[i]
        
        # Average
        num_states = len(states)
        if num_states > 0:
            averaged_output = [x / num_states for x in averaged_output]
        
        return averaged_output
    
    def _get_wavelength_allocation_summary(self) -> Dict[str, Any]:
        """Get summary of wavelength channel allocation."""
        return {
            'total_channels': self.config.num_channels,
            'base_wavelength': self.config.base_wavelength,
            'channel_spacing': self.config.channel_spacing,
            'allocation_strategy': self.config.wavelength_allocation.value,
            'channel_utilization': self.metrics['channel_utilization'],
            'active_channels': len(self.metrics['channel_utilization'])
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            'timing': {
                'encoding_time': self.metrics['encoding_time'],
                'interference_time': self.metrics['interference_time'],
                'fusion_time': self.metrics['fusion_time'],
                'total_time': self.metrics['total_processing_time']
            },
            'quality': {
                'encoding_fidelities': self.metrics['encoding_fidelities'],
                'average_fidelity': sum(self.metrics['encoding_fidelities'].values()) / len(self.metrics['encoding_fidelities']) if self.metrics['encoding_fidelities'] else 0.0
            },
            'utilization': {
                'channel_utilization': self.metrics['channel_utilization'],
                'average_utilization': sum(self.metrics['channel_utilization'].values()) / len(self.metrics['channel_utilization']) if self.metrics['channel_utilization'] else 0.0
            }
        }


def create_wdm_processor(
    num_channels: int = 8,
    qubits_per_channel: int = 4,
    wavelength_allocation: str = "dynamic",
    fusion_strategy: str = "spectral_fusion"
) -> WDMQuantumMultimodalProcessor:
    """Create a configured WDM quantum multimodal processor."""
    
    config = WDMQuantumConfig(
        num_channels=num_channels,
        qubits_per_channel=qubits_per_channel,
        wavelength_allocation=WavelengthAllocation(wavelength_allocation),
        fusion_strategy=fusion_strategy
    )
    
    return WDMQuantumMultimodalProcessor(config)


def demo_wdm_processing():
    """Demonstrate WDM quantum multimodal processing."""
    print("🌈 WDM Quantum Multimodal Processing Demo")
    print("=" * 60)
    
    # Create WDM processor
    processor = create_wdm_processor(
        num_channels=6,
        qubits_per_channel=3,
        wavelength_allocation="dynamic",
        fusion_strategy="spectral_fusion"
    )
    
    # Create multimodal test data
    multimodal_data = {
        InputModality.TEXT: [0.8, -0.3, 0.5, 0.2, -0.1, 0.7, 0.4, -0.2],
        InputModality.AUDIO: [0.1, 0.9, -0.4, 0.6, 0.3, -0.5],
        InputModality.VISUAL: [-0.2, 0.4, 0.8, -0.1, 0.6, 0.2, -0.3, 0.9, 0.1],
        InputModality.TEMPORAL: [0.5, -0.8, 0.2, 0.7, -0.3]
    }
    
    print("🔬 Processing multimodal input:")
    for modality, data in multimodal_data.items():
        print(f"  {modality.value}: {len(data)} features")
    
    # Process through WDM system
    results = processor.process_multimodal_input(multimodal_data)
    
    # Display results
    print(f"\n✅ Processing Complete!")
    print(f"📡 Active wavelength channels: {results['wavelength_allocation']['active_channels']}")
    print(f"🎯 Fused output: {results['fused_output']}")
    
    # Performance metrics
    metrics = results['performance_metrics']
    
    print(f"\n📊 Performance Metrics:")
    print(f"  Encoding time: {metrics['timing']['encoding_time']:.4f}s")
    print(f"  Interference time: {metrics['timing']['interference_time']:.4f}s")
    print(f"  Fusion time: {metrics['timing']['fusion_time']:.4f}s")
    print(f"  Total time: {metrics['timing']['total_time']:.4f}s")
    
    print(f"\n🎛️ Channel Utilization:")
    for channel, utilization in metrics['utilization']['channel_utilization'].items():
        print(f"  {channel}: {utilization:.3f}")
    
    print(f"\n🔍 Encoding Fidelities:")
    for modality, fidelity in metrics['quality']['encoding_fidelities'].items():
        print(f"  {modality}: {fidelity:.3f}")
    
    # Wavelength allocation summary
    allocation = results['wavelength_allocation']
    print(f"\n🌈 Wavelength Allocation:")
    print(f"  Strategy: {allocation['allocation_strategy']}")
    print(f"  Base wavelength: {allocation['base_wavelength']}nm")
    print(f"  Channel spacing: {allocation['channel_spacing']}nm")
    print(f"  Utilization efficiency: {metrics['utilization']['average_utilization']:.3f}")
    
    # Sentiment prediction
    fused_output = results['fused_output']
    if fused_output and len(fused_output) >= 3:
        # Apply softmax for probabilities
        max_val = max(fused_output)
        exp_vals = [math.exp(x - max_val) for x in fused_output]
        sum_exp = sum(exp_vals)
        probabilities = [x / sum_exp for x in exp_vals]
        
        predicted_class = probabilities.index(max(probabilities))
        confidence = probabilities[predicted_class]
        
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        print(f"\n🎯 Multimodal Sentiment Prediction: {sentiment_labels[predicted_class]} ({confidence:.3f})")
        
        print(f"📊 Class Probabilities:")
        for i, (label, prob) in enumerate(zip(sentiment_labels, probabilities)):
            print(f"  {label}: {prob:.3f}")
    
    return processor, results


if __name__ == "__main__":
    demo_wdm_processing()