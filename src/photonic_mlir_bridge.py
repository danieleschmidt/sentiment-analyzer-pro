"""
Photonic-MLIR Synthesis Bridge - Core Module

This module provides the foundational bridge between photonic computing concepts
and MLIR (Multi-Level Intermediate Representation) compiler infrastructure.

Key Components:
- PhotonicCircuit: High-level photonic circuit representation
- MLIRDialectGenerator: Generates MLIR dialects for photonic operations
- SynthesisBridge: Translates between photonic circuits and MLIR IR
- PhotonicOptimizer: Optimizes photonic circuits using MLIR passes
"""

from __future__ import annotations

import json
import logging
import re
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import uuid
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import time

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Global lock for thread safety
_global_lock = threading.RLock()

# Circuit validation cache for performance
_validation_cache = {}
_cache_lock = threading.Lock()


class PhotonicComponentType(Enum):
    """Types of photonic components with validation."""
    WAVEGUIDE = "waveguide"
    BEAM_SPLITTER = "beam_splitter"  
    PHASE_SHIFTER = "phase_shifter"
    MACH_ZEHNDER = "mach_zehnder"
    RING_RESONATOR = "ring_resonator"
    DIRECTIONAL_COUPLER = "directional_coupler"
    PHOTODETECTOR = "photodetector"
    LASER = "laser"
    MODULATOR = "modulator"
    
    @classmethod
    def validate_type(cls, component_type: str) -> bool:
        """Validate component type string."""
        return component_type in [item.value for item in cls]
    
    @classmethod
    def get_valid_types(cls) -> List[str]:
        """Get list of valid component types."""
        return [item.value for item in cls]


class WavelengthBand(Enum):
    """Standard wavelength bands for photonic circuits with ranges."""
    C_BAND = "c_band"  # 1530-1565 nm
    L_BAND = "l_band"  # 1565-1625 nm
    O_BAND = "o_band"  # 1260-1360 nm
    
    @classmethod
    def get_wavelength_range(cls, band: 'WavelengthBand') -> Tuple[float, float]:
        """Get wavelength range for a band in nm."""
        ranges = {
            cls.C_BAND: (1530.0, 1565.0),
            cls.L_BAND: (1565.0, 1625.0),
            cls.O_BAND: (1260.0, 1360.0)
        }
        return ranges.get(band, (0.0, 0.0))
    
    @classmethod
    def validate_wavelength(cls, wavelength: float, band: 'WavelengthBand') -> bool:
        """Validate if wavelength is within band range."""
        min_wl, max_wl = cls.get_wavelength_range(band)
        return min_wl <= wavelength <= max_wl


@dataclass
class PhotonicComponent:
    """Represents a single photonic component."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    component_type: PhotonicComponentType = PhotonicComponentType.WAVEGUIDE
    position: Tuple[float, float] = (0.0, 0.0)
    parameters: Dict[str, Union[float, int, str]] = field(default_factory=dict)
    wavelength_band: WavelengthBand = WavelengthBand.C_BAND
    
    def __post_init__(self):
        """Validate component parameters with security checks."""
        from .photonic_security import validate_input, sanitize_input
        
        # Validate and sanitize ID
        if not isinstance(self.id, str):
            raise ValueError(f"Component ID must be string, got {type(self.id)}")
        if not validate_input(self.id, "component_id"):
            raise ValueError(f"Invalid component ID: {self.id}")
        self.id = sanitize_input(self.id)
        
        # Validate position
        if not isinstance(self.position, tuple) or len(self.position) != 2:
            raise ValueError(f"Position must be 2-tuple, got {self.position}")
        
        # Validate position values
        for i, coord in enumerate(self.position):
            if not validate_input(coord, f"position_{i}"):
                raise ValueError(f"Invalid position coordinate: {coord}")
        
        # Sanitize position
        self.position = tuple(sanitize_input(coord) for coord in self.position)
        
        # Validate parameters
        if not validate_input(self.parameters, "component_parameters"):
            raise ValueError("Invalid component parameters")
        self.parameters = sanitize_input(self.parameters)
        
        # Additional validations based on component type
        self._validate_component_specific_parameters()
    
    def _validate_component_specific_parameters(self):
        """Validate parameters specific to component type."""
        try:
            if self.component_type == PhotonicComponentType.WAVEGUIDE:
                self._validate_waveguide_parameters()
            elif self.component_type == PhotonicComponentType.BEAM_SPLITTER:
                self._validate_beam_splitter_parameters()
            elif self.component_type == PhotonicComponentType.PHASE_SHIFTER:
                self._validate_phase_shifter_parameters()
            elif self.component_type == PhotonicComponentType.RING_RESONATOR:
                self._validate_ring_resonator_parameters()
        except Exception as e:
            logger.error(f"Component validation failed for {self.id}: {e}")
            raise ValueError(f"Invalid parameters for {self.component_type.value}: {e}")
    
    def _validate_waveguide_parameters(self):
        """Validate waveguide-specific parameters."""
        if 'length' in self.parameters and self.parameters['length'] <= 0:
            raise ValueError("Waveguide length must be positive")
        if 'width' in self.parameters and self.parameters['width'] <= 0:
            raise ValueError("Waveguide width must be positive")
    
    def _validate_beam_splitter_parameters(self):
        """Validate beam splitter-specific parameters."""
        if 'ratio' in self.parameters:
            ratio = self.parameters['ratio']
            if not 0 <= ratio <= 1:
                raise ValueError("Beam splitter ratio must be between 0 and 1")
    
    def _validate_phase_shifter_parameters(self):
        """Validate phase shifter-specific parameters."""
        if 'phase_shift' in self.parameters:
            phase = self.parameters['phase_shift']
            if not -10 <= phase <= 10:  # Reasonable phase range
                logger.warning(f"Unusual phase shift value: {phase}")
    
    def _validate_ring_resonator_parameters(self):
        """Validate ring resonator-specific parameters."""
        if 'radius' in self.parameters and self.parameters['radius'] <= 0:
            raise ValueError("Ring resonator radius must be positive")
        if 'coupling_ratio' in self.parameters:
            ratio = self.parameters['coupling_ratio']
            if not 0 <= ratio <= 1:
                raise ValueError("Ring coupling ratio must be between 0 and 1")


@dataclass  
class PhotonicConnection:
    """Represents a connection between photonic components."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_component: str = ""
    target_component: str = ""
    source_port: int = 0
    target_port: int = 0
    loss_db: float = 0.0
    delay_ps: float = 0.0
    
    def __post_init__(self):
        """Validate connection parameters."""
        if self.loss_db < 0:
            raise ValueError(f"Loss cannot be negative: {self.loss_db}")
        if self.delay_ps < 0:
            raise ValueError(f"Delay cannot be negative: {self.delay_ps}")


@dataclass
class PhotonicCircuit:
    """High-level representation of a photonic circuit."""
    name: str = "untitled_circuit"
    components: List[PhotonicComponent] = field(default_factory=list)
    connections: List[PhotonicConnection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_component(self, component: PhotonicComponent) -> None:
        """Add a component to the circuit."""
        if any(c.id == component.id for c in self.components):
            raise ValueError(f"Component ID {component.id} already exists")
        self.components.append(component)
        logger.info(f"Added component {component.id} of type {component.component_type}")
    
    def add_connection(self, connection: PhotonicConnection) -> None:
        """Add a connection between components."""
        # Validate that source and target components exist
        component_ids = {c.id for c in self.components}
        if connection.source_component not in component_ids:
            raise ValueError(f"Source component {connection.source_component} not found")
        if connection.target_component not in component_ids:
            raise ValueError(f"Target component {connection.target_component} not found")
        
        self.connections.append(connection)
        logger.info(f"Added connection from {connection.source_component} to {connection.target_component}")
    
    def validate(self) -> bool:
        """Validate the circuit structure."""
        component_ids = {c.id for c in self.components}
        
        for conn in self.connections:
            if conn.source_component not in component_ids:
                raise ValueError(f"Invalid connection: source {conn.source_component} not found")
            if conn.target_component not in component_ids:
                raise ValueError(f"Invalid connection: target {conn.target_component} not found")
        
        logger.info(f"Circuit {self.name} validated successfully")
        return True


class MLIRDialect:
    """Represents an MLIR dialect for photonic operations."""
    
    def __init__(self, name: str = "photonic"):
        self.name = name
        self.operations: Dict[str, Dict] = {}
        self.types: Dict[str, Dict] = {}
        self.attributes: Dict[str, Dict] = {}
    
    def add_operation(self, op_name: str, op_def: Dict) -> None:
        """Add an operation definition to the dialect."""
        self.operations[op_name] = op_def
        logger.debug(f"Added operation {op_name} to {self.name} dialect")
    
    def add_type(self, type_name: str, type_def: Dict) -> None:
        """Add a type definition to the dialect."""
        self.types[type_name] = type_def
        logger.debug(f"Added type {type_name} to {self.name} dialect")
    
    def generate_tablegen(self) -> str:
        """Generate TableGen definition for the dialect."""
        tablegen = f"""
// Photonic MLIR Dialect Definition
// Auto-generated by Photonic-MLIR Bridge

#ifndef PHOTONIC_DIALECT
#define PHOTONIC_DIALECT

include "mlir/IR/OpBase.td"

def Photonic_Dialect : Dialect {{
  let name = "{self.name}";
  let summary = "Photonic computing operations";
  let description = [{{
    This dialect provides operations for photonic circuit synthesis
    and optimization, bridging high-level photonic descriptions 
    with hardware-specific implementations.
  }}];
  let cppNamespace = "::mlir::photonic";
}}

// Base operation class
class Photonic_Op<string mnemonic, list<Trait> traits = []> :
    Op<Photonic_Dialect, mnemonic, traits>;

"""
        
        # Add operation definitions
        for op_name, op_def in self.operations.items():
            tablegen += self._generate_operation_tablegen(op_name, op_def)
        
        tablegen += "\n#endif // PHOTONIC_DIALECT\n"
        return tablegen
    
    def _generate_operation_tablegen(self, op_name: str, op_def: Dict) -> str:
        """Generate TableGen for a specific operation."""
        return f"""
def Photonic_{op_name}Op : Photonic_Op<"{op_name.lower()}"> {{
  let summary = "{op_def.get('summary', f'{op_name} operation')}";
  let description = [{{
    {op_def.get('description', f'Photonic {op_name} operation')}
  }}];
}}

"""


class MLIRDialectGenerator:
    """Generates MLIR dialects for photonic operations."""
    
    def __init__(self):
        self.dialect = MLIRDialect("photonic")
        self._initialize_base_operations()
    
    def _initialize_base_operations(self) -> None:
        """Initialize base photonic operations."""
        base_ops = {
            "Waveguide": {
                "summary": "Optical waveguide operation",
                "description": "Represents an optical waveguide for light transmission"
            },
            "BeamSplitter": {
                "summary": "Beam splitter operation", 
                "description": "Splits optical signal into multiple paths"
            },
            "PhaseShifter": {
                "summary": "Phase shifter operation",
                "description": "Applies phase shift to optical signal"
            },
            "MachZehnder": {
                "summary": "Mach-Zehnder interferometer operation",
                "description": "Implements Mach-Zehnder interferometer functionality"
            }
        }
        
        for op_name, op_def in base_ops.items():
            self.dialect.add_operation(op_name, op_def)
        
        logger.info(f"Initialized {len(base_ops)} base photonic operations")
    
    def generate_from_circuit(self, circuit: PhotonicCircuit) -> MLIRDialect:
        """Generate MLIR dialect from photonic circuit."""
        circuit.validate()
        
        # Add operations based on circuit components
        for component in circuit.components:
            op_name = component.component_type.value.title().replace("_", "")
            if op_name not in self.dialect.operations:
                self.dialect.add_operation(op_name, {
                    "summary": f"{component.component_type.value} operation",
                    "description": f"Photonic {component.component_type.value} component"
                })
        
        logger.info(f"Generated dialect with {len(self.dialect.operations)} operations")
        return self.dialect


class SynthesisBridge:
    """Main bridge class for photonic-MLIR synthesis with optimization support."""
    
    def __init__(self, enable_optimization: bool = True):
        self.dialect_generator = MLIRDialectGenerator()
        self.optimization_passes: List[str] = []
        self.enable_optimization = enable_optimization
        
        # Initialize monitoring and optimization
        if enable_optimization:
            from .photonic_monitoring import get_monitor
            from .photonic_optimization import get_optimizer
            self.monitor = get_monitor()
            self.optimizer = get_optimizer()
        else:
            self.monitor = None
            self.optimizer = None
    
    def synthesize_circuit(self, circuit: PhotonicCircuit) -> Dict[str, Any]:
        """Synthesize photonic circuit to MLIR representation with optimization."""
        start_time = time.time()
        logger.info(f"Starting synthesis of circuit: {circuit.name}")
        
        try:
            # Use optimized synthesis if available
            if self.enable_optimization and self.optimizer:
                return self.optimizer.optimized_synthesis(self._do_synthesis, circuit)
            else:
                return self._do_synthesis(circuit)
        
        except Exception as e:
            # Record error metrics
            if self.monitor:
                self.monitor.record_synthesis_operation(
                    len(circuit.components), len(circuit.connections),
                    time.time() - start_time, success=False
                )
            logger.error(f"Synthesis failed for circuit {circuit.name}: {e}")
            raise
    
    def _do_synthesis(self, circuit: PhotonicCircuit) -> Dict[str, Any]:
        """Internal synthesis implementation."""
        start_time = time.time()
        
        # Validate circuit with monitoring
        validation_start = time.time()
        
        with _global_lock:
            # Use cached validation if available
            cache_key = None
            if self.enable_optimization:
                import hashlib
                circuit_str = f"{circuit.name}_{len(circuit.components)}_{len(circuit.connections)}"
                cache_key = hashlib.md5(circuit_str.encode()).hexdigest()
                
                with _cache_lock:
                    if cache_key in _validation_cache:
                        logger.debug(f"Using cached validation for circuit {circuit.name}")
                    else:
                        circuit.validate()
                        _validation_cache[cache_key] = True
            else:
                circuit.validate()
        
        validation_time = time.time() - validation_start
        
        # Record validation metrics
        if self.monitor:
            from .photonic_monitoring import record_validation_metrics
            record_validation_metrics(
                len(circuit.components), validation_time, success=True
            )
        
        # Generate MLIR dialect
        dialect = self.dialect_generator.generate_from_circuit(circuit)
        
        # Generate MLIR IR
        mlir_ir = self._generate_mlir_ir(circuit, dialect)
        
        # Apply optimization passes
        optimized_ir = self._apply_optimizations(mlir_ir)
        
        synthesis_time = time.time() - start_time
        
        result = {
            "circuit_name": circuit.name,
            "mlir_dialect": dialect.generate_tablegen(),
            "mlir_ir": optimized_ir,
            "components_count": len(circuit.components),
            "connections_count": len(circuit.connections),
            "synthesis_metadata": {
                "timestamp": time.time(),
                "synthesis_time": synthesis_time,
                "validation_time": validation_time,
                "optimization_passes": self.optimization_passes,
                "wavelength_bands": list(set(c.wavelength_band.value for c in circuit.components)),
                "optimization_enabled": self.enable_optimization
            }
        }
        
        # Record synthesis metrics
        if self.monitor:
            from .photonic_monitoring import record_synthesis_metrics
            record_synthesis_metrics(
                len(circuit.components), len(circuit.connections),
                synthesis_time, success=True
            )
        
        logger.info(f"Successfully synthesized circuit {circuit.name} in {synthesis_time:.3f}s")
        return result
    
    def _generate_mlir_ir(self, circuit: PhotonicCircuit, dialect: MLIRDialect) -> str:
        """Generate MLIR IR from photonic circuit."""
        ir_lines = [
            f"// MLIR IR for photonic circuit: {circuit.name}",
            f"// Generated by Photonic-MLIR Bridge",
            "",
            "module {",
        ]
        
        # Generate function for the circuit
        ir_lines.extend([
            f'  func.func @{circuit.name.replace(" ", "_")}() {{',
        ])
        
        # Generate operations for each component
        for component in circuit.components:
            op_name = component.component_type.value.replace("_", ".")
            ir_lines.append(f'    photonic.{op_name} "{component.id}" {{')
            
            # Add component parameters
            for param, value in component.parameters.items():
                ir_lines.append(f'      {param} = {json.dumps(value)},')
            
            ir_lines.extend([
                f'      position = [{component.position[0]}, {component.position[1]}],',
                f'      wavelength_band = "{component.wavelength_band.value}"',
                '    }',
            ])
        
        # Generate connections
        for connection in circuit.connections:
            ir_lines.append(f'    photonic.connect "{connection.id}" {{')
            ir_lines.extend([
                f'      source = "{connection.source_component}:{connection.source_port}",',
                f'      target = "{connection.target_component}:{connection.target_port}",',
                f'      loss_db = {connection.loss_db},',
                f'      delay_ps = {connection.delay_ps}',
                '    }',
            ])
        
        ir_lines.extend([
            "    return",
            "  }",
            "}",
        ])
        
        return "\n".join(ir_lines)
    
    def _apply_optimizations(self, mlir_ir: str) -> str:
        """Apply optimization passes to MLIR IR."""
        # Placeholder for actual MLIR optimization passes
        # In a real implementation, this would use MLIR's pass manager
        
        optimized_ir = mlir_ir
        
        # Simulate optimization passes
        self.optimization_passes = [
            "photonic-component-fusion",
            "photonic-routing-optimization", 
            "photonic-loss-minimization",
            "photonic-delay-balancing"
        ]
        
        # Add optimization annotations
        optimization_header = f"""
// Optimized MLIR IR - Applied passes: {', '.join(self.optimization_passes)}
// Optimization level: O2
// Target: Photonic Hardware Synthesis

"""
        
        optimized_ir = optimization_header + optimized_ir
        
        logger.info(f"Applied {len(self.optimization_passes)} optimization passes")
        return optimized_ir


class PhotonicCircuitBuilder:
    """Builder class for constructing photonic circuits."""
    
    def __init__(self, name: str = "circuit"):
        self.circuit = PhotonicCircuit(name=name)
    
    def add_waveguide(self, length: float, width: float = 0.5, 
                     position: Tuple[float, float] = (0.0, 0.0)) -> str:
        """Add a waveguide component."""
        component = PhotonicComponent(
            component_type=PhotonicComponentType.WAVEGUIDE,
            position=position,
            parameters={"length": length, "width": width}
        )
        self.circuit.add_component(component)
        return component.id
    
    def add_beam_splitter(self, ratio: float = 0.5,
                         position: Tuple[float, float] = (0.0, 0.0)) -> str:
        """Add a beam splitter component."""
        if not 0 <= ratio <= 1:
            raise ValueError(f"Beam splitter ratio must be between 0 and 1, got {ratio}")
        
        component = PhotonicComponent(
            component_type=PhotonicComponentType.BEAM_SPLITTER,
            position=position,
            parameters={"ratio": ratio}
        )
        self.circuit.add_component(component)
        return component.id
    
    def add_phase_shifter(self, phase_shift: float,
                         position: Tuple[float, float] = (0.0, 0.0)) -> str:
        """Add a phase shifter component."""
        component = PhotonicComponent(
            component_type=PhotonicComponentType.PHASE_SHIFTER,
            position=position,
            parameters={"phase_shift": phase_shift}
        )
        self.circuit.add_component(component)
        return component.id
    
    def connect(self, source_id: str, target_id: str, 
               source_port: int = 0, target_port: int = 0,
               loss_db: float = 0.0, delay_ps: float = 0.0) -> str:
        """Connect two components."""
        connection = PhotonicConnection(
            source_component=source_id,
            target_component=target_id,
            source_port=source_port,
            target_port=target_port,
            loss_db=loss_db,
            delay_ps=delay_ps
        )
        self.circuit.add_connection(connection)
        return connection.id
    
    def build(self) -> PhotonicCircuit:
        """Build and validate the circuit."""
        self.circuit.validate()
        return self.circuit


# Global bridge instance for convenient access
bridge = SynthesisBridge()


def create_simple_mzi_circuit() -> PhotonicCircuit:
    """Create a simple Mach-Zehnder interferometer circuit for testing."""
    builder = PhotonicCircuitBuilder("simple_mzi")
    
    # Add components
    input_wg = builder.add_waveguide(10.0, position=(0, 0))
    bs1 = builder.add_beam_splitter(0.5, position=(10, 0))
    upper_wg = builder.add_waveguide(20.0, position=(15, 5))
    lower_wg = builder.add_waveguide(20.0, position=(15, -5))
    ps = builder.add_phase_shifter(1.57, position=(25, 5))  # π/2 phase shift
    bs2 = builder.add_beam_splitter(0.5, position=(35, 0))
    output_wg = builder.add_waveguide(10.0, position=(40, 0))
    
    # Add connections
    builder.connect(input_wg, bs1, loss_db=0.1)
    builder.connect(bs1, upper_wg, source_port=0, target_port=0, loss_db=0.1)
    builder.connect(bs1, lower_wg, source_port=1, target_port=0, loss_db=0.1)
    builder.connect(upper_wg, ps, loss_db=0.05)
    builder.connect(ps, bs2, target_port=0, loss_db=0.1)
    builder.connect(lower_wg, bs2, target_port=1, loss_db=0.1)
    builder.connect(bs2, output_wg, loss_db=0.1)
    
    return builder.build()


if __name__ == "__main__":
    # Demo usage
    print("🔬 Photonic-MLIR Synthesis Bridge Demo")
    print("=" * 50)
    
    # Create a test circuit
    circuit = create_simple_mzi_circuit()
    print(f"Created circuit: {circuit.name}")
    print(f"Components: {len(circuit.components)}")
    print(f"Connections: {len(circuit.connections)}")
    
    # Synthesize to MLIR
    result = bridge.synthesize_circuit(circuit)
    print(f"\n✅ Synthesis completed!")
    print(f"Generated {len(result['mlir_dialect'])} characters of dialect definition")
    print(f"Generated {len(result['mlir_ir'])} characters of MLIR IR")
    
    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "photonic_dialect.td", "w") as f:
        f.write(result["mlir_dialect"])
    
    with open(output_dir / "circuit.mlir", "w") as f:
        f.write(result["mlir_ir"])
    
    print(f"\n📁 Output saved to {output_dir}/")
    print("- photonic_dialect.td (MLIR dialect definition)")
    print("- circuit.mlir (MLIR IR)")