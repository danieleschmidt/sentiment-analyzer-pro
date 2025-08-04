"""
Tests for Photonic-MLIR Bridge functionality.

Comprehensive test suite covering all aspects of the photonic-MLIR synthesis bridge,
including component creation, circuit validation, MLIR generation, and optimization.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.photonic_mlir_bridge import (
    PhotonicComponent, PhotonicConnection, PhotonicCircuit,
    PhotonicComponentType, WavelengthBand, PhotonicCircuitBuilder,
    MLIRDialect, MLIRDialectGenerator, SynthesisBridge,
    create_simple_mzi_circuit
)


class TestPhotonicComponent:
    """Test PhotonicComponent class."""
    
    def test_component_creation(self):
        """Test basic component creation."""
        component = PhotonicComponent(
            component_type=PhotonicComponentType.WAVEGUIDE,
            position=(10.0, 5.0),
            parameters={"length": 20.0, "width": 0.5}
        )
        
        assert component.component_type == PhotonicComponentType.WAVEGUIDE
        assert component.position == (10.0, 5.0)
        assert component.parameters["length"] == 20.0
        assert component.wavelength_band == WavelengthBand.C_BAND
        assert isinstance(component.id, str)
        assert len(component.id) > 0
    
    def test_component_validation(self):
        """Test component parameter validation."""
        # Test invalid position
        with pytest.raises(ValueError, match="Position must be 2-tuple"):
            PhotonicComponent(position=(1.0,))  # Single element tuple
        
        with pytest.raises(ValueError, match="Position must be 2-tuple"):
            PhotonicComponent(position=(1.0, 2.0, 3.0))  # Three elements
    
    def test_component_default_values(self):
        """Test component default values."""
        component = PhotonicComponent()
        
        assert component.component_type == PhotonicComponentType.WAVEGUIDE
        assert component.position == (0.0, 0.0)
        assert component.parameters == {}
        assert component.wavelength_band == WavelengthBand.C_BAND


class TestPhotonicConnection:
    """Test PhotonicConnection class."""
    
    def test_connection_creation(self):
        """Test basic connection creation."""
        connection = PhotonicConnection(
            source_component="comp1",
            target_component="comp2",
            loss_db=0.5,
            delay_ps=1.2
        )
        
        assert connection.source_component == "comp1"
        assert connection.target_component == "comp2"
        assert connection.loss_db == 0.5
        assert connection.delay_ps == 1.2
        assert connection.source_port == 0
        assert connection.target_port == 0
    
    def test_connection_validation(self):
        """Test connection parameter validation."""
        # Test negative loss
        with pytest.raises(ValueError, match="Loss cannot be negative"):
            PhotonicConnection(loss_db=-1.0)
        
        # Test negative delay
        with pytest.raises(ValueError, match="Delay cannot be negative"):
            PhotonicConnection(delay_ps=-1.0)


class TestPhotonicCircuit:
    """Test PhotonicCircuit class."""
    
    def test_circuit_creation(self):
        """Test basic circuit creation."""
        circuit = PhotonicCircuit(name="test_circuit")
        
        assert circuit.name == "test_circuit"
        assert circuit.components == []
        assert circuit.connections == []
        assert circuit.metadata == {}
    
    def test_add_component(self):
        """Test adding components to circuit."""
        circuit = PhotonicCircuit()
        component = PhotonicComponent(component_type=PhotonicComponentType.BEAM_SPLITTER)
        
        circuit.add_component(component)
        
        assert len(circuit.components) == 1
        assert circuit.components[0] == component
    
    def test_add_duplicate_component(self):
        """Test adding duplicate component IDs."""
        circuit = PhotonicCircuit()
        component1 = PhotonicComponent(id="test_id")
        component2 = PhotonicComponent(id="test_id")
        
        circuit.add_component(component1)
        
        with pytest.raises(ValueError, match="Component ID test_id already exists"):
            circuit.add_component(component2)
    
    def test_add_connection(self):
        """Test adding connections to circuit."""
        circuit = PhotonicCircuit()
        comp1 = PhotonicComponent(id="comp1")
        comp2 = PhotonicComponent(id="comp2")
        
        circuit.add_component(comp1)
        circuit.add_component(comp2)
        
        connection = PhotonicConnection(source_component="comp1", target_component="comp2")
        circuit.add_connection(connection)
        
        assert len(circuit.connections) == 1
        assert circuit.connections[0] == connection
    
    def test_add_invalid_connection(self):
        """Test adding connection with non-existent components."""
        circuit = PhotonicCircuit()
        comp1 = PhotonicComponent(id="comp1")
        circuit.add_component(comp1)
        
        # Connection to non-existent component
        connection = PhotonicConnection(source_component="comp1", target_component="comp2")
        
        with pytest.raises(ValueError, match="Target component comp2 not found"):
            circuit.add_connection(connection)
    
    def test_circuit_validation(self):
        """Test circuit validation."""
        circuit = PhotonicCircuit()
        comp1 = PhotonicComponent(id="comp1")
        comp2 = PhotonicComponent(id="comp2")
        
        circuit.add_component(comp1)
        circuit.add_component(comp2)
        
        connection = PhotonicConnection(source_component="comp1", target_component="comp2")
        circuit.add_connection(connection)
        
        assert circuit.validate() is True
    
    def test_circuit_validation_invalid_connection(self):
        """Test circuit validation with invalid connections."""
        circuit = PhotonicCircuit()
        comp1 = PhotonicComponent(id="comp1")
        circuit.add_component(comp1)
        
        # Manually add invalid connection (bypassing add_connection validation)
        invalid_connection = PhotonicConnection(source_component="comp1", target_component="nonexistent")
        circuit.connections.append(invalid_connection)
        
        with pytest.raises(ValueError, match="Invalid connection: target nonexistent not found"):
            circuit.validate()


class TestPhotonicCircuitBuilder:
    """Test PhotonicCircuitBuilder class."""
    
    def test_builder_creation(self):
        """Test builder creation."""
        builder = PhotonicCircuitBuilder("test_circuit")
        assert builder.circuit.name == "test_circuit"
    
    def test_add_waveguide(self):
        """Test adding waveguide through builder."""
        builder = PhotonicCircuitBuilder()
        wg_id = builder.add_waveguide(length=10.0, width=0.5, position=(1.0, 2.0))
        
        assert len(builder.circuit.components) == 1
        component = builder.circuit.components[0]
        assert component.id == wg_id
        assert component.component_type == PhotonicComponentType.WAVEGUIDE
        assert component.parameters["length"] == 10.0
        assert component.parameters["width"] == 0.5
        assert component.position == (1.0, 2.0)
    
    def test_add_beam_splitter(self):
        """Test adding beam splitter through builder."""
        builder = PhotonicCircuitBuilder()
        bs_id = builder.add_beam_splitter(ratio=0.3, position=(5.0, 5.0))
        
        assert len(builder.circuit.components) == 1
        component = builder.circuit.components[0]
        assert component.id == bs_id
        assert component.component_type == PhotonicComponentType.BEAM_SPLITTER
        assert component.parameters["ratio"] == 0.3
    
    def test_beam_splitter_invalid_ratio(self):
        """Test beam splitter with invalid ratio.""" 
        builder = PhotonicCircuitBuilder()
        
        with pytest.raises(ValueError, match="Beam splitter ratio must be between 0 and 1"):
            builder.add_beam_splitter(ratio=1.5)
    
    def test_add_phase_shifter(self):
        """Test adding phase shifter through builder."""
        builder = PhotonicCircuitBuilder()
        ps_id = builder.add_phase_shifter(phase_shift=1.57, position=(3.0, 4.0))
        
        assert len(builder.circuit.components) == 1
        component = builder.circuit.components[0]
        assert component.id == ps_id
        assert component.component_type == PhotonicComponentType.PHASE_SHIFTER
        assert component.parameters["phase_shift"] == 1.57
    
    def test_connect_components(self):
        """Test connecting components through builder."""
        builder = PhotonicCircuitBuilder()
        comp1_id = builder.add_waveguide(10.0)
        comp2_id = builder.add_beam_splitter(0.5)
        
        conn_id = builder.connect(comp1_id, comp2_id, loss_db=0.1, delay_ps=0.5)
        
        assert len(builder.circuit.connections) == 1
        connection = builder.circuit.connections[0]
        assert connection.id == conn_id
        assert connection.source_component == comp1_id
        assert connection.target_component == comp2_id
        assert connection.loss_db == 0.1
        assert connection.delay_ps == 0.5
    
    def test_build_circuit(self):
        """Test building and validating circuit."""
        builder = PhotonicCircuitBuilder("complete_circuit")
        comp1_id = builder.add_waveguide(10.0)
        comp2_id = builder.add_beam_splitter(0.5)
        builder.connect(comp1_id, comp2_id)
        
        circuit = builder.build()
        
        assert circuit.name == "complete_circuit"
        assert len(circuit.components) == 2
        assert len(circuit.connections) == 1
        assert circuit.validate() is True


class TestMLIRDialect:
    """Test MLIRDialect class."""
    
    def test_dialect_creation(self):
        """Test MLIR dialect creation."""
        dialect = MLIRDialect("test_dialect")
        
        assert dialect.name == "test_dialect"
        assert dialect.operations == {}
        assert dialect.types == {}
        assert dialect.attributes == {}
    
    def test_add_operation(self):
        """Test adding operation to dialect."""
        dialect = MLIRDialect()
        op_def = {"summary": "Test operation", "description": "Test description"}
        
        dialect.add_operation("TestOp", op_def)
        
        assert "TestOp" in dialect.operations
        assert dialect.operations["TestOp"] == op_def
    
    def test_add_type(self):
        """Test adding type to dialect."""
        dialect = MLIRDialect()
        type_def = {"description": "Test type"}
        
        dialect.add_type("TestType", type_def)
        
        assert "TestType" in dialect.types
        assert dialect.types["TestType"] == type_def
    
    def test_generate_tablegen(self):
        """Test TableGen generation."""
        dialect = MLIRDialect("photonic")
        dialect.add_operation("Waveguide", {"summary": "Waveguide op", "description": "A waveguide"})
        
        tablegen = dialect.generate_tablegen()
        
        assert "Photonic_Dialect" in tablegen
        assert "photonic" in tablegen
        assert "Photonic_WaveguideOp" in tablegen
        assert "waveguide" in tablegen
        assert "#ifndef PHOTONIC_DIALECT" in tablegen
        assert "#endif // PHOTONIC_DIALECT" in tablegen


class TestMLIRDialectGenerator:
    """Test MLIRDialectGenerator class."""
    
    def test_generator_initialization(self):
        """Test dialect generator initialization."""
        generator = MLIRDialectGenerator()
        
        assert generator.dialect.name == "photonic"
        assert len(generator.dialect.operations) > 0
        assert "Waveguide" in generator.dialect.operations
        assert "BeamSplitter" in generator.dialect.operations
    
    def test_generate_from_circuit(self):
        """Test generating dialect from circuit."""
        generator = MLIRDialectGenerator()
        circuit = create_simple_mzi_circuit()
        
        dialect = generator.generate_from_circuit(circuit)
        
        assert isinstance(dialect, MLIRDialect)
        assert len(dialect.operations) >= 3  # At least waveguide, beam_splitter, phase_shifter


class TestSynthesisBridge:
    """Test SynthesisBridge class."""
    
    def test_bridge_creation(self):
        """Test synthesis bridge creation."""
        bridge = SynthesisBridge()
        
        assert isinstance(bridge.dialect_generator, MLIRDialectGenerator)
        assert bridge.optimization_passes == []
    
    def test_synthesize_circuit(self):
        """Test circuit synthesis."""
        bridge = SynthesisBridge()
        circuit = create_simple_mzi_circuit()
        
        result = bridge.synthesize_circuit(circuit)
        
        assert isinstance(result, dict)
        assert "circuit_name" in result
        assert "mlir_dialect" in result
        assert "mlir_ir" in result
        assert "components_count" in result
        assert "connections_count" in result
        assert "synthesis_metadata" in result
        
        assert result["circuit_name"] == circuit.name
        assert result["components_count"] == len(circuit.components)
        assert result["connections_count"] == len(circuit.connections)
        assert len(result["mlir_dialect"]) > 0
        assert len(result["mlir_ir"]) > 0
    
    def test_mlir_ir_generation(self):
        """Test MLIR IR content."""
        bridge = SynthesisBridge()
        circuit = create_simple_mzi_circuit()
        
        result = bridge.synthesize_circuit(circuit)
        mlir_ir = result["mlir_ir"]
        
        # Check for expected MLIR structure
        assert "module {" in mlir_ir
        assert "func.func @" in mlir_ir
        assert "photonic." in mlir_ir
        assert "return" in mlir_ir
        
        # Check for optimization annotations
        assert "Optimized MLIR IR" in mlir_ir
        assert "photonic-component-fusion" in mlir_ir
    
    def test_optimization_passes(self):
        """Test optimization pass application."""
        bridge = SynthesisBridge()
        circuit = create_simple_mzi_circuit()
        
        result = bridge.synthesize_circuit(circuit)
        
        expected_passes = [
            "photonic-component-fusion",
            "photonic-routing-optimization",
            "photonic-loss-minimization", 
            "photonic-delay-balancing"
        ]
        
        assert len(bridge.optimization_passes) == len(expected_passes)
        for expected_pass in expected_passes:
            assert expected_pass in bridge.optimization_passes
        
        metadata = result["synthesis_metadata"]
        assert metadata["optimization_passes"] == bridge.optimization_passes


class TestCreateSimpleMZICircuit:
    """Test the demo MZI circuit creation."""
    
    def test_mzi_circuit_creation(self):
        """Test MZI circuit creation."""
        circuit = create_simple_mzi_circuit()
        
        assert circuit.name == "simple_mzi"
        assert len(circuit.components) == 7  # input_wg, bs1, upper_wg, lower_wg, ps, bs2, output_wg
        assert len(circuit.connections) == 7
        
        # Verify component types
        component_types = [comp.component_type for comp in circuit.components]
        assert component_types.count(PhotonicComponentType.WAVEGUIDE) == 4
        assert component_types.count(PhotonicComponentType.BEAM_SPLITTER) == 2
        assert component_types.count(PhotonicComponentType.PHASE_SHIFTER) == 1
    
    def test_mzi_circuit_validation(self):
        """Test MZI circuit validation."""
        circuit = create_simple_mzi_circuit()
        assert circuit.validate() is True
    
    def test_mzi_synthesis(self):
        """Test MZI circuit synthesis."""
        bridge = SynthesisBridge()
        circuit = create_simple_mzi_circuit()
        
        result = bridge.synthesize_circuit(circuit)
        
        assert result["circuit_name"] == "simple_mzi"
        assert result["components_count"] == 7
        assert result["connections_count"] == 7


class TestIntegration:
    """Integration tests for the complete synthesis flow."""
    
    def test_end_to_end_synthesis(self):
        """Test complete end-to-end synthesis."""
        # Create circuit
        builder = PhotonicCircuitBuilder("integration_test")
        wg1 = builder.add_waveguide(10.0)
        bs = builder.add_beam_splitter(0.5)
        wg2 = builder.add_waveguide(10.0)
        builder.connect(wg1, bs)
        builder.connect(bs, wg2)
        
        circuit = builder.build()
        
        # Synthesize
        bridge = SynthesisBridge()
        result = bridge.synthesize_circuit(circuit)
        
        # Verify result structure
        assert all(key in result for key in [
            "circuit_name", "mlir_dialect", "mlir_ir", 
            "components_count", "connections_count", "synthesis_metadata"
        ])
        
        # Verify MLIR content
        mlir_ir = result["mlir_ir"]
        assert "photonic.waveguide" in mlir_ir
        assert "photonic.beam.splitter" in mlir_ir
        assert "photonic.connect" in mlir_ir
    
    def test_error_handling(self):
        """Test error handling in synthesis."""
        bridge = SynthesisBridge()
        
        # Create invalid circuit (no components)
        circuit = PhotonicCircuit("empty_circuit")
        
        # Should synthesize successfully even with empty circuit
        result = bridge.synthesize_circuit(circuit)
        assert result["components_count"] == 0
        assert result["connections_count"] == 0
    
    def test_wavelength_band_handling(self):
        """Test wavelength band metadata extraction."""
        builder = PhotonicCircuitBuilder("wavelength_test")
        
        # Add components with different wavelength bands
        comp1 = PhotonicComponent(
            component_type=PhotonicComponentType.WAVEGUIDE,
            wavelength_band=WavelengthBand.C_BAND
        )
        comp2 = PhotonicComponent(
            component_type=PhotonicComponentType.BEAM_SPLITTER,
            wavelength_band=WavelengthBand.L_BAND
        )
        
        builder.circuit.add_component(comp1)
        builder.circuit.add_component(comp2)
        
        circuit = builder.build()
        bridge = SynthesisBridge()
        result = bridge.synthesize_circuit(circuit)
        
        wavelength_bands = result["synthesis_metadata"]["wavelength_bands"]
        assert "c_band" in wavelength_bands
        assert "l_band" in wavelength_bands


@pytest.fixture
def sample_circuit():
    """Fixture providing a sample photonic circuit."""
    builder = PhotonicCircuitBuilder("sample_circuit")
    wg1 = builder.add_waveguide(10.0, position=(0, 0))
    bs = builder.add_beam_splitter(0.5, position=(10, 0))
    wg2 = builder.add_waveguide(15.0, position=(15, 5))
    wg3 = builder.add_waveguide(15.0, position=(15, -5))
    builder.connect(wg1, bs, loss_db=0.1)
    builder.connect(bs, wg2, source_port=0, loss_db=0.05)
    builder.connect(bs, wg3, source_port=1, loss_db=0.05)
    return builder.build()


class TestWithFixtures:
    """Tests using pytest fixtures."""
    
    def test_fixture_circuit(self, sample_circuit):
        """Test using the sample circuit fixture."""
        assert sample_circuit.name == "sample_circuit"
        assert len(sample_circuit.components) == 4
        assert len(sample_circuit.connections) == 3
        assert sample_circuit.validate() is True
    
    def test_synthesis_with_fixture(self, sample_circuit):
        """Test synthesis using fixture circuit."""
        bridge = SynthesisBridge()
        result = bridge.synthesize_circuit(sample_circuit)
        
        assert result["circuit_name"] == "sample_circuit"
        assert result["components_count"] == 4
        assert result["connections_count"] == 3