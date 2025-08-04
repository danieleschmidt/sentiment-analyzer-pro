"""
Basic tests for photonic-MLIR bridge without external dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that core modules can be imported."""
    try:
        from src.photonic_mlir_bridge import (
            PhotonicComponent, PhotonicComponentType, WavelengthBand,
            PhotonicConnection, PhotonicCircuit, PhotonicCircuitBuilder
        )
        print("✅ Core imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_component_creation():
    """Test basic component creation."""
    try:
        from src.photonic_mlir_bridge import PhotonicComponent, PhotonicComponentType
        
        component = PhotonicComponent(
            component_type=PhotonicComponentType.WAVEGUIDE,
            position=(1.0, 2.0),
            parameters={"length": 10.0}
        )
        
        assert component.component_type == PhotonicComponentType.WAVEGUIDE
        assert component.position == (1.0, 2.0)
        assert component.parameters["length"] == 10.0
        
        print("✅ Component creation test passed")
        return True
    except Exception as e:
        print(f"❌ Component creation failed: {e}")
        return False

def test_circuit_builder():
    """Test circuit builder functionality."""
    try:
        from src.photonic_mlir_bridge import PhotonicCircuitBuilder
        
        builder = PhotonicCircuitBuilder("test_circuit")
        wg1 = builder.add_waveguide(10.0)
        wg2 = builder.add_waveguide(5.0)
        builder.connect(wg1, wg2)
        
        circuit = builder.build()
        
        assert circuit.name == "test_circuit"
        assert len(circuit.components) == 2
        assert len(circuit.connections) == 1
        
        print("✅ Circuit builder test passed")
        return True
    except Exception as e:
        print(f"❌ Circuit builder failed: {e}")
        return False

def test_validation():
    """Test circuit validation."""
    try:
        from src.photonic_mlir_bridge import PhotonicCircuitBuilder
        
        builder = PhotonicCircuitBuilder("validation_test")
        wg1 = builder.add_waveguide(10.0)
        wg2 = builder.add_waveguide(5.0)
        builder.connect(wg1, wg2)
        
        circuit = builder.build()
        result = circuit.validate()
        
        assert result == True
        
        print("✅ Validation test passed")
        return True
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def test_mzi_creation():
    """Test MZI circuit creation."""
    try:
        from src.photonic_mlir_bridge import create_simple_mzi_circuit
        
        circuit = create_simple_mzi_circuit()
        
        assert circuit.name == "simple_mzi"
        assert len(circuit.components) > 0
        assert len(circuit.connections) > 0
        
        # Validate the circuit
        circuit.validate()
        
        print("✅ MZI creation test passed")
        return True
    except Exception as e:
        print(f"❌ MZI creation failed: {e}")
        return False

def run_basic_tests():
    """Run all basic tests."""
    print("🔬 Running Basic Photonic-MLIR Bridge Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_component_creation,
        test_circuit_builder,
        test_validation,
        test_mzi_creation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All basic tests passed!")
        return True
    else:
        print("⚠️  Some tests failed")
        return False

if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)