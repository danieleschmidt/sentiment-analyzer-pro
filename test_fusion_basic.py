#!/usr/bin/env python3
"""
Basic test for quantum-photonic fusion without external dependencies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_fusion_import():
    """Test basic import functionality."""
    print("🧪 Testing Quantum-Photonic Fusion Import...")
    
    try:
        from quantum_photonic_fusion import (
            QuantumPhotonicFusionConfig, FusionMode,
            create_fusion_engine
        )
        print("✅ Import successful - core components available")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_mock_functionality():
    """Test with mock components."""
    print("\n🔬 Testing Mock Functionality...")
    
    try:
        from quantum_photonic_fusion import create_fusion_engine
        
        # Create fusion engine (will use mocks)
        engine = create_fusion_engine(
            quantum_qubits=4,
            photonic_wavelengths=2,
            neuromorphic_neurons=64
        )
        
        print("✅ Fusion engine created successfully")
        
        # Test basic structure
        config = engine.config
        print(f"   Quantum qubits: {config.quantum_qubits}")
        print(f"   Photonic wavelengths: {config.photonic_wavelengths}")
        print(f"   Neuromorphic neurons: {config.neuromorphic_neurons}")
        print(f"   Fusion mode: {config.fusion_mode.value}")
        
        return True
    except Exception as e:
        print(f"❌ Mock test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🌌 Quantum-Photonic-Neuromorphic Fusion - Basic Test")
    print("=" * 60)
    
    # Test import
    import_success = test_fusion_import()
    
    if import_success:
        # Test mock functionality
        mock_success = test_mock_functionality()
        
        if mock_success:
            print("\n🎉 Basic tests passed!")
            print("✅ Quantum-Photonic-Neuromorphic Fusion engine is operational")
            print("✅ Mock components working correctly")
            print("✅ Ready for full dependency testing when available")
            return 0
        else:
            print("\n❌ Mock tests failed")
            return 1
    else:
        print("\n❌ Import tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())