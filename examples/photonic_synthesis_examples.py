"""
Photonic-MLIR Synthesis Bridge Examples

This module provides comprehensive examples demonstrating the capabilities
of the photonic-MLIR synthesis bridge, including various circuit topologies,
optimization strategies, and integration patterns.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any

from src.photonic_mlir_bridge import (
    PhotonicCircuit, PhotonicComponent, PhotonicConnection,
    PhotonicComponentType, WavelengthBand, PhotonicCircuitBuilder,
    SynthesisBridge, create_simple_mzi_circuit
)


def create_ring_resonator_filter(resonance_wavelength: float = 1550.0, 
                               coupling_ratio: float = 0.1,
                               ring_radius: float = 5.0) -> PhotonicCircuit:
    """
    Create a ring resonator filter circuit.
    
    Args:
        resonance_wavelength: Target resonance wavelength in nm
        coupling_ratio: Coupling ratio between waveguide and ring (0-1)
        ring_radius: Ring radius in micrometers
        
    Returns:
        PhotonicCircuit: Complete ring resonator filter
    """
    builder = PhotonicCircuitBuilder(f"ring_filter_{resonance_wavelength}nm")
    
    # Calculate ring circumference for target wavelength
    ring_circumference = 2 * 3.14159 * ring_radius
    
    # Input section
    input_wg = builder.add_waveguide(
        length=20.0, 
        width=0.45, 
        position=(0, 0)
    )
    
    # Coupling region
    directional_coupler = PhotonicComponent(
        component_type=PhotonicComponentType.DIRECTIONAL_COUPLER,
        position=(25, 0),
        parameters={
            "coupling_length": 10.0,
            "gap": 0.2,
            "coupling_ratio": coupling_ratio
        }
    )
    builder.circuit.add_component(directional_coupler)
    
    # Ring waveguide
    ring_wg = builder.add_waveguide(
        length=ring_circumference,
        width=0.45,
        position=(30, 5)
    )
    
    # Through port waveguide
    through_wg = builder.add_waveguide(
        length=20.0,
        width=0.45, 
        position=(35, 0)
    )
    
    # Drop port waveguide
    drop_wg = builder.add_waveguide(
        length=20.0,
        width=0.45,
        position=(35, -10)
    )
    
    # Connections
    builder.connect(input_wg.id, directional_coupler.id, loss_db=0.05)
    builder.connect(directional_coupler.id, ring_wg.id, source_port=1, target_port=0, loss_db=0.02)
    builder.connect(ring_wg.id, directional_coupler.id, source_port=1, target_port=1, loss_db=0.02)
    builder.connect(directional_coupler.id, through_wg.id, source_port=0, target_port=0, loss_db=0.05)
    builder.connect(directional_coupler.id, drop_wg.id, source_port=2, target_port=0, loss_db=0.05)
    
    # Add metadata
    builder.circuit.metadata = {
        "resonance_wavelength_nm": resonance_wavelength,
        "coupling_ratio": coupling_ratio,
        "ring_radius_um": ring_radius,
        "free_spectral_range_nm": resonance_wavelength**2 / (2 * ring_circumference * 2.4),  # Approx for silicon
        "circuit_type": "ring_resonator_filter"
    }
    
    return builder.build()


def create_4x4_optical_switch(switch_state: str = "cross") -> PhotonicCircuit:
    """
    Create a 4x4 optical switch using Mach-Zehnder interferometers.
    
    Args:
        switch_state: Switch configuration ("bar", "cross", "broadcast")
        
    Returns:
        PhotonicCircuit: 4x4 optical switch circuit
    """
    builder = PhotonicCircuitBuilder(f"4x4_switch_{switch_state}")
    
    # Input waveguides
    inputs = []
    for i in range(4):
        wg = builder.add_waveguide(10.0, position=(0, i * 10))
        inputs.append(wg)
    
    # First stage - 2x2 switches
    switches_stage1 = []
    for i in range(2):
        # Each 2x2 switch is implemented as a pair of MZIs
        bs1 = builder.add_beam_splitter(0.5, position=(20, i * 20 + 5))
        ps1 = builder.add_phase_shifter(0 if switch_state == "bar" else 3.14159, 
                                      position=(30, i * 20 + 2))
        ps2 = builder.add_phase_shifter(0, position=(30, i * 20 + 8))
        bs2 = builder.add_beam_splitter(0.5, position=(40, i * 20 + 5))
        
        switches_stage1.append({
            "bs1": bs1, "ps1": ps1, "ps2": ps2, "bs2": bs2
        })
    
    # Second stage - 2x2 switches
    switches_stage2 = []
    for i in range(2):
        bs1 = builder.add_beam_splitter(0.5, position=(60, i * 20 + 15))
        ps1 = builder.add_phase_shifter(0 if switch_state == "cross" else 3.14159,
                                      position=(70, i * 20 + 12))
        ps2 = builder.add_phase_shifter(0, position=(70, i * 20 + 18))
        bs2 = builder.add_beam_splitter(0.5, position=(80, i * 20 + 15))
        
        switches_stage2.append({
            "bs1": bs1, "ps1": ps1, "ps2": ps2, "bs2": bs2
        })
    
    # Output waveguides
    outputs = []
    for i in range(4):
        wg = builder.add_waveguide(10.0, position=(100, i * 10))
        outputs.append(wg)
    
    # Connect stage 1
    for i, switch in enumerate(switches_stage1):
        # Connect inputs to first beam splitter
        builder.connect(inputs[i*2], switch["bs1"], target_port=0, loss_db=0.1)
        builder.connect(inputs[i*2+1], switch["bs1"], target_port=1, loss_db=0.1)
        
        # Connect beam splitter to phase shifters
        builder.connect(switch["bs1"], switch["ps1"], source_port=0, loss_db=0.05)
        builder.connect(switch["bs1"], switch["ps2"], source_port=1, loss_db=0.05)
        
        # Connect phase shifters to second beam splitter
        builder.connect(switch["ps1"], switch["bs2"], target_port=0, loss_db=0.05)
        builder.connect(switch["ps2"], switch["bs2"], target_port=1, loss_db=0.05)
    
    # Connect between stages (crossover connections)
    builder.connect(switches_stage1[0]["bs2"], switches_stage2[0]["bs1"], 
                   source_port=0, target_port=0, loss_db=0.1)
    builder.connect(switches_stage1[0]["bs2"], switches_stage2[1]["bs1"],
                   source_port=1, target_port=0, loss_db=0.1)
    builder.connect(switches_stage1[1]["bs2"], switches_stage2[0]["bs1"],
                   source_port=0, target_port=1, loss_db=0.1)
    builder.connect(switches_stage1[1]["bs2"], switches_stage2[1]["bs1"],
                   source_port=1, target_port=1, loss_db=0.1)
    
    # Connect stage 2 to outputs
    for i, switch in enumerate(switches_stage2):
        # Connect to phase shifters
        builder.connect(switch["bs1"], switch["ps1"], source_port=0, loss_db=0.05)
        builder.connect(switch["bs1"], switch["ps2"], source_port=1, loss_db=0.05)
        
        # Connect to final outputs
        builder.connect(switch["ps1"], switch["bs2"], target_port=0, loss_db=0.05)
        builder.connect(switch["ps2"], switch["bs2"], target_port=1, loss_db=0.05)
        
        builder.connect(switch["bs2"], outputs[i*2], source_port=0, loss_db=0.1)
        builder.connect(switch["bs2"], outputs[i*2+1], source_port=1, loss_db=0.1)
    
    # Add metadata
    builder.circuit.metadata = {
        "switch_state": switch_state,
        "switch_size": "4x4",
        "num_phase_shifters": 8,
        "estimated_insertion_loss_db": 2.0,
        "circuit_type": "optical_switch"
    }
    
    return builder.build()


def create_wavelength_division_multiplexer(channel_count: int = 4,
                                         channel_spacing: float = 1.6) -> PhotonicCircuit:
    """
    Create a wavelength division multiplexer (WDM) using ring resonators.
    
    Args:
        channel_count: Number of wavelength channels
        channel_spacing: Channel spacing in nm
        
    Returns:
        PhotonicCircuit: WDM circuit
    """
    builder = PhotonicCircuitBuilder(f"wdm_{channel_count}ch")
    
    base_wavelength = 1550.0  # C-band center
    
    # Input bus waveguide
    bus_wg = builder.add_waveguide(
        length=50.0 + channel_count * 10,
        width=0.45,
        position=(0, 0)
    )
    
    # Create ring resonators for each channel
    rings = []
    drop_ports = []
    
    for i in range(channel_count):
        wavelength = base_wavelength + i * channel_spacing
        x_pos = 15 + i * 15
        
        # Ring resonator
        ring_radius = wavelength / (2 * 3.14159 * 2.4)  # Effective index ~2.4 for silicon
        ring_wg = builder.add_waveguide(
            length=2 * 3.14159 * ring_radius,
            width=0.45,
            position=(x_pos, 8)
        )
        
        # Coupling region
        coupler = PhotonicComponent(
            component_type=PhotonicComponentType.DIRECTIONAL_COUPLER,
            position=(x_pos, 4),
            parameters={
                "coupling_length": 5.0,
                "gap": 0.15,
                "coupling_ratio": 0.05,  # Weak coupling for high Q
                "target_wavelength": wavelength
            },
            wavelength_band=WavelengthBand.C_BAND
        )
        builder.circuit.add_component(coupler)
        
        # Drop port waveguide
        drop_wg = builder.add_waveguide(
            length=20.0,
            width=0.45,
            position=(x_pos, -8)
        )
        
        # Connect ring to coupler
        builder.connect(ring_wg, coupler.id, loss_db=0.01)
        builder.connect(coupler.id, ring_wg, source_port=1, target_port=1, loss_db=0.01)
        builder.connect(coupler.id, drop_wg, source_port=2, loss_db=0.05)
        
        rings.append(ring_wg)
        drop_ports.append(drop_wg)
    
    # Through port (remaining wavelengths)
    through_wg = builder.add_waveguide(
        length=20.0,
        width=0.45,
        position=(70 + channel_count * 10, 0)
    )
    
    # Add photodetectors at drop ports (optional)
    detectors = []
    for i, drop_port in enumerate(drop_ports):
        detector = PhotonicComponent(
            component_type=PhotonicComponentType.PHOTODETECTOR,
            position=(15 + i * 15, -15),
            parameters={
                "responsivity": 0.8,  # A/W
                "dark_current": 1e-9,  # A
                "bandwidth": 25e9     # Hz
            }
        )
        builder.circuit.add_component(detector)
        builder.connect(drop_port, detector.id, loss_db=0.2)
        detectors.append(detector.id)
    
    # Add metadata
    builder.circuit.metadata = {
        "channel_count": channel_count,
        "channel_spacing_nm": channel_spacing,
        "wavelength_range": f"{base_wavelength} - {base_wavelength + (channel_count-1) * channel_spacing}",
        "estimated_insertion_loss_db": 0.5,
        "estimated_crosstalk_db": -30,
        "circuit_type": "wavelength_division_multiplexer"
    }
    
    return builder.build()


def create_optical_neural_network_layer(input_size: int = 4, 
                                       output_size: int = 4,
                                       activation: str = "linear") -> PhotonicCircuit:
    """
    Create an optical neural network layer using programmable photonic circuits.
    
    Args:
        input_size: Number of input nodes
        output_size: Number of output nodes  
        activation: Activation function type
        
    Returns:
        PhotonicCircuit: Optical neural network layer
    """
    builder = PhotonicCircuitBuilder(f"onn_layer_{input_size}x{output_size}")
    
    # Input modulators (encode input data)
    input_modulators = []
    for i in range(input_size):
        laser = PhotonicComponent(
            component_type=PhotonicComponentType.LASER,
            position=(0, i * 8),
            parameters={
                "wavelength": 1550.0 + i * 0.8,  # Different wavelengths
                "power_dbm": 0
            }
        )
        
        modulator = PhotonicComponent(
            component_type=PhotonicComponentType.MODULATOR,
            position=(10, i * 8),
            parameters={
                "modulation_depth": 0.9,
                "bandwidth": 25e9,
                "vpi": 2.5  # V·π voltage
            }
        )
        
        builder.circuit.add_component(laser)
        builder.circuit.add_component(modulator)
        builder.connect(laser.id, modulator.id, loss_db=0.1)
        input_modulators.append(modulator.id)
    
    # Weight matrix implementation using programmable MZI mesh
    weight_mzis = []
    for i in range(output_size):
        for j in range(input_size):
            # Each weight is implemented as a programmable MZI
            bs1 = builder.add_beam_splitter(0.5, position=(30 + j * 15, i * 10))
            
            # Programmable phase shifters for weight control
            phase_shifter = builder.add_phase_shifter(
                phase_shift=0.0,  # Will be programmed during training
                position=(40 + j * 15, i * 10 - 2)
            )
            
            bs2 = builder.add_beam_splitter(0.5, position=(50 + j * 15, i * 10))
            
            weight_mzis.append({
                "input_idx": j,
                "output_idx": i,
                "bs1": bs1,
                "phase_shifter": phase_shifter,
                "bs2": bs2
            })
    
    # Connect input modulators to weight matrix
    for i, modulator in enumerate(input_modulators):
        # Fan out to all relevant MZIs
        for mzi in weight_mzis:
            if mzi["input_idx"] == i:
                builder.connect(modulator, mzi["bs1"], loss_db=0.05)
                builder.connect(mzi["bs1"], mzi["phase_shifter"], source_port=0, loss_db=0.02)
                builder.connect(mzi["phase_shifter"], mzi["bs2"], target_port=0, loss_db=0.02)
    
    # Summing networks (combine weighted inputs for each output)
    output_combiners = []
    for i in range(output_size):
        # Simple power combiner for now
        combiner = PhotonicComponent(
            component_type=PhotonicComponentType.DIRECTIONAL_COUPLER,
            position=(80, i * 10),
            parameters={
                "num_inputs": input_size,
                "combining_loss": 10 * 3.14159 / input_size  # log10(N) loss
            }
        )
        builder.circuit.add_component(combiner)
        output_combiners.append(combiner.id)
    
    # Optical activation functions (if nonlinear)
    activation_elements = []
    if activation == "sigmoid":
        for i in range(output_size):
            # Nonlinear element using cross-phase modulation
            activation_elem = PhotonicComponent(
                component_type=PhotonicComponentType.PHASE_SHIFTER,
                position=(90, i * 10),
                parameters={
                    "activation_type": "sigmoid",
                    "nonlinear_coefficient": 1e-18  # Kerr coefficient
                }
            )
            builder.circuit.add_component(activation_elem)
            builder.connect(output_combiners[i], activation_elem.id, loss_db=0.1)
            activation_elements.append(activation_elem.id)
    
    # Output photodetectors
    output_detectors = []
    for i in range(output_size):
        detector = PhotonicComponent(
            component_type=PhotonicComponentType.PHOTODETECTOR,
            position=(100, i * 10),
            parameters={
                "responsivity": 1.0,
                "bandwidth": 50e9,
                "output_type": "electrical"
            }
        )
        builder.circuit.add_component(detector)
        
        if activation_elements:
            builder.connect(activation_elements[i], detector.id, loss_db=0.2)
        else:
            builder.connect(output_combiners[i], detector.id, loss_db=0.2)
        
        output_detectors.append(detector.id)
    
    # Add metadata
    builder.circuit.metadata = {
        "layer_type": "optical_neural_network",
        "input_size": input_size,
        "output_size": output_size,
        "activation_function": activation,
        "num_programmable_weights": len(weight_mzis),
        "estimated_latency_ns": 0.1,  # Speed of light delay
        "estimated_power_mw": input_size * 10 + output_size * 5
    }
    
    return builder.build()


def benchmark_synthesis_performance(circuit_sizes: List[int] = [10, 50, 100, 500]) -> Dict[str, Any]:
    """
    Benchmark synthesis performance for different circuit sizes.
    
    Args:
        circuit_sizes: List of component counts to benchmark
        
    Returns:
        Dict containing benchmark results
    """
    bridge = SynthesisBridge()
    results = {
        "benchmark_timestamp": time.time(),
        "circuit_sizes": circuit_sizes,
        "results": []
    }
    
    print("🔬 Running Synthesis Performance Benchmark")
    print("=" * 50)
    
    for size in circuit_sizes:
        print(f"Benchmarking {size} components...")
        
        # Create a random circuit of specified size
        builder = PhotonicCircuitBuilder(f"benchmark_{size}")
        
        # Add random components
        component_ids = []
        for i in range(size):
            if i % 3 == 0:
                comp_id = builder.add_waveguide(10.0, position=(i, 0))
            elif i % 3 == 1:
                comp_id = builder.add_beam_splitter(0.5, position=(i, 5))
            else:
                comp_id = builder.add_phase_shifter(1.57, position=(i, -5))
            component_ids.append(comp_id)
        
        # Add connections (linear chain)
        for i in range(size - 1):
            builder.connect(component_ids[i], component_ids[i + 1], loss_db=0.1)
        
        circuit = builder.build()
        
        # Time the synthesis
        start_time = time.time()
        synthesis_result = bridge.synthesize_circuit(circuit)
        synthesis_time = time.time() - start_time
        
        # Collect metrics
        result = {
            "component_count": size,
            "connection_count": size - 1,
            "synthesis_time_s": synthesis_time,
            "mlir_ir_size_chars": len(synthesis_result["mlir_ir"]),
            "mlir_dialect_size_chars": len(synthesis_result["mlir_dialect"]),
            "components_per_second": size / synthesis_time if synthesis_time > 0 else float('inf')
        }
        
        results["results"].append(result)
        print(f"  Completed in {synthesis_time:.3f}s ({result['components_per_second']:.1f} components/s)")
    
    # Calculate summary statistics
    synthesis_times = [r["synthesis_time_s"] for r in results["results"]]
    throughputs = [r["components_per_second"] for r in results["results"]]
    
    results["summary"] = {
        "avg_synthesis_time_s": sum(synthesis_times) / len(synthesis_times),
        "max_synthesis_time_s": max(synthesis_times),
        "min_synthesis_time_s": min(synthesis_times),
        "avg_throughput_comp_per_s": sum(throughputs) / len(throughputs),
        "max_throughput_comp_per_s": max(throughputs)
    }
    
    print(f"\n📊 Benchmark Summary:")
    print(f"Average synthesis time: {results['summary']['avg_synthesis_time_s']:.3f}s")
    print(f"Average throughput: {results['summary']['avg_throughput_comp_per_s']:.1f} components/s")
    print(f"Max throughput: {results['summary']['max_throughput_comp_per_s']:.1f} components/s")
    
    return results


def demonstrate_advanced_features():
    """Demonstrate advanced features of the photonic-MLIR bridge."""
    print("🚀 Advanced Photonic-MLIR Bridge Features Demo")
    print("=" * 60)
    
    bridge = SynthesisBridge()
    examples = []
    
    # 1. Ring Resonator Filter
    print("\n1. Ring Resonator Filter (1550nm)")
    ring_circuit = create_ring_resonator_filter(1550.0, 0.1, 5.0)
    result = bridge.synthesize_circuit(ring_circuit)
    examples.append(("ring_filter", ring_circuit, result))
    print(f"   Components: {result['components_count']}, Connections: {result['connections_count']}")
    
    # 2. 4x4 Optical Switch
    print("\n2. 4x4 Optical Switch (Cross State)")
    switch_circuit = create_4x4_optical_switch("cross")
    result = bridge.synthesize_circuit(switch_circuit)
    examples.append(("4x4_switch", switch_circuit, result))
    print(f"   Components: {result['components_count']}, Connections: {result['connections_count']}")
    
    # 3. WDM Multiplexer
    print("\n3. 4-Channel WDM Multiplexer")
    wdm_circuit = create_wavelength_division_multiplexer(4, 1.6)
    result = bridge.synthesize_circuit(wdm_circuit)
    examples.append(("wdm_4ch", wdm_circuit, result))
    print(f"   Components: {result['components_count']}, Connections: {result['connections_count']}")
    
    # 4. Optical Neural Network Layer
    print("\n4. 4x4 Optical Neural Network Layer")
    onn_circuit = create_optical_neural_network_layer(4, 4, "linear")
    result = bridge.synthesize_circuit(onn_circuit)
    examples.append(("onn_4x4", onn_circuit, result))
    print(f"   Components: {result['components_count']}, Connections: {result['connections_count']}")
    
    # Save all examples
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, circuit, synthesis_result in examples:
        # Save circuit description
        circuit_data = {
            "name": circuit.name,
            "metadata": circuit.metadata,
            "components": [
                {
                    "id": comp.id,
                    "type": comp.component_type.value,
                    "position": comp.position,
                    "parameters": comp.parameters,
                    "wavelength_band": comp.wavelength_band.value
                }
                for comp in circuit.components
            ],
            "connections": [
                {
                    "id": conn.id,
                    "source": conn.source_component,
                    "target": conn.target_component,
                    "source_port": conn.source_port,
                    "target_port": conn.target_port,
                    "loss_db": conn.loss_db,
                    "delay_ps": conn.delay_ps
                }
                for conn in circuit.connections
            ]
        }
        
        with open(output_dir / f"{name}_circuit.json", "w") as f:
            json.dump(circuit_data, f, indent=2)
        
        # Save MLIR output
        with open(output_dir / f"{name}_circuit.mlir", "w") as f:
            f.write(synthesis_result["mlir_ir"])
        
        # Save dialect definition
        with open(output_dir / f"{name}_dialect.td", "w") as f:
            f.write(synthesis_result["mlir_dialect"])
    
    print(f"\n📁 All examples saved to {output_dir}/")
    print("   - Circuit descriptions (JSON)")
    print("   - MLIR IR files")
    print("   - Dialect definitions (TableGen)")
    
    return examples


def run_comprehensive_demo():
    """Run a comprehensive demonstration of all features."""
    print("🔬 Photonic-MLIR Synthesis Bridge - Comprehensive Demo")
    print("=" * 70)
    
    # 1. Basic functionality demo
    print("\n📋 PART 1: Basic Functionality")
    circuit = create_simple_mzi_circuit()
    bridge = SynthesisBridge()
    result = bridge.synthesize_circuit(circuit)
    print(f"✅ Basic MZI synthesis: {result['components_count']} components, {result['connections_count']} connections")
    
    # 2. Advanced circuits demo
    print("\n📋 PART 2: Advanced Circuit Examples")
    demonstrate_advanced_features()
    
    # 3. Performance benchmark
    print("\n📋 PART 3: Performance Benchmark")
    benchmark_results = benchmark_synthesis_performance([10, 25, 50, 100])
    
    # 4. Save comprehensive results
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)
    
    # 5. Summary
    print("\n📋 PART 4: Demo Summary")
    print("✅ All photonic-MLIR bridge features demonstrated successfully!")
    print(f"✅ Performance: {benchmark_results['summary']['avg_throughput_comp_per_s']:.1f} avg components/s")
    print(f"✅ Maximum throughput: {benchmark_results['summary']['max_throughput_comp_per_s']:.1f} components/s")
    print(f"✅ Output files saved to: {output_dir}/")
    
    return {
        "basic_demo": result,
        "advanced_examples": True,
        "benchmark_results": benchmark_results
    }


if __name__ == "__main__":
    # Run the comprehensive demo
    results = run_comprehensive_demo()
    print("\n🎉 Comprehensive demo completed successfully!")