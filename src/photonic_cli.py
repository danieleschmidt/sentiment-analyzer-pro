"""
Photonic-MLIR Bridge CLI Interface

Command-line interface for photonic circuit synthesis and MLIR generation.
Provides comprehensive tooling for circuit design, validation, and optimization.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from .photonic_mlir_bridge import (
    PhotonicCircuit, PhotonicComponent, PhotonicConnection,
    PhotonicComponentType, WavelengthBand, PhotonicCircuitBuilder,
    SynthesisBridge, create_simple_mzi_circuit
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhotonicCLI:
    """Command-line interface for photonic circuit operations."""
    
    def __init__(self):
        self.bridge = SynthesisBridge()
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            prog='photonic-cli',
            description='Photonic-MLIR Synthesis Bridge CLI',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Synthesize a simple MZI circuit
  photonic-cli synthesize --demo mzi --output mzi_circuit.mlir
  
  # Load and synthesize from JSON
  photonic-cli synthesize --input circuit.json --output result.mlir
  
  # Validate a circuit design
  photonic-cli validate --input circuit.json
  
  # Generate example circuits
  photonic-cli examples --type mzi --output example_mzi.json
  
  # Benchmark synthesis performance
  photonic-cli benchmark --circuits 10 --components 100
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Synthesize command
        synth_parser = subparsers.add_parser('synthesize', help='Synthesize photonic circuit to MLIR')
        synth_group = synth_parser.add_mutually_exclusive_group(required=True)
        synth_group.add_argument('--input', '-i', type=str, help='Input circuit JSON file')
        synth_group.add_argument('--demo', choices=['mzi', 'ring', 'lattice'], help='Use demo circuit')
        synth_parser.add_argument('--output', '-o', type=str, required=True, help='Output MLIR file')
        synth_parser.add_argument('--dialect-output', type=str, help='Output dialect definition file')
        synth_parser.add_argument('--optimization-level', '-O', type=int, choices=[0, 1, 2, 3], 
                                default=2, help='Optimization level (default: 2)')
        synth_parser.add_argument('--target', choices=['hardware', 'simulation'], 
                                default='hardware', help='Synthesis target')
        
        # Validate command
        validate_parser = subparsers.add_parser('validate', help='Validate photonic circuit')
        validate_parser.add_argument('--input', '-i', type=str, required=True, help='Input circuit JSON file')
        validate_parser.add_argument('--strict', action='store_true', help='Enable strict validation')
        
        # Examples command
        examples_parser = subparsers.add_parser('examples', help='Generate example circuits')
        examples_parser.add_argument('--type', choices=['mzi', 'ring', 'lattice', 'all'], 
                                   default='mzi', help='Example circuit type')
        examples_parser.add_argument('--output', '-o', type=str, help='Output directory')
        examples_parser.add_argument('--format', choices=['json', 'yaml'], default='json', 
                                   help='Output format')
        
        # Benchmark command
        benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark synthesis performance')
        benchmark_parser.add_argument('--circuits', type=int, default=10, help='Number of circuits')
        benchmark_parser.add_argument('--components', type=int, default=50, help='Components per circuit')
        benchmark_parser.add_argument('--output', '-o', type=str, help='Benchmark results file')
        
        # Info command
        info_parser = subparsers.add_parser('info', help='Display system information')
        info_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
        
        # Convert command
        convert_parser = subparsers.add_parser('convert', help='Convert between formats')
        convert_parser.add_argument('--input', '-i', type=str, required=True, help='Input file')
        convert_parser.add_argument('--output', '-o', type=str, required=True, help='Output file')
        convert_parser.add_argument('--from-format', choices=['json', 'yaml', 'mlir'], 
                                  help='Input format (auto-detected if not specified)')
        convert_parser.add_argument('--to-format', choices=['json', 'yaml', 'mlir'], 
                                  required=True, help='Output format')
        
        return parser
    
    def run(self, args: Optional[list] = None) -> int:
        """Run the CLI with given arguments."""
        try:
            parsed_args = self.parser.parse_args(args)
            
            if not parsed_args.command:
                self.parser.print_help()
                return 1
            
            # Execute the appropriate command
            if parsed_args.command == 'synthesize':
                return self._synthesize(parsed_args)
            elif parsed_args.command == 'validate':
                return self._validate(parsed_args)
            elif parsed_args.command == 'examples':
                return self._examples(parsed_args)
            elif parsed_args.command == 'benchmark':
                return self._benchmark(parsed_args)
            elif parsed_args.command == 'info':
                return self._info(parsed_args)
            elif parsed_args.command == 'convert':
                return self._convert(parsed_args)
            else:
                logger.error(f"Unknown command: {parsed_args.command}")
                return 1
                
        except KeyboardInterrupt:
            logger.info("Operation cancelled by user")
            return 130
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            if logger.level <= logging.DEBUG:
                import traceback
                traceback.print_exc()
            return 1
    
    def _synthesize(self, args) -> int:
        """Synthesize a photonic circuit to MLIR."""
        logger.info("Starting circuit synthesis...")
        
        # Load or create circuit
        if args.demo:
            circuit = self._create_demo_circuit(args.demo)
        else:
            circuit = self._load_circuit_from_file(args.input)
        
        if not circuit:
            return 1
        
        # Perform synthesis
        try:
            result = self.bridge.synthesize_circuit(circuit)
            
            # Save MLIR IR
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(result['mlir_ir'])
            
            logger.info(f"MLIR IR saved to: {output_path}")
            
            # Save dialect definition if requested
            if args.dialect_output:
                dialect_path = Path(args.dialect_output)
                dialect_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(dialect_path, 'w') as f:
                    f.write(result['mlir_dialect'])
                
                logger.info(f"Dialect definition saved to: {dialect_path}")
            
            # Print synthesis summary
            print(f"\n✅ Synthesis completed successfully!")
            print(f"Circuit: {result['circuit_name']}")
            print(f"Components: {result['components_count']}")
            print(f"Connections: {result['connections_count']}")
            print(f"Optimization passes: {len(result['synthesis_metadata']['optimization_passes'])}")
            print(f"Target wavelength bands: {', '.join(result['synthesis_metadata']['wavelength_bands'])}")
            
            return 0
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return 1
    
    def _validate(self, args) -> int:
        """Validate a photonic circuit."""
        logger.info("Validating circuit...")
        
        circuit = self._load_circuit_from_file(args.input)
        if not circuit:
            return 1
        
        try:
            is_valid = circuit.validate()
            
            if is_valid:
                print(f"✅ Circuit '{circuit.name}' is valid")
                print(f"Components: {len(circuit.components)}")
                print(f"Connections: {len(circuit.connections)}")
                
                if args.strict:
                    # Additional strict validation checks
                    self._strict_validation(circuit)
                
                return 0
            else:
                print(f"❌ Circuit '{circuit.name}' is invalid")
                return 1
                
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return 1
    
    def _examples(self, args) -> int:
        """Generate example circuits."""
        logger.info("Generating example circuits...")
        
        output_dir = Path(args.output) if args.output else Path("examples")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        examples = {}
        
        if args.type in ['mzi', 'all']:
            examples['mzi'] = create_simple_mzi_circuit()
        
        if args.type in ['ring', 'all']:
            examples['ring'] = self._create_ring_resonator_circuit()
        
        if args.type in ['lattice', 'all']:
            examples['lattice'] = self._create_lattice_circuit()
        
        # Save examples
        for name, circuit in examples.items():
            if args.format == 'json':
                filename = output_dir / f"{name}_circuit.json"
                self._save_circuit_as_json(circuit, filename)
            elif args.format == 'yaml':
                filename = output_dir / f"{name}_circuit.yaml"
                self._save_circuit_as_yaml(circuit, filename)
            
            logger.info(f"Saved {name} example to: {filename}")
        
        print(f"✅ Generated {len(examples)} example circuit(s) in {output_dir}")
        return 0
    
    def _benchmark(self, args) -> int:
        """Benchmark synthesis performance."""
        import time
        
        logger.info(f"Running benchmark: {args.circuits} circuits, {args.components} components each")
        
        results = {
            "benchmark_config": {
                "circuits": args.circuits,
                "components_per_circuit": args.components,
                "timestamp": time.time()
            },
            "results": []
        }
        
        total_start = time.time()
        
        for i in range(args.circuits):
            # Create a random circuit
            circuit = self._create_random_circuit(f"benchmark_circuit_{i}", args.components)
            
            # Time the synthesis
            start_time = time.time()
            synthesis_result = self.bridge.synthesize_circuit(circuit)
            synthesis_time = time.time() - start_time
            
            results["results"].append({
                "circuit_id": i,
                "synthesis_time": synthesis_time,
                "components": len(circuit.components),
                "connections": len(circuit.connections),
                "mlir_ir_size": len(synthesis_result["mlir_ir"])
            })
            
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{args.circuits} circuits")
        
        total_time = time.time() - total_start
        
        # Calculate statistics
        synthesis_times = [r["synthesis_time"] for r in results["results"]]
        avg_time = sum(synthesis_times) / len(synthesis_times)
        min_time = min(synthesis_times)
        max_time = max(synthesis_times)
        
        results["summary"] = {
            "total_time": total_time,
            "average_synthesis_time": avg_time,
            "min_synthesis_time": min_time,
            "max_synthesis_time": max_time,
            "circuits_per_second": args.circuits / total_time
        }
        
        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Benchmark results saved to: {output_path}")
        
        # Print summary
        print(f"\n📊 Benchmark Results")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average synthesis time: {avg_time:.4f}s")
        print(f"Min/Max synthesis time: {min_time:.4f}s / {max_time:.4f}s")
        print(f"Throughput: {results['summary']['circuits_per_second']:.2f} circuits/s")
        
        return 0
    
    def _info(self, args) -> int:
        """Display system information."""
        print("🔬 Photonic-MLIR Synthesis Bridge")
        print("=" * 40)
        print(f"CLI Version: 1.0.0")
        print(f"Bridge Version: 1.0.0")
        
        print(f"\nSupported Components:")
        for comp_type in PhotonicComponentType:
            print(f"  - {comp_type.value}")
        
        print(f"\nSupported Wavelength Bands:")
        for band in WavelengthBand:
            print(f"  - {band.value}")
        
        if args.verbose:
            print(f"\nOptimization Passes:")
            dummy_bridge = SynthesisBridge()
            dummy_circuit = create_simple_mzi_circuit()
            dummy_bridge.synthesize_circuit(dummy_circuit)
            for pass_name in dummy_bridge.optimization_passes:
                print(f"  - {pass_name}")
        
        return 0
    
    def _convert(self, args) -> int:
        """Convert between different formats."""
        logger.info(f"Converting {args.input} to {args.to_format}")
        
        # This would implement format conversion
        # For now, just a placeholder
        logger.warning("Format conversion not yet implemented")
        return 1
    
    def _create_demo_circuit(self, demo_type: str) -> Optional[PhotonicCircuit]:
        """Create a demo circuit of the specified type."""
        if demo_type == 'mzi':
            return create_simple_mzi_circuit()
        elif demo_type == 'ring':
            return self._create_ring_resonator_circuit()
        elif demo_type == 'lattice':
            return self._create_lattice_circuit()
        else:
            logger.error(f"Unknown demo type: {demo_type}")
            return None
    
    def _create_ring_resonator_circuit(self) -> PhotonicCircuit:
        """Create a ring resonator circuit."""
        builder = PhotonicCircuitBuilder("ring_resonator")
        
        # Add components for ring resonator
        input_wg = builder.add_waveguide(10.0, position=(0, 0))
        coupler = builder.add_beam_splitter(0.1, position=(10, 0))  # 10% coupling
        ring_wg = builder.add_waveguide(31.4, position=(15, 5))  # Circumference for resonance
        output_wg = builder.add_waveguide(10.0, position=(20, 0))
        
        # Connect components
        builder.connect(input_wg, coupler, loss_db=0.1)
        builder.connect(coupler, ring_wg, source_port=1, loss_db=0.05)
        builder.connect(ring_wg, coupler, target_port=1, loss_db=0.05)
        builder.connect(coupler, output_wg, source_port=0, loss_db=0.1)
        
        return builder.build()
    
    def _create_lattice_circuit(self) -> PhotonicCircuit:
        """Create a lattice filter circuit."""
        builder = PhotonicCircuitBuilder("lattice_filter")
        
        # Simple 2x2 lattice structure
        input1 = builder.add_waveguide(5.0, position=(0, 0))
        input2 = builder.add_waveguide(5.0, position=(0, 10))
        
        coupler1 = builder.add_beam_splitter(0.5, position=(10, 5))
        coupler2 = builder.add_beam_splitter(0.5, position=(30, 5))
        
        upper_wg = builder.add_waveguide(15.0, position=(15, 2))
        lower_wg = builder.add_waveguide(15.0, position=(15, 8))
        
        output1 = builder.add_waveguide(5.0, position=(35, 0))
        output2 = builder.add_waveguide(5.0, position=(35, 10))
        
        # Connect the lattice
        builder.connect(input1, coupler1, target_port=0, loss_db=0.1)
        builder.connect(input2, coupler1, target_port=1, loss_db=0.1)
        
        builder.connect(coupler1, upper_wg, source_port=0, loss_db=0.05)
        builder.connect(coupler1, lower_wg, source_port=1, loss_db=0.05)
        
        builder.connect(upper_wg, coupler2, target_port=0, loss_db=0.05)
        builder.connect(lower_wg, coupler2, target_port=1, loss_db=0.05)
        
        builder.connect(coupler2, output1, source_port=0, loss_db=0.1)
        builder.connect(coupler2, output2, source_port=1, loss_db=0.1)
        
        return builder.build()
    
    def _create_random_circuit(self, name: str, num_components: int) -> PhotonicCircuit:
        """Create a random circuit for benchmarking."""
        import random
        
        builder = PhotonicCircuitBuilder(name)
        component_ids = []
        
        # Add random components
        for i in range(num_components):
            comp_type = random.choice(list(PhotonicComponentType))
            x = random.uniform(0, 100)
            y = random.uniform(0, 100)
            
            if comp_type == PhotonicComponentType.WAVEGUIDE:
                comp_id = builder.add_waveguide(
                    length=random.uniform(1, 50),
                    width=random.uniform(0.3, 1.0),
                    position=(x, y)
                )
            elif comp_type == PhotonicComponentType.BEAM_SPLITTER:
                comp_id = builder.add_beam_splitter(
                    ratio=random.uniform(0.1, 0.9),
                    position=(x, y)
                )
            elif comp_type == PhotonicComponentType.PHASE_SHIFTER:
                comp_id = builder.add_phase_shifter(
                    phase_shift=random.uniform(0, 2 * 3.14159),
                    position=(x, y)
                )
            else:
                # Default to waveguide for other types
                comp_id = builder.add_waveguide(
                    length=random.uniform(1, 20),
                    position=(x, y)
                )
            
            component_ids.append(comp_id)
        
        # Add random connections (about 50% of possible connections)
        num_connections = min(num_components - 1, random.randint(1, num_components // 2))
        connected = set()
        
        for _ in range(num_connections):
            if len(component_ids) < 2:
                break
                
            source = random.choice(component_ids)
            target = random.choice([c for c in component_ids if c != source])
            
            connection_key = (source, target)
            if connection_key not in connected:
                builder.connect(
                    source, target,
                    loss_db=random.uniform(0, 1.0),
                    delay_ps=random.uniform(0, 10.0)
                )
                connected.add(connection_key)
        
        return builder.build()
    
    def _load_circuit_from_file(self, filename: str) -> Optional[PhotonicCircuit]:
        """Load a photonic circuit from a JSON file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Create circuit from JSON data
            circuit = PhotonicCircuit(
                name=data.get('name', 'loaded_circuit'),
                metadata=data.get('metadata', {})
            )
            
            # Load components
            for comp_data in data.get('components', []):
                component = PhotonicComponent(
                    id=comp_data['id'],
                    component_type=PhotonicComponentType(comp_data['component_type']),
                    position=tuple(comp_data['position']),
                    parameters=comp_data.get('parameters', {}),
                    wavelength_band=WavelengthBand(comp_data.get('wavelength_band', 'c_band'))
                )
                circuit.add_component(component)
            
            # Load connections
            for conn_data in data.get('connections', []):
                connection = PhotonicConnection(
                    id=conn_data['id'],
                    source_component=conn_data['source_component'],
                    target_component=conn_data['target_component'],
                    source_port=conn_data.get('source_port', 0),
                    target_port=conn_data.get('target_port', 0),
                    loss_db=conn_data.get('loss_db', 0.0),
                    delay_ps=conn_data.get('delay_ps', 0.0)
                )
                circuit.add_connection(connection)
            
            logger.info(f"Loaded circuit from {filename}")
            return circuit
            
        except Exception as e:
            logger.error(f"Failed to load circuit from {filename}: {e}")
            return None
    
    def _save_circuit_as_json(self, circuit: PhotonicCircuit, filename: Path) -> None:
        """Save a circuit as JSON."""
        data = {
            'name': circuit.name,
            'metadata': circuit.metadata,
            'components': [
                {
                    'id': comp.id,
                    'component_type': comp.component_type.value,
                    'position': list(comp.position),
                    'parameters': comp.parameters,
                    'wavelength_band': comp.wavelength_band.value
                }
                for comp in circuit.components
            ],
            'connections': [
                {
                    'id': conn.id,
                    'source_component': conn.source_component,
                    'target_component': conn.target_component,
                    'source_port': conn.source_port,
                    'target_port': conn.target_port,
                    'loss_db': conn.loss_db,
                    'delay_ps': conn.delay_ps
                }
                for conn in circuit.connections
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_circuit_as_yaml(self, circuit: PhotonicCircuit, filename: Path) -> None:
        """Save a circuit as YAML."""
        # Would require PyYAML - placeholder for now
        logger.warning("YAML export not implemented - saving as JSON")
        json_filename = filename.with_suffix('.json')
        self._save_circuit_as_json(circuit, json_filename)
    
    def _strict_validation(self, circuit: PhotonicCircuit) -> None:
        """Perform strict validation checks."""
        logger.info("Performing strict validation...")
        
        # Check for isolated components
        connected_components = set()
        for conn in circuit.connections:
            connected_components.add(conn.source_component)
            connected_components.add(conn.target_component)
        
        isolated = [comp.id for comp in circuit.components if comp.id not in connected_components]
        if isolated:
            logger.warning(f"Found {len(isolated)} isolated components: {isolated}")
        
        # Check for excessive losses
        high_loss_connections = [conn for conn in circuit.connections if conn.loss_db > 3.0]
        if high_loss_connections:
            logger.warning(f"Found {len(high_loss_connections)} high-loss connections (>3dB)")
        
        # Check component density
        positions = [comp.position for comp in circuit.components]
        if len(set(positions)) < len(positions):
            logger.warning("Multiple components at same position detected")


def main():
    """Main CLI entry point."""
    cli = PhotonicCLI()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())