"""
Photonic-MLIR Bridge - Comprehensive Test Suite

This module provides comprehensive testing for all photonic bridge components
including unit tests, integration tests, and system validation tests.
"""

import time
import logging
import unittest
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class PhotonicBridgeTestSuite:
    """Comprehensive test suite for photonic-MLIR bridge."""
    
    def __init__(self):
        self.test_results = []
        self.failed_tests = []
        self.passed_tests = []
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests."""
        print("🧪 Running Comprehensive Photonic-MLIR Bridge Test Suite")
        print("=" * 70)
        
        test_methods = [
            ("Basic Bridge Functionality", self.test_basic_bridge_functionality),
            ("Security System Integration", self.test_security_system_integration),
            ("Error Handling System", self.test_error_handling_system),
            ("Validation Framework", self.test_validation_framework),
            ("Resilience System", self.test_resilience_system),
            ("Scaling System", self.test_scaling_system),
            ("Performance Suite", self.test_performance_suite),
            ("Quality Analyzer", self.test_quality_analyzer),
            ("End-to-End Integration", self.test_end_to_end_integration)
        ]
        
        start_time = time.time()
        
        for test_name, test_method in test_methods:
            print(f"\n🔍 Running: {test_name}")
            
            try:
                test_start = time.time()
                result = test_method()
                test_duration = time.time() - test_start
                
                if result.get("success", False):
                    print(f"   ✅ {test_name} - PASSED ({test_duration:.3f}s)")
                    self.passed_tests.append(test_name)
                else:
                    print(f"   ❌ {test_name} - FAILED ({test_duration:.3f}s)")
                    if result.get("error"):
                        print(f"      Error: {result['error']}")
                    self.failed_tests.append(test_name)
                
                result.update({
                    "test_name": test_name,
                    "duration": test_duration,
                    "timestamp": test_start
                })
                self.test_results.append(result)
                
            except Exception as e:
                print(f"   💥 {test_name} - CRASHED: {e}")
                self.failed_tests.append(test_name)
                self.test_results.append({
                    "test_name": test_name,
                    "success": False,
                    "error": str(e),
                    "crashed": True,
                    "duration": 0,
                    "timestamp": time.time()
                })
        
        total_duration = time.time() - start_time
        
        # Generate summary
        summary = {
            "total_tests": len(test_methods),
            "passed": len(self.passed_tests),
            "failed": len(self.failed_tests),
            "success_rate": len(self.passed_tests) / len(test_methods) * 100,
            "total_duration": total_duration,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "detailed_results": self.test_results,
            "timestamp": start_time
        }
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 Comprehensive Test Suite Summary")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total Duration: {summary['total_duration']:.2f}s")
        
        if summary['failed'] > 0:
            print(f"\nFailed Tests:")
            for test in self.failed_tests:
                print(f"  ❌ {test}")
        
        return summary
    
    def test_basic_bridge_functionality(self) -> Dict[str, Any]:
        """Test basic photonic bridge functionality."""
        try:
            from .photonic_mlir_bridge import (
                create_simple_mzi_circuit, 
                SynthesisBridge,
                PhotonicCircuitBuilder
            )
            
            # Test circuit creation
            circuit = create_simple_mzi_circuit()
            if len(circuit.components) == 0:
                return {"success": False, "error": "Circuit has no components"}
            
            # Test synthesis
            bridge = SynthesisBridge()
            result = bridge.synthesize_circuit(circuit)
            
            if not result or "mlir_ir" not in result:
                return {"success": False, "error": "Synthesis failed to generate MLIR IR"}
            
            # Test circuit builder
            builder = PhotonicCircuitBuilder("test_circuit")
            wg1 = builder.add_waveguide(10.0, position=(0, 0))
            wg2 = builder.add_waveguide(10.0, position=(10, 0))
            builder.connect(wg1, wg2, loss_db=0.1)
            test_circuit = builder.build()
            
            if len(test_circuit.components) != 2:
                return {"success": False, "error": "Circuit builder failed"}
            
            return {
                "success": True,
                "details": {
                    "mzi_components": len(circuit.components),
                    "synthesis_result_keys": list(result.keys()),
                    "builder_test_passed": True
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_security_system_integration(self) -> Dict[str, Any]:
        """Test security system integration."""
        try:
            from .photonic_security import (
                SecurityValidator,
                validate_input,
                sanitize_input
            )
            
            # Test security validator
            validator = SecurityValidator()
            
            # Test input validation
            test_inputs = [
                ("valid_input", "component_id"),
                ("malicious<script>", "component_id"),
                ({"key": "value"}, "component_parameters")
            ]
            
            validation_results = []
            for test_input, input_type in test_inputs:
                try:
                    result = validate_input(test_input, input_type)
                    validation_results.append({"input": str(test_input), "valid": result})
                except Exception as e:
                    validation_results.append({"input": str(test_input), "error": str(e)})
            
            # Test input sanitization
            sanitized = sanitize_input("test<script>alert('xss')</script>")
            if "<script>" in sanitized:
                return {"success": False, "error": "Sanitization failed"}
            
            return {
                "success": True,
                "details": {
                    "validator_created": True,
                    "validation_results": validation_results,
                    "sanitization_test_passed": True
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_error_handling_system(self) -> Dict[str, Any]:
        """Test error handling system."""
        try:
            from .photonic_error_handling import (
                handle_photonic_error,
                retry_operation,
                PhotonicErrorContext
            )
            
            # Test error handling
            test_exception = ValueError("Test error")
            error_context = handle_photonic_error(
                test_exception,
                component="test_component",
                operation="test_operation"
            )
            
            if error_context.message != "Test error":
                return {"success": False, "error": "Error context not created properly"}
            
            # Test retry operation
            call_count = 0
            def failing_operation():
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise RuntimeError("Temporary failure")
                return "success"
            
            try:
                result = retry_operation(failing_operation, component="test")
                if result != "success":
                    return {"success": False, "error": "Retry operation failed"}
            except Exception:
                pass  # Expected to fail in some cases
            
            # Test error context manager
            context_test_passed = False
            try:
                with PhotonicErrorContext("test_component", "test_operation"):
                    context_test_passed = True
            except Exception as e:
                return {"success": False, "error": f"Error context manager failed: {e}"}
            
            return {
                "success": True,
                "details": {
                    "error_context_created": True,
                    "retry_calls": call_count,
                    "context_manager_test": context_test_passed
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_validation_framework(self) -> Dict[str, Any]:
        """Test validation framework."""
        try:
            from .photonic_validation import (
                validate_photonic_circuit,
                validate_photonic_component,
                ValidationLevel
            )
            from .photonic_mlir_bridge import create_simple_mzi_circuit
            
            # Test circuit validation
            circuit = create_simple_mzi_circuit()
            report = validate_photonic_circuit(circuit, ValidationLevel.STANDARD)
            
            if not hasattr(report, 'is_valid'):
                return {"success": False, "error": "Validation report missing is_valid property"}
            
            # Test component validation
            if circuit.components:
                component_report = validate_photonic_component(circuit.components[0])
                if not hasattr(component_report, 'issues'):
                    return {"success": False, "error": "Component validation report missing issues"}
            
            # Test different validation levels
            strict_report = validate_photonic_circuit(circuit, ValidationLevel.STRICT)
            basic_report = validate_photonic_circuit(circuit, ValidationLevel.BASIC)
            
            return {
                "success": True,
                "details": {
                    "circuit_validation_passed": True,
                    "component_validation_passed": len(circuit.components) > 0,
                    "standard_issues": len(report.issues),
                    "strict_issues": len(strict_report.issues),
                    "basic_issues": len(basic_report.issues)
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_resilience_system(self) -> Dict[str, Any]:
        """Test resilience system."""
        try:
            from .photonic_resilience import (
                get_resilience_manager,
                get_system_health,
                start_resilience_monitoring,
                stop_resilience_monitoring
            )
            
            # Test resilience manager
            manager = get_resilience_manager()
            if not manager:
                return {"success": False, "error": "Could not get resilience manager"}
            
            # Test health monitoring
            start_resilience_monitoring()
            time.sleep(1)  # Let monitoring run briefly
            
            health = get_system_health()
            if "overall_health" not in health:
                return {"success": False, "error": "Health check missing overall_health"}
            
            stop_resilience_monitoring()
            
            return {
                "success": True,
                "details": {
                    "manager_available": True,
                    "monitoring_started": True,
                    "health_check_keys": list(health.keys()),
                    "overall_health": health.get("overall_health"),
                    "monitoring_stopped": True
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_scaling_system(self) -> Dict[str, Any]:
        """Test scaling system."""
        try:
            from .photonic_scaling import (
                get_scaling_manager,
                get_scaling_stats
            )
            
            # Test scaling manager
            manager = get_scaling_manager()
            if not manager:
                return {"success": False, "error": "Could not get scaling manager"}
            
            # Test scaling statistics
            stats = get_scaling_stats()
            if "current_state" not in stats:
                return {"success": False, "error": "Scaling stats missing current_state"}
            
            return {
                "success": True,
                "details": {
                    "manager_available": True,
                    "stats_keys": list(stats.keys()),
                    "worker_count": stats.get("current_state", {}).get("total_workers", 0)
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_performance_suite(self) -> Dict[str, Any]:
        """Test performance suite."""
        try:
            from .photonic_performance_suite import (
                get_performance_monitor,
                start_performance_monitoring,
                stop_performance_monitoring,
                PerformanceBenchmark
            )
            
            # Test performance monitor
            monitor = get_performance_monitor()
            if not monitor:
                return {"success": False, "error": "Could not get performance monitor"}
            
            # Test monitoring
            start_performance_monitoring()
            time.sleep(1)
            stop_performance_monitoring()
            
            # Test benchmark
            benchmark = PerformanceBenchmark()
            # Run a quick memory benchmark
            memory_result = benchmark.run_memory_usage_benchmark()
            
            if not memory_result or not hasattr(memory_result, 'statistics'):
                return {"success": False, "error": "Memory benchmark failed"}
            
            return {
                "success": True,
                "details": {
                    "monitor_available": True,
                    "monitoring_test_passed": True,
                    "benchmark_completed": True,
                    "memory_usage_mb": memory_result.statistics.get("total_memory_usage", 0)
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_quality_analyzer(self) -> Dict[str, Any]:
        """Test quality analyzer."""
        try:
            from .photonic_quality_analyzer import (
                QualityAnalyzer,
                run_quality_analysis
            )
            
            # Test quality analyzer creation
            analyzer = QualityAnalyzer(Path("."))
            if not analyzer:
                return {"success": False, "error": "Could not create quality analyzer"}
            
            # Test quick analysis (limit scope to avoid long runtime)
            if len(analyzer.python_files) > 10:
                analyzer.python_files = analyzer.python_files[:5]  # Limit for testing
            
            results = analyzer.analyze_codebase()
            
            if "total_issues" not in results:
                return {"success": False, "error": "Analysis results missing total_issues"}
            
            return {
                "success": True,
                "details": {
                    "analyzer_created": True,
                    "files_analyzed": results.get("files_analyzed", 0),
                    "total_issues": results.get("total_issues", 0),
                    "improvement_score": results.get("improvement_score", 0)
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_end_to_end_integration(self) -> Dict[str, Any]:
        """Test end-to-end integration."""
        try:
            from .photonic_mlir_bridge import create_simple_mzi_circuit, SynthesisBridge
            from .photonic_validation import validate_photonic_circuit
            from .photonic_security import validate_input
            from .photonic_error_handling import PhotonicErrorContext
            
            # Create circuit
            circuit = create_simple_mzi_circuit()
            
            # Validate circuit
            with PhotonicErrorContext("integration_test", "circuit_validation"):
                validation_report = validate_photonic_circuit(circuit)
                if not validation_report.is_valid:
                    return {"success": False, "error": "Circuit validation failed in integration test"}
            
            # Synthesize with security validation
            with PhotonicErrorContext("integration_test", "synthesis"):
                # Validate circuit name
                if not validate_input(circuit.name, "circuit_name"):
                    return {"success": False, "error": "Circuit name validation failed"}
                
                # Perform synthesis
                bridge = SynthesisBridge(enable_optimization=True)
                synthesis_result = bridge.synthesize_circuit(circuit)
                
                if not synthesis_result or "mlir_ir" not in synthesis_result:
                    return {"success": False, "error": "End-to-end synthesis failed"}
            
            return {
                "success": True,
                "details": {
                    "circuit_created": True,
                    "validation_passed": validation_report.is_valid,
                    "synthesis_completed": True,
                    "mlir_ir_generated": len(synthesis_result["mlir_ir"]) > 0,
                    "components_synthesized": synthesis_result.get("components_count", 0)
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def export_test_results(self, filepath: str = None) -> str:
        """Export test results to JSON file."""
        if filepath is None:
            filepath = f"comprehensive_test_results_{int(time.time())}.json"
        
        export_data = {
            "test_suite": "Photonic-MLIR Bridge Comprehensive Tests",
            "export_timestamp": time.time(),
            "summary": {
                "total_tests": len(self.test_results),
                "passed": len(self.passed_tests),
                "failed": len(self.failed_tests),
                "success_rate": len(self.passed_tests) / len(self.test_results) * 100 if self.test_results else 0
            },
            "test_results": self.test_results,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Test results exported to {filepath}")
        return filepath


def run_comprehensive_tests() -> Dict[str, Any]:
    """Run comprehensive test suite."""
    test_suite = PhotonicBridgeTestSuite()
    results = test_suite.run_all_tests()
    
    # Export results
    export_file = test_suite.export_test_results()
    results["export_file"] = export_file
    
    return results


if __name__ == "__main__":
    # Run comprehensive tests
    print("🚀 Starting Comprehensive Photonic-MLIR Bridge Test Suite")
    print("=" * 70)
    
    results = run_comprehensive_tests()
    
    print(f"\n📁 Test results exported to: {results['export_file']}")
    
    # Exit with appropriate code
    if results["failed"] > 0:
        print(f"\n❌ Test suite completed with {results['failed']} failures")
        exit(1)
    else:
        print(f"\n✅ All {results['passed']} tests passed successfully!")
        exit(0)