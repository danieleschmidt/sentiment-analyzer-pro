#!/usr/bin/env python3
"""
Comprehensive Quality Gates Runner
Executes all quality checks, security scans, and performance tests.
"""

import os
import sys
import time
import subprocess
import json
from typing import Dict, List, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QualityGateRunner:
    """Comprehensive quality gate execution system."""
    
    def __init__(self):
        self.results = {}
        self.overall_status = "PASS"
        self.start_time = time.time()
        
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates and return comprehensive results."""
        logger.info("🚀 Starting Comprehensive Quality Gates")
        
        # Core quality gates
        self.results["syntax_check"] = self._run_syntax_check()
        self.results["import_validation"] = self._run_import_validation()
        self.results["security_scan"] = self._run_security_scan()
        self.results["performance_test"] = self._run_performance_test()
        self.results["integration_test"] = self._run_integration_test()
        self.results["code_quality"] = self._run_code_quality_check()
        
        # Calculate overall status
        self._calculate_overall_status()
        
        # Generate final report
        execution_time = time.time() - self.start_time
        
        final_report = {
            "timestamp": time.time(),
            "execution_time_seconds": execution_time,
            "overall_status": self.overall_status,
            "gate_results": self.results,
            "summary": self._generate_summary()
        }
        
        self._save_report(final_report)
        self._print_summary(final_report)
        
        return final_report
    
    def _run_syntax_check(self) -> Dict[str, Any]:
        """Check Python syntax for all source files."""
        logger.info("🔍 Running syntax check...")
        
        results = {"status": "PASS", "errors": [], "files_checked": 0}
        
        try:
            # Find all Python files
            python_files = []
            for root, dirs, files in os.walk("src"):
                for file in files:
                    if file.endswith(".py"):
                        python_files.append(os.path.join(root, file))
            
            results["files_checked"] = len(python_files)
            
            # Check syntax for each file
            for file_path in python_files:
                try:
                    with open(file_path, 'r') as f:
                        code = f.read()
                    
                    # Compile to check syntax
                    compile(code, file_path, 'exec')
                    
                except SyntaxError as e:
                    results["errors"].append({
                        "file": file_path,
                        "error": str(e),
                        "line": e.lineno
                    })
                    results["status"] = "FAIL"
                except Exception as e:
                    results["errors"].append({
                        "file": file_path,
                        "error": f"Compilation error: {str(e)}"
                    })
                    results["status"] = "FAIL"
            
            logger.info(f"✅ Syntax check: {results['status']} ({results['files_checked']} files)")
            
        except Exception as e:
            results["status"] = "ERROR"
            results["error"] = str(e)
            logger.error(f"❌ Syntax check failed: {e}")
        
        return results
    
    def _run_import_validation(self) -> Dict[str, Any]:
        """Validate that all imports can be resolved."""
        logger.info("📦 Running import validation...")
        
        results = {"status": "PASS", "errors": [], "modules_tested": 0}
        
        # Key modules to test
        test_modules = [
            "src.adaptive_agi_engine",
            "src.enterprise_security_framework",
            "src.intelligent_error_recovery_v2",
            "src.quantum_performance_accelerator"
        ]
        
        results["modules_tested"] = len(test_modules)
        
        for module in test_modules:
            try:
                __import__(module)
                logger.info(f"  ✅ {module}")
            except ImportError as e:
                results["errors"].append({
                    "module": module,
                    "error": str(e)
                })
                results["status"] = "FAIL"
                logger.error(f"  ❌ {module}: {e}")
            except Exception as e:
                results["errors"].append({
                    "module": module,
                    "error": f"Unexpected error: {str(e)}"
                })
                results["status"] = "FAIL"
                logger.error(f"  ❌ {module}: {e}")
        
        logger.info(f"✅ Import validation: {results['status']}")
        return results
    
    def _run_security_scan(self) -> Dict[str, Any]:
        """Run security vulnerability scanning."""
        logger.info("🔒 Running security scan...")
        
        results = {"status": "PASS", "vulnerabilities": [], "security_score": 100}
        
        try:
            # Basic security checks
            security_issues = []
            
            # Check for common security anti-patterns
            for root, dirs, files in os.walk("src"):
                for file in files:
                    if file.endswith(".py"):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r') as f:
                            content = f.read()
                        
                        # Check for potential security issues
                        if "eval(" in content:
                            security_issues.append({
                                "file": file_path,
                                "issue": "Use of eval() function",
                                "severity": "HIGH"
                            })
                        
                        if "exec(" in content:
                            security_issues.append({
                                "file": file_path,
                                "issue": "Use of exec() function",
                                "severity": "HIGH"
                            })
                        
                        if "subprocess.call" in content and "shell=True" in content:
                            security_issues.append({
                                "file": file_path,
                                "issue": "Subprocess call with shell=True",
                                "severity": "MEDIUM"
                            })
            
            results["vulnerabilities"] = security_issues
            
            if security_issues:
                high_severity = sum(1 for issue in security_issues if issue["severity"] == "HIGH")
                medium_severity = sum(1 for issue in security_issues if issue["severity"] == "MEDIUM")
                
                # Calculate security score
                results["security_score"] = max(0, 100 - (high_severity * 20) - (medium_severity * 10))
                
                if high_severity > 0:
                    results["status"] = "FAIL"
                elif medium_severity > 3:
                    results["status"] = "WARN"
            
            logger.info(f"🔒 Security scan: {results['status']} (Score: {results['security_score']}/100)")
            
        except Exception as e:
            results["status"] = "ERROR"
            results["error"] = str(e)
            logger.error(f"❌ Security scan failed: {e}")
        
        return results
    
    def _run_performance_test(self) -> Dict[str, Any]:
        """Run basic performance tests."""
        logger.info("⚡ Running performance tests...")
        
        results = {"status": "PASS", "benchmarks": {}, "performance_score": 100}
        
        try:
            # Test AGI Engine initialization performance
            start_time = time.time()
            from src.adaptive_agi_engine import AdaptiveAGIEngine
            engine = AdaptiveAGIEngine()
            init_time = time.time() - start_time
            
            results["benchmarks"]["agi_init_time"] = init_time
            
            # Test basic processing performance
            start_time = time.time()
            # Simulate processing without async for now
            neural_input = engine._text_to_vector("Test performance input")
            neural_output = engine.neural_network.forward(neural_input.reshape(1, -1))
            process_time = time.time() - start_time
            
            results["benchmarks"]["neural_process_time"] = process_time
            
            # Performance criteria
            if init_time > 5.0:  # Should initialize in under 5 seconds
                results["status"] = "FAIL"
                results["performance_score"] -= 30
            elif init_time > 2.0:
                results["status"] = "WARN"
                results["performance_score"] -= 10
            
            if process_time > 1.0:  # Should process in under 1 second
                results["status"] = "FAIL"
                results["performance_score"] -= 30
            elif process_time > 0.5:
                results["status"] = "WARN"
                results["performance_score"] -= 10
            
            logger.info(f"⚡ Performance test: {results['status']} (Score: {results['performance_score']}/100)")
            logger.info(f"  AGI Init: {init_time:.3f}s, Neural Process: {process_time:.3f}s")
            
        except Exception as e:
            results["status"] = "ERROR"
            results["error"] = str(e)
            logger.error(f"❌ Performance test failed: {e}")
        
        return results
    
    def _run_integration_test(self) -> Dict[str, Any]:
        """Run integration tests between components."""
        logger.info("🔗 Running integration tests...")
        
        results = {"status": "PASS", "tests": [], "coverage": 0}
        
        try:
            # Test 1: AGI Engine + Security Framework integration
            test_result = self._test_agi_security_integration()
            results["tests"].append(test_result)
            
            # Test 2: Error Recovery integration
            test_result = self._test_error_recovery_integration()
            results["tests"].append(test_result)
            
            # Test 3: Quantum Accelerator integration
            test_result = self._test_quantum_integration()
            results["tests"].append(test_result)
            
            # Calculate coverage
            passed_tests = sum(1 for test in results["tests"] if test["status"] == "PASS")
            results["coverage"] = (passed_tests / len(results["tests"])) * 100
            
            if results["coverage"] < 50:
                results["status"] = "FAIL"
            elif results["coverage"] < 80:
                results["status"] = "WARN"
            
            logger.info(f"🔗 Integration tests: {results['status']} ({results['coverage']:.1f}% coverage)")
            
        except Exception as e:
            results["status"] = "ERROR"
            results["error"] = str(e)
            logger.error(f"❌ Integration test failed: {e}")
        
        return results
    
    def _test_agi_security_integration(self) -> Dict[str, Any]:
        """Test AGI Engine with Security Framework integration."""
        try:
            from src.adaptive_agi_engine import create_agi_engine
            from src.enterprise_security_framework import create_security_framework
            
            agi = create_agi_engine()
            security = create_security_framework()
            
            # Test security validation
            validation = security.validate_request(
                {"text": "Test integration"}, 
                user_id="test_user",
                source_ip="127.0.0.1"
            )
            
            assert validation["allowed"] == True
            assert "security_level" in validation
            
            return {"name": "AGI-Security Integration", "status": "PASS"}
            
        except Exception as e:
            return {"name": "AGI-Security Integration", "status": "FAIL", "error": str(e)}
    
    def _test_error_recovery_integration(self) -> Dict[str, Any]:
        """Test Error Recovery system integration."""
        try:
            from src.intelligent_error_recovery_v2 import create_error_recovery_system
            
            recovery = create_error_recovery_system()
            
            # Test error handling
            test_error = Exception("Integration test error")
            event = recovery.handle_error(test_error, {"test": True})
            
            assert event.error_type == "Exception"
            assert event.context["test"] == True
            
            return {"name": "Error Recovery Integration", "status": "PASS"}
            
        except Exception as e:
            return {"name": "Error Recovery Integration", "status": "FAIL", "error": str(e)}
    
    def _test_quantum_integration(self) -> Dict[str, Any]:
        """Test Quantum Accelerator integration."""
        try:
            from src.quantum_performance_accelerator import create_quantum_accelerator
            
            quantum = create_quantum_accelerator()
            
            # Test function optimization
            @quantum.optimize_function
            def test_function(x):
                return x * 2
            
            result = test_function(5)
            assert result == 10
            
            return {"name": "Quantum Accelerator Integration", "status": "PASS"}
            
        except Exception as e:
            return {"name": "Quantum Accelerator Integration", "status": "FAIL", "error": str(e)}
    
    def _run_code_quality_check(self) -> Dict[str, Any]:
        """Run code quality analysis."""
        logger.info("📊 Running code quality check...")
        
        results = {"status": "PASS", "metrics": {}, "quality_score": 100}
        
        try:
            # Count lines of code
            total_lines = 0
            total_files = 0
            empty_lines = 0
            comment_lines = 0
            
            for root, dirs, files in os.walk("src"):
                for file in files:
                    if file.endswith(".py"):
                        total_files += 1
                        file_path = os.path.join(root, file)
                        
                        with open(file_path, 'r') as f:
                            lines = f.readlines()
                            total_lines += len(lines)
                            
                            for line in lines:
                                stripped = line.strip()
                                if not stripped:
                                    empty_lines += 1
                                elif stripped.startswith('#'):
                                    comment_lines += 1
            
            code_lines = total_lines - empty_lines - comment_lines
            
            results["metrics"] = {
                "total_files": total_files,
                "total_lines": total_lines,
                "code_lines": code_lines,
                "comment_lines": comment_lines,
                "comment_ratio": (comment_lines / total_lines) * 100 if total_lines > 0 else 0,
                "avg_lines_per_file": total_lines / total_files if total_files > 0 else 0
            }
            
            # Quality scoring
            comment_ratio = results["metrics"]["comment_ratio"]
            avg_lines = results["metrics"]["avg_lines_per_file"]
            
            if comment_ratio < 5:  # Less than 5% comments
                results["quality_score"] -= 20
                results["status"] = "WARN"
            
            if avg_lines > 1000:  # Very large files
                results["quality_score"] -= 15
                results["status"] = "WARN"
            
            logger.info(f"📊 Code quality: {results['status']} (Score: {results['quality_score']}/100)")
            logger.info(f"  Files: {total_files}, Lines: {total_lines}, Comments: {comment_ratio:.1f}%")
            
        except Exception as e:
            results["status"] = "ERROR"
            results["error"] = str(e)
            logger.error(f"❌ Code quality check failed: {e}")
        
        return results
    
    def _calculate_overall_status(self):
        """Calculate overall quality gate status."""
        failed_gates = [name for name, result in self.results.items() 
                       if result.get("status") == "FAIL"]
        error_gates = [name for name, result in self.results.items() 
                      if result.get("status") == "ERROR"]
        warn_gates = [name for name, result in self.results.items() 
                     if result.get("status") == "WARN"]
        
        if error_gates or failed_gates:
            self.overall_status = "FAIL"
        elif len(warn_gates) > 2:  # Too many warnings
            self.overall_status = "WARN"
        else:
            self.overall_status = "PASS"
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate quality gate summary."""
        gate_statuses = {result.get("status", "UNKNOWN"): 0 for result in self.results.values()}
        for result in self.results.values():
            status = result.get("status", "UNKNOWN")
            gate_statuses[status] = gate_statuses.get(status, 0) + 1
        
        return {
            "total_gates": len(self.results),
            "passed": gate_statuses.get("PASS", 0),
            "warnings": gate_statuses.get("WARN", 0),
            "failed": gate_statuses.get("FAIL", 0),
            "errors": gate_statuses.get("ERROR", 0),
            "success_rate": (gate_statuses.get("PASS", 0) / len(self.results)) * 100
        }
    
    def _save_report(self, report: Dict[str, Any]):
        """Save quality gate report to file."""
        try:
            timestamp = int(time.time())
            filename = f"quality_gates_report_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📄 Report saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    def _print_summary(self, report: Dict[str, Any]):
        """Print quality gate summary."""
        print("\n" + "="*80)
        print("🎯 QUALITY GATES SUMMARY")
        print("="*80)
        
        summary = report["summary"]
        
        print(f"⏱️  Execution Time: {report['execution_time_seconds']:.2f} seconds")
        print(f"🎪 Overall Status: {report['overall_status']}")
        print(f"📊 Success Rate: {summary['success_rate']:.1f}%")
        print()
        
        print("📋 Gate Results:")
        for gate_name, result in report["gate_results"].items():
            status = result.get("status", "UNKNOWN")
            emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "ERROR": "💥"}.get(status, "❓")
            print(f"  {emoji} {gate_name.replace('_', ' ').title()}: {status}")
            
            # Show additional details for failed/error gates
            if status in ["FAIL", "ERROR"]:
                if "error" in result:
                    print(f"    Error: {result['error']}")
                if "errors" in result and result["errors"]:
                    print(f"    Issues: {len(result['errors'])}")
        
        print()
        
        # Performance metrics
        if "performance_test" in report["gate_results"]:
            perf = report["gate_results"]["performance_test"]
            if "benchmarks" in perf:
                print("⚡ Performance Benchmarks:")
                for metric, value in perf["benchmarks"].items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.3f}s")
                    else:
                        print(f"  {metric}: {value}")
                print()
        
        # Security metrics
        if "security_scan" in report["gate_results"]:
            sec = report["gate_results"]["security_scan"]
            print(f"🔒 Security Score: {sec.get('security_score', 0)}/100")
            if sec.get("vulnerabilities"):
                print(f"   Vulnerabilities: {len(sec['vulnerabilities'])}")
            print()
        
        # Code quality metrics
        if "code_quality" in report["gate_results"]:
            qual = report["gate_results"]["code_quality"]
            if "metrics" in qual:
                metrics = qual["metrics"]
                print("📊 Code Quality Metrics:")
                print(f"  Files: {metrics.get('total_files', 0)}")
                print(f"  Lines of Code: {metrics.get('code_lines', 0)}")
                print(f"  Comment Ratio: {metrics.get('comment_ratio', 0):.1f}%")
                print()
        
        print("="*80)
        
        if report['overall_status'] == "PASS":
            print("🎉 ALL QUALITY GATES PASSED! 🎉")
        elif report['overall_status'] == "WARN":
            print("⚠️  Quality gates passed with warnings")
        else:
            print("❌ Quality gates FAILED - Please review and fix issues")
        
        print("="*80)


def main():
    """Run quality gates."""
    runner = QualityGateRunner()
    
    try:
        report = runner.run_all_gates()
        
        # Exit with appropriate code
        if report['overall_status'] == "FAIL":
            sys.exit(1)
        elif report['overall_status'] == "WARN":
            sys.exit(2)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Quality gate runner failed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()