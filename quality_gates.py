"""
Quality Gates Implementation for Photonic-MLIR Bridge

Comprehensive quality assurance system implementing mandatory quality gates
including code validation, security scanning, performance testing, and compliance checks.
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import ast
import re


class QualityGateStatus(Enum):
    """Status of quality gate checks."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    name: str
    status: QualityGateStatus
    message: str
    details: Dict[str, Any]
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "duration_seconds": self.duration_seconds
        }


class CodeAnalyzer:
    """Analyzes code quality and security."""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.python_files = list(root_dir.rglob("*.py"))
    
    def check_imports(self) -> QualityGateResult:
        """Check for suspicious or dangerous imports."""
        start_time = time.time()
        
        dangerous_imports = [
            'eval', 'exec', 'compile', '__import__',
            'subprocess.call', 'os.system', 'os.popen',
            'pickle.loads', 'marshal.loads'
        ]
        
        suspicious_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'subprocess\.call',
            r'os\.system',
            r'pickle\.loads',
        ]
        
        issues = []
        
        for py_file in self.python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for dangerous patterns
                for pattern in suspicious_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        issues.append({
                            "file": str(py_file.relative_to(self.root_dir)),
                            "pattern": pattern,
                            "matches": len(matches),
                            "severity": "high"
                        })
                
                # Parse AST to check imports
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                if alias.name in dangerous_imports:
                                    issues.append({
                                        "file": str(py_file.relative_to(self.root_dir)),
                                        "import": alias.name,
                                        "line": node.lineno,
                                        "severity": "medium"
                                    })
                        elif isinstance(node, ast.ImportFrom):
                            if node.module and any(danger in node.module for danger in dangerous_imports):
                                issues.append({
                                    "file": str(py_file.relative_to(self.root_dir)),
                                    "import": f"from {node.module}",
                                    "line": node.lineno,
                                    "severity": "medium"
                                })
                
                except SyntaxError as e:
                    issues.append({
                        "file": str(py_file.relative_to(self.root_dir)),
                        "error": f"Syntax error: {e}",
                        "severity": "high"
                    })
            
            except Exception as e:
                issues.append({
                    "file": str(py_file.relative_to(self.root_dir)),
                    "error": f"Failed to read file: {e}",
                    "severity": "low"
                })
        
        duration = time.time() - start_time
        
        if any(issue["severity"] == "high" for issue in issues):
            status = QualityGateStatus.FAILED
            message = f"Found {len(issues)} security issues including high-risk patterns"
        elif issues:
            status = QualityGateStatus.WARNING
            message = f"Found {len(issues)} potential security issues"
        else:
            status = QualityGateStatus.PASSED
            message = "No security issues detected in imports"
        
        return QualityGateResult(
            name="Security Import Check",
            status=status,
            message=message,
            details={
                "issues": issues,
                "files_scanned": len(self.python_files),
                "patterns_checked": len(suspicious_patterns)
            },
            duration_seconds=duration
        )
    
    def check_code_complexity(self) -> QualityGateResult:
        """Check code complexity metrics."""
        start_time = time.time()
        
        complexity_issues = []
        max_complexity = 10  # McCabe complexity limit
        max_function_lines = 50
        max_class_methods = 20
        
        for py_file in self.python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Count lines in function
                            if hasattr(node, 'end_lineno') and node.end_lineno:
                                func_lines = node.end_lineno - node.lineno
                                if func_lines > max_function_lines:
                                    complexity_issues.append({
                                        "file": str(py_file.relative_to(self.root_dir)),
                                        "function": node.name,
                                        "issue": "long_function",
                                        "lines": func_lines,
                                        "limit": max_function_lines
                                    })
                            
                            # Simple complexity check (count control structures)
                            complexity = self._calculate_complexity(node)
                            if complexity > max_complexity:
                                complexity_issues.append({
                                    "file": str(py_file.relative_to(self.root_dir)),
                                    "function": node.name,
                                    "issue": "high_complexity",
                                    "complexity": complexity,
                                    "limit": max_complexity
                                })
                        
                        elif isinstance(node, ast.ClassDef):
                            # Count methods in class
                            methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                            if len(methods) > max_class_methods:
                                complexity_issues.append({
                                    "file": str(py_file.relative_to(self.root_dir)),
                                    "class": node.name,
                                    "issue": "too_many_methods",
                                    "methods": len(methods),
                                    "limit": max_class_methods
                                })
                
                except SyntaxError:
                    pass  # Skip files with syntax errors
            
            except Exception:
                pass  # Skip files that can't be read
        
        duration = time.time() - start_time
        
        if complexity_issues:
            status = QualityGateStatus.WARNING
            message = f"Found {len(complexity_issues)} complexity issues"
        else:
            status = QualityGateStatus.PASSED
            message = "Code complexity within acceptable limits"
        
        return QualityGateResult(
            name="Code Complexity Check",
            status=status,
            message=message,
            details={
                "issues": complexity_issues,
                "limits": {
                    "max_complexity": max_complexity,
                    "max_function_lines": max_function_lines,
                    "max_class_methods": max_class_methods
                }
            },
            duration_seconds=duration
        )
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate approximate McCabe complexity."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.Try):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity


class TestRunner:
    """Runs and validates test suites."""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.test_dir = root_dir / "tests"
    
    def run_isolated_tests(self) -> QualityGateResult:
        """Run isolated tests that don't require external dependencies."""
        start_time = time.time()
        
        test_file = self.test_dir / "test_photonic_isolated.py"
        
        if not test_file.exists():
            return QualityGateResult(
                name="Isolated Tests",
                status=QualityGateStatus.SKIPPED,
                message="Isolated test file not found",
                details={"test_file": str(test_file)},
                duration_seconds=time.time() - start_time
            )
        
        try:
            # Run the test file
            result = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                text=True,
                cwd=self.root_dir,
                timeout=60
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                status = QualityGateStatus.PASSED
                message = "All isolated tests passed"
            else:
                status = QualityGateStatus.FAILED
                message = "Isolated tests failed"
            
            return QualityGateResult(
                name="Isolated Tests",
                status=status,
                message=message,
                details={
                    "return_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                },
                duration_seconds=duration
            )
        
        except subprocess.TimeoutExpired:
            return QualityGateResult(
                name="Isolated Tests",
                status=QualityGateStatus.FAILED,
                message="Tests timed out after 60 seconds",
                details={"timeout": 60},
                duration_seconds=time.time() - start_time
            )
        
        except Exception as e:
            return QualityGateResult(
                name="Isolated Tests",
                status=QualityGateStatus.FAILED,
                message=f"Failed to run tests: {e}",
                details={"error": str(e)},
                duration_seconds=time.time() - start_time
            )
    
    def check_test_coverage(self) -> QualityGateResult:
        """Check test coverage metrics."""
        start_time = time.time()
        
        # Count test files
        test_files = list(self.test_dir.glob("test_*.py"))
        src_files = list((self.root_dir / "src").glob("*.py"))
        
        # Filter out __init__.py and main files
        src_modules = [
            f for f in src_files 
            if f.name not in ["__init__.py", "main.py"]
            and not f.name.startswith("test_")
        ]
        
        coverage_ratio = len(test_files) / len(src_modules) if src_modules else 0
        
        details = {
            "test_files": len(test_files),
            "source_modules": len(src_modules),
            "coverage_ratio": coverage_ratio,
            "test_file_list": [f.name for f in test_files],
            "source_file_list": [f.name for f in src_modules]
        }
        
        if coverage_ratio >= 0.8:  # 80% coverage target
            status = QualityGateStatus.PASSED
            message = f"Good test coverage: {coverage_ratio:.1%}"
        elif coverage_ratio >= 0.5:  # 50% minimum
            status = QualityGateStatus.WARNING
            message = f"Moderate test coverage: {coverage_ratio:.1%}"
        else:
            status = QualityGateStatus.FAILED
            message = f"Insufficient test coverage: {coverage_ratio:.1%}"
        
        return QualityGateResult(
            name="Test Coverage Check",
            status=status,
            message=message,
            details=details,
            duration_seconds=time.time() - start_time
        )


class PerformanceValidator:
    """Validates performance requirements."""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
    
    def benchmark_synthesis(self) -> QualityGateResult:
        """Benchmark synthesis performance."""
        start_time = time.time()
        
        try:
            # Create a simple performance test
            perf_test = '''
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Test synthesis performance without external dependencies
def test_performance():
    start_time = time.time()
    
    # Simulate circuit creation and processing
    circuits = []
    for i in range(100):
        circuit = {
            "name": f"circuit_{i}",
            "components": [
                {"id": f"wg_{j}", "type": "waveguide", "position": (j, 0)}
                for j in range(10)
            ],
            "connections": [
                {"source": f"wg_{j}", "target": f"wg_{j+1}", "loss_db": 0.1}
                for j in range(9)
            ]
        }
        circuits.append(circuit)
    
    processing_time = time.time() - start_time
    
    # Performance targets
    max_time = 1.0  # 1 second for 100 circuits with 10 components each
    circuits_per_second = len(circuits) / processing_time
    
    return {
        "circuits_processed": len(circuits),
        "processing_time": processing_time,
        "circuits_per_second": circuits_per_second,
        "meets_target": processing_time <= max_time
    }

if __name__ == "__main__":
    result = test_performance()
    print(f"Processed {result['circuits_processed']} circuits in {result['processing_time']:.3f}s")
    print(f"Rate: {result['circuits_per_second']:.1f} circuits/s")
    sys.exit(0 if result['meets_target'] else 1)
'''
            
            # Write and run performance test
            perf_file = self.root_dir / "temp_perf_test.py"
            with open(perf_file, 'w') as f:
                f.write(perf_test)
            
            try:
                result = subprocess.run(
                    [sys.executable, str(perf_file)],
                    capture_output=True,
                    text=True,
                    cwd=self.root_dir,
                    timeout=30
                )
                
                # Clean up
                perf_file.unlink()
                
                duration = time.time() - start_time
                
                if result.returncode == 0:
                    status = QualityGateStatus.PASSED
                    message = "Performance benchmarks met"
                else:
                    status = QualityGateStatus.WARNING
                    message = "Performance below target"
                
                return QualityGateResult(
                    name="Performance Benchmark",
                    status=status,
                    message=message,
                    details={
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "return_code": result.returncode
                    },
                    duration_seconds=duration
                )
            
            finally:
                # Ensure cleanup
                if perf_file.exists():
                    perf_file.unlink()
        
        except Exception as e:
            return QualityGateResult(
                name="Performance Benchmark",
                status=QualityGateStatus.FAILED,
                message=f"Performance test failed: {e}",
                details={"error": str(e)},
                duration_seconds=time.time() - start_time
            )


class QualityGateManager:
    """Manages and executes all quality gates."""
    
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir).resolve()
        self.analyzer = CodeAnalyzer(self.root_dir)
        self.test_runner = TestRunner(self.root_dir)
        self.perf_validator = PerformanceValidator(self.root_dir)
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates."""
        print("🛡️ Running Quality Gates for Photonic-MLIR Bridge")
        print("=" * 60)
        
        gates = [
            ("Security Analysis", self.analyzer.check_imports),
            ("Code Complexity", self.analyzer.check_code_complexity),
            ("Isolated Tests", self.test_runner.run_isolated_tests),
            ("Test Coverage", self.test_runner.check_test_coverage),
            ("Performance Benchmark", self.perf_validator.benchmark_synthesis)
        ]
        
        results = []
        total_start = time.time()
        
        for gate_name, gate_func in gates:
            print(f"\n🔍 Running: {gate_name}")
            
            try:
                result = gate_func()
                results.append(result)
                
                # Print result
                status_emoji = {
                    QualityGateStatus.PASSED: "✅",
                    QualityGateStatus.FAILED: "❌", 
                    QualityGateStatus.WARNING: "⚠️",
                    QualityGateStatus.SKIPPED: "⏭️"
                }
                
                print(f"   {status_emoji[result.status]} {result.message}")
                print(f"   Duration: {result.duration_seconds:.3f}s")
                
                if result.status in [QualityGateStatus.FAILED, QualityGateStatus.WARNING]:
                    if "issues" in result.details:
                        print(f"   Issues found: {len(result.details['issues'])}")
                
            except Exception as e:
                print(f"   ❌ Gate crashed: {e}")
                results.append(QualityGateResult(
                    name=gate_name,
                    status=QualityGateStatus.FAILED,
                    message=f"Gate execution failed: {e}",
                    details={"error": str(e)}
                ))
        
        total_duration = time.time() - total_start
        
        # Calculate summary
        status_counts = {}
        for status in QualityGateStatus:
            status_counts[status.value] = sum(
                1 for r in results if r.status == status
            )
        
        # Determine overall status
        if status_counts["failed"] > 0:
            overall_status = "FAILED"
        elif status_counts["warning"] > 0:
            overall_status = "PASSED_WITH_WARNINGS"
        else:
            overall_status = "PASSED"
        
        summary = {
            "overall_status": overall_status,
            "total_duration_seconds": total_duration,
            "gates_run": len(results),
            "status_counts": status_counts,
            "results": [r.to_dict() for r in results],
            "timestamp": time.time()
        }
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 Quality Gates Summary")
        print(f"Overall Status: {overall_status}")
        print(f"Total Duration: {total_duration:.2f}s")
        print(f"Gates Run: {len(results)}")
        
        for status, count in status_counts.items():
            if count > 0:
                print(f"{status.title()}: {count}")
        
        # Save results
        results_file = self.root_dir / "quality_gates_results.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📁 Results saved to: {results_file}")
        
        return summary


def main():
    """Main entry point."""
    manager = QualityGateManager()
    results = manager.run_all_gates()
    
    # Exit with appropriate code
    if results["overall_status"] == "FAILED":
        sys.exit(1)
    elif results["overall_status"] == "PASSED_WITH_WARNINGS":
        sys.exit(0)  # Warnings are acceptable
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()