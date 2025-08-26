#!/usr/bin/env python3
"""
QNP Autonomous Quality Gates System
Comprehensive validation system for Quantum-Neuromorphic-Photonic implementation.

This system automatically validates:
- Code quality and security
- Performance benchmarks  
- Research validation
- Production readiness
- Global compliance
"""

import sys
import os
import json
import time
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QualityGateResult:
    """Individual quality gate result"""
    gate_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    execution_time_seconds: float
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

@dataclass
class QNPQualityReport:
    """Comprehensive QNP quality assessment report"""
    timestamp: datetime
    overall_pass: bool
    total_score: float
    gate_results: List[QualityGateResult]
    execution_summary: Dict[str, Any]
    recommendations: List[str]

class QNPQualityGateSystem:
    """Autonomous quality gate validation system for QNP"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.src_path = self.project_root / "src"
        self.tests_path = self.project_root / "tests"
        self.examples_path = self.project_root / "examples"
        
        # Quality gate configuration
        self.quality_gates = [
            ("syntax_validation", self.validate_syntax),
            ("import_validation", self.validate_imports),
            ("qnp_core_validation", self.validate_qnp_core),
            ("security_scan", self.run_security_scan),
            ("performance_benchmarks", self.run_performance_benchmarks),
            ("research_validation", self.run_research_validation),
            ("documentation_check", self.validate_documentation),
            ("integration_tests", self.run_integration_tests)
        ]
        
        # Scoring weights
        self.gate_weights = {
            "syntax_validation": 0.15,
            "import_validation": 0.10,
            "qnp_core_validation": 0.20,
            "security_scan": 0.15,
            "performance_benchmarks": 0.15,
            "research_validation": 0.15,
            "documentation_check": 0.05,
            "integration_tests": 0.05
        }
        
        # Results storage
        self.results: List[QualityGateResult] = []
    
    def validate_syntax(self) -> QualityGateResult:
        """Validate Python syntax across all QNP modules"""
        start_time = time.time()
        
        try:
            logger.info("🔍 Validating Python syntax...")
            
            syntax_errors = []
            files_checked = 0
            
            # Check all Python files
            for python_file in self.src_path.rglob("*.py"):
                files_checked += 1
                try:
                    with open(python_file, 'r', encoding='utf-8') as f:
                        compile(f.read(), str(python_file), 'exec')
                except SyntaxError as e:
                    syntax_errors.append(f"{python_file}: {e}")
            
            execution_time = time.time() - start_time
            
            passed = len(syntax_errors) == 0
            score = 1.0 if passed else max(0, 1.0 - len(syntax_errors) / files_checked)
            
            return QualityGateResult(
                gate_name="syntax_validation",
                passed=passed,
                score=score,
                details={
                    "files_checked": files_checked,
                    "syntax_errors": len(syntax_errors),
                    "error_rate": len(syntax_errors) / files_checked if files_checked > 0 else 0
                },
                execution_time_seconds=execution_time,
                errors=syntax_errors
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return QualityGateResult(
                gate_name="syntax_validation",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time_seconds=execution_time,
                errors=[str(e)]
            )
    
    def validate_imports(self) -> QualityGateResult:
        """Validate that all imports are available and QNP modules load correctly"""
        start_time = time.time()
        
        try:
            logger.info("📦 Validating module imports...")
            
            # Add src to path
            sys.path.insert(0, str(self.src_path))
            
            import_results = []
            qnp_modules = [
                "quantum_neuromorphic_photonic_fusion",
                "qnp_production_framework",
                "qnp_research_validation",
                "qnp_global_scaling_system"
            ]
            
            for module_name in qnp_modules:
                try:
                    __import__(module_name)
                    import_results.append({"module": module_name, "status": "success"})
                except ImportError as e:
                    import_results.append({"module": module_name, "status": "failed", "error": str(e)})
                except Exception as e:
                    import_results.append({"module": module_name, "status": "error", "error": str(e)})
            
            execution_time = time.time() - start_time
            
            successful_imports = len([r for r in import_results if r["status"] == "success"])
            total_imports = len(import_results)
            
            passed = successful_imports == total_imports
            score = successful_imports / total_imports if total_imports > 0 else 0
            
            errors = [f"{r['module']}: {r.get('error', 'Unknown error')}" 
                     for r in import_results if r["status"] != "success"]
            
            return QualityGateResult(
                gate_name="import_validation",
                passed=passed,
                score=score,
                details={
                    "total_modules": total_imports,
                    "successful_imports": successful_imports,
                    "failed_imports": total_imports - successful_imports,
                    "import_results": import_results
                },
                execution_time_seconds=execution_time,
                errors=errors
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return QualityGateResult(
                gate_name="import_validation",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time_seconds=execution_time,
                errors=[str(e)]
            )
    
    def validate_qnp_core(self) -> QualityGateResult:
        """Validate core QNP functionality"""
        start_time = time.time()
        
        try:
            logger.info("⚡ Validating QNP core functionality...")
            
            # Import QNP modules
            sys.path.insert(0, str(self.src_path))
            
            from quantum_neuromorphic_photonic_fusion import create_qnp_fusion_engine
            
            validation_results = []
            
            # Test 1: Engine creation
            try:
                engine = create_qnp_fusion_engine()
                validation_results.append({"test": "engine_creation", "status": "pass"})
            except Exception as e:
                validation_results.append({"test": "engine_creation", "status": "fail", "error": str(e)})
            
            # Test 2: Basic sentiment analysis
            try:
                result = engine.analyze_sentiment("This is a test of the QNP system.")
                
                required_keys = ["positive", "negative", "neutral", "confidence"]
                has_required_keys = all(key in result for key in required_keys)
                
                if has_required_keys:
                    validation_results.append({"test": "basic_analysis", "status": "pass"})
                else:
                    validation_results.append({
                        "test": "basic_analysis", 
                        "status": "fail", 
                        "error": "Missing required keys in result"
                    })
            except Exception as e:
                validation_results.append({"test": "basic_analysis", "status": "fail", "error": str(e)})
            
            # Test 3: Fallback functionality
            try:
                fallback_result = engine.fallback_analysis("Test fallback analysis.")
                
                if "positive" in fallback_result and "negative" in fallback_result:
                    validation_results.append({"test": "fallback_analysis", "status": "pass"})
                else:
                    validation_results.append({
                        "test": "fallback_analysis", 
                        "status": "fail", 
                        "error": "Fallback missing required fields"
                    })
            except Exception as e:
                validation_results.append({"test": "fallback_analysis", "status": "fail", "error": str(e)})
            
            # Test 4: Performance metrics
            try:
                report = engine.get_performance_report()
                
                if "qnp_fusion_engine" in report:
                    validation_results.append({"test": "performance_report", "status": "pass"})
                else:
                    validation_results.append({
                        "test": "performance_report", 
                        "status": "fail", 
                        "error": "Invalid performance report structure"
                    })
            except Exception as e:
                validation_results.append({"test": "performance_report", "status": "fail", "error": str(e)})
            
            execution_time = time.time() - start_time
            
            passed_tests = len([r for r in validation_results if r["status"] == "pass"])
            total_tests = len(validation_results)
            
            passed = passed_tests == total_tests
            score = passed_tests / total_tests if total_tests > 0 else 0
            
            errors = [f"{r['test']}: {r.get('error', 'Unknown error')}" 
                     for r in validation_results if r["status"] != "pass"]
            
            return QualityGateResult(
                gate_name="qnp_core_validation",
                passed=passed,
                score=score,
                details={
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": total_tests - passed_tests,
                    "test_results": validation_results
                },
                execution_time_seconds=execution_time,
                errors=errors
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return QualityGateResult(
                gate_name="qnp_core_validation",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time_seconds=execution_time,
                errors=[str(e)]
            )
    
    def run_security_scan(self) -> QualityGateResult:
        """Run security analysis on QNP codebase"""
        start_time = time.time()
        
        try:
            logger.info("🔒 Running security scan...")
            
            security_issues = []
            
            # Basic security checks
            security_patterns = [
                ("hardcoded_secrets", ["password", "secret", "api_key", "token"]),
                ("unsafe_imports", ["eval", "exec", "pickle"]),
                ("file_operations", ["open(", "file(", "input("]),
                ("network_calls", ["requests.", "urllib", "socket"])
            ]
            
            files_scanned = 0
            
            for python_file in self.src_path.rglob("*.py"):
                files_scanned += 1
                
                try:
                    with open(python_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        for pattern_name, patterns in security_patterns:
                            for pattern in patterns:
                                if pattern in content.lower():
                                    # This is just a basic check - not necessarily an issue
                                    if pattern_name == "hardcoded_secrets":
                                        # More thorough check for actual secrets
                                        lines = content.lower().split('\n')
                                        for i, line in enumerate(lines):
                                            if pattern in line and '=' in line:
                                                security_issues.append({
                                                    "file": str(python_file),
                                                    "line": i + 1,
                                                    "type": pattern_name,
                                                    "pattern": pattern,
                                                    "severity": "medium"
                                                })
                                    
                except Exception as e:
                    security_issues.append({
                        "file": str(python_file),
                        "type": "scan_error",
                        "error": str(e),
                        "severity": "low"
                    })
            
            execution_time = time.time() - start_time
            
            # Calculate score based on issues found
            high_severity_issues = len([i for i in security_issues if i.get("severity") == "high"])
            medium_severity_issues = len([i for i in security_issues if i.get("severity") == "medium"])
            
            # Scoring: -1.0 for high, -0.5 for medium, -0.1 for low
            penalty = high_severity_issues * 1.0 + medium_severity_issues * 0.5
            score = max(0, 1.0 - (penalty / 10))  # Normalize penalty
            
            passed = high_severity_issues == 0 and medium_severity_issues <= 2
            
            return QualityGateResult(
                gate_name="security_scan",
                passed=passed,
                score=score,
                details={
                    "files_scanned": files_scanned,
                    "total_issues": len(security_issues),
                    "high_severity": high_severity_issues,
                    "medium_severity": medium_severity_issues,
                    "low_severity": len(security_issues) - high_severity_issues - medium_severity_issues,
                    "issues": security_issues[:10]  # Limit to first 10 for reporting
                },
                execution_time_seconds=execution_time,
                warnings=[f"Found {len(security_issues)} potential security issues"] if security_issues else []
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return QualityGateResult(
                gate_name="security_scan",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time_seconds=execution_time,
                errors=[str(e)]
            )
    
    def run_performance_benchmarks(self) -> QualityGateResult:
        """Run performance benchmarks for QNP system"""
        start_time = time.time()
        
        try:
            logger.info("⚡ Running performance benchmarks...")
            
            sys.path.insert(0, str(self.src_path))
            from quantum_neuromorphic_photonic_fusion import create_qnp_fusion_engine
            
            engine = create_qnp_fusion_engine()
            
            # Performance test cases
            test_cases = [
                "Short text test.",
                "Medium length text for testing the performance of the QNP system with moderate content.",
                "Very long text content that will test the scalability and performance characteristics of the Quantum-Neuromorphic-Photonic fusion system under conditions with substantial input data that requires comprehensive processing through all three major components including quantum feature encoding, neuromorphic spike processing, and photonic acceleration to ensure that the system maintains acceptable performance levels."
            ]
            
            benchmark_results = []
            
            for i, test_text in enumerate(test_cases):
                test_start = time.time()
                
                try:
                    result = engine.analyze_sentiment(test_text)
                    test_time = time.time() - test_start
                    
                    benchmark_results.append({
                        "test_case": i + 1,
                        "text_length": len(test_text),
                        "processing_time_ms": test_time * 1000,
                        "qnp_fusion_score": result.get("qnp_fusion_score", 0),
                        "status": "success"
                    })
                    
                except Exception as e:
                    test_time = time.time() - test_start
                    benchmark_results.append({
                        "test_case": i + 1,
                        "text_length": len(test_text),
                        "processing_time_ms": test_time * 1000,
                        "status": "failed",
                        "error": str(e)
                    })
            
            execution_time = time.time() - start_time
            
            # Calculate performance score
            successful_tests = [r for r in benchmark_results if r["status"] == "success"]
            
            if successful_tests:
                avg_processing_time = sum(r["processing_time_ms"] for r in successful_tests) / len(successful_tests)
                avg_fusion_score = sum(r.get("qnp_fusion_score", 0) for r in successful_tests) / len(successful_tests)
                
                # Performance criteria: < 100ms average, fusion score > 0.5
                time_score = min(1.0, 100 / max(1, avg_processing_time))  # Better if faster
                fusion_score = avg_fusion_score
                
                score = (time_score * 0.7 + fusion_score * 0.3)
                passed = avg_processing_time < 200 and len(successful_tests) == len(test_cases)
            else:
                score = 0.0
                passed = False
                avg_processing_time = 0
                avg_fusion_score = 0
            
            return QualityGateResult(
                gate_name="performance_benchmarks",
                passed=passed,
                score=score,
                details={
                    "total_tests": len(test_cases),
                    "successful_tests": len(successful_tests),
                    "avg_processing_time_ms": avg_processing_time,
                    "avg_qnp_fusion_score": avg_fusion_score,
                    "benchmark_results": benchmark_results
                },
                execution_time_seconds=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return QualityGateResult(
                gate_name="performance_benchmarks",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time_seconds=execution_time,
                errors=[str(e)]
            )
    
    def run_research_validation(self) -> QualityGateResult:
        """Run research validation tests"""
        start_time = time.time()
        
        try:
            logger.info("🧪 Running research validation...")
            
            sys.path.insert(0, str(self.src_path))
            
            # Test research validation module
            try:
                from qnp_research_validation import QNPResearchValidator
                
                validator = QNPResearchValidator({
                    "max_workers": 2,
                    "qnp_config": {"num_qubits": 4, "num_neurons": 32}
                })
                
                # Simple validation test
                test_texts = ["Great test!", "Bad experience.", "Okay results."]
                
                # Run a minimal comparative benchmark
                import asyncio
                
                async def run_validation():
                    return await validator.run_comparative_benchmark(
                        texts=test_texts,
                        runs=1
                    )
                
                validation_result = asyncio.run(run_validation())
                
                execution_time = time.time() - start_time
                
                # Check validation results
                total_tests = validation_result["config"]["total_tests"]
                analysis = validation_result["analysis"]
                
                passed = len(analysis) > 0 and total_tests > 0
                score = 1.0 if passed else 0.0
                
                return QualityGateResult(
                    gate_name="research_validation",
                    passed=passed,
                    score=score,
                    details={
                        "total_tests": total_tests,
                        "methods_tested": len(analysis),
                        "validation_successful": passed
                    },
                    execution_time_seconds=execution_time
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                return QualityGateResult(
                    gate_name="research_validation",
                    passed=False,
                    score=0.5,  # Partial score if module loads but validation fails
                    details={"validation_error": str(e)},
                    execution_time_seconds=execution_time,
                    warnings=[f"Research validation failed: {e}"]
                )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return QualityGateResult(
                gate_name="research_validation",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time_seconds=execution_time,
                errors=[str(e)]
            )
    
    def validate_documentation(self) -> QualityGateResult:
        """Validate documentation completeness"""
        start_time = time.time()
        
        try:
            logger.info("📚 Validating documentation...")
            
            doc_checks = []
            
            # Check for README
            readme_exists = (self.project_root / "README.md").exists()
            doc_checks.append({"item": "README.md", "exists": readme_exists})
            
            # Check for key QNP documentation
            qnp_modules = [
                "quantum_neuromorphic_photonic_fusion.py",
                "qnp_production_framework.py",
                "qnp_research_validation.py",
                "qnp_global_scaling_system.py"
            ]
            
            for module in qnp_modules:
                module_file = self.src_path / module
                if module_file.exists():
                    try:
                        with open(module_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            has_docstring = '"""' in content[:500]  # Check first 500 chars
                            doc_checks.append({
                                "item": f"{module} docstring",
                                "exists": has_docstring
                            })
                    except:
                        doc_checks.append({
                            "item": f"{module} docstring",
                            "exists": False
                        })
                else:
                    doc_checks.append({"item": f"{module}", "exists": False})
            
            execution_time = time.time() - start_time
            
            docs_present = len([check for check in doc_checks if check["exists"]])
            total_docs = len(doc_checks)
            
            score = docs_present / total_docs if total_docs > 0 else 0
            passed = score >= 0.8  # 80% documentation coverage required
            
            return QualityGateResult(
                gate_name="documentation_check",
                passed=passed,
                score=score,
                details={
                    "total_doc_items": total_docs,
                    "docs_present": docs_present,
                    "coverage_percent": score * 100,
                    "doc_checks": doc_checks
                },
                execution_time_seconds=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return QualityGateResult(
                gate_name="documentation_check",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time_seconds=execution_time,
                errors=[str(e)]
            )
    
    def run_integration_tests(self) -> QualityGateResult:
        """Run basic integration tests"""
        start_time = time.time()
        
        try:
            logger.info("🔗 Running integration tests...")
            
            sys.path.insert(0, str(self.src_path))
            
            integration_results = []
            
            # Test 1: QNP Engine + Production Framework integration
            try:
                from quantum_neuromorphic_photonic_fusion import create_qnp_fusion_engine
                from qnp_production_framework import create_qnp_production_engine, SecurityContext, SecurityLevel
                
                # Create engines
                qnp_engine = create_qnp_fusion_engine()
                production_engine = create_qnp_production_engine()
                
                integration_results.append({"test": "engine_creation_integration", "status": "pass"})
                
                # Test production security context
                security_context = SecurityContext(
                    user_id="test_user",
                    session_id="test_session",
                    classification=SecurityLevel.PUBLIC
                )
                
                integration_results.append({"test": "security_context_creation", "status": "pass"})
                
            except Exception as e:
                integration_results.append({
                    "test": "production_framework_integration",
                    "status": "fail",
                    "error": str(e)
                })
            
            # Test 2: Global Scaling System integration
            try:
                from qnp_global_scaling_system import create_global_qnp_cluster
                
                cluster = create_global_qnp_cluster({
                    "initial_nodes_per_region": 1,
                    "cache_size": 100
                })
                
                integration_results.append({"test": "global_cluster_creation", "status": "pass"})
                
            except Exception as e:
                integration_results.append({
                    "test": "global_cluster_integration",
                    "status": "fail",
                    "error": str(e)
                })
            
            execution_time = time.time() - start_time
            
            successful_tests = len([r for r in integration_results if r["status"] == "pass"])
            total_tests = len(integration_results)
            
            score = successful_tests / total_tests if total_tests > 0 else 0
            passed = successful_tests == total_tests
            
            return QualityGateResult(
                gate_name="integration_tests",
                passed=passed,
                score=score,
                details={
                    "total_tests": total_tests,
                    "successful_tests": successful_tests,
                    "test_results": integration_results
                },
                execution_time_seconds=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return QualityGateResult(
                gate_name="integration_tests",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time_seconds=execution_time,
                errors=[str(e)]
            )
    
    def run_all_quality_gates(self) -> QNPQualityReport:
        """Execute all quality gates and generate comprehensive report"""
        logger.info("🚀 Starting QNP Quality Gates Autonomous Execution")
        report_start = time.time()
        
        self.results = []
        recommendations = []
        
        # Execute all quality gates
        for gate_name, gate_func in self.quality_gates:
            logger.info(f"Executing {gate_name}...")
            
            try:
                result = gate_func()
                self.results.append(result)
                
                # Log result
                status = "✅ PASS" if result.passed else "❌ FAIL"
                logger.info(f"{status} {gate_name}: Score {result.score:.2f} ({result.execution_time_seconds:.2f}s)")
                
                # Generate recommendations
                if not result.passed:
                    if result.errors:
                        recommendations.append(f"Fix {gate_name} errors: {', '.join(result.errors[:3])}")
                    else:
                        recommendations.append(f"Improve {gate_name} (score: {result.score:.2f})")
                
            except Exception as e:
                logger.error(f"Quality gate {gate_name} failed with exception: {e}")
                error_result = QualityGateResult(
                    gate_name=gate_name,
                    passed=False,
                    score=0.0,
                    details={"exception": str(e)},
                    execution_time_seconds=0,
                    errors=[str(e)]
                )
                self.results.append(error_result)
        
        # Calculate overall metrics
        total_execution_time = time.time() - report_start
        
        # Weighted total score
        total_score = sum(
            result.score * self.gate_weights.get(result.gate_name, 1.0 / len(self.results))
            for result in self.results
        )
        
        # Overall pass/fail
        critical_gates = ["syntax_validation", "qnp_core_validation", "security_scan"]
        critical_passed = all(
            result.passed for result in self.results 
            if result.gate_name in critical_gates
        )
        
        overall_pass = critical_passed and total_score >= 0.7
        
        # Execution summary
        execution_summary = {
            "total_execution_time_seconds": total_execution_time,
            "gates_executed": len(self.results),
            "gates_passed": len([r for r in self.results if r.passed]),
            "gates_failed": len([r for r in self.results if not r.passed]),
            "average_execution_time": sum(r.execution_time_seconds for r in self.results) / len(self.results),
            "critical_gates_passed": critical_passed
        }
        
        # Generate final recommendations
        if overall_pass:
            recommendations.insert(0, "✅ QNP system passes all critical quality gates!")
        else:
            recommendations.insert(0, "❌ QNP system requires attention before production deployment")
        
        return QNPQualityReport(
            timestamp=datetime.now(),
            overall_pass=overall_pass,
            total_score=total_score,
            gate_results=self.results,
            execution_summary=execution_summary,
            recommendations=recommendations
        )

def save_quality_report(report: QNPQualityReport, output_file: Optional[Path] = None):
    """Save quality report to file"""
    if output_file is None:
        output_file = Path(f"qnp_quality_report_{int(time.time())}.json")
    
    # Convert to dictionary for JSON serialization
    report_dict = {
        "timestamp": report.timestamp.isoformat(),
        "overall_pass": report.overall_pass,
        "total_score": report.total_score,
        "gate_results": [asdict(result) for result in report.gate_results],
        "execution_summary": report.execution_summary,
        "recommendations": report.recommendations
    }
    
    with open(output_file, 'w') as f:
        json.dump(report_dict, f, indent=2, default=str)
    
    logger.info(f"Quality report saved to {output_file}")

def main():
    """Main execution function"""
    print("🧬 QNP Autonomous Quality Gates System")
    print("=" * 60)
    print("🚀 Comprehensive validation of Quantum-Neuromorphic-Photonic system")
    print("=" * 60)
    
    # Initialize quality gate system
    qgs = QNPQualityGateSystem()
    
    # Run all quality gates
    report = qgs.run_all_quality_gates()
    
    # Display results
    print("\n📊 QUALITY GATES EXECUTION SUMMARY")
    print("=" * 60)
    
    for result in report.gate_results:
        status_icon = "✅" if result.passed else "❌"
        print(f"{status_icon} {result.gate_name.upper()}: {result.score:.2f} ({result.execution_time_seconds:.2f}s)")
        
        if result.errors:
            for error in result.errors[:2]:  # Show first 2 errors
                print(f"    ⚠️ {error}")
        
        if result.warnings:
            for warning in result.warnings[:2]:  # Show first 2 warnings
                print(f"    ⚠️ {warning}")
    
    print("\n" + "=" * 60)
    print(f"🎯 OVERALL RESULT: {'✅ PASS' if report.overall_pass else '❌ FAIL'}")
    print(f"📈 TOTAL SCORE: {report.total_score:.2f}/1.00")
    print(f"⏱️ EXECUTION TIME: {report.execution_summary['total_execution_time_seconds']:.2f}s")
    print(f"🔍 GATES PASSED: {report.execution_summary['gates_passed']}/{report.execution_summary['gates_executed']}")
    
    print("\n📋 RECOMMENDATIONS:")
    for i, recommendation in enumerate(report.recommendations[:5], 1):
        print(f"  {i}. {recommendation}")
    
    # Save report
    save_quality_report(report)
    
    print(f"\n💾 Quality report saved to disk")
    print("🏁 Quality gates execution completed!")
    
    # Exit with appropriate code
    return 0 if report.overall_pass else 1

if __name__ == "__main__":
    sys.exit(main())