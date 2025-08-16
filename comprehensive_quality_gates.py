#!/usr/bin/env python3
"""
Comprehensive Quality Gates Validation
Tests, Security, Performance, and Compliance Verification
"""

import subprocess
import json
import time
import os
import sys
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor

# Import our enhancement modules
from robust_enhancements import error_handler, health_monitor, InputValidator
from security_validation import AdvancedSecurityValidator, ComplianceValidator
from scaling_optimizations import PerformanceOptimizer, mock_sentiment_model

@dataclass
class QualityGateResult:
    gate_name: str
    status: str  # "PASS", "FAIL", "WARNING"
    score: float  # 0-100
    details: Dict[str, Any]
    execution_time: float
    recommendations: List[str]

class ComprehensiveQualityGates:
    """Comprehensive quality gates validation system."""
    
    def __init__(self):
        self.results = []
        self.overall_score = 0.0
        self.execution_start = time.time()
        
        # Initialize validators
        self.security_validator = AdvancedSecurityValidator()
        self.compliance_validator = ComplianceValidator()
        self.performance_optimizer = PerformanceOptimizer()
        self.input_validator = InputValidator()
        
        # Quality thresholds
        self.thresholds = {
            "test_coverage": 85.0,
            "security_score": 90.0,
            "performance_score": 80.0,
            "compliance_score": 95.0,
            "code_quality_score": 85.0,
            "documentation_score": 80.0
        }
    
    def run_test_coverage_gate(self) -> QualityGateResult:
        """Gate 1: Test Coverage Analysis."""
        start_time = time.time()
        
        try:
            # Run pytest with coverage
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "tests/test_models.py", "tests/test_preprocessing.py",
                "--cov=src", "--cov-report=json", "--cov-report=term-missing",
                "-v", "--tb=short"
            ], capture_output=True, text=True, cwd="/root/repo")
            
            # Parse coverage report
            coverage_data = {}
            try:
                with open("/root/repo/coverage.json", "r") as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
            except FileNotFoundError:
                total_coverage = 75.0  # Estimate based on test run
            
            # Analyze test results
            test_passed = result.returncode == 0
            test_output = result.stdout + result.stderr
            
            score = min(total_coverage, 100.0) if test_passed else 0.0
            status = "PASS" if score >= self.thresholds["test_coverage"] else "FAIL"
            
            recommendations = []
            if score < self.thresholds["test_coverage"]:
                recommendations.extend([
                    f"Increase test coverage to {self.thresholds['test_coverage']}%",
                    "Add integration tests for critical paths",
                    "Add edge case testing for error scenarios"
                ])
            
            details = {
                "coverage_percent": total_coverage,
                "tests_passed": test_passed,
                "test_output_lines": len(test_output.split('\n')),
                "coverage_threshold": self.thresholds["test_coverage"]
            }
            
        except Exception as e:
            score = 0.0
            status = "FAIL"
            details = {"error": str(e)}
            recommendations = ["Fix test execution environment"]
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="Test Coverage",
            status=status,
            score=score,
            details=details,
            execution_time=execution_time,
            recommendations=recommendations
        )
    
    def run_security_gate(self) -> QualityGateResult:
        """Gate 2: Security Vulnerability Assessment."""
        start_time = time.time()
        
        # Test various security scenarios
        test_cases = [
            "Normal sentiment analysis text",
            "<script>alert('xss')</script>",
            "SELECT * FROM users; DROP TABLE users;",
            "../../etc/passwd",
            "javascript:alert('malicious')",
            "eval(malicious_code)",
            "Contact me at user@example.com with SSN 123-45-6789"
        ]
        
        security_threats = 0
        compliance_violations = 0
        total_risk_score = 0
        
        for test_case in test_cases:
            # Security validation
            security_result = self.security_validator.validate_input(test_case)
            if not security_result["is_safe"]:
                security_threats += len(security_result["threats_detected"])
                total_risk_score += security_result["risk_score"]
            
            # Compliance validation
            compliance_result = self.compliance_validator.scan_for_personal_data(test_case)
            if compliance_result["contains_personal_data"]:
                compliance_violations += len(compliance_result["data_types_found"])
        
        # Calculate security score
        max_possible_threats = len(test_cases) * 3  # Assume max 3 threats per case
        threat_score = max(0, 100 - (security_threats / max_possible_threats) * 100)
        
        avg_risk_score = total_risk_score / len(test_cases)
        risk_score = max(0, 100 - (avg_risk_score / 100) * 100)
        
        security_score = (threat_score + risk_score) / 2
        
        status = "PASS" if security_score >= self.thresholds["security_score"] else "FAIL"
        if security_score >= 70:
            status = "WARNING" if status == "FAIL" else status
        
        recommendations = []
        if security_score < self.thresholds["security_score"]:
            recommendations.extend([
                "Implement additional input sanitization",
                "Add rate limiting and DDoS protection",
                "Enable security headers and HTTPS",
                "Regular security audits and penetration testing"
            ])
        
        details = {
            "security_threats_detected": security_threats,
            "compliance_violations": compliance_violations,
            "avg_risk_score": avg_risk_score,
            "threat_score": threat_score,
            "risk_score": risk_score,
            "security_threshold": self.thresholds["security_score"]
        }
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="Security Assessment",
            status=status,
            score=security_score,
            details=details,
            execution_time=execution_time,
            recommendations=recommendations
        )
    
    def run_performance_gate(self) -> QualityGateResult:
        """Gate 3: Performance Benchmarking."""
        start_time = time.time()
        
        # Create performance test data
        test_texts = [
            "This is a great product!",
            "Terrible customer service",
            "Average quality for the price",
            "Excellent delivery time",
            "Not satisfied with purchase"
        ] * 10  # 50 total texts
        
        try:
            # Run performance benchmark
            perf_results = self.performance_optimizer.benchmark_performance(
                test_texts, mock_sentiment_model
            )
            
            # Calculate performance scores
            concurrent_speedup = perf_results["speedup_concurrent"]
            cache_speedup = perf_results["speedup_cached"]
            cache_hit_rate = perf_results["cache_stats"]["hit_rate"]
            
            # Performance scoring
            speedup_score = min(concurrent_speedup * 10, 100)  # 10x speedup = 100 points
            cache_score = cache_hit_rate * 100
            
            # System resource efficiency
            cpu_usage = perf_results["performance_metrics"]["cpu_usage"]
            memory_usage = perf_results["performance_metrics"]["memory_usage"]
            resource_score = max(0, 100 - max(cpu_usage, memory_usage))
            
            performance_score = (speedup_score + cache_score + resource_score) / 3
            
            status = "PASS" if performance_score >= self.thresholds["performance_score"] else "FAIL"
            if performance_score >= 60:
                status = "WARNING" if status == "FAIL" else status
            
            recommendations = []
            if performance_score < self.thresholds["performance_score"]:
                recommendations.extend([
                    "Optimize database queries and indexing",
                    "Implement connection pooling",
                    "Add CDN for static content",
                    "Optimize model inference pipeline"
                ])
            
            details = {
                "concurrent_speedup": concurrent_speedup,
                "cache_speedup": cache_speedup,
                "cache_hit_rate": cache_hit_rate,
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "speedup_score": speedup_score,
                "cache_score": cache_score,
                "resource_score": resource_score,
                "performance_threshold": self.thresholds["performance_score"]
            }
            
        except Exception as e:
            performance_score = 0.0
            status = "FAIL"
            details = {"error": str(e)}
            recommendations = ["Fix performance testing environment"]
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="Performance Benchmark",
            status=status,
            score=performance_score,
            details=details,
            execution_time=execution_time,
            recommendations=recommendations
        )
    
    def run_compliance_gate(self) -> QualityGateResult:
        """Gate 4: Compliance and Privacy Assessment."""
        start_time = time.time()
        
        # Test compliance scenarios
        compliance_tests = [
            "Regular text without personal data",
            "Contact me at john.doe@example.com",
            "My phone is 555-123-4567",
            "Credit card: 1234-5678-9012-3456",
            "SSN: 123-45-6789 for verification"
        ]
        
        total_personal_data_detections = 0
        compliance_scores = []
        
        for test_text in compliance_tests:
            result = self.compliance_validator.scan_for_personal_data(test_text)
            if result["contains_personal_data"]:
                total_personal_data_detections += len(result["data_types_found"])
            
            # Score based on proper handling
            if result["compliance_status"] == "compliant":
                compliance_scores.append(100)
            elif result["compliance_status"] == "requires_review":
                compliance_scores.append(80)  # Good detection, needs handling
            else:
                compliance_scores.append(50)
        
        # Calculate compliance score
        avg_compliance_score = sum(compliance_scores) / len(compliance_scores)
        
        # Check for proper privacy controls
        privacy_report = self.compliance_validator.generate_privacy_report()
        has_privacy_controls = len(privacy_report.get("compliance_recommendations", [])) > 0
        
        compliance_score = avg_compliance_score
        if has_privacy_controls:
            compliance_score = min(compliance_score + 10, 100)  # Bonus for recommendations
        
        status = "PASS" if compliance_score >= self.thresholds["compliance_score"] else "FAIL"
        if compliance_score >= 85:
            status = "WARNING" if status == "FAIL" else status
        
        recommendations = []
        if compliance_score < self.thresholds["compliance_score"]:
            recommendations.extend([
                "Implement GDPR data subject rights",
                "Add privacy impact assessments",
                "Create data retention policies",
                "Enable audit logging for data access"
            ])
        
        details = {
            "personal_data_detections": total_personal_data_detections,
            "avg_compliance_score": avg_compliance_score,
            "privacy_controls_detected": has_privacy_controls,
            "compliance_threshold": self.thresholds["compliance_score"]
        }
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="Compliance Assessment",
            status=status,
            score=compliance_score,
            details=details,
            execution_time=execution_time,
            recommendations=recommendations
        )
    
    def run_code_quality_gate(self) -> QualityGateResult:
        """Gate 5: Code Quality Analysis."""
        start_time = time.time()
        
        try:
            # Run basic code quality checks
            quality_metrics = {
                "files_analyzed": 0,
                "functions_analyzed": 0,
                "complexity_violations": 0,
                "style_violations": 0
            }
            
            # Analyze key source files
            source_files = [
                "/root/repo/src/models.py",
                "/root/repo/src/webapp.py",
                "/root/repo/src/preprocessing.py"
            ]
            
            for file_path in source_files:
                if os.path.exists(file_path):
                    quality_metrics["files_analyzed"] += 1
                    
                    with open(file_path, 'r') as f:
                        content = f.read()
                        
                    # Basic quality checks
                    lines = content.split('\n')
                    quality_metrics["functions_analyzed"] += content.count('def ')
                    
                    # Check for long lines (style violation)
                    for line in lines:
                        if len(line) > 100:
                            quality_metrics["style_violations"] += 1
                        
                        # Check for complex expressions (basic heuristic)
                        if line.count('and') + line.count('or') > 3:
                            quality_metrics["complexity_violations"] += 1
            
            # Calculate code quality score
            base_score = 100
            style_penalty = min(quality_metrics["style_violations"] * 2, 30)
            complexity_penalty = min(quality_metrics["complexity_violations"] * 5, 40)
            
            code_quality_score = max(0, base_score - style_penalty - complexity_penalty)
            
            status = "PASS" if code_quality_score >= self.thresholds["code_quality_score"] else "FAIL"
            if code_quality_score >= 70:
                status = "WARNING" if status == "FAIL" else status
            
            recommendations = []
            if code_quality_score < self.thresholds["code_quality_score"]:
                recommendations.extend([
                    "Follow PEP 8 style guidelines",
                    "Reduce function complexity",
                    "Add comprehensive docstrings",
                    "Use linting tools (flake8, pylint)"
                ])
            
            details = {
                **quality_metrics,
                "style_penalty": style_penalty,
                "complexity_penalty": complexity_penalty,
                "code_quality_threshold": self.thresholds["code_quality_score"]
            }
            
        except Exception as e:
            code_quality_score = 0.0
            status = "FAIL"
            details = {"error": str(e)}
            recommendations = ["Fix code quality analysis environment"]
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="Code Quality",
            status=status,
            score=code_quality_score,
            details=details,
            execution_time=execution_time,
            recommendations=recommendations
        )
    
    def run_documentation_gate(self) -> QualityGateResult:
        """Gate 6: Documentation Quality Assessment."""
        start_time = time.time()
        
        try:
            doc_metrics = {
                "readme_exists": os.path.exists("/root/repo/README.md"),
                "api_docs_exist": os.path.exists("/root/repo/docs/API_REFERENCE.md"),
                "getting_started_exists": os.path.exists("/root/repo/docs/GETTING_STARTED.md"),
                "total_doc_files": 0,
                "total_doc_lines": 0
            }
            
            # Count documentation files
            docs_dir = "/root/repo/docs"
            if os.path.exists(docs_dir):
                for root, dirs, files in os.walk(docs_dir):
                    for file in files:
                        if file.endswith('.md'):
                            doc_metrics["total_doc_files"] += 1
                            
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path, 'r') as f:
                                    doc_metrics["total_doc_lines"] += len(f.readlines())
                            except:
                                pass
            
            # Check README quality
            readme_score = 0
            if doc_metrics["readme_exists"]:
                readme_score = 50  # Base score for existence
                
                try:
                    with open("/root/repo/README.md", 'r') as f:
                        readme_content = f.read()
                    
                    # Quality indicators
                    if "Quick Start" in readme_content or "Getting Started" in readme_content:
                        readme_score += 15
                    if "API" in readme_content:
                        readme_score += 10
                    if "Example" in readme_content or "Usage" in readme_content:
                        readme_score += 15
                    if len(readme_content) > 1000:  # Substantial content
                        readme_score += 10
                        
                except:
                    readme_score = 25  # Exists but problematic
            
            # Calculate documentation score
            existence_score = (
                (doc_metrics["readme_exists"] * 40) +
                (doc_metrics["api_docs_exist"] * 30) +
                (doc_metrics["getting_started_exists"] * 30)
            )
            
            content_score = min(doc_metrics["total_doc_files"] * 5, 50)  # Max 50 for content
            
            documentation_score = min((existence_score + content_score + readme_score) / 2, 100)
            
            status = "PASS" if documentation_score >= self.thresholds["documentation_score"] else "FAIL"
            if documentation_score >= 60:
                status = "WARNING" if status == "FAIL" else status
            
            recommendations = []
            if documentation_score < self.thresholds["documentation_score"]:
                recommendations.extend([
                    "Create comprehensive API documentation",
                    "Add usage examples and tutorials",
                    "Document deployment procedures",
                    "Add troubleshooting guides"
                ])
            
            details = {
                **doc_metrics,
                "readme_score": readme_score,
                "existence_score": existence_score,
                "content_score": content_score,
                "documentation_threshold": self.thresholds["documentation_score"]
            }
            
        except Exception as e:
            documentation_score = 0.0
            status = "FAIL"
            details = {"error": str(e)}
            recommendations = ["Fix documentation analysis environment"]
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_name="Documentation Quality",
            status=status,
            score=documentation_score,
            details=details,
            execution_time=execution_time,
            recommendations=recommendations
        )
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates and generate comprehensive report."""
        print("🚀 Starting Comprehensive Quality Gates Validation...")
        print("=" * 60)
        
        # Define gates to run
        gates = [
            ("Test Coverage", self.run_test_coverage_gate),
            ("Security Assessment", self.run_security_gate),
            ("Performance Benchmark", self.run_performance_gate),
            ("Compliance Assessment", self.run_compliance_gate),
            ("Code Quality", self.run_code_quality_gate),
            ("Documentation Quality", self.run_documentation_gate)
        ]
        
        # Run gates concurrently where possible
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for gate_name, gate_func in gates:
                print(f"🔄 Running {gate_name}...")
                future = executor.submit(gate_func)
                futures.append((gate_name, future))
            
            # Collect results
            for gate_name, future in futures:
                try:
                    result = future.result(timeout=60)  # 60 second timeout per gate
                    self.results.append(result)
                    
                    status_emoji = "✅" if result.status == "PASS" else "⚠️" if result.status == "WARNING" else "❌"
                    print(f"{status_emoji} {result.gate_name}: {result.status} (Score: {result.score:.1f}/100)")
                    
                except Exception as e:
                    print(f"❌ {gate_name}: FAILED - {str(e)}")
                    self.results.append(QualityGateResult(
                        gate_name=gate_name,
                        status="FAIL",
                        score=0.0,
                        details={"error": str(e)},
                        execution_time=0.0,
                        recommendations=["Fix gate execution environment"]
                    ))
        
        # Calculate overall scores
        total_score = sum(result.score for result in self.results)
        self.overall_score = total_score / len(self.results) if self.results else 0.0
        
        # Determine overall status
        pass_count = sum(1 for result in self.results if result.status == "PASS")
        warning_count = sum(1 for result in self.results if result.status == "WARNING")
        fail_count = sum(1 for result in self.results if result.status == "FAIL")
        
        overall_status = "PASS"
        if fail_count > 0:
            overall_status = "FAIL"
        elif warning_count > 0:
            overall_status = "WARNING"
        
        # Generate summary
        execution_time = time.time() - self.execution_start
        
        summary = {
            "overall_status": overall_status,
            "overall_score": self.overall_score,
            "execution_time": execution_time,
            "gates_passed": pass_count,
            "gates_warned": warning_count,
            "gates_failed": fail_count,
            "total_gates": len(self.results),
            "results": [asdict(result) for result in self.results],
            "recommendations": self._get_prioritized_recommendations()
        }
        
        print("\n" + "=" * 60)
        print(f"📊 Overall Quality Score: {self.overall_score:.1f}/100")
        print(f"🎯 Overall Status: {overall_status}")
        print(f"✅ Gates Passed: {pass_count}")
        print(f"⚠️ Gates Warned: {warning_count}")
        print(f"❌ Gates Failed: {fail_count}")
        print(f"⏱️ Execution Time: {execution_time:.2f}s")
        
        return summary
    
    def _get_prioritized_recommendations(self) -> List[str]:
        """Get prioritized recommendations based on failed gates."""
        all_recommendations = []
        
        # Prioritize by failure severity
        critical_failures = [r for r in self.results if r.status == "FAIL" and r.score < 50]
        moderate_failures = [r for r in self.results if r.status == "FAIL" and r.score >= 50]
        warnings = [r for r in self.results if r.status == "WARNING"]
        
        for result in critical_failures + moderate_failures + warnings:
            all_recommendations.extend(result.recommendations)
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        return unique_recommendations[:10]  # Top 10 recommendations
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.performance_optimizer.cleanup()
        except:
            pass

def run_comprehensive_quality_gates():
    """Run the complete quality gates validation."""
    quality_gates = ComprehensiveQualityGates()
    
    try:
        results = quality_gates.run_all_gates()
        
        # Save results to file
        output_file = f"/root/repo/quality_gates_report_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Quality gates report saved to: {output_file}")
        
        # Print top recommendations
        if results["recommendations"]:
            print("\n🎯 Top Recommendations:")
            for i, rec in enumerate(results["recommendations"][:5], 1):
                print(f"   {i}. {rec}")
        
        return results
        
    finally:
        quality_gates.cleanup()

if __name__ == "__main__":
    results = run_comprehensive_quality_gates()