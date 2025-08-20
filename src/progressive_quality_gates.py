"""
Progressive Quality Gates System v4.0 - Autonomous SDLC Execution
Implements intelligent, adaptive quality gates that evolve through development phases.
"""
from __future__ import annotations

import asyncio
import json
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable, Any, Union
import logging
import subprocess
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import importlib.util

logger = logging.getLogger(__name__)


class DevelopmentPhase(Enum):
    """Development phases with increasing quality requirements."""
    GENERATION_1 = "make_it_work"      # Simple, basic functionality
    GENERATION_2 = "make_it_robust"    # Reliable, error handling
    GENERATION_3 = "make_it_scale"     # Optimized, performance


class QualityGateStatus(Enum):
    """Status of individual quality gates."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class GateType(Enum):
    """Types of quality gates."""
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    SECURITY_SCAN = "security_scan"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    CODE_COVERAGE = "code_coverage"
    LINT_CHECK = "lint_check"
    TYPE_CHECK = "type_check"
    DEPENDENCY_SCAN = "dependency_scan"
    DOCUMENTATION = "documentation"
    COMPLIANCE = "compliance"
    LOAD_TEST = "load_test"
    CHAOS_TEST = "chaos_test"
    RESEARCH_VALIDATION = "research_validation"


@dataclass
class QualityGate:
    """Individual quality gate definition."""
    name: str
    gate_type: GateType
    phase: DevelopmentPhase
    command: str
    timeout: int = 300
    retries: int = 2
    required: bool = True
    threshold: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    
    # Runtime state
    status: QualityGateStatus = QualityGateStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0


@dataclass
class GateResult:
    """Result of executing a quality gate."""
    success: bool
    duration: float
    output: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    coverage: Optional[float] = None
    issues: List[str] = field(default_factory=list)


@dataclass
class PhaseMetrics:
    """Metrics for a development phase."""
    phase: DevelopmentPhase
    total_gates: int = 0
    passed_gates: int = 0
    failed_gates: int = 0
    skipped_gates: int = 0
    total_duration: float = 0.0
    success_rate: float = 0.0
    performance_score: float = 0.0
    security_score: float = 0.0
    quality_score: float = 0.0


class ProgressiveQualityGates:
    """
    Progressive Quality Gates System for Autonomous SDLC.
    
    Implements intelligent quality gates that adapt based on development phase,
    automatically escalating requirements and sophistication as the project matures.
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.current_phase = DevelopmentPhase.GENERATION_1
        self.gates: Dict[str, QualityGate] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.phase_metrics: Dict[DevelopmentPhase, PhaseMetrics] = {}
        self.config = self._load_config()
        self._initialize_gates()
        
        # Adaptive learning
        self.gate_performance_history: Dict[str, List[float]] = {}
        self.failure_patterns: Dict[str, List[str]] = {}
        self.optimization_cache: Dict[str, Any] = {}
        
        # Execution tracking
        self.current_execution_id = None
        self.parallel_executor = ThreadPoolExecutor(max_workers=4)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from various sources."""
        config = {
            "min_coverage": {
                DevelopmentPhase.GENERATION_1: 70.0,
                DevelopmentPhase.GENERATION_2: 85.0,
                DevelopmentPhase.GENERATION_3: 95.0
            },
            "performance_thresholds": {
                DevelopmentPhase.GENERATION_1: {"response_time": 1000},
                DevelopmentPhase.GENERATION_2: {"response_time": 500},
                DevelopmentPhase.GENERATION_3: {"response_time": 200}
            },
            "security_requirements": {
                DevelopmentPhase.GENERATION_1: ["basic_scan"],
                DevelopmentPhase.GENERATION_2: ["vulnerability_scan", "dependency_check"],
                DevelopmentPhase.GENERATION_3: ["full_security_audit", "penetration_test"]
            }
        }
        
        # Load from file if exists
        config_file = self.project_root / "quality_gates_config.json"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    file_config = json.load(f)
                    config.update(file_config)
            except Exception as e:
                logger.warning(f"Could not load config file: {e}")
                
        return config
    
    def _initialize_gates(self):
        """Initialize quality gates for all phases."""
        
        # Generation 1: MAKE IT WORK (Simple)
        gen1_gates = [
            QualityGate(
                name="basic_unit_tests",
                gate_type=GateType.UNIT_TEST,
                phase=DevelopmentPhase.GENERATION_1,
                command="python3 -m pytest tests/unit/ -v --tb=short",
                timeout=180,
                description="Run basic unit tests to ensure core functionality works"
            ),
            QualityGate(
                name="basic_linting",
                gate_type=GateType.LINT_CHECK,
                phase=DevelopmentPhase.GENERATION_1,
                command="python3 -m ruff check src/ --select=E,F --exit-zero",
                timeout=60,
                description="Basic code style and syntax checks"
            ),
            QualityGate(
                name="import_validation",
                gate_type=GateType.UNIT_TEST,
                phase=DevelopmentPhase.GENERATION_1,
                command="python3 -c 'import src; print(\"Core imports successful\")'",
                timeout=30,
                description="Validate that core modules can be imported"
            )
        ]
        
        # Generation 2: MAKE IT ROBUST (Reliable)
        gen2_gates = [
            QualityGate(
                name="comprehensive_tests",
                gate_type=GateType.UNIT_TEST,
                phase=DevelopmentPhase.GENERATION_2,
                command="python3 -m pytest tests/ -v --cov=src --cov-report=json --cov-fail-under=85",
                timeout=300,
                threshold=85.0,
                description="Comprehensive test suite with coverage requirements"
            ),
            QualityGate(
                name="security_scan",
                gate_type=GateType.SECURITY_SCAN,
                phase=DevelopmentPhase.GENERATION_2,
                command="python3 -m bandit -r src/ -f json -o bandit_report.json",
                timeout=120,
                description="Security vulnerability scanning"
            ),
            QualityGate(
                name="integration_tests",
                gate_type=GateType.INTEGRATION_TEST,
                phase=DevelopmentPhase.GENERATION_2,
                command="python3 -m pytest tests/integration/ -v",
                timeout=600,
                description="Integration tests for component interactions"
            ),
            QualityGate(
                name="dependency_audit",
                gate_type=GateType.DEPENDENCY_SCAN,
                phase=DevelopmentPhase.GENERATION_2,
                command="python3 -m safety check --json",
                timeout=60,
                description="Audit dependencies for known vulnerabilities"
            ),
            QualityGate(
                name="comprehensive_linting",
                gate_type=GateType.LINT_CHECK,
                phase=DevelopmentPhase.GENERATION_2,
                command="python3 -m ruff check src/ tests/",
                timeout=120,
                description="Comprehensive code quality and style checks"
            )
        ]
        
        # Generation 3: MAKE IT SCALE (Optimized)
        gen3_gates = [
            QualityGate(
                name="performance_benchmarks",
                gate_type=GateType.PERFORMANCE_BENCHMARK,
                phase=DevelopmentPhase.GENERATION_3,
                command="python3 -m pytest tests/performance/ -v --benchmark-only",
                timeout=900,
                description="Performance benchmarking and optimization validation"
            ),
            QualityGate(
                name="load_testing",
                gate_type=GateType.LOAD_TEST,
                phase=DevelopmentPhase.GENERATION_3,
                command="python3 -c 'from tests.load_test import run_load_test; run_load_test()'",
                timeout=1200,
                description="Load testing for scalability validation"
            ),
            QualityGate(
                name="full_security_audit",
                gate_type=GateType.SECURITY_SCAN,
                phase=DevelopmentPhase.GENERATION_3,
                command="python3 -c 'from src.enterprise_security_framework import run_full_audit; run_full_audit()'",
                timeout=600,
                description="Comprehensive security audit and penetration testing"
            ),
            QualityGate(
                name="research_validation",
                gate_type=GateType.RESEARCH_VALIDATION,
                phase=DevelopmentPhase.GENERATION_3,
                command="python3 -c 'from src.quantum_research_framework import validate_research; validate_research()'",
                timeout=1800,
                description="Validate research algorithms and statistical significance"
            ),
            QualityGate(
                name="chaos_engineering",
                gate_type=GateType.CHAOS_TEST,
                phase=DevelopmentPhase.GENERATION_3,
                command="python3 -c 'from src.resilience_framework import run_chaos_tests; run_chaos_tests()'",
                timeout=900,
                description="Chaos engineering tests for resilience validation"
            )
        ]
        
        # Register all gates
        for gate in gen1_gates + gen2_gates + gen3_gates:
            self.gates[gate.name] = gate
    
    async def execute_phase(self, phase: DevelopmentPhase, parallel: bool = True) -> PhaseMetrics:
        """Execute all gates for a specific phase."""
        logger.info(f"🚀 Starting execution of {phase.value} phase")
        
        phase_gates = [g for g in self.gates.values() if g.phase == phase]
        if not phase_gates:
            logger.warning(f"No gates defined for phase {phase.value}")
            return PhaseMetrics(phase=phase)
        
        self.current_phase = phase
        self.current_execution_id = f"{phase.value}_{int(time.time())}"
        
        start_time = time.time()
        metrics = PhaseMetrics(phase=phase, total_gates=len(phase_gates))
        
        if parallel:
            results = await self._execute_gates_parallel(phase_gates)
        else:
            results = await self._execute_gates_sequential(phase_gates)
        
        # Process results
        for gate_name, result in results.items():
            gate = self.gates[gate_name]
            if result.success:
                metrics.passed_gates += 1
            else:
                metrics.failed_gates += 1
                if not gate.required:
                    metrics.skipped_gates += 1
        
        metrics.total_duration = time.time() - start_time
        metrics.success_rate = metrics.passed_gates / metrics.total_gates * 100
        
        # Calculate composite scores
        metrics.quality_score = self._calculate_quality_score(results)
        metrics.performance_score = self._calculate_performance_score(results)
        metrics.security_score = self._calculate_security_score(results)
        
        self.phase_metrics[phase] = metrics
        
        # Save execution history
        self._save_execution_history(phase, results, metrics)
        
        logger.info(f"✅ Phase {phase.value} completed: {metrics.success_rate:.1f}% success rate")
        return metrics
    
    async def _execute_gates_parallel(self, gates: List[QualityGate]) -> Dict[str, GateResult]:
        """Execute gates in parallel where possible."""
        results = {}
        dependency_graph = self._build_dependency_graph(gates)
        
        # Execute gates in topological order, parallelizing independent gates
        completed = set()
        
        while len(completed) < len(gates):
            # Find gates that can be executed (dependencies satisfied)
            ready_gates = [
                gate for gate in gates 
                if gate.name not in completed and 
                all(dep in completed for dep in gate.dependencies)
            ]
            
            if not ready_gates:
                logger.error("Circular dependency detected in quality gates")
                break
            
            # Execute ready gates in parallel
            tasks = []
            for gate in ready_gates:
                task = asyncio.create_task(self._execute_single_gate(gate))
                tasks.append((gate.name, task))
            
            # Wait for completion
            for gate_name, task in tasks:
                result = await task
                results[gate_name] = result
                completed.add(gate_name)
        
        return results
    
    async def _execute_gates_sequential(self, gates: List[QualityGate]) -> Dict[str, GateResult]:
        """Execute gates sequentially."""
        results = {}
        
        for gate in gates:
            result = await self._execute_single_gate(gate)
            results[gate.name] = result
            
            # Stop on critical failures
            if not result.success and gate.required:
                logger.error(f"Critical gate {gate.name} failed, stopping execution")
                break
        
        return results
    
    async def _execute_single_gate(self, gate: QualityGate) -> GateResult:
        """Execute a single quality gate."""
        logger.info(f"🔄 Executing gate: {gate.name}")
        
        gate.status = QualityGateStatus.RUNNING
        gate.start_time = datetime.now()
        
        start_time = time.time()
        
        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(gate.environment_vars)
            
            # Execute command with timeout
            process = await asyncio.create_subprocess_shell(
                gate.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root,
                env=env
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=gate.timeout
                )
                return_code = process.returncode
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError(f"Gate {gate.name} timed out after {gate.timeout}s")
            
            duration = time.time() - start_time
            output = stdout.decode() + stderr.decode()
            
            # Determine success
            success = return_code == 0
            
            # Extract metrics based on gate type
            metrics = self._extract_metrics(gate, output, success)
            
            # Check thresholds
            if gate.threshold and 'coverage' in metrics:
                success = success and metrics['coverage'] >= gate.threshold
            
            result = GateResult(
                success=success,
                duration=duration,
                output=output,
                metrics=metrics,
                coverage=metrics.get('coverage'),
                issues=self._extract_issues(gate, output)
            )
            
            gate.status = QualityGateStatus.PASSED if success else QualityGateStatus.FAILED
            gate.result = result.__dict__
            
            if success:
                logger.info(f"✅ Gate {gate.name} passed in {duration:.2f}s")
            else:
                logger.error(f"❌ Gate {gate.name} failed in {duration:.2f}s")
                if not gate.required:
                    logger.info(f"⚠️  Gate {gate.name} is optional, continuing...")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            gate.status = QualityGateStatus.FAILED
            gate.error_message = error_msg
            
            logger.error(f"❌ Gate {gate.name} failed with error: {error_msg}")
            
            # Retry logic
            if gate.retry_count < gate.retries:
                gate.retry_count += 1
                logger.info(f"🔄 Retrying gate {gate.name} (attempt {gate.retry_count + 1})")
                await asyncio.sleep(2 ** gate.retry_count)  # Exponential backoff
                return await self._execute_single_gate(gate)
            
            return GateResult(
                success=False,
                duration=duration,
                output=error_msg,
                issues=[error_msg]
            )
        
        finally:
            gate.end_time = datetime.now()
    
    def _extract_metrics(self, gate: QualityGate, output: str, success: bool) -> Dict[str, Any]:
        """Extract metrics from gate execution output."""
        metrics = {}
        
        if gate.gate_type == GateType.CODE_COVERAGE:
            # Extract coverage percentage
            import re
            coverage_match = re.search(r'TOTAL.*?(\d+)%', output)
            if coverage_match:
                metrics['coverage'] = float(coverage_match.group(1))
        
        elif gate.gate_type == GateType.SECURITY_SCAN:
            # Extract security findings
            if 'bandit' in gate.command.lower():
                try:
                    bandit_file = self.project_root / "bandit_report.json"
                    if bandit_file.exists():
                        with open(bandit_file) as f:
                            bandit_data = json.load(f)
                            metrics['high_severity'] = len([r for r in bandit_data.get('results', []) if r.get('issue_severity') == 'HIGH'])
                            metrics['medium_severity'] = len([r for r in bandit_data.get('results', []) if r.get('issue_severity') == 'MEDIUM'])
                            metrics['total_issues'] = len(bandit_data.get('results', []))
                except Exception:
                    pass
        
        elif gate.gate_type == GateType.PERFORMANCE_BENCHMARK:
            # Extract performance metrics
            import re
            time_matches = re.findall(r'(\d+\.?\d*)\s*(ms|s)', output.lower())
            if time_matches:
                total_time = sum(
                    float(t[0]) * (1000 if t[1] == 's' else 1) 
                    for t in time_matches
                )
                metrics['avg_response_time'] = total_time / len(time_matches)
        
        return metrics
    
    def _extract_issues(self, gate: QualityGate, output: str) -> List[str]:
        """Extract issues/warnings from gate output."""
        issues = []
        
        # Common patterns for different tools
        patterns = [
            r'ERROR:.*',
            r'FAILED.*',
            r'WARNING:.*',
            r'E\d+:.*',  # Flake8/ruff errors
            r'F\d+:.*',  # Flake8/ruff failures
        ]
        
        import re
        for pattern in patterns:
            matches = re.findall(pattern, output, re.MULTILINE)
            issues.extend(matches[:10])  # Limit to first 10 issues
        
        return issues
    
    def _calculate_quality_score(self, results: Dict[str, GateResult]) -> float:
        """Calculate composite quality score."""
        if not results:
            return 0.0
        
        total_score = 0.0
        weights = {
            GateType.UNIT_TEST: 0.3,
            GateType.INTEGRATION_TEST: 0.2,
            GateType.CODE_COVERAGE: 0.2,
            GateType.LINT_CHECK: 0.1,
            GateType.SECURITY_SCAN: 0.2
        }
        
        total_weight = 0.0
        
        for gate_name, result in results.items():
            gate = self.gates[gate_name]
            weight = weights.get(gate.gate_type, 0.1)
            score = 100.0 if result.success else 0.0
            
            # Adjust score based on coverage if available
            if result.coverage:
                score = result.coverage
            
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_performance_score(self, results: Dict[str, GateResult]) -> float:
        """Calculate performance score."""
        perf_results = [
            r for name, r in results.items() 
            if self.gates[name].gate_type in [GateType.PERFORMANCE_BENCHMARK, GateType.LOAD_TEST]
        ]
        
        if not perf_results:
            return 100.0
        
        scores = []
        for result in perf_results:
            if result.success:
                # Performance score based on response time thresholds
                response_time = result.metrics.get('avg_response_time', 1000)
                threshold = self.config['performance_thresholds'][self.current_phase]['response_time']
                score = max(0, 100 - (response_time / threshold) * 100)
                scores.append(score)
            else:
                scores.append(0)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_security_score(self, results: Dict[str, GateResult]) -> float:
        """Calculate security score."""
        security_results = [
            r for name, r in results.items()
            if self.gates[name].gate_type == GateType.SECURITY_SCAN
        ]
        
        if not security_results:
            return 100.0
        
        scores = []
        for result in security_results:
            if result.success:
                # Security score based on severity of findings
                high_issues = result.metrics.get('high_severity', 0)
                medium_issues = result.metrics.get('medium_severity', 0)
                score = max(0, 100 - (high_issues * 20 + medium_issues * 5))
                scores.append(score)
            else:
                scores.append(0)
        
        return sum(scores) / len(scores) if scores else 100.0
    
    def _build_dependency_graph(self, gates: List[QualityGate]) -> Dict[str, List[str]]:
        """Build dependency graph for gates."""
        graph = {gate.name: gate.dependencies for gate in gates}
        return graph
    
    def _save_execution_history(self, phase: DevelopmentPhase, results: Dict[str, GateResult], metrics: PhaseMetrics):
        """Save execution history for analysis."""
        execution_record = {
            'execution_id': self.current_execution_id,
            'phase': phase.value,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics.__dict__,
            'gate_results': {
                name: {
                    'success': result.success,
                    'duration': result.duration,
                    'coverage': result.coverage,
                    'issues_count': len(result.issues)
                }
                for name, result in results.items()
            }
        }
        
        self.execution_history.append(execution_record)
        
        # Save to file
        history_file = self.project_root / "quality_gates_history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump(self.execution_history, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save execution history: {e}")
    
    async def execute_autonomous_sdlc(self) -> Dict[DevelopmentPhase, PhaseMetrics]:
        """Execute complete autonomous SDLC through all phases."""
        logger.info("🚀 Starting Autonomous SDLC Execution")
        
        results = {}
        
        for phase in [DevelopmentPhase.GENERATION_1, DevelopmentPhase.GENERATION_2, DevelopmentPhase.GENERATION_3]:
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 PHASE: {phase.value.upper()}")
            logger.info(f"{'='*60}")
            
            metrics = await self.execute_phase(phase, parallel=True)
            results[phase] = metrics
            
            # Check if phase passed minimum requirements
            if not self._phase_passed_requirements(phase, metrics):
                logger.error(f"❌ Phase {phase.value} failed minimum requirements")
                if phase == DevelopmentPhase.GENERATION_1:
                    logger.error("Cannot proceed without basic functionality working")
                    break
                else:
                    logger.warning(f"Proceeding to next phase despite failures in {phase.value}")
            else:
                logger.info(f"✅ Phase {phase.value} passed all requirements")
            
            # Brief pause between phases
            await asyncio.sleep(2)
        
        # Generate final report
        self._generate_final_report(results)
        
        return results
    
    def _phase_passed_requirements(self, phase: DevelopmentPhase, metrics: PhaseMetrics) -> bool:
        """Check if phase meets minimum requirements."""
        requirements = {
            DevelopmentPhase.GENERATION_1: {
                'min_success_rate': 80.0,
                'required_gates': ['basic_unit_tests', 'import_validation']
            },
            DevelopmentPhase.GENERATION_2: {
                'min_success_rate': 85.0,
                'min_coverage': 85.0,
                'max_security_issues': 5
            },
            DevelopmentPhase.GENERATION_3: {
                'min_success_rate': 90.0,
                'min_performance_score': 80.0,
                'min_security_score': 95.0
            }
        }
        
        phase_req = requirements.get(phase, {})
        
        # Check success rate
        if metrics.success_rate < phase_req.get('min_success_rate', 80.0):
            return False
        
        # Check specific requirements for each phase
        if phase == DevelopmentPhase.GENERATION_2:
            if metrics.quality_score < phase_req.get('min_coverage', 85.0):
                return False
        
        elif phase == DevelopmentPhase.GENERATION_3:
            if (metrics.performance_score < phase_req.get('min_performance_score', 80.0) or
                metrics.security_score < phase_req.get('min_security_score', 95.0)):
                return False
        
        return True
    
    def _generate_final_report(self, results: Dict[DevelopmentPhase, PhaseMetrics]):
        """Generate comprehensive final report."""
        report = {
            'autonomous_sdlc_execution': {
                'execution_id': f"autonomous_sdlc_{int(time.time())}",
                'timestamp': datetime.now().isoformat(),
                'total_duration': sum(m.total_duration for m in results.values()),
                'overall_success': all(
                    self._phase_passed_requirements(phase, metrics)
                    for phase, metrics in results.items()
                ),
                'phases': {
                    phase.value: {
                        'success_rate': metrics.success_rate,
                        'quality_score': metrics.quality_score,
                        'performance_score': metrics.performance_score,
                        'security_score': metrics.security_score,
                        'duration': metrics.total_duration,
                        'gates_executed': metrics.total_gates,
                        'gates_passed': metrics.passed_gates,
                        'gates_failed': metrics.failed_gates
                    }
                    for phase, metrics in results.items()
                },
                'recommendations': self._generate_recommendations(results)
            }
        }
        
        # Save report
        report_file = self.project_root / f"autonomous_sdlc_report_{int(time.time())}.json"
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"📊 Final report saved to {report_file}")
        except Exception as e:
            logger.error(f"Could not save final report: {e}")
        
        # Print summary
        self._print_execution_summary(results)
    
    def _generate_recommendations(self, results: Dict[DevelopmentPhase, PhaseMetrics]) -> List[str]:
        """Generate recommendations based on execution results."""
        recommendations = []
        
        for phase, metrics in results.items():
            if metrics.success_rate < 90:
                recommendations.append(f"Improve {phase.value} phase reliability - current success rate: {metrics.success_rate:.1f}%")
            
            if metrics.quality_score < 85:
                recommendations.append(f"Enhance code quality in {phase.value} phase - current score: {metrics.quality_score:.1f}")
            
            if metrics.performance_score < 80:
                recommendations.append(f"Optimize performance in {phase.value} phase - current score: {metrics.performance_score:.1f}")
            
            if metrics.security_score < 95:
                recommendations.append(f"Strengthen security in {phase.value} phase - current score: {metrics.security_score:.1f}")
        
        if not recommendations:
            recommendations.append("Excellent! All phases meet or exceed quality standards.")
        
        return recommendations
    
    def _print_execution_summary(self, results: Dict[DevelopmentPhase, PhaseMetrics]):
        """Print execution summary to console."""
        print("\n" + "="*80)
        print("🎯 AUTONOMOUS SDLC EXECUTION SUMMARY")
        print("="*80)
        
        for phase, metrics in results.items():
            status = "✅ PASSED" if self._phase_passed_requirements(phase, metrics) else "❌ FAILED"
            print(f"\n📋 {phase.value.upper()} {status}")
            print(f"   Success Rate: {metrics.success_rate:.1f}%")
            print(f"   Quality Score: {metrics.quality_score:.1f}")
            print(f"   Performance Score: {metrics.performance_score:.1f}")
            print(f"   Security Score: {metrics.security_score:.1f}")
            print(f"   Duration: {metrics.total_duration:.1f}s")
            print(f"   Gates: {metrics.passed_gates}/{metrics.total_gates} passed")
        
        overall_success = all(
            self._phase_passed_requirements(phase, metrics)
            for phase, metrics in results.items()
        )
        
        print(f"\n🎉 OVERALL RESULT: {'SUCCESS' if overall_success else 'NEEDS IMPROVEMENT'}")
        print("="*80)


# Factory function for easy instantiation
def create_progressive_quality_gates(project_root: str = ".") -> ProgressiveQualityGates:
    """Create and configure Progressive Quality Gates system."""
    return ProgressiveQualityGates(project_root)


# CLI interface for standalone execution
async def main():
    """Main execution function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Progressive Quality Gates for Autonomous SDLC")
    parser.add_argument("--phase", choices=["gen1", "gen2", "gen3", "all"], default="all",
                       help="Which phase to execute")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--parallel", action="store_true", default=True,
                       help="Execute gates in parallel")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    gates = create_progressive_quality_gates(args.project_root)
    
    if args.phase == "all":
        await gates.execute_autonomous_sdlc()
    else:
        phase_map = {
            "gen1": DevelopmentPhase.GENERATION_1,
            "gen2": DevelopmentPhase.GENERATION_2,
            "gen3": DevelopmentPhase.GENERATION_3
        }
        await gates.execute_phase(phase_map[args.phase], args.parallel)


if __name__ == "__main__":
    asyncio.run(main())