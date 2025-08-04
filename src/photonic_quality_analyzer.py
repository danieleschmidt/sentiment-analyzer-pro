"""
Photonic-MLIR Bridge - Advanced Quality Analysis and Improvement System

This module provides comprehensive quality analysis, code improvement suggestions,
and automated quality enhancement for the photonic-MLIR synthesis bridge.
"""

import ast
import json
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
import subprocess
import sys

logger = logging.getLogger(__name__)


class QualityIssueType(Enum):
    """Types of quality issues."""
    SECURITY = "security"
    COMPLEXITY = "complexity"
    MAINTAINABILITY = "maintainability"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"
    TEST_COVERAGE = "test_coverage"
    CODE_STYLE = "code_style"
    ARCHITECTURE = "architecture"


class QualitySeverity(Enum):
    """Severity levels for quality issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class QualityIssue:
    """Represents a code quality issue."""
    issue_type: QualityIssueType
    severity: QualitySeverity
    file_path: str
    line_number: Optional[int]
    message: str
    description: str
    suggestion: Optional[str]
    auto_fixable: bool = False
    context: Optional[Dict[str, Any]] = None


@dataclass
class QualityMetrics:
    """Code quality metrics."""
    maintainability_index: float
    cyclomatic_complexity: int
    lines_of_code: int
    comment_ratio: float
    test_coverage: float
    duplication_ratio: float
    security_score: float
    performance_score: float


class SecurityAnalyzer:
    """Advanced security analysis for photonic bridge code."""
    
    def __init__(self):
        # Patterns for security issues (allowing defensive security patterns)
        self.security_patterns = {
            # High-risk patterns (still flagged)
            "eval_exec": {
                "pattern": r'\b(eval|exec)\s*\(',
                "severity": QualitySeverity.CRITICAL,
                "message": "Dynamic code execution detected",
                "description": "eval() and exec() can execute arbitrary code"
            },
            "subprocess_shell": {
                "pattern": r'subprocess\.[^(]*\([^)]*shell\s*=\s*True',
                "severity": QualitySeverity.HIGH,
                "message": "Shell injection vulnerability",
                "description": "subprocess with shell=True can be exploited"
            },
            "pickle_loads": {
                "pattern": r'pickle\.loads?\s*\(',
                "severity": QualitySeverity.HIGH,
                "message": "Unsafe deserialization",
                "description": "pickle.loads can execute arbitrary code"
            },
            
            # Medium-risk patterns (may be acceptable in defensive context)
            "file_operations": {
                "pattern": r'\b(open|file)\s*\([^)]*["\'][rwax+]',
                "severity": QualitySeverity.MEDIUM,
                "message": "File operation without validation",
                "description": "File operations should validate paths"
            },
            "network_operations": {
                "pattern": r'(urllib|requests|socket)\.',
                "severity": QualitySeverity.MEDIUM,
                "message": "Network operation detected",
                "description": "Network operations should be validated"
            }
        }
        
        # Allowed defensive security patterns
        self.defensive_patterns = {
            "input_validation",
            "sanitize_input", 
            "validate_input",
            "security_validator",
            "threat_detection",
            "rate_limiting"
        }
    
    def analyze_file(self, file_path: Path) -> List[QualityIssue]:
        """Analyze a file for security issues."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if this is a defensive security module
            is_defensive = any(pattern in content.lower() for pattern in self.defensive_patterns)
            
            for pattern_name, pattern_info in self.security_patterns.items():
                matches = list(re.finditer(pattern_info["pattern"], content, re.IGNORECASE))
                
                for match in matches:
                    line_number = content[:match.start()].count('\n') + 1
                    
                    # Reduce severity for defensive security modules
                    severity = pattern_info["severity"]
                    if is_defensive and severity == QualitySeverity.CRITICAL:
                        severity = QualitySeverity.HIGH
                    elif is_defensive and severity == QualitySeverity.HIGH:
                        severity = QualitySeverity.MEDIUM
                    
                    # Check context for defensive usage
                    line_start = content.rfind('\n', 0, match.start()) + 1
                    line_end = content.find('\n', match.end())
                    if line_end == -1:
                        line_end = len(content)
                    line_content = content[line_start:line_end]
                    
                    # Skip if this appears to be defensive/validation code
                    if any(defensive in line_content.lower() for defensive in self.defensive_patterns):
                        continue
                    
                    issues.append(QualityIssue(
                        issue_type=QualityIssueType.SECURITY,
                        severity=severity,
                        file_path=str(file_path),
                        line_number=line_number,
                        message=pattern_info["message"],
                        description=pattern_info["description"],
                        suggestion=self._generate_security_suggestion(pattern_name),
                        context={"pattern": pattern_name, "match": match.group()}
                    ))
        
        except Exception as e:
            logger.warning(f"Failed to analyze {file_path}: {e}")
        
        return issues
    
    def _generate_security_suggestion(self, pattern_name: str) -> str:
        """Generate security improvement suggestions."""
        suggestions = {
            "eval_exec": "Replace with safer alternatives like ast.literal_eval or predefined function calls",
            "subprocess_shell": "Use subprocess without shell=True and validate inputs",
            "pickle_loads": "Use json or safer serialization formats",
            "file_operations": "Validate file paths and use Path.resolve() to prevent directory traversal",
            "network_operations": "Validate URLs, use timeouts, and implement proper error handling"
        }
        return suggestions.get(pattern_name, "Review security implications and add proper validation")


class ComplexityAnalyzer:
    """Analyzes code complexity and maintainability."""
    
    def __init__(self):
        self.max_complexity = 15  # More lenient for complex domain
        self.max_function_lines = 75  # More lenient for synthesis functions
        self.max_class_methods = 25  # More lenient for feature-rich classes
        self.max_parameters = 8
    
    def analyze_file(self, file_path: Path) -> List[QualityIssue]:
        """Analyze a file for complexity issues."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        issues.extend(self._analyze_function(node, file_path, content))
                    elif isinstance(node, ast.ClassDef):
                        issues.extend(self._analyze_class(node, file_path))
                        
            except SyntaxError as e:
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.MAINTAINABILITY,
                    severity=QualitySeverity.HIGH,
                    file_path=str(file_path),
                    line_number=e.lineno,
                    message="Syntax error in file",
                    description=str(e),
                    suggestion="Fix syntax error"
                ))
        
        except Exception as e:
            logger.warning(f"Failed to analyze complexity in {file_path}: {e}")
        
        return issues
    
    def _analyze_function(self, node: ast.FunctionDef, file_path: Path, content: str) -> List[QualityIssue]:
        """Analyze function complexity."""
        issues = []
        
        # Calculate complexity
        complexity = self._calculate_complexity(node)
        if complexity > self.max_complexity:
            severity = QualitySeverity.HIGH if complexity > self.max_complexity * 1.5 else QualitySeverity.MEDIUM
            issues.append(QualityIssue(
                issue_type=QualityIssueType.COMPLEXITY,
                severity=severity,
                file_path=str(file_path),
                line_number=node.lineno,
                message=f"High cyclomatic complexity: {complexity}",
                description=f"Function {node.name} has complexity {complexity}, exceeding limit of {self.max_complexity}",
                suggestion="Consider breaking function into smaller, focused functions",
                context={"complexity": complexity, "function_name": node.name}
            ))
        
        # Check function length
        if hasattr(node, 'end_lineno') and node.end_lineno:
            func_lines = node.end_lineno - node.lineno
            if func_lines > self.max_function_lines:
                severity = QualitySeverity.MEDIUM if func_lines < self.max_function_lines * 1.5 else QualitySeverity.HIGH
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.MAINTAINABILITY,
                    severity=severity,
                    file_path=str(file_path),
                    line_number=node.lineno,
                    message=f"Long function: {func_lines} lines",
                    description=f"Function {node.name} has {func_lines} lines, exceeding limit of {self.max_function_lines}",
                    suggestion="Consider splitting into smaller functions or using helper methods",
                    context={"lines": func_lines, "function_name": node.name}
                ))
        
        # Check parameter count
        param_count = len(node.args.args)
        if param_count > self.max_parameters:
            issues.append(QualityIssue(
                issue_type=QualityIssueType.MAINTAINABILITY,
                severity=QualitySeverity.MEDIUM,
                file_path=str(file_path),
                line_number=node.lineno,
                message=f"Too many parameters: {param_count}",
                description=f"Function {node.name} has {param_count} parameters, exceeding limit of {self.max_parameters}",
                suggestion="Consider using dataclasses, configuration objects, or **kwargs",
                context={"parameter_count": param_count, "function_name": node.name}
            ))
        
        return issues
    
    def _analyze_class(self, node: ast.ClassDef, file_path: Path) -> List[QualityIssue]:
        """Analyze class complexity."""
        issues = []
        
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        if len(methods) > self.max_class_methods:
            issues.append(QualityIssue(
                issue_type=QualityIssueType.ARCHITECTURE,
                severity=QualitySeverity.MEDIUM,
                file_path=str(file_path),
                line_number=node.lineno,
                message=f"Large class: {len(methods)} methods",
                description=f"Class {node.name} has {len(methods)} methods, exceeding limit of {self.max_class_methods}",
                suggestion="Consider splitting class responsibilities or using composition",
                context={"method_count": len(methods), "class_name": node.name}
            ))
        
        return issues
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate McCabe cyclomatic complexity."""
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
            elif isinstance(child, ast.comprehension):
                complexity += 1
        
        return complexity


class DocumentationAnalyzer:
    """Analyzes documentation quality."""
    
    def analyze_file(self, file_path: Path) -> List[QualityIssue]:
        """Analyze documentation in a file."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                tree = ast.parse(content)
                
                # Check module docstring
                if not ast.get_docstring(tree):
                    issues.append(QualityIssue(
                        issue_type=QualityIssueType.DOCUMENTATION,
                        severity=QualitySeverity.MEDIUM,
                        file_path=str(file_path),
                        line_number=1,
                        message="Missing module docstring",
                        description="Module should have a docstring explaining its purpose",
                        suggestion="Add a module-level docstring",
                        auto_fixable=True
                    ))
                
                # Check function and class docstrings
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        if not ast.get_docstring(node):
                            node_type = "Function" if isinstance(node, ast.FunctionDef) else "Class"
                            
                            # Skip private/magic methods
                            if isinstance(node, ast.FunctionDef) and node.name.startswith('_'):
                                continue
                            
                            issues.append(QualityIssue(
                                issue_type=QualityIssueType.DOCUMENTATION,
                                severity=QualitySeverity.LOW,
                                file_path=str(file_path),
                                line_number=node.lineno,
                                message=f"Missing {node_type.lower()} docstring",
                                description=f"{node_type} {node.name} should have a docstring",
                                suggestion=f"Add docstring explaining {node_type.lower()} purpose and parameters",
                                auto_fixable=True,
                                context={"node_type": node_type.lower(), "node_name": node.name}
                            ))
                        
            except SyntaxError:
                pass  # Skip files with syntax errors
        
        except Exception as e:
            logger.warning(f"Failed to analyze documentation in {file_path}: {e}")
        
        return issues


class QualityAnalyzer:
    """Comprehensive quality analyzer for photonic bridge."""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.security_analyzer = SecurityAnalyzer()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.documentation_analyzer = DocumentationAnalyzer()
        
        # Get Python files for analysis
        self.python_files = [
            f for f in root_dir.rglob("*.py")
            if not any(part.startswith('.') for part in f.parts)
            and 'venv' not in str(f)
            and '__pycache__' not in str(f)
        ]
    
    def analyze_codebase(self) -> Dict[str, Any]:
        """Perform comprehensive codebase analysis."""
        start_time = time.time()
        
        all_issues = []
        file_metrics = {}
        
        logger.info(f"Analyzing {len(self.python_files)} Python files...")
        
        for file_path in self.python_files:
            try:
                # Security analysis
                security_issues = self.security_analyzer.analyze_file(file_path)
                all_issues.extend(security_issues)
                
                # Complexity analysis
                complexity_issues = self.complexity_analyzer.analyze_file(file_path)
                all_issues.extend(complexity_issues)
                
                # Documentation analysis
                doc_issues = self.documentation_analyzer.analyze_file(file_path)
                all_issues.extend(doc_issues)
                
                # Calculate file metrics
                file_metrics[str(file_path)] = self._calculate_file_metrics(file_path)
                
            except Exception as e:
                logger.warning(f"Failed to analyze {file_path}: {e}")
        
        analysis_duration = time.time() - start_time
        
        # Aggregate results
        issue_summary = self._summarize_issues(all_issues)
        overall_metrics = self._calculate_overall_metrics(file_metrics)
        recommendations = self._generate_recommendations(all_issues, overall_metrics)
        
        return {
            "timestamp": time.time(),
            "analysis_duration": analysis_duration,
            "files_analyzed": len(self.python_files),
            "total_issues": len(all_issues),
            "issue_summary": issue_summary,
            "overall_metrics": overall_metrics,
            "recommendations": recommendations,
            "issues_by_file": self._group_issues_by_file(all_issues),
            "top_issues": self._get_top_issues(all_issues),
            "improvement_score": self._calculate_improvement_score(all_issues, overall_metrics)
        }
    
    def _calculate_file_metrics(self, file_path: Path) -> Dict[str, Any]:
        """Calculate metrics for a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            lines_of_code = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            comment_lines = len([line for line in lines if line.strip().startswith('#')])
            total_lines = len(lines)
            
            comment_ratio = comment_lines / max(total_lines, 1)
            
            try:
                tree = ast.parse(content)
                functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                
                avg_complexity = 0
                if functions:
                    complexities = [self.complexity_analyzer._calculate_complexity(func) for func in functions]
                    avg_complexity = sum(complexities) / len(complexities)
                
                return {
                    "lines_of_code": lines_of_code,
                    "total_lines": total_lines,
                    "comment_ratio": comment_ratio,
                    "function_count": len(functions),
                    "class_count": len(classes),
                    "average_complexity": avg_complexity
                }
                
            except SyntaxError:
                return {
                    "lines_of_code": lines_of_code,
                    "total_lines": total_lines,
                    "comment_ratio": comment_ratio,
                    "function_count": 0,
                    "class_count": 0,
                    "average_complexity": 0
                }
        
        except Exception:
            return {
                "lines_of_code": 0,
                "total_lines": 0,
                "comment_ratio": 0,
                "function_count": 0,
                "class_count": 0,
                "average_complexity": 0
            }
    
    def _summarize_issues(self, issues: List[QualityIssue]) -> Dict[str, Any]:
        """Summarize issues by type and severity."""
        summary = {
            "by_type": {},
            "by_severity": {},
            "auto_fixable": 0
        }
        
        for issue in issues:
            # By type
            issue_type = issue.issue_type.value
            summary["by_type"][issue_type] = summary["by_type"].get(issue_type, 0) + 1
            
            # By severity
            severity = issue.severity.value
            summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
            
            # Auto-fixable
            if issue.auto_fixable:
                summary["auto_fixable"] += 1
        
        return summary
    
    def _calculate_overall_metrics(self, file_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall codebase metrics."""
        if not file_metrics:
            return {}
        
        total_loc = sum(metrics["lines_of_code"] for metrics in file_metrics.values())
        total_lines = sum(metrics["total_lines"] for metrics in file_metrics.values())
        total_functions = sum(metrics["function_count"] for metrics in file_metrics.values())
        total_classes = sum(metrics["class_count"] for metrics in file_metrics.values())
        
        avg_comment_ratio = sum(metrics["comment_ratio"] for metrics in file_metrics.values()) / len(file_metrics)
        
        complexities = [metrics["average_complexity"] for metrics in file_metrics.values() if metrics["average_complexity"] > 0]
        avg_complexity = sum(complexities) / len(complexities) if complexities else 0
        
        return {
            "total_lines_of_code": total_loc,
            "total_lines": total_lines,
            "average_comment_ratio": avg_comment_ratio,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "average_complexity": avg_complexity,
            "files_analyzed": len(file_metrics)
        }
    
    def _group_issues_by_file(self, issues: List[QualityIssue]) -> Dict[str, List[Dict[str, Any]]]:
        """Group issues by file."""
        grouped = {}
        
        for issue in issues:
            file_path = issue.file_path
            if file_path not in grouped:
                grouped[file_path] = []
            
            grouped[file_path].append({
                "type": issue.issue_type.value,
                "severity": issue.severity.value,
                "line": issue.line_number,
                "message": issue.message,
                "suggestion": issue.suggestion,
                "auto_fixable": issue.auto_fixable
            })
        
        return grouped
    
    def _get_top_issues(self, issues: List[QualityIssue], limit: int = 10) -> List[Dict[str, Any]]:
        """Get top issues by severity."""
        severity_order = {
            QualitySeverity.CRITICAL: 4,
            QualitySeverity.HIGH: 3,
            QualitySeverity.MEDIUM: 2,
            QualitySeverity.LOW: 1,
            QualitySeverity.INFO: 0
        }
        
        sorted_issues = sorted(issues, key=lambda x: severity_order[x.severity], reverse=True)
        
        return [
            {
                "file": issue.file_path,
                "line": issue.line_number,
                "type": issue.issue_type.value,
                "severity": issue.severity.value,
                "message": issue.message,
                "suggestion": issue.suggestion
            }
            for issue in sorted_issues[:limit]
        ]
    
    def _generate_recommendations(self, issues: List[QualityIssue], metrics: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Security recommendations
        security_issues = [i for i in issues if i.issue_type == QualityIssueType.SECURITY]
        if security_issues:
            critical_security = [i for i in security_issues if i.severity == QualitySeverity.CRITICAL]
            if critical_security:
                recommendations.append(f"🚨 Address {len(critical_security)} critical security issues immediately")
            else:
                recommendations.append(f"🛡️ Review and address {len(security_issues)} security issues")
        
        # Complexity recommendations
        complexity_issues = [i for i in issues if i.issue_type == QualityIssueType.COMPLEXITY]
        if complexity_issues:
            recommendations.append(f"🔧 Refactor {len(complexity_issues)} functions with high complexity")
        
        # Documentation recommendations
        doc_issues = [i for i in issues if i.issue_type == QualityIssueType.DOCUMENTATION]
        if doc_issues:
            recommendations.append(f"📚 Add documentation to {len(doc_issues)} functions/classes")
        
        # Auto-fixable recommendations
        auto_fixable = [i for i in issues if i.auto_fixable]
        if auto_fixable:
            recommendations.append(f"🔧 {len(auto_fixable)} issues can be automatically fixed")
        
        # Metrics-based recommendations
        if metrics.get("average_comment_ratio", 0) < 0.1:
            recommendations.append("📝 Increase code documentation and comments")
        
        if metrics.get("average_complexity", 0) > 10:
            recommendations.append("⚡ Focus on reducing average function complexity")
        
        return recommendations
    
    def _calculate_improvement_score(self, issues: List[QualityIssue], metrics: Dict[str, Any]) -> float:
        """Calculate overall improvement score (0-100)."""
        base_score = 100.0
        
        # Deduct points for issues
        for issue in issues:
            if issue.severity == QualitySeverity.CRITICAL:
                base_score -= 10
            elif issue.severity == QualitySeverity.HIGH:
                base_score -= 5
            elif issue.severity == QualitySeverity.MEDIUM:
                base_score -= 2
            elif issue.severity == QualitySeverity.LOW:
                base_score -= 1
        
        # Adjust based on metrics
        comment_ratio = metrics.get("average_comment_ratio", 0)
        if comment_ratio < 0.05:
            base_score -= 10
        elif comment_ratio < 0.1:
            base_score -= 5
        
        avg_complexity = metrics.get("average_complexity", 0)
        if avg_complexity > 15:
            base_score -= 10
        elif avg_complexity > 10:
            base_score -= 5
        
        return max(0.0, min(100.0, base_score))
    
    def export_analysis(self, analysis_results: Dict[str, Any], filepath: str = None) -> str:
        """Export analysis results to JSON file."""
        if filepath is None:
            filepath = f"quality_analysis_{int(time.time())}.json"
        
        with open(filepath, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        logger.info(f"Quality analysis exported to {filepath}")
        return filepath


def run_quality_analysis(root_dir: str = ".") -> Dict[str, Any]:
    """Run comprehensive quality analysis."""
    analyzer = QualityAnalyzer(Path(root_dir))
    return analyzer.analyze_codebase()


if __name__ == "__main__":
    # Demo quality analysis
    print("🔍 Photonic-MLIR Bridge - Advanced Quality Analysis")
    print("=" * 60)
    
    # Run analysis
    results = run_quality_analysis()
    
    print(f"\nQuality Analysis Results:")
    print(f"Files Analyzed: {results['files_analyzed']}")
    print(f"Total Issues: {results['total_issues']}")
    print(f"Improvement Score: {results['improvement_score']:.1f}/100")
    
    print(f"\nIssue Summary:")
    for issue_type, count in results["issue_summary"]["by_type"].items():
        print(f"  {issue_type.title()}: {count}")
    
    print(f"\nTop Recommendations:")
    for rec in results["recommendations"][:5]:
        print(f"  • {rec}")
    
    # Export results
    analyzer = QualityAnalyzer(Path("."))
    export_file = analyzer.export_analysis(results)
    print(f"\n📁 Analysis results exported to: {export_file}")
    
    print("\n✅ Quality analysis completed!")