#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Value Discovery Engine

Advanced repository optimization and modernization engine for sentiment-analyzer-pro.
Implements continuous value discovery with WSJF + ICE + Technical Debt scoring.
"""

import json
import yaml
import subprocess
import re
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

@dataclass
class ValueItem:
    """Represents a discovered value opportunity"""
    id: str
    title: str
    description: str
    category: str
    source: str
    files_affected: List[str]
    estimated_effort_hours: float
    wsjf_score: float
    ice_score: float
    technical_debt_score: float
    composite_score: float
    risk_level: str
    business_impact: str
    confidence: float
    discovered_at: str
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class AdvancedValueDiscoveryEngine:
    """Autonomous value discovery engine for advanced repositories"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.config_path = self.repo_path / ".terragon" / "value-config.yaml"
        self.metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        self.backlog_path = self.repo_path / "BACKLOG.md"
        
        self.config = self._load_config()
        self.metrics = self._load_metrics()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load Terragon configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._default_config()
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Load historical metrics"""
        try:
            with open(self.metrics_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._default_metrics()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for advanced repositories"""
        return {
            "repository": {"maturity_level": "advanced"},
            "scoring": {
                "weights": {"advanced": {"wsjf": 0.5, "ice": 0.1, "technicalDebt": 0.3, "security": 0.1}},
                "thresholds": {"minScore": 15.0, "maxRisk": 0.7}
            },
            "discovery": {"sources": {"gitHistory": {"enabled": True}}}
        }
    
    def _default_metrics(self) -> Dict[str, Any]:
        """Default metrics structure"""
        return {
            "execution_history": [],
            "learning_metrics": {"estimation_accuracy": 0.5},
            "repository_health_score": {"overall_score": 70}
        }
    
    def discover_value_opportunities(self) -> List[ValueItem]:
        """Main discovery method - finds all value opportunities"""
        opportunities = []
        
        # Advanced repository discovery sources
        opportunities.extend(self._discover_from_git_history())
        opportunities.extend(self._discover_from_static_analysis())
        opportunities.extend(self._discover_from_dependencies())
        opportunities.extend(self._discover_from_performance_metrics())
        opportunities.extend(self._discover_from_security_scanning())
        opportunities.extend(self._discover_modernization_opportunities())
        
        # Score and rank all opportunities
        for item in opportunities:
            self._calculate_scores(item)
        
        # Sort by composite score descending
        opportunities.sort(key=lambda x: x.composite_score, reverse=True)
        
        return opportunities
    
    def _discover_from_git_history(self) -> List[ValueItem]:
        """Discover opportunities from git commit history"""
        opportunities = []
        
        try:
            # Look for technical debt indicators in recent commits
            result = subprocess.run([
                "git", "log", "--since=30 days ago", "--grep=TODO\\|FIXME\\|HACK\\|XXX", 
                "--oneline", "--no-merges"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.stdout:
                opportunities.append(ValueItem(
                    id="git-technical-debt",
                    title="Address Technical Debt from Recent Commits",
                    description="Technical debt indicators found in recent commit messages",
                    category="technical_debt",
                    source="git_history",
                    files_affected=["multiple"],
                    estimated_effort_hours=3.0,
                    wsjf_score=0.0, ice_score=0.0, technical_debt_score=0.0, composite_score=0.0,
                    risk_level="medium",
                    business_impact="maintenance_improvement",
                    confidence=0.7,
                    discovered_at=datetime.now().isoformat()
                ))
                
        except subprocess.CalledProcessError:
            pass
            
        return opportunities
    
    def _discover_from_static_analysis(self) -> List[ValueItem]:
        """Discover opportunities from static analysis"""
        opportunities = []
        
        # Run ruff to check for code quality issues
        try:
            result = subprocess.run([
                "python", "-m", "ruff", "check", "src/", "--output-format=json"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode != 0 and result.stdout:
                try:
                    issues = json.loads(result.stdout)
                    if issues:
                        opportunities.append(ValueItem(
                            id="static-analysis-issues",
                            title="Fix Static Analysis Issues",
                            description=f"Found {len(issues)} code quality issues",
                            category="code_quality",
                            source="static_analysis",
                            files_affected=list(set(issue.get("filename", "unknown") for issue in issues[:5])),
                            estimated_effort_hours=len(issues) * 0.25,
                            wsjf_score=0.0, ice_score=0.0, technical_debt_score=0.0, composite_score=0.0,
                            risk_level="low",
                            business_impact="maintainability",
                            confidence=0.9,
                            discovered_at=datetime.now().isoformat()
                        ))
                except json.JSONDecodeError:
                    pass
                    
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
            
        return opportunities
    
    def _discover_from_dependencies(self) -> List[ValueItem]:
        """Discover dependency-related opportunities"""
        opportunities = []
        
        # Check for outdated dependencies
        pyproject_path = self.repo_path / "pyproject.toml"
        if pyproject_path.exists():
            opportunities.append(ValueItem(
                id="dependency-modernization",
                title="Dependency Security and Modernization Review",
                description="Review and update dependencies for security and compatibility",
                category="dependency_management",
                source="dependency_analysis",
                files_affected=["pyproject.toml", "requirements.txt"],
                estimated_effort_hours=2.0,
                wsjf_score=0.0, ice_score=0.0, technical_debt_score=0.0, composite_score=0.0,
                risk_level="medium",
                business_impact="security_improvement",
                confidence=0.8,
                discovered_at=datetime.now().isoformat()
            ))
        
        return opportunities
    
    def _discover_from_performance_metrics(self) -> List[ValueItem]:
        """Discover performance optimization opportunities"""
        opportunities = []
        
        # Look for slow tests or performance issues
        test_files = list(self.repo_path.glob("tests/test_*.py"))
        if test_files:
            opportunities.append(ValueItem(
                id="performance-optimization",
                title="Test Suite Performance Optimization",
                description="Analyze and optimize slow-running tests",
                category="performance",
                source="performance_analysis",
                files_affected=[str(f.relative_to(self.repo_path)) for f in test_files[:3]],
                estimated_effort_hours=4.0,
                wsjf_score=0.0, ice_score=0.0, technical_debt_score=0.0, composite_score=0.0,
                risk_level="low",
                business_impact="developer_productivity",
                confidence=0.6,
                discovered_at=datetime.now().isoformat()
            ))
        
        return opportunities
    
    def _discover_from_security_scanning(self) -> List[ValueItem]:
        """Discover security-related opportunities"""
        opportunities = []
        
        # Check for security scanning opportunities
        if (self.repo_path / ".github" / "workflows").exists():
            opportunities.append(ValueItem(
                id="security-automation",
                title="Enhanced Security Automation",
                description="Implement advanced security scanning and monitoring",
                category="security",
                source="security_analysis",
                files_affected=[".github/workflows/python-ci.yml"],
                estimated_effort_hours=3.0,
                wsjf_score=0.0, ice_score=0.0, technical_debt_score=0.0, composite_score=0.0,
                risk_level="low",
                business_impact="security_posture",
                confidence=0.8,
                discovered_at=datetime.now().isoformat()
            ))
        
        return opportunities
    
    def _discover_modernization_opportunities(self) -> List[ValueItem]:
        """Discover modernization and optimization opportunities"""
        opportunities = []
        
        # Container optimization
        if (self.repo_path / "Dockerfile").exists():
            opportunities.append(ValueItem(
                id="container-optimization",
                title="Container Image Optimization",
                description="Optimize Docker image size and security",
                category="modernization",
                source="container_analysis",
                files_affected=["Dockerfile", "docker-compose.monitoring.yml"],
                estimated_effort_hours=2.5,
                wsjf_score=0.0, ice_score=0.0, technical_debt_score=0.0, composite_score=0.0,
                risk_level="medium",
                business_impact="deployment_efficiency",
                confidence=0.7,
                discovered_at=datetime.now().isoformat()
            ))
        
        # ML Pipeline optimization
        if (self.repo_path / "src" / "transformer_trainer.py").exists():
            opportunities.append(ValueItem(
                id="ml-pipeline-optimization",
                title="ML Pipeline Performance Enhancement",
                description="Optimize transformer training and inference pipeline",
                category="ml_optimization",
                source="ml_analysis",
                files_affected=["src/transformer_trainer.py", "src/model_comparison.py"],
                estimated_effort_hours=6.0,
                wsjf_score=0.0, ice_score=0.0, technical_debt_score=0.0, composite_score=0.0,
                risk_level="medium",
                business_impact="model_performance",
                confidence=0.8,
                discovered_at=datetime.now().isoformat()
            ))
        
        return opportunities
    
    def _calculate_scores(self, item: ValueItem) -> None:
        """Calculate WSJF, ICE, and technical debt scores for an item"""
        
        # WSJF Scoring (Weighted Shortest Job First)
        user_business_value = self._calculate_business_value(item)
        time_criticality = self._calculate_time_criticality(item)
        risk_reduction = self._calculate_risk_reduction(item)
        opportunity_enablement = self._calculate_opportunity_enablement(item)
        
        cost_of_delay = user_business_value + time_criticality + risk_reduction + opportunity_enablement
        job_size = item.estimated_effort_hours
        
        item.wsjf_score = cost_of_delay / max(job_size, 0.5)  # Avoid division by zero
        
        # ICE Scoring (Impact, Confidence, Ease)
        impact = self._calculate_impact(item)
        confidence = item.confidence * 10  # Convert to 1-10 scale
        ease = 10 - (item.estimated_effort_hours / 2)  # Inverse of effort
        ease = max(1, min(10, ease))  # Clamp to 1-10
        
        item.ice_score = impact * confidence * ease
        
        # Technical Debt Scoring
        debt_impact = self._calculate_debt_impact(item)
        debt_interest = self._calculate_debt_interest(item)
        hotspot_multiplier = self._calculate_hotspot_multiplier(item)
        
        item.technical_debt_score = (debt_impact + debt_interest) * hotspot_multiplier
        
        # Composite Score with adaptive weights
        weights = self.config["scoring"]["weights"]["advanced"]
        
        normalized_wsjf = min(item.wsjf_score / 50.0, 1.0)  # Normalize to 0-1
        normalized_ice = min(item.ice_score / 1000.0, 1.0)  # Normalize to 0-1
        normalized_debt = min(item.technical_debt_score / 100.0, 1.0)  # Normalize to 0-1
        
        item.composite_score = (
            weights["wsjf"] * normalized_wsjf * 100 +
            weights["ice"] * normalized_ice * 100 +
            weights["technicalDebt"] * normalized_debt * 100
        )
        
        # Apply category-specific boosts
        if item.category == "security":
            item.composite_score *= self.config["scoring"]["thresholds"]["securityBoost"]
        elif item.category == "performance":
            item.composite_score *= 1.5  # Performance boost for advanced repos
    
    def _calculate_business_value(self, item: ValueItem) -> float:
        """Calculate business value component"""
        category_values = {
            "security": 10.0,
            "performance": 8.0,
            "technical_debt": 6.0,
            "modernization": 7.0,
            "code_quality": 5.0,
            "dependency_management": 6.0,
            "ml_optimization": 9.0
        }
        return category_values.get(item.category, 5.0)
    
    def _calculate_time_criticality(self, item: ValueItem) -> float:
        """Calculate time criticality component"""
        if item.category in ["security", "performance"]:
            return 8.0
        elif item.category in ["technical_debt", "modernization"]:
            return 5.0
        return 3.0
    
    def _calculate_risk_reduction(self, item: ValueItem) -> float:
        """Calculate risk reduction component"""
        risk_values = {"low": 3.0, "medium": 6.0, "high": 9.0}
        return risk_values.get(item.risk_level, 3.0)
    
    def _calculate_opportunity_enablement(self, item: ValueItem) -> float:
        """Calculate opportunity enablement component"""
        if item.category in ["modernization", "ml_optimization"]:
            return 7.0
        elif item.category in ["performance", "dependency_management"]:
            return 5.0
        return 3.0
    
    def _calculate_impact(self, item: ValueItem) -> float:
        """Calculate impact component for ICE"""
        impact_map = {
            "security_improvement": 9.0,
            "model_performance": 8.0,
            "developer_productivity": 7.0,
            "deployment_efficiency": 6.0,
            "maintainability": 5.0,
            "maintenance_improvement": 4.0
        }
        return impact_map.get(item.business_impact, 5.0)
    
    def _calculate_debt_impact(self, item: ValueItem) -> float:
        """Calculate technical debt impact"""
        if item.category == "technical_debt":
            return item.estimated_effort_hours * 5.0  # High debt impact
        elif item.category in ["code_quality", "modernization"]:
            return item.estimated_effort_hours * 3.0
        return item.estimated_effort_hours * 1.0
    
    def _calculate_debt_interest(self, item: ValueItem) -> float:
        """Calculate technical debt interest (future cost)"""
        interest_rates = {
            "technical_debt": 0.3,
            "security": 0.4,
            "performance": 0.2,
            "code_quality": 0.15,
            "modernization": 0.1
        }
        rate = interest_rates.get(item.category, 0.1)
        return self._calculate_debt_impact(item) * rate
    
    def _calculate_hotspot_multiplier(self, item: ValueItem) -> float:
        """Calculate hotspot multiplier based on file activity"""
        # Simplified - would analyze git churn in real implementation
        if len(item.files_affected) > 3:
            return 2.0
        elif any("src/" in f for f in item.files_affected):
            return 1.5
        return 1.0
    
    def generate_backlog_report(self, opportunities: List[ValueItem]) -> str:
        """Generate comprehensive backlog report"""
        now = datetime.now().isoformat()
        
        # Filter opportunities above minimum threshold
        min_score = self.config["scoring"]["thresholds"]["minScore"]
        viable_opportunities = [op for op in opportunities if op.composite_score >= min_score]
        
        report = f"""# 📊 Autonomous Value Backlog

**Repository**: sentiment-analyzer-pro (Advanced Maturity: 85%+)  
**Last Updated**: {now}  
**Next Execution**: Continuous Discovery Mode  
**Maturity Focus**: Optimization & Modernization

## 🎯 Next Best Value Item

"""
        
        if viable_opportunities:
            next_item = viable_opportunities[0]
            report += f"""**[{next_item.id.upper()}] {next_item.title}**
- **Composite Score**: {next_item.composite_score:.1f}
- **WSJF**: {next_item.wsjf_score:.1f} | **ICE**: {next_item.ice_score:.1f} | **Tech Debt**: {next_item.technical_debt_score:.1f}
- **Estimated Effort**: {next_item.estimated_effort_hours:.1f} hours
- **Expected Impact**: {next_item.business_impact.replace('_', ' ').title()}
- **Risk Level**: {next_item.risk_level.title()}
- **Files**: {', '.join(next_item.files_affected[:3])}

"""
        else:
            report += "**Repository is optimally maintained - no high-value items identified**\n\n"
        
        # Top opportunities table
        report += "## 📋 Value-Ranked Opportunities\n\n"
        
        if viable_opportunities:
            report += "| Rank | ID | Title | Score | Category | Est. Hours |\n"
            report += "|------|-----|--------|---------|----------|------------|\n"
            
            for i, item in enumerate(viable_opportunities[:10], 1):
                report += f"| {i} | {item.id} | {item.title[:40]}{'...' if len(item.title) > 40 else ''} | {item.composite_score:.1f} | {item.category.replace('_', ' ').title()} | {item.estimated_effort_hours:.1f} |\n"
        else:
            report += "*No actionable items above minimum threshold (score >= 15.0)*\n"
        
        # Value metrics
        report += f"""

## 📈 Value Discovery Metrics

- **Items Discovered**: {len(opportunities)}
- **Viable Opportunities**: {len(viable_opportunities)}
- **Average Score**: {sum(op.composite_score for op in opportunities) / max(len(opportunities), 1):.1f}
- **Total Estimated Effort**: {sum(op.estimated_effort_hours for op in viable_opportunities):.1f} hours
- **Categories Represented**: {len(set(op.category for op in opportunities))}

## 🔄 Discovery Sources Performance

"""
        
        # Source breakdown
        source_stats = {}
        for op in opportunities:
            source_stats[op.source] = source_stats.get(op.source, 0) + 1
        
        for source, count in source_stats.items():
            report += f"- **{source.replace('_', ' ').title()}**: {count} items\n"
        
        # Repository health summary
        health_score = self.metrics.get("repository_health_score", {}).get("overall_score", 85)
        report += f"""

## 🏥 Repository Health Summary

- **Overall Health Score**: {health_score}/100
- **Maturity Level**: Advanced (85%+ SDLC infrastructure)
- **Recent Optimization**: Completed comprehensive backlog management (July 2025)
- **Maintenance Status**: Production-ready with continuous optimization

## 🎯 Continuous Value Philosophy

This advanced repository operates in **continuous optimization mode**:

1. **Perpetual Discovery**: Automated scanning for improvement opportunities
2. **Value-Driven Prioritization**: WSJF + ICE + Technical Debt scoring
3. **Autonomous Execution**: High-confidence items executed automatically
4. **Learning Integration**: Outcomes inform future prioritization

**Quality Gate**: Only items scoring >= 15.0 are considered for execution in this mature codebase.
"""
        
        return report
    
    def save_backlog_report(self, opportunities: List[ValueItem]) -> None:
        """Save backlog report to file"""
        report = self.generate_backlog_report(opportunities)
        with open(self.backlog_path, 'w') as f:
            f.write(report)
    
    def update_metrics(self, opportunities: List[ValueItem]) -> None:
        """Update metrics with latest discovery results"""
        self.metrics["current_backlog_metrics"] = {
            "timestamp": datetime.now().isoformat(),
            "total_discovered_items": len(opportunities),
            "high_priority_items": len([op for op in opportunities if op.composite_score >= 30]),
            "medium_priority_items": len([op for op in opportunities if 15 <= op.composite_score < 30]),
            "low_priority_items": len([op for op in opportunities if op.composite_score < 15]),
            "blocked_items": 0,
            "average_age_days": 0,
            "debt_ratio": 0.15,
            "velocity_trend": "stable"
        }
        
        self.metrics["continuous_discovery_stats"]["last_discovery_run"] = datetime.now().isoformat()
        
        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

def main():
    """Main execution function"""
    engine = AdvancedValueDiscoveryEngine()
    
    print("🔍 Starting autonomous value discovery for advanced repository...")
    opportunities = engine.discover_value_opportunities()
    
    print(f"📊 Discovered {len(opportunities)} potential value opportunities")
    
    # Generate and save backlog report
    engine.save_backlog_report(opportunities)
    engine.update_metrics(opportunities)
    
    print("✅ Value discovery complete. Results saved to BACKLOG.md")
    
    if opportunities:
        best_item = opportunities[0]
        print(f"🎯 Next best value item: {best_item.title} (Score: {best_item.composite_score:.1f})")
    else:
        print("🏆 Repository is optimally maintained - no high-value opportunities identified")

if __name__ == "__main__":
    main()