# app/improvement_engine.py
"""
Improvement suggestion engine for analyzing code and generating optimization recommendations
"""

import ast
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from .code_analyzer import CodeAnalyzer, QualityReport, CodeIssue, IssueSeverity, IssueCategory

logger = logging.getLogger(__name__)

class ImprovementType(Enum):
    """Types of improvements that can be suggested"""
    PERFORMANCE = "performance"
    CODE_QUALITY = "code_quality"
    MAINTAINABILITY = "maintainability"
    SECURITY = "security"
    ARCHITECTURE = "architecture"
    TESTING = "testing"

class ImprovementPriority(Enum):
    """Priority levels for improvements"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class RiskLevel(Enum):
    """Risk levels for implementing improvements"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class CodePattern:
    """Represents a code pattern that can be optimized"""
    pattern_id: str
    name: str
    description: str
    detection_regex: Optional[str] = None
    ast_node_types: List[str] = field(default_factory=list)
    improvement_suggestion: str = ""
    example_before: str = ""
    example_after: str = ""
    impact_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW

@dataclass
class Improvement:
    """Represents a specific improvement suggestion"""
    id: str
    type: ImprovementType
    priority: ImprovementPriority
    risk_level: RiskLevel
    title: str
    description: str
    affected_files: List[str]
    affected_lines: List[Tuple[str, int]] = field(default_factory=list)
    proposed_changes: Dict[str, str] = field(default_factory=dict)
    expected_benefit: str = ""
    implementation_effort: str = ""
    impact_score: float = 0.0
    confidence_score: float = 0.0
    related_issues: List[str] = field(default_factory=list)
    code_examples: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "type": self.type.value,
            "priority": self.priority.value,
            "risk_level": self.risk_level.value,
            "title": self.title,
            "description": self.description,
            "affected_files": self.affected_files,
            "affected_lines": self.affected_lines,
            "proposed_changes": self.proposed_changes,
            "expected_benefit": self.expected_benefit,
            "implementation_effort": self.implementation_effort,
            "impact_score": self.impact_score,
            "confidence_score": self.confidence_score,
            "related_issues": self.related_issues,
            "code_examples": self.code_examples,
            "created_at": self.created_at.isoformat()
        }

class PatternDetector:
    """Detects common code patterns that can be optimized"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> List[CodePattern]:
        """Initialize common optimization patterns"""
        return [
            CodePattern(
                pattern_id="string_concat_loop",
                name="String Concatenation in Loop",
                description="Inefficient string concatenation using += in loops",
                ast_node_types=["For", "While"],
                improvement_suggestion="Use list.append() and ''.join() for better performance",
                example_before="result = ''\nfor item in items:\n    result += str(item)",
                example_after="parts = []\nfor item in items:\n    parts.append(str(item))\nresult = ''.join(parts)",
                impact_score=8.0,
                risk_level=RiskLevel.LOW
            ),
            CodePattern(
                pattern_id="list_comprehension_optimization",
                name="List Comprehension Optimization",
                description="Replace loops with list comprehensions for better performance",
                detection_regex=r"\.append\(",
                improvement_suggestion="Use list comprehension instead of explicit loop with append",
                example_before="result = []\nfor x in items:\n    result.append(x * 2)",
                example_after="result = [x * 2 for x in items]",
                impact_score=6.0,
                risk_level=RiskLevel.LOW
            ),
            CodePattern(
                pattern_id="dict_get_default",
                name="Dictionary Get with Default",
                description="Use dict.get() with default instead of key checking",
                detection_regex=r"if\s+\w+\s+in\s+\w+:",
                improvement_suggestion="Use dict.get(key, default) instead of explicit key checking",
                example_before="if key in my_dict:\n    value = my_dict[key]\nelse:\n    value = default",
                example_after="value = my_dict.get(key, default)",
                impact_score=4.0,
                risk_level=RiskLevel.LOW
            ),
            CodePattern(
                pattern_id="exception_handling_optimization",
                name="Exception Handling Optimization",
                description="Optimize exception handling for better performance",
                ast_node_types=["Try"],
                improvement_suggestion="Use EAFP (Easier to Ask for Forgiveness than Permission) pattern",
                example_before="if hasattr(obj, 'method'):\n    obj.method()",
                example_after="try:\n    obj.method()\nexcept AttributeError:\n    pass",
                impact_score=5.0,
                risk_level=RiskLevel.MEDIUM
            ),
            CodePattern(
                pattern_id="generator_expression",
                name="Generator Expression Optimization",
                description="Use generator expressions for memory efficiency",
                improvement_suggestion="Replace list comprehensions with generator expressions when possible",
                example_before="sum([x * 2 for x in large_list])",
                example_after="sum(x * 2 for x in large_list)",
                impact_score=7.0,
                risk_level=RiskLevel.LOW
            ),
            CodePattern(
                pattern_id="function_caching",
                name="Function Result Caching",
                description="Add caching to expensive function calls",
                improvement_suggestion="Use functools.lru_cache for expensive computations",
                example_before="def expensive_function(n):\n    # expensive computation\n    return result",
                example_after="from functools import lru_cache\n\n@lru_cache(maxsize=128)\ndef expensive_function(n):\n    # expensive computation\n    return result",
                impact_score=9.0,
                risk_level=RiskLevel.MEDIUM
            ),
            CodePattern(
                pattern_id="set_membership",
                name="Set Membership Testing",
                description="Use sets for membership testing instead of lists",
                improvement_suggestion="Convert lists to sets for O(1) membership testing",
                example_before="if item in long_list:",
                example_after="long_set = set(long_list)\nif item in long_set:",
                impact_score=8.0,
                risk_level=RiskLevel.LOW
            ),
            CodePattern(
                pattern_id="context_manager",
                name="Context Manager Usage",
                description="Use context managers for resource management",
                improvement_suggestion="Use 'with' statements for file operations and resource management",
                example_before="f = open('file.txt')\ndata = f.read()\nf.close()",
                example_after="with open('file.txt') as f:\n    data = f.read()",
                impact_score=6.0,
                risk_level=RiskLevel.LOW
            )
        ]
    
    def detect_patterns(self, file_path: str, content: str) -> List[Tuple[CodePattern, List[int]]]:
        """Detect optimization patterns in code content"""
        detected = []
        lines = content.split('\n')
        
        for pattern in self.patterns:
            line_numbers = []
            
            # Regex-based detection
            if pattern.detection_regex:
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern.detection_regex, line):
                        line_numbers.append(line_num)
            
            # AST-based detection
            if pattern.ast_node_types:
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if node.__class__.__name__ in pattern.ast_node_types:
                            if hasattr(node, 'lineno'):
                                # Additional pattern-specific logic
                                if self._matches_pattern_context(pattern, node, lines):
                                    line_numbers.append(node.lineno)
                except SyntaxError:
                    logger.warning(f"Could not parse {file_path} for AST analysis")
            
            if line_numbers:
                detected.append((pattern, line_numbers))
        
        return detected
    
    def _matches_pattern_context(self, pattern: CodePattern, node: ast.AST, lines: List[str]) -> bool:
        """Check if AST node matches pattern-specific context"""
        if pattern.pattern_id == "string_concat_loop":
            # Check for string concatenation in loop
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if isinstance(child, ast.AugAssign) and isinstance(child.op, ast.Add):
                        return True
        
        elif pattern.pattern_id == "exception_handling_optimization":
            # Check for broad exception handling
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    if handler.type is None or (
                        isinstance(handler.type, ast.Name) and 
                        handler.type.id == "Exception"
                    ):
                        return True
        
        return False

class ImprovementEngine:
    """Main engine for generating improvement suggestions"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.code_analyzer = CodeAnalyzer(project_root)
        self.pattern_detector = PatternDetector()
        
        logger.info(f"Improvement engine initialized for project: {self.project_root}")
    
    def analyze_and_suggest_improvements(self, file_paths: Optional[List[str]] = None) -> List[Improvement]:
        """Analyze code and generate improvement suggestions"""
        improvements = []
        
        if file_paths:
            # Analyze specific files
            for file_path in file_paths:
                file_improvements = self._analyze_file_for_improvements(file_path)
                improvements.extend(file_improvements)
        else:
            # Analyze entire project
            quality_reports = self.code_analyzer.analyze_project()
            for file_path, report in quality_reports.items():
                file_improvements = self._generate_improvements_from_report(file_path, report)
                improvements.extend(file_improvements)
        
        # Prioritize improvements
        prioritized_improvements = self._prioritize_improvements(improvements)
        
        logger.info(f"Generated {len(prioritized_improvements)} improvement suggestions")
        return prioritized_improvements
    
    def _analyze_file_for_improvements(self, file_path: str) -> List[Improvement]:
        """Analyze a single file and generate improvements"""
        improvements = []
        
        # Get quality report
        report = self.code_analyzer.analyze_file(file_path)
        if not report:
            return improvements
        
        # Generate improvements from quality issues
        improvements.extend(self._generate_improvements_from_report(file_path, report))
        
        # Detect optimization patterns
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            pattern_improvements = self._generate_pattern_improvements(file_path, content)
            improvements.extend(pattern_improvements)
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
        
        return improvements
    
    def _generate_improvements_from_report(self, file_path: str, report: QualityReport) -> List[Improvement]:
        """Generate improvements from quality report issues"""
        improvements = []
        
        # Group issues by category for better suggestions
        issues_by_category = {}
        for issue in report.issues:
            category = issue.category.value
            if category not in issues_by_category:
                issues_by_category[category] = []
            issues_by_category[category].append(issue)
        
        # Generate improvements for each category
        for category, issues in issues_by_category.items():
            if len(issues) >= 3:  # Only suggest if multiple similar issues
                improvement = self._create_category_improvement(file_path, category, issues)
                if improvement:
                    improvements.append(improvement)
        
        # Generate specific improvements for high-severity issues
        for issue in report.issues:
            if issue.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH]:
                improvement = self._create_issue_improvement(file_path, issue)
                if improvement:
                    improvements.append(improvement)
        
        return improvements
    
    def _generate_pattern_improvements(self, file_path: str, content: str) -> List[Improvement]:
        """Generate improvements based on detected patterns"""
        improvements = []
        
        detected_patterns = self.pattern_detector.detect_patterns(file_path, content)
        
        for pattern, line_numbers in detected_patterns:
            improvement = Improvement(
                id=f"pattern_{pattern.pattern_id}_{hash(file_path)}",
                type=ImprovementType.PERFORMANCE,
                priority=self._calculate_priority_from_impact(pattern.impact_score),
                risk_level=pattern.risk_level,
                title=f"Optimize {pattern.name}",
                description=f"{pattern.description}\n\n{pattern.improvement_suggestion}",
                affected_files=[file_path],
                affected_lines=[(file_path, line_num) for line_num in line_numbers],
                expected_benefit=f"Performance improvement with impact score: {pattern.impact_score}/10",
                implementation_effort=self._estimate_effort(pattern.risk_level, len(line_numbers)),
                impact_score=pattern.impact_score,
                confidence_score=8.0,  # High confidence for pattern-based suggestions
                code_examples={
                    "before": pattern.example_before,
                    "after": pattern.example_after
                }
            )
            improvements.append(improvement)
        
        return improvements
    
    def _create_category_improvement(self, file_path: str, category: str, issues: List[CodeIssue]) -> Optional[Improvement]:
        """Create improvement suggestion for a category of issues"""
        if category == "performance":
            return Improvement(
                id=f"perf_category_{hash(file_path)}_{category}",
                type=ImprovementType.PERFORMANCE,
                priority=ImprovementPriority.HIGH,
                risk_level=RiskLevel.MEDIUM,
                title=f"Address Performance Issues in {Path(file_path).name}",
                description=f"Multiple performance issues detected ({len(issues)} issues). "
                           f"Consider optimizing algorithms, reducing complexity, and improving efficiency.",
                affected_files=[file_path],
                affected_lines=[(file_path, issue.line_number) for issue in issues],
                expected_benefit="Improved application performance and reduced resource usage",
                implementation_effort="Medium - requires careful analysis and testing",
                impact_score=7.0,
                confidence_score=6.0,
                related_issues=[issue.rule_id for issue in issues]
            )
        
        elif category == "complexity":
            return Improvement(
                id=f"complexity_category_{hash(file_path)}_{category}",
                type=ImprovementType.CODE_QUALITY,
                priority=ImprovementPriority.MEDIUM,
                risk_level=RiskLevel.MEDIUM,
                title=f"Reduce Code Complexity in {Path(file_path).name}",
                description=f"Multiple complexity issues detected ({len(issues)} issues). "
                           f"Consider breaking down complex functions and reducing nesting.",
                affected_files=[file_path],
                affected_lines=[(file_path, issue.line_number) for issue in issues],
                expected_benefit="Improved code readability, maintainability, and reduced bug risk",
                implementation_effort="Medium to High - requires refactoring",
                impact_score=6.0,
                confidence_score=7.0,
                related_issues=[issue.rule_id for issue in issues]
            )
        
        elif category == "maintainability":
            return Improvement(
                id=f"maint_category_{hash(file_path)}_{category}",
                type=ImprovementType.MAINTAINABILITY,
                priority=ImprovementPriority.LOW,
                risk_level=RiskLevel.LOW,
                title=f"Improve Code Maintainability in {Path(file_path).name}",
                description=f"Multiple maintainability issues detected ({len(issues)} issues). "
                           f"Consider adding documentation, removing magic numbers, and improving code style.",
                affected_files=[file_path],
                affected_lines=[(file_path, issue.line_number) for issue in issues],
                expected_benefit="Better code documentation and easier maintenance",
                implementation_effort="Low to Medium - mostly documentation and style fixes",
                impact_score=4.0,
                confidence_score=8.0,
                related_issues=[issue.rule_id for issue in issues]
            )
        
        return None
    
    def _create_issue_improvement(self, file_path: str, issue: CodeIssue) -> Optional[Improvement]:
        """Create improvement suggestion for a specific high-severity issue"""
        improvement_type = ImprovementType.CODE_QUALITY
        if issue.category == IssueCategory.PERFORMANCE:
            improvement_type = ImprovementType.PERFORMANCE
        elif issue.category == IssueCategory.SECURITY:
            improvement_type = ImprovementType.SECURITY
        
        priority = ImprovementPriority.CRITICAL if issue.severity == IssueSeverity.CRITICAL else ImprovementPriority.HIGH
        
        return Improvement(
            id=f"issue_{issue.rule_id}_{hash(file_path)}_{issue.line_number}",
            type=improvement_type,
            priority=priority,
            risk_level=RiskLevel.LOW if issue.suggestion else RiskLevel.MEDIUM,
            title=f"Fix {issue.severity.value.title()} Issue: {issue.message}",
            description=f"Issue: {issue.message}\n\n"
                       f"Location: {file_path}:{issue.line_number}\n\n"
                       f"Suggestion: {issue.suggestion or 'Manual review required'}",
            affected_files=[file_path],
            affected_lines=[(file_path, issue.line_number)],
            expected_benefit=f"Resolve {issue.severity.value} {issue.category.value} issue",
            implementation_effort="Low - specific fix required",
            impact_score=self._severity_to_impact_score(issue.severity),
            confidence_score=9.0,  # High confidence for specific issues
            code_examples={"current": issue.code_snippet} if issue.code_snippet else {}
        )
    
    def _prioritize_improvements(self, improvements: List[Improvement]) -> List[Improvement]:
        """Prioritize improvements based on impact, risk, and effort"""
        def priority_score(improvement: Improvement) -> float:
            # Base score from priority
            priority_scores = {
                ImprovementPriority.CRITICAL: 100,
                ImprovementPriority.HIGH: 80,
                ImprovementPriority.MEDIUM: 60,
                ImprovementPriority.LOW: 40
            }
            
            score = priority_scores[improvement.priority]
            
            # Adjust for impact
            score += improvement.impact_score * 5
            
            # Adjust for confidence
            score += improvement.confidence_score * 2
            
            # Penalize for risk
            risk_penalties = {
                RiskLevel.LOW: 0,
                RiskLevel.MEDIUM: -10,
                RiskLevel.HIGH: -20,
                RiskLevel.VERY_HIGH: -40
            }
            score += risk_penalties[improvement.risk_level]
            
            # Bonus for affecting multiple files (architectural improvements)
            if len(improvement.affected_files) > 1:
                score += 10
            
            return score
        
        # Sort by priority score (descending)
        prioritized = sorted(improvements, key=priority_score, reverse=True)
        
        # Remove duplicates based on similar improvements
        unique_improvements = self._remove_duplicate_improvements(prioritized)
        
        return unique_improvements
    
    def _remove_duplicate_improvements(self, improvements: List[Improvement]) -> List[Improvement]:
        """Remove duplicate or very similar improvements"""
        unique_improvements = []
        seen_combinations = set()
        
        for improvement in improvements:
            # Create a signature for the improvement
            signature = (
                improvement.type.value,
                tuple(sorted(improvement.affected_files)),
                improvement.title.lower()
            )
            
            if signature not in seen_combinations:
                seen_combinations.add(signature)
                unique_improvements.append(improvement)
        
        return unique_improvements
    
    def _calculate_priority_from_impact(self, impact_score: float) -> ImprovementPriority:
        """Calculate priority based on impact score"""
        if impact_score >= 8.0:
            return ImprovementPriority.HIGH
        elif impact_score >= 6.0:
            return ImprovementPriority.MEDIUM
        else:
            return ImprovementPriority.LOW
    
    def _estimate_effort(self, risk_level: RiskLevel, affected_lines: int) -> str:
        """Estimate implementation effort"""
        base_effort = "Low"
        
        if risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
            base_effort = "High"
        elif risk_level == RiskLevel.MEDIUM:
            base_effort = "Medium"
        
        if affected_lines > 10:
            if base_effort == "Low":
                base_effort = "Medium"
            elif base_effort == "Medium":
                base_effort = "High"
        
        return f"{base_effort} - estimated based on risk level and scope"
    
    def _severity_to_impact_score(self, severity: IssueSeverity) -> float:
        """Convert issue severity to impact score"""
        severity_scores = {
            IssueSeverity.CRITICAL: 10.0,
            IssueSeverity.HIGH: 8.0,
            IssueSeverity.MEDIUM: 6.0,
            IssueSeverity.LOW: 4.0
        }
        return severity_scores.get(severity, 5.0)
    
    def generate_improvement_report(self, improvements: List[Improvement]) -> Dict[str, Any]:
        """Generate a comprehensive improvement report"""
        if not improvements:
            return {"message": "No improvements suggested", "improvements": []}
        
        # Categorize improvements
        by_type = {}
        by_priority = {}
        by_risk = {}
        
        for improvement in improvements:
            # By type
            type_key = improvement.type.value
            if type_key not in by_type:
                by_type[type_key] = []
            by_type[type_key].append(improvement)
            
            # By priority
            priority_key = improvement.priority.value
            if priority_key not in by_priority:
                by_priority[priority_key] = []
            by_priority[priority_key].append(improvement)
            
            # By risk
            risk_key = improvement.risk_level.value
            if risk_key not in by_risk:
                by_risk[risk_key] = []
            by_risk[risk_key].append(improvement)
        
        # Calculate statistics
        total_impact = sum(imp.impact_score for imp in improvements)
        avg_confidence = sum(imp.confidence_score for imp in improvements) / len(improvements)
        
        # Get top recommendations
        top_improvements = improvements[:5]  # Already prioritized
        
        return {
            "summary": {
                "total_improvements": len(improvements),
                "total_impact_score": round(total_impact, 2),
                "average_confidence": round(avg_confidence, 2),
                "affected_files": len(set(file for imp in improvements for file in imp.affected_files))
            },
            "by_type": {k: len(v) for k, v in by_type.items()},
            "by_priority": {k: len(v) for k, v in by_priority.items()},
            "by_risk": {k: len(v) for k, v in by_risk.items()},
            "top_recommendations": [imp.to_dict() for imp in top_improvements],
            "all_improvements": [imp.to_dict() for imp in improvements]
        }