# app/code_analyzer.py
"""
Code quality analyzer for static code analysis and optimization detection
"""

import ast
import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import json

logger = logging.getLogger(__name__)

class IssueSeverity(Enum):
    """Code issue severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IssueCategory(Enum):
    """Code issue categories"""
    PERFORMANCE = "performance"
    COMPLEXITY = "complexity"
    MAINTAINABILITY = "maintainability"
    SECURITY = "security"
    STYLE = "style"
    BUG_RISK = "bug_risk"

@dataclass
class CodeIssue:
    """Represents a code quality issue"""
    file_path: str
    line_number: int
    column: int
    severity: IssueSeverity
    category: IssueCategory
    message: str
    rule_id: str
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column": self.column,
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "rule_id": self.rule_id,
            "suggestion": self.suggestion,
            "code_snippet": self.code_snippet
        }

@dataclass
class ComplexityMetrics:
    """Code complexity metrics"""
    cyclomatic_complexity: int
    cognitive_complexity: int
    lines_of_code: int
    function_count: int
    class_count: int
    max_nesting_depth: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "cognitive_complexity": self.cognitive_complexity,
            "lines_of_code": self.lines_of_code,
            "function_count": self.function_count,
            "class_count": self.class_count,
            "max_nesting_depth": self.max_nesting_depth
        }

@dataclass
class QualityReport:
    """Comprehensive code quality report"""
    file_path: str
    issues: List[CodeIssue]
    complexity_metrics: ComplexityMetrics
    quality_score: float
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "file_path": self.file_path,
            "issues": [issue.to_dict() for issue in self.issues],
            "complexity_metrics": self.complexity_metrics.to_dict(),
            "quality_score": self.quality_score,
            "recommendations": self.recommendations
        }

class PythonASTAnalyzer:
    """AST-based Python code analyzer"""
    
    def __init__(self):
        self.issues = []
        self.complexity_metrics = None
        
    def analyze_file(self, file_path: str) -> QualityReport:
        """Analyze a Python file and return quality report"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content, filename=file_path)
            
            # Reset state
            self.issues = []
            
            # Run analysis
            self._analyze_ast(tree, file_path, content)
            complexity_metrics = self._calculate_complexity(tree, content)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(self.issues, complexity_metrics)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(self.issues, complexity_metrics)
            
            return QualityReport(
                file_path=file_path,
                issues=self.issues,
                complexity_metrics=complexity_metrics,
                quality_score=quality_score,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return QualityReport(
                file_path=file_path,
                issues=[CodeIssue(
                    file_path=file_path,
                    line_number=1,
                    column=1,
                    severity=IssueSeverity.HIGH,
                    category=IssueCategory.BUG_RISK,
                    message=f"Failed to parse file: {str(e)}",
                    rule_id="PARSE_ERROR"
                )],
                complexity_metrics=ComplexityMetrics(0, 0, 0, 0, 0, 0),
                quality_score=0.0
            )
    
    def _analyze_ast(self, tree: ast.AST, file_path: str, content: str):
        """Analyze AST for various code quality issues"""
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            # Check for performance anti-patterns
            self._check_performance_issues(node, file_path, lines)
            
            # Check for complexity issues
            self._check_complexity_issues(node, file_path, lines)
            
            # Check for maintainability issues
            self._check_maintainability_issues(node, file_path, lines)
            
            # Check for potential bugs
            self._check_bug_risks(node, file_path, lines)
        
        # Check for TODO/FIXME comments separately
        self._check_todo_comments(file_path, lines)
    
    def _check_performance_issues(self, node: ast.AST, file_path: str, lines: List[str]):
        """Check for performance-related issues"""
        
        # Inefficient string concatenation in loops
        if isinstance(node, (ast.For, ast.While)):
            for child in ast.walk(node):
                if isinstance(child, ast.AugAssign) and isinstance(child.op, ast.Add):
                    if isinstance(child.target, ast.Name):
                        # Check if we're concatenating strings (simplified check)
                        if hasattr(child, 'lineno'):
                            self.issues.append(CodeIssue(
                                file_path=file_path,
                                line_number=child.lineno,
                                column=getattr(child, 'col_offset', 0),
                                severity=IssueSeverity.MEDIUM,
                                category=IssueCategory.PERFORMANCE,
                                message="Inefficient string concatenation in loop",
                                rule_id="PERF001",
                                suggestion="Use list.append() and ''.join() instead",
                                code_snippet=lines[child.lineno - 1] if child.lineno <= len(lines) else None
                            ))
        
        # Inefficient list comprehensions
        if isinstance(node, ast.ListComp) and hasattr(node, 'lineno'):
            # Check for nested loops in list comprehension
            if len(node.generators) > 2:
                self.issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=node.lineno,
                    column=getattr(node, 'col_offset', 0),
                    severity=IssueSeverity.MEDIUM,
                    category=IssueCategory.PERFORMANCE,
                    message="Complex nested list comprehension may impact performance",
                    rule_id="PERF002",
                    suggestion="Consider breaking into multiple steps or using generator expressions",
                    code_snippet=lines[node.lineno - 1] if node.lineno <= len(lines) else None
                ))
        
        # Global variable access in loops (simplified check)
        if isinstance(node, (ast.For, ast.While)) and hasattr(node, 'lineno'):
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and hasattr(child, 'lineno'):
                    # This is a simplified check - in practice, you'd need scope analysis
                    if child.id.isupper() and len(child.id) > 2:  # Convention for constants/globals
                        self.issues.append(CodeIssue(
                            file_path=file_path,
                            line_number=child.lineno,
                            column=getattr(child, 'col_offset', 0),
                            severity=IssueSeverity.LOW,
                            category=IssueCategory.PERFORMANCE,
                            message="Global variable access in loop may impact performance",
                            rule_id="PERF003",
                            suggestion="Consider caching global variables in local scope",
                            code_snippet=lines[child.lineno - 1] if child.lineno <= len(lines) else None
                        ))
        
        # Inefficient exception handling
        if isinstance(node, ast.Try) and hasattr(node, 'lineno'):
            if len(node.handlers) == 1 and node.handlers[0].type:
                if isinstance(node.handlers[0].type, ast.Name) and node.handlers[0].type.id == "Exception":
                    self.issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        column=getattr(node, 'col_offset', 0),
                        severity=IssueSeverity.MEDIUM,
                        category=IssueCategory.PERFORMANCE,
                        message="Catching broad Exception type may hide performance issues",
                        rule_id="PERF004",
                        suggestion="Catch specific exception types",
                        code_snippet=lines[node.lineno - 1] if node.lineno <= len(lines) else None
                    ))
    
    def _check_complexity_issues(self, node: ast.AST, file_path: str, lines: List[str]):
        """Check for complexity-related issues"""
        
        # Function with too many parameters
        if isinstance(node, ast.FunctionDef) and hasattr(node, 'lineno'):
            param_count = len(node.args.args)
            if hasattr(node.args, 'posonlyargs'):
                param_count += len(node.args.posonlyargs)
            if hasattr(node.args, 'kwonlyargs'):
                param_count += len(node.args.kwonlyargs)
            if node.args.vararg:
                param_count += 1
            if node.args.kwarg:
                param_count += 1
                
            if param_count > 7:
                self.issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=node.lineno,
                    column=getattr(node, 'col_offset', 0),
                    severity=IssueSeverity.MEDIUM,
                    category=IssueCategory.COMPLEXITY,
                    message=f"Function has too many parameters ({param_count})",
                    rule_id="COMP001",
                    suggestion="Consider using a configuration object or breaking into smaller functions",
                    code_snippet=lines[node.lineno - 1] if node.lineno <= len(lines) else None
                ))
        
        # Deep nesting (only check for specific nesting nodes)
        if isinstance(node, (ast.If, ast.For, ast.While, ast.With)) and hasattr(node, 'lineno'):
            nesting_depth = self._calculate_nesting_depth(node)
            if nesting_depth > 4:
                self.issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=node.lineno,
                    column=getattr(node, 'col_offset', 0),
                    severity=IssueSeverity.HIGH,
                    category=IssueCategory.COMPLEXITY,
                    message=f"Excessive nesting depth ({nesting_depth})",
                    rule_id="COMP002",
                    suggestion="Extract nested logic into separate functions",
                    code_snippet=lines[node.lineno - 1] if node.lineno <= len(lines) else None
                ))
        
        # Long function
        if isinstance(node, ast.FunctionDef) and hasattr(node, 'lineno'):
            if hasattr(node, 'end_lineno') and node.end_lineno:
                function_length = node.end_lineno - node.lineno
                if function_length > 50:
                    self.issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        column=getattr(node, 'col_offset', 0),
                        severity=IssueSeverity.MEDIUM,
                        category=IssueCategory.COMPLEXITY,
                        message=f"Function is too long ({function_length} lines)",
                        rule_id="COMP003",
                        suggestion="Break function into smaller, focused functions",
                        code_snippet=lines[node.lineno - 1] if node.lineno <= len(lines) else None
                    ))
    
    def _check_maintainability_issues(self, node: ast.AST, file_path: str, lines: List[str]):
        """Check for maintainability issues"""
        
        # Missing docstrings
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and hasattr(node, 'lineno'):
            if not ast.get_docstring(node):
                self.issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=node.lineno,
                    column=getattr(node, 'col_offset', 0),
                    severity=IssueSeverity.LOW,
                    category=IssueCategory.MAINTAINABILITY,
                    message=f"{node.__class__.__name__.lower().replace('def', '')} '{node.name}' missing docstring",
                    rule_id="MAINT001",
                    suggestion="Add descriptive docstring",
                    code_snippet=lines[node.lineno - 1] if node.lineno <= len(lines) else None
                ))
        
        # Magic numbers (updated for modern Python AST)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and hasattr(node, 'lineno'):
            if not isinstance(node.value, bool) and abs(node.value) > 1 and node.value not in [0, 1, -1, 2, 10, 100, 1000]:
                self.issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=node.lineno,
                    column=getattr(node, 'col_offset', 0),
                    severity=IssueSeverity.LOW,
                    category=IssueCategory.MAINTAINABILITY,
                    message=f"Magic number {node.value} should be a named constant",
                    rule_id="MAINT002",
                    suggestion="Define as a named constant with descriptive name",
                    code_snippet=lines[node.lineno - 1] if node.lineno <= len(lines) else None
                ))
        

    
    def _check_bug_risks(self, node: ast.AST, file_path: str, lines: List[str]):
        """Check for potential bug risks"""
        
        # Mutable default arguments
        if isinstance(node, ast.FunctionDef) and hasattr(node, 'lineno'):
            for default in node.args.defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    self.issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        column=getattr(node, 'col_offset', 0),
                        severity=IssueSeverity.HIGH,
                        category=IssueCategory.BUG_RISK,
                        message="Mutable default argument can cause unexpected behavior",
                        rule_id="BUG001",
                        suggestion="Use None as default and create mutable object inside function",
                        code_snippet=lines[node.lineno - 1] if node.lineno <= len(lines) else None
                    ))
        
        # Bare except clauses
        if isinstance(node, ast.ExceptHandler) and node.type is None and hasattr(node, 'lineno'):
            self.issues.append(CodeIssue(
                file_path=file_path,
                line_number=node.lineno,
                column=getattr(node, 'col_offset', 0),
                severity=IssueSeverity.HIGH,
                category=IssueCategory.BUG_RISK,
                message="Bare except clause can hide bugs",
                rule_id="BUG002",
                suggestion="Catch specific exception types",
                code_snippet=lines[node.lineno - 1] if node.lineno <= len(lines) else None
            ))
        
        # Comparison with None using == instead of is
        if isinstance(node, ast.Compare) and hasattr(node, 'lineno'):
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and comparator.value is None:
                    for op in node.ops:
                        if isinstance(op, (ast.Eq, ast.NotEq)):
                            self.issues.append(CodeIssue(
                                file_path=file_path,
                                line_number=node.lineno,
                                column=getattr(node, 'col_offset', 0),
                                severity=IssueSeverity.MEDIUM,
                                category=IssueCategory.BUG_RISK,
                                message="Use 'is' or 'is not' when comparing with None",
                                rule_id="BUG003",
                                suggestion="Replace == None with 'is None' and != None with 'is not None'",
                                code_snippet=lines[node.lineno - 1] if node.lineno <= len(lines) else None
                            ))
    
    def _check_todo_comments(self, file_path: str, lines: List[str]):
        """Check for TODO/FIXME comments in the code"""
        for line_num, line_content in enumerate(lines, 1):
            if re.search(r'#.*\b(TODO|FIXME|HACK)\b', line_content, re.IGNORECASE):
                self.issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=line_num,
                    column=0,
                    severity=IssueSeverity.LOW,
                    category=IssueCategory.MAINTAINABILITY,
                    message="TODO/FIXME comment found",
                    rule_id="MAINT003",
                    suggestion="Address the TODO/FIXME or create a proper issue",
                    code_snippet=line_content.strip()
                ))
    
    def _calculate_nesting_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth for a node"""
        max_depth = current_depth
        
        # Nodes that increase nesting depth
        nesting_nodes = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.FunctionDef, ast.ClassDef)
        
        if isinstance(node, nesting_nodes):
            current_depth += 1
            max_depth = current_depth
        
        # Recursively check children
        for child in ast.iter_child_nodes(node):
            child_depth = self._calculate_nesting_depth(child, current_depth)
            max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def _calculate_complexity(self, tree: ast.AST, content: str) -> ComplexityMetrics:
        """Calculate various complexity metrics"""
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        
        function_count = 0
        class_count = 0
        cyclomatic_complexity = 1  # Base complexity
        max_nesting_depth = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_count += 1
            elif isinstance(node, ast.ClassDef):
                class_count += 1
            elif isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                cyclomatic_complexity += 1
            elif isinstance(node, ast.BoolOp):
                cyclomatic_complexity += len(node.values) - 1
            
            # Calculate nesting depth
            nesting_depth = self._calculate_nesting_depth(node)
            max_nesting_depth = max(max_nesting_depth, nesting_depth)
        
        # Simplified cognitive complexity (similar to cyclomatic but with weights)
        cognitive_complexity = cyclomatic_complexity
        
        return ComplexityMetrics(
            cyclomatic_complexity=cyclomatic_complexity,
            cognitive_complexity=cognitive_complexity,
            lines_of_code=len(non_empty_lines),
            function_count=function_count,
            class_count=class_count,
            max_nesting_depth=max_nesting_depth
        )
    
    def _calculate_quality_score(self, issues: List[CodeIssue], metrics: ComplexityMetrics) -> float:
        """Calculate overall quality score (0-100)"""
        base_score = 100.0
        
        # Deduct points for issues
        for issue in issues:
            if issue.severity == IssueSeverity.CRITICAL:
                base_score -= 20
            elif issue.severity == IssueSeverity.HIGH:
                base_score -= 10
            elif issue.severity == IssueSeverity.MEDIUM:
                base_score -= 5
            elif issue.severity == IssueSeverity.LOW:
                base_score -= 2
        
        # Deduct points for complexity
        if metrics.cyclomatic_complexity > 10:
            base_score -= (metrics.cyclomatic_complexity - 10) * 2
        
        if metrics.max_nesting_depth > 4:
            base_score -= (metrics.max_nesting_depth - 4) * 5
        
        # Ensure score is between 0 and 100
        return max(0.0, min(100.0, base_score))
    
    def _generate_recommendations(self, issues: List[CodeIssue], metrics: ComplexityMetrics) -> List[str]:
        """Generate recommendations based on issues and metrics"""
        recommendations = []
        
        # Issue-based recommendations
        critical_issues = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
        high_issues = [i for i in issues if i.severity == IssueSeverity.HIGH]
        
        if critical_issues:
            recommendations.append(f"Address {len(critical_issues)} critical issues immediately")
        
        if high_issues:
            recommendations.append(f"Fix {len(high_issues)} high-severity issues")
        
        # Performance recommendations
        perf_issues = [i for i in issues if i.category == IssueCategory.PERFORMANCE]
        if perf_issues:
            recommendations.append("Optimize performance bottlenecks identified")
        
        # Complexity recommendations
        if metrics.cyclomatic_complexity > 15:
            recommendations.append("Reduce cyclomatic complexity by breaking down complex functions")
        
        if metrics.max_nesting_depth > 5:
            recommendations.append("Reduce nesting depth by extracting nested logic")
        
        # Maintainability recommendations
        maint_issues = [i for i in issues if i.category == IssueCategory.MAINTAINABILITY]
        if len(maint_issues) > 5:
            recommendations.append("Improve code maintainability by addressing documentation and style issues")
        
        if not recommendations:
            recommendations.append("Code quality is good - continue following best practices")
        
        return recommendations

class CodeAnalyzer:
    """Main code analyzer that coordinates different analysis tools"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.python_analyzer = PythonASTAnalyzer()
        
        # File patterns to analyze
        self.python_patterns = ["*.py"]
        self.exclude_patterns = [
            "__pycache__",
            ".git",
            ".pytest_cache",
            "venv",
            "env",
            "node_modules",
            "*.pyc"
        ]
        
        logger.info(f"Code analyzer initialized for project: {self.project_root}")
    
    def analyze_project(self) -> Dict[str, QualityReport]:
        """Analyze entire project and return quality reports for all files"""
        reports = {}
        
        python_files = self._find_python_files()
        
        for file_path in python_files:
            try:
                report = self.python_analyzer.analyze_file(str(file_path))
                reports[str(file_path.relative_to(self.project_root))] = report
                logger.info(f"Analyzed {file_path}: Quality score {report.quality_score:.1f}")
            except Exception as e:
                logger.error(f"Failed to analyze {file_path}: {e}")
        
        return reports
    
    def analyze_file(self, file_path: str) -> Optional[QualityReport]:
        """Analyze a single file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        if file_path.suffix == '.py':
            return self.python_analyzer.analyze_file(str(file_path))
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            return None
    
    def get_project_summary(self, reports: Dict[str, QualityReport]) -> Dict[str, Any]:
        """Generate project-wide quality summary"""
        if not reports:
            return {"error": "No reports to summarize"}
        
        total_issues = 0
        total_files = len(reports)
        quality_scores = []
        issues_by_severity = {severity.value: 0 for severity in IssueSeverity}
        issues_by_category = {category.value: 0 for category in IssueCategory}
        
        for report in reports.values():
            total_issues += len(report.issues)
            quality_scores.append(report.quality_score)
            
            for issue in report.issues:
                issues_by_severity[issue.severity.value] += 1
                issues_by_category[issue.category.value] += 1
        
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Find files with lowest quality scores
        worst_files = sorted(reports.items(), key=lambda x: x[1].quality_score)[:5]
        
        return {
            "total_files_analyzed": total_files,
            "total_issues": total_issues,
            "average_quality_score": round(avg_quality_score, 2),
            "issues_by_severity": issues_by_severity,
            "issues_by_category": issues_by_category,
            "worst_files": [
                {"file": file_path, "score": report.quality_score, "issues": len(report.issues)}
                for file_path, report in worst_files
            ],
            "recommendations": self._generate_project_recommendations(reports)
        }
    
    def _find_python_files(self) -> List[Path]:
        """Find all Python files in the project"""
        python_files = []
        
        for pattern in self.python_patterns:
            for file_path in self.project_root.rglob(pattern):
                # Check if file should be excluded
                if any(exclude in str(file_path) for exclude in self.exclude_patterns):
                    continue
                
                python_files.append(file_path)
        
        return python_files
    
    def _generate_project_recommendations(self, reports: Dict[str, QualityReport]) -> List[str]:
        """Generate project-wide recommendations"""
        recommendations = []
        
        # Count critical and high issues across project
        critical_count = sum(
            len([i for i in report.issues if i.severity == IssueSeverity.CRITICAL])
            for report in reports.values()
        )
        high_count = sum(
            len([i for i in report.issues if i.severity == IssueSeverity.HIGH])
            for report in reports.values()
        )
        
        if critical_count > 0:
            recommendations.append(f"URGENT: Address {critical_count} critical issues across the project")
        
        if high_count > 5:
            recommendations.append(f"Focus on fixing {high_count} high-severity issues")
        
        # Performance recommendations
        perf_issues = sum(
            len([i for i in report.issues if i.category == IssueCategory.PERFORMANCE])
            for report in reports.values()
        )
        if perf_issues > 10:
            recommendations.append("Consider performance optimization - multiple performance issues detected")
        
        # Complexity recommendations
        avg_complexity = sum(
            report.complexity_metrics.cyclomatic_complexity
            for report in reports.values()
        ) / len(reports) if reports else 0
        
        if avg_complexity > 8:
            recommendations.append("Reduce overall code complexity by refactoring complex functions")
        
        # Quality score recommendations
        low_quality_files = [
            file_path for file_path, report in reports.items()
            if report.quality_score < 70
        ]
        
        if len(low_quality_files) > len(reports) * 0.2:  # More than 20% of files
            recommendations.append("Focus on improving code quality in low-scoring files")
        
        if not recommendations:
            recommendations.append("Overall code quality is good - maintain current standards")
        
        return recommendations

# Global code analyzer instance
code_analyzer = CodeAnalyzer()