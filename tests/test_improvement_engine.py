# tests/test_improvement_engine.py
"""
Unit tests for the improvement suggestion engine
"""

import unittest
from unittest.mock import Mock, patch, mock_open
import tempfile
import os
from pathlib import Path
from datetime import datetime

from app.improvement_engine import (
    ImprovementEngine, PatternDetector, CodePattern, Improvement,
    ImprovementType, ImprovementPriority, RiskLevel
)
from app.code_analyzer import CodeIssue, IssueSeverity, IssueCategory, QualityReport, ComplexityMetrics


class TestCodePattern(unittest.TestCase):
    """Test CodePattern data class"""
    
    def test_code_pattern_creation(self):
        """Test creating a code pattern"""
        pattern = CodePattern(
            pattern_id="test_pattern",
            name="Test Pattern",
            description="A test pattern",
            improvement_suggestion="Use better approach",
            impact_score=7.5,
            risk_level=RiskLevel.LOW
        )
        
        self.assertEqual(pattern.pattern_id, "test_pattern")
        self.assertEqual(pattern.name, "Test Pattern")
        self.assertEqual(pattern.impact_score, 7.5)
        self.assertEqual(pattern.risk_level, RiskLevel.LOW)


class TestImprovement(unittest.TestCase):
    """Test Improvement data class"""
    
    def test_improvement_creation(self):
        """Test creating an improvement suggestion"""
        improvement = Improvement(
            id="test_improvement",
            type=ImprovementType.PERFORMANCE,
            priority=ImprovementPriority.HIGH,
            risk_level=RiskLevel.MEDIUM,
            title="Test Improvement",
            description="A test improvement",
            affected_files=["test.py"],
            expected_benefit="Better performance",
            impact_score=8.0,
            confidence_score=9.0
        )
        
        self.assertEqual(improvement.id, "test_improvement")
        self.assertEqual(improvement.type, ImprovementType.PERFORMANCE)
        self.assertEqual(improvement.priority, ImprovementPriority.HIGH)
        self.assertEqual(improvement.impact_score, 8.0)
        self.assertEqual(improvement.affected_files, ["test.py"])
    
    def test_improvement_to_dict(self):
        """Test converting improvement to dictionary"""
        improvement = Improvement(
            id="test_improvement",
            type=ImprovementType.PERFORMANCE,
            priority=ImprovementPriority.HIGH,
            risk_level=RiskLevel.MEDIUM,
            title="Test Improvement",
            description="A test improvement",
            affected_files=["test.py"],
            impact_score=8.0,
            confidence_score=9.0
        )
        
        result = improvement.to_dict()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["id"], "test_improvement")
        self.assertEqual(result["type"], "performance")
        self.assertEqual(result["priority"], "high")
        self.assertEqual(result["risk_level"], "medium")
        self.assertIn("created_at", result)


class TestPatternDetector(unittest.TestCase):
    """Test PatternDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = PatternDetector()
    
    def test_initialization(self):
        """Test pattern detector initialization"""
        self.assertIsInstance(self.detector.patterns, list)
        self.assertGreater(len(self.detector.patterns), 0)
        
        # Check that all patterns have required fields
        for pattern in self.detector.patterns:
            self.assertIsInstance(pattern, CodePattern)
            self.assertTrue(pattern.pattern_id)
            self.assertTrue(pattern.name)
            self.assertTrue(pattern.description)
    
    def test_string_concatenation_detection(self):
        """Test detection of string concatenation in loops"""
        code = '''
result = ""
for item in items:
    result += str(item)
'''
        
        detected = self.detector.detect_patterns("test.py", code)
        
        # Should detect string concatenation pattern
        pattern_ids = [pattern.pattern_id for pattern, _ in detected]
        self.assertIn("string_concat_loop", pattern_ids)
    
    def test_list_comprehension_detection(self):
        """Test detection of list comprehension opportunities"""
        code = '''
result = []
for x in items:
    result.append(x * 2)
'''
        
        detected = self.detector.detect_patterns("test.py", code)
        
        # Should detect list comprehension pattern
        pattern_ids = [pattern.pattern_id for pattern, _ in detected]
        self.assertIn("list_comprehension_optimization", pattern_ids)
    
    def test_dict_get_detection(self):
        """Test detection of dictionary get optimization"""
        code = '''
if key in my_dict:
    value = my_dict[key]
else:
    value = default_value
'''
        
        detected = self.detector.detect_patterns("test.py", code)
        
        # Should detect dict get pattern
        pattern_ids = [pattern.pattern_id for pattern, _ in detected]
        self.assertIn("dict_get_default", pattern_ids)
    
    def test_no_patterns_in_clean_code(self):
        """Test that clean code doesn't trigger false positives"""
        code = '''
def clean_function(items):
    """A well-written function."""
    return [x * 2 for x in items if x > 0]

result = my_dict.get(key, default_value)
'''
        
        detected = self.detector.detect_patterns("test.py", code)
        
        # Should not detect any patterns in clean code
        self.assertEqual(len(detected), 0)
    
    def test_pattern_context_matching(self):
        """Test pattern context matching for AST nodes"""
        # Test string concatenation context
        code_with_concat = '''
for i in range(10):
    result += str(i)
'''
        
        code_without_concat = '''
for i in range(10):
    print(i)
'''
        
        detected_with = self.detector.detect_patterns("test.py", code_with_concat)
        detected_without = self.detector.detect_patterns("test.py", code_without_concat)
        
        # Should detect pattern in first case but not second
        with_pattern_ids = [pattern.pattern_id for pattern, _ in detected_with]
        without_pattern_ids = [pattern.pattern_id for pattern, _ in detected_without]
        
        self.assertIn("string_concat_loop", with_pattern_ids)
        self.assertNotIn("string_concat_loop", without_pattern_ids)


class TestImprovementEngine(unittest.TestCase):
    """Test ImprovementEngine class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.engine = ImprovementEngine(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test improvement engine initialization"""
        self.assertEqual(self.engine.project_root, Path(self.temp_dir))
        self.assertIsNotNone(self.engine.code_analyzer)
        self.assertIsNotNone(self.engine.pattern_detector)
    
    @patch('app.improvement_engine.ImprovementEngine._analyze_file_for_improvements')
    def test_analyze_specific_files(self, mock_analyze_file):
        """Test analyzing specific files"""
        mock_analyze_file.return_value = [
            Improvement(
                id="test_improvement",
                type=ImprovementType.PERFORMANCE,
                priority=ImprovementPriority.HIGH,
                risk_level=RiskLevel.LOW,
                title="Test Improvement",
                description="Test description",
                affected_files=["test.py"],
                impact_score=8.0,
                confidence_score=9.0
            )
        ]
        
        result = self.engine.analyze_and_suggest_improvements(["test.py"])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].title, "Test Improvement")
        mock_analyze_file.assert_called_once_with("test.py")
    
    def test_create_category_improvement_performance(self):
        """Test creating category improvement for performance issues"""
        issues = [
            CodeIssue(
                file_path="test.py",
                line_number=10,
                column=0,
                severity=IssueSeverity.MEDIUM,
                category=IssueCategory.PERFORMANCE,
                message="Performance issue 1",
                rule_id="PERF001"
            ),
            CodeIssue(
                file_path="test.py",
                line_number=20,
                column=0,
                severity=IssueSeverity.MEDIUM,
                category=IssueCategory.PERFORMANCE,
                message="Performance issue 2",
                rule_id="PERF002"
            ),
            CodeIssue(
                file_path="test.py",
                line_number=30,
                column=0,
                severity=IssueSeverity.HIGH,
                category=IssueCategory.PERFORMANCE,
                message="Performance issue 3",
                rule_id="PERF003"
            )
        ]
        
        improvement = self.engine._create_category_improvement("test.py", "performance", issues)
        
        self.assertIsNotNone(improvement)
        self.assertEqual(improvement.type, ImprovementType.PERFORMANCE)
        self.assertEqual(improvement.priority, ImprovementPriority.HIGH)
        self.assertEqual(len(improvement.affected_lines), 3)
        self.assertIn("Performance Issues", improvement.title)
    
    def test_create_category_improvement_complexity(self):
        """Test creating category improvement for complexity issues"""
        issues = [
            CodeIssue(
                file_path="test.py",
                line_number=10,
                column=0,
                severity=IssueSeverity.MEDIUM,
                category=IssueCategory.COMPLEXITY,
                message="Complex function",
                rule_id="COMP001"
            ),
            CodeIssue(
                file_path="test.py",
                line_number=20,
                column=0,
                severity=IssueSeverity.HIGH,
                category=IssueCategory.COMPLEXITY,
                message="Deep nesting",
                rule_id="COMP002"
            ),
            CodeIssue(
                file_path="test.py",
                line_number=30,
                column=0,
                severity=IssueSeverity.MEDIUM,
                category=IssueCategory.COMPLEXITY,
                message="Long function",
                rule_id="COMP003"
            )
        ]
        
        improvement = self.engine._create_category_improvement("test.py", "complexity", issues)
        
        self.assertIsNotNone(improvement)
        self.assertEqual(improvement.type, ImprovementType.CODE_QUALITY)
        self.assertEqual(improvement.priority, ImprovementPriority.MEDIUM)
        self.assertIn("Complexity", improvement.title)
    
    def test_create_issue_improvement(self):
        """Test creating improvement for specific issue"""
        issue = CodeIssue(
            file_path="test.py",
            line_number=15,
            column=4,
            severity=IssueSeverity.CRITICAL,
            category=IssueCategory.SECURITY,
            message="SQL injection vulnerability",
            rule_id="SEC001",
            suggestion="Use parameterized queries",
            code_snippet="cursor.execute(f'SELECT * FROM users WHERE id = {user_id}')"
        )
        
        improvement = self.engine._create_issue_improvement("test.py", issue)
        
        self.assertIsNotNone(improvement)
        self.assertEqual(improvement.type, ImprovementType.SECURITY)
        self.assertEqual(improvement.priority, ImprovementPriority.CRITICAL)
        self.assertEqual(improvement.affected_lines, [("test.py", 15)])
        self.assertIn("SQL injection", improvement.title)
        self.assertIn("parameterized queries", improvement.description)
    
    def test_prioritize_improvements(self):
        """Test improvement prioritization"""
        improvements = [
            Improvement(
                id="low_priority",
                type=ImprovementType.MAINTAINABILITY,
                priority=ImprovementPriority.LOW,
                risk_level=RiskLevel.LOW,
                title="Low Priority",
                description="Low priority improvement",
                affected_files=["test.py"],
                impact_score=3.0,
                confidence_score=5.0
            ),
            Improvement(
                id="high_priority",
                type=ImprovementType.PERFORMANCE,
                priority=ImprovementPriority.CRITICAL,
                risk_level=RiskLevel.LOW,
                title="High Priority",
                description="High priority improvement",
                affected_files=["test.py"],
                impact_score=9.0,
                confidence_score=9.0
            ),
            Improvement(
                id="medium_priority",
                type=ImprovementType.CODE_QUALITY,
                priority=ImprovementPriority.MEDIUM,
                risk_level=RiskLevel.MEDIUM,
                title="Medium Priority",
                description="Medium priority improvement",
                affected_files=["test.py"],
                impact_score=6.0,
                confidence_score=7.0
            )
        ]
        
        prioritized = self.engine._prioritize_improvements(improvements)
        
        # Should be ordered by priority (high to low)
        self.assertEqual(prioritized[0].id, "high_priority")
        self.assertEqual(prioritized[1].id, "medium_priority")
        self.assertEqual(prioritized[2].id, "low_priority")
    
    def test_remove_duplicate_improvements(self):
        """Test removing duplicate improvements"""
        improvements = [
            Improvement(
                id="improvement1",
                type=ImprovementType.PERFORMANCE,
                priority=ImprovementPriority.HIGH,
                risk_level=RiskLevel.LOW,
                title="Performance Improvement",
                description="First improvement",
                affected_files=["test.py"],
                impact_score=8.0,
                confidence_score=9.0
            ),
            Improvement(
                id="improvement2",
                type=ImprovementType.PERFORMANCE,
                priority=ImprovementPriority.HIGH,
                risk_level=RiskLevel.LOW,
                title="Performance Improvement",  # Same title
                description="Second improvement",
                affected_files=["test.py"],  # Same files
                impact_score=7.0,
                confidence_score=8.0
            ),
            Improvement(
                id="improvement3",
                type=ImprovementType.CODE_QUALITY,
                priority=ImprovementPriority.MEDIUM,
                risk_level=RiskLevel.LOW,
                title="Quality Improvement",
                description="Different improvement",
                affected_files=["other.py"],
                impact_score=6.0,
                confidence_score=7.0
            )
        ]
        
        unique = self.engine._remove_duplicate_improvements(improvements)
        
        # Should remove the duplicate performance improvement
        self.assertEqual(len(unique), 2)
        titles = [imp.title for imp in unique]
        self.assertIn("Performance Improvement", titles)
        self.assertIn("Quality Improvement", titles)
    
    def test_calculate_priority_from_impact(self):
        """Test priority calculation from impact score"""
        self.assertEqual(
            self.engine._calculate_priority_from_impact(9.0),
            ImprovementPriority.HIGH
        )
        self.assertEqual(
            self.engine._calculate_priority_from_impact(7.0),
            ImprovementPriority.MEDIUM
        )
        self.assertEqual(
            self.engine._calculate_priority_from_impact(4.0),
            ImprovementPriority.LOW
        )
    
    def test_estimate_effort(self):
        """Test effort estimation"""
        # Low risk, few lines
        effort = self.engine._estimate_effort(RiskLevel.LOW, 5)
        self.assertIn("Low", effort)
        
        # High risk, many lines
        effort = self.engine._estimate_effort(RiskLevel.HIGH, 20)
        self.assertIn("High", effort)
        
        # Medium risk, medium lines
        effort = self.engine._estimate_effort(RiskLevel.MEDIUM, 8)
        self.assertIn("Medium", effort)
    
    def test_severity_to_impact_score(self):
        """Test converting severity to impact score"""
        self.assertEqual(
            self.engine._severity_to_impact_score(IssueSeverity.CRITICAL),
            10.0
        )
        self.assertEqual(
            self.engine._severity_to_impact_score(IssueSeverity.HIGH),
            8.0
        )
        self.assertEqual(
            self.engine._severity_to_impact_score(IssueSeverity.MEDIUM),
            6.0
        )
        self.assertEqual(
            self.engine._severity_to_impact_score(IssueSeverity.LOW),
            4.0
        )
    
    def test_generate_improvement_report(self):
        """Test generating improvement report"""
        improvements = [
            Improvement(
                id="perf_improvement",
                type=ImprovementType.PERFORMANCE,
                priority=ImprovementPriority.HIGH,
                risk_level=RiskLevel.LOW,
                title="Performance Improvement",
                description="Improve performance",
                affected_files=["test.py"],
                impact_score=8.0,
                confidence_score=9.0
            ),
            Improvement(
                id="quality_improvement",
                type=ImprovementType.CODE_QUALITY,
                priority=ImprovementPriority.MEDIUM,
                risk_level=RiskLevel.MEDIUM,
                title="Quality Improvement",
                description="Improve quality",
                affected_files=["test.py"],
                impact_score=6.0,
                confidence_score=7.0
            )
        ]
        
        report = self.engine.generate_improvement_report(improvements)
        
        self.assertIn("summary", report)
        self.assertIn("by_type", report)
        self.assertIn("by_priority", report)
        self.assertIn("by_risk", report)
        self.assertIn("top_recommendations", report)
        self.assertIn("all_improvements", report)
        
        # Check summary
        summary = report["summary"]
        self.assertEqual(summary["total_improvements"], 2)
        self.assertEqual(summary["total_impact_score"], 14.0)
        self.assertEqual(summary["average_confidence"], 8.0)
        self.assertEqual(summary["affected_files"], 1)
        
        # Check categorization
        self.assertEqual(report["by_type"]["performance"], 1)
        self.assertEqual(report["by_type"]["code_quality"], 1)
        self.assertEqual(report["by_priority"]["high"], 1)
        self.assertEqual(report["by_priority"]["medium"], 1)
    
    def test_generate_improvement_report_empty(self):
        """Test generating report with no improvements"""
        report = self.engine.generate_improvement_report([])
        
        self.assertIn("message", report)
        self.assertEqual(report["message"], "No improvements suggested")
        self.assertEqual(report["improvements"], [])
    
    @patch('builtins.open', new_callable=mock_open, read_data='''
def inefficient_function():
    result = ""
    for i in range(100):
        result += str(i)
    return result
''')
    @patch('app.improvement_engine.CodeAnalyzer.analyze_file')
    def test_analyze_file_for_improvements(self, mock_analyze_file, mock_file):
        """Test analyzing a file for improvements"""
        # Mock quality report
        mock_report = QualityReport(
            file_path="test.py",
            issues=[
                CodeIssue(
                    file_path="test.py",
                    line_number=4,
                    column=8,
                    severity=IssueSeverity.MEDIUM,
                    category=IssueCategory.PERFORMANCE,
                    message="Inefficient string concatenation",
                    rule_id="PERF001"
                )
            ],
            complexity_metrics=ComplexityMetrics(5, 5, 10, 1, 0, 2),
            quality_score=75.0
        )
        mock_analyze_file.return_value = mock_report
        
        improvements = self.engine._analyze_file_for_improvements("test.py")
        
        self.assertGreater(len(improvements), 0)
        
        # Should detect both quality issues and patterns
        improvement_types = [imp.type for imp in improvements]
        self.assertIn(ImprovementType.PERFORMANCE, improvement_types)


if __name__ == '__main__':
    unittest.main()