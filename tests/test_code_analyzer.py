# tests/test_code_analyzer.py
"""
Unit tests for code analyzer
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from app.code_analyzer import (
    CodeAnalyzer, PythonASTAnalyzer, CodeIssue, ComplexityMetrics, QualityReport,
    IssueSeverity, IssueCategory
)

class TestPythonASTAnalyzer:
    """Test Python AST analyzer functionality"""
    
    @pytest.fixture
    def analyzer(self):
        """Create Python AST analyzer"""
        return PythonASTAnalyzer()
    
    @pytest.fixture
    def temp_file(self):
        """Create temporary Python file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            yield f
        os.unlink(f.name)
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.issues == []
        assert analyzer.complexity_metrics is None
    
    def test_analyze_simple_file(self, analyzer, temp_file):
        """Test analyzing a simple Python file"""
        # Write simple Python code
        temp_file.write("""
def hello_world():
    \"\"\"Simple function\"\"\"
    print("Hello, World!")
    return "Hello"

class SimpleClass:
    \"\"\"Simple class\"\"\"
    def __init__(self):
        self.value = 42
""")
        temp_file.flush()
        
        report = analyzer.analyze_file(temp_file.name)
        
        assert isinstance(report, QualityReport)
        assert report.file_path == temp_file.name
        assert report.quality_score > 80  # Should be high quality
        assert report.complexity_metrics.function_count == 2  # hello_world + __init__
        assert report.complexity_metrics.class_count == 1
    
    def test_detect_performance_issues(self, analyzer, temp_file):
        """Test detection of performance issues"""
        # Write code with performance issues
        temp_file.write("""
def bad_string_concat():
    result = ""
    for i in range(100):
        result += str(i)  # Inefficient string concatenation
    return result

def complex_list_comp():
    return [x*y*z for x in range(10) for y in range(10) for z in range(10)]  # Complex nested comprehension
""")
        temp_file.flush()
        
        report = analyzer.analyze_file(temp_file.name)
        
        # Should detect performance issues
        perf_issues = [i for i in report.issues if i.category == IssueCategory.PERFORMANCE]
        assert len(perf_issues) > 0
        
        # Check for specific performance issues
        string_concat_issues = [i for i in perf_issues if "string concatenation" in i.message]
        assert len(string_concat_issues) > 0
        
        complex_comp_issues = [i for i in perf_issues if "nested list comprehension" in i.message]
        assert len(complex_comp_issues) > 0
    
    def test_detect_complexity_issues(self, analyzer, temp_file):
        """Test detection of complexity issues"""
        # Write code with complexity issues
        temp_file.write("""
def too_many_params(a, b, c, d, e, f, g, h, i):  # Too many parameters
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:  # Deep nesting
                        return f + g + h + i
    return 0

def very_long_function():
    # This function is intentionally long
    x = 1
    y = 2
    z = 3
    # ... imagine 50+ lines of code here
    for i in range(50):
        print(f"Line {i}")
    return x + y + z
""")
        temp_file.flush()
        
        report = analyzer.analyze_file(temp_file.name)
        
        # Should detect complexity issues
        comp_issues = [i for i in report.issues if i.category == IssueCategory.COMPLEXITY]
        assert len(comp_issues) > 0
        
        # Check for specific complexity issues
        param_issues = [i for i in comp_issues if "too many parameters" in i.message]
        assert len(param_issues) > 0
        
        nesting_issues = [i for i in comp_issues if "nesting depth" in i.message]
        assert len(nesting_issues) > 0
    
    def test_detect_maintainability_issues(self, analyzer, temp_file):
        """Test detection of maintainability issues"""
        # Write code with maintainability issues
        temp_file.write("""
def no_docstring_function():  # Missing docstring
    magic_number = 42  # Magic number
    # TODO: Fix this later
    return magic_number * 3.14159  # Another magic number

class NoDocstringClass:  # Missing docstring
    def method(self):
        return 100  # Magic number
""")
        temp_file.flush()
        
        report = analyzer.analyze_file(temp_file.name)
        
        # Should detect maintainability issues
        maint_issues = [i for i in report.issues if i.category == IssueCategory.MAINTAINABILITY]
        assert len(maint_issues) > 0
        
        # Check for specific maintainability issues
        docstring_issues = [i for i in maint_issues if "missing docstring" in i.message]
        assert len(docstring_issues) > 0
        
        magic_number_issues = [i for i in maint_issues if "Magic number" in i.message]
        assert len(magic_number_issues) > 0
        
        todo_issues = [i for i in maint_issues if "TODO" in i.message]
        assert len(todo_issues) > 0
    
    def test_detect_bug_risks(self, analyzer, temp_file):
        """Test detection of potential bug risks"""
        # Write code with bug risks
        temp_file.write("""
def mutable_default(items=[]):  # Mutable default argument
    items.append(1)
    return items

def bad_exception_handling():
    try:
        risky_operation()
    except:  # Bare except
        pass

def none_comparison(value):
    if value == None:  # Should use 'is None'
        return True
    return False
""")
        temp_file.flush()
        
        report = analyzer.analyze_file(temp_file.name)
        
        # Should detect bug risks
        bug_issues = [i for i in report.issues if i.category == IssueCategory.BUG_RISK]
        assert len(bug_issues) > 0
        
        # Check for specific bug risks
        mutable_default_issues = [i for i in bug_issues if "Mutable default argument" in i.message]
        assert len(mutable_default_issues) > 0
        
        bare_except_issues = [i for i in bug_issues if "Bare except" in i.message]
        assert len(bare_except_issues) > 0
        
        none_comparison_issues = [i for i in bug_issues if "Use 'is'" in i.message]
        assert len(none_comparison_issues) > 0
    
    def test_complexity_metrics_calculation(self, analyzer, temp_file):
        """Test complexity metrics calculation"""
        # Write code with known complexity
        temp_file.write("""
def complex_function(x):
    \"\"\"Function with known complexity\"\"\"
    if x > 0:  # +1 cyclomatic
        for i in range(x):  # +1 cyclomatic
            if i % 2 == 0:  # +1 cyclomatic
                print(i)
    elif x < 0:  # +1 cyclomatic
        while x < 0:  # +1 cyclomatic
            x += 1
    return x

class TestClass:
    \"\"\"Test class\"\"\"
    def method1(self):
        pass
    
    def method2(self):
        pass
""")
        temp_file.flush()
        
        report = analyzer.analyze_file(temp_file.name)
        metrics = report.complexity_metrics
        
        assert metrics.function_count == 3  # complex_function + 2 methods
        assert metrics.class_count == 1
        assert metrics.cyclomatic_complexity >= 5  # Base + conditions
        assert metrics.lines_of_code > 0
        assert metrics.max_nesting_depth >= 2  # if + for
    
    def test_quality_score_calculation(self, analyzer, temp_file):
        """Test quality score calculation"""
        # Write high-quality code
        temp_file.write("""
def well_written_function(name: str) -> str:
    \"\"\"
    A well-written function with proper documentation.
    
    Args:
        name: The name to greet
        
    Returns:
        A greeting message
    \"\"\"
    if not name:
        raise ValueError("Name cannot be empty")
    
    return f"Hello, {name}!"

class WellWrittenClass:
    \"\"\"A well-documented class.\"\"\"
    
    def __init__(self, value: int):
        \"\"\"Initialize with a value.\"\"\"
        self.value = value
    
    def get_value(self) -> int:
        \"\"\"Get the stored value.\"\"\"
        return self.value
""")
        temp_file.flush()
        
        report = analyzer.analyze_file(temp_file.name)
        
        # High-quality code should have a high score
        assert report.quality_score > 85
        assert len(report.issues) <= 2  # Should have very few issues
    
    def test_recommendations_generation(self, analyzer, temp_file):
        """Test recommendations generation"""
        # Write code with various issues
        temp_file.write("""
def problematic_function():
    try:
        result = ""
        for i in range(1000):
            result += str(i)
    except:
        pass
    return result
""")
        temp_file.flush()
        
        report = analyzer.analyze_file(temp_file.name)
        
        assert len(report.recommendations) > 0
        assert any("performance" in rec.lower() or "optimization" in rec.lower() 
                  for rec in report.recommendations)
    
    def test_invalid_file_handling(self, analyzer):
        """Test handling of invalid Python files"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Write invalid Python syntax
            f.write("def invalid_syntax(\n  # Missing closing parenthesis")
            f.flush()
            
            report = analyzer.analyze_file(f.name)
            
            # Should handle parse errors gracefully
            assert isinstance(report, QualityReport)
            assert report.quality_score == 0.0
            assert len(report.issues) > 0
            assert any("parse" in issue.message.lower() for issue in report.issues)
        
        os.unlink(f.name)

class TestCodeAnalyzer:
    """Test main code analyzer functionality"""
    
    @pytest.fixture
    def temp_project(self):
        """Create temporary project directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Create some Python files
            (project_path / "main.py").write_text("""
def main():
    \"\"\"Main function\"\"\"
    print("Hello, World!")

if __name__ == "__main__":
    main()
""")
            
            (project_path / "utils.py").write_text("""
def helper_function():
    return 42

class UtilityClass:
    def __init__(self):
        self.value = 100
""")
            
            # Create subdirectory with more files
            sub_dir = project_path / "submodule"
            sub_dir.mkdir()
            (sub_dir / "module.py").write_text("""
def sub_function():
    \"\"\"Function in submodule\"\"\"
    return "sub"
""")
            
            # Create files to exclude
            (project_path / "__pycache__").mkdir()
            (project_path / "__pycache__" / "main.cpython-39.pyc").write_text("compiled")
            
            yield project_path
    
    def test_analyzer_initialization(self, temp_project):
        """Test code analyzer initialization"""
        analyzer = CodeAnalyzer(str(temp_project))
        
        assert analyzer.project_root == temp_project
        assert isinstance(analyzer.python_analyzer, PythonASTAnalyzer)
        assert "*.py" in analyzer.python_patterns
        assert "__pycache__" in analyzer.exclude_patterns
    
    def test_find_python_files(self, temp_project):
        """Test finding Python files in project"""
        analyzer = CodeAnalyzer(str(temp_project))
        python_files = analyzer._find_python_files()
        
        # Should find 3 Python files (main.py, utils.py, submodule/module.py)
        assert len(python_files) == 3
        
        file_names = [f.name for f in python_files]
        assert "main.py" in file_names
        assert "utils.py" in file_names
        assert "module.py" in file_names
        
        # Should not include __pycache__ files
        assert not any("__pycache__" in str(f) for f in python_files)
    
    def test_analyze_project(self, temp_project):
        """Test analyzing entire project"""
        analyzer = CodeAnalyzer(str(temp_project))
        reports = analyzer.analyze_project()
        
        # Should have reports for all Python files
        assert len(reports) == 3
        
        # Check that all reports are valid
        for file_path, report in reports.items():
            assert isinstance(report, QualityReport)
            assert report.quality_score >= 0
            assert report.complexity_metrics is not None
    
    def test_analyze_single_file(self, temp_project):
        """Test analyzing a single file"""
        analyzer = CodeAnalyzer(str(temp_project))
        main_file = temp_project / "main.py"
        
        report = analyzer.analyze_file(str(main_file))
        
        assert isinstance(report, QualityReport)
        assert report.file_path == str(main_file)
        assert report.quality_score > 0
    
    def test_analyze_nonexistent_file(self, temp_project):
        """Test analyzing non-existent file"""
        analyzer = CodeAnalyzer(str(temp_project))
        
        report = analyzer.analyze_file("nonexistent.py")
        
        assert report is None
    
    def test_analyze_unsupported_file(self, temp_project):
        """Test analyzing unsupported file type"""
        analyzer = CodeAnalyzer(str(temp_project))
        
        # Create a non-Python file
        txt_file = temp_project / "readme.txt"
        txt_file.write_text("This is a text file")
        
        report = analyzer.analyze_file(str(txt_file))
        
        assert report is None
    
    def test_project_summary_generation(self, temp_project):
        """Test project summary generation"""
        analyzer = CodeAnalyzer(str(temp_project))
        reports = analyzer.analyze_project()
        summary = analyzer.get_project_summary(reports)
        
        assert "total_files_analyzed" in summary
        assert "total_issues" in summary
        assert "average_quality_score" in summary
        assert "issues_by_severity" in summary
        assert "issues_by_category" in summary
        assert "worst_files" in summary
        assert "recommendations" in summary
        
        assert summary["total_files_analyzed"] == 3
        assert summary["average_quality_score"] >= 0
        assert isinstance(summary["recommendations"], list)
    
    def test_project_summary_empty_reports(self, temp_project):
        """Test project summary with empty reports"""
        analyzer = CodeAnalyzer(str(temp_project))
        summary = analyzer.get_project_summary({})
        
        assert "error" in summary
    
    def test_project_recommendations(self, temp_project):
        """Test project-wide recommendations"""
        analyzer = CodeAnalyzer(str(temp_project))
        
        # Create a file with issues to test recommendations
        bad_file = temp_project / "bad_code.py"
        bad_file.write_text("""
def bad_function():
    try:
        result = ""
        for i in range(1000):
            result += str(i)
    except:
        pass
    return result

def another_bad_function(a, b, c, d, e, f, g, h):
    if a:
        if b:
            if c:
                if d:
                    if e:
                        return f + g + h
""")
        
        reports = analyzer.analyze_project()
        summary = analyzer.get_project_summary(reports)
        
        recommendations = summary["recommendations"]
        assert len(recommendations) > 0
        
        # Should have recommendations for the issues we introduced
        rec_text = " ".join(recommendations).lower()
        assert any(keyword in rec_text for keyword in ["performance", "complexity", "quality"])

class TestCodeIssue:
    """Test CodeIssue data class"""
    
    def test_code_issue_creation(self):
        """Test creating a code issue"""
        issue = CodeIssue(
            file_path="test.py",
            line_number=10,
            column=5,
            severity=IssueSeverity.HIGH,
            category=IssueCategory.PERFORMANCE,
            message="Test issue",
            rule_id="TEST001",
            suggestion="Fix this",
            code_snippet="print('test')"
        )
        
        assert issue.file_path == "test.py"
        assert issue.line_number == 10
        assert issue.severity == IssueSeverity.HIGH
        assert issue.category == IssueCategory.PERFORMANCE
        assert issue.suggestion == "Fix this"
    
    def test_code_issue_to_dict(self):
        """Test converting code issue to dictionary"""
        issue = CodeIssue(
            file_path="test.py",
            line_number=10,
            column=5,
            severity=IssueSeverity.HIGH,
            category=IssueCategory.PERFORMANCE,
            message="Test issue",
            rule_id="TEST001"
        )
        
        issue_dict = issue.to_dict()
        
        assert issue_dict["file_path"] == "test.py"
        assert issue_dict["line_number"] == 10
        assert issue_dict["severity"] == "high"
        assert issue_dict["category"] == "performance"
        assert issue_dict["message"] == "Test issue"
        assert issue_dict["rule_id"] == "TEST001"

class TestComplexityMetrics:
    """Test ComplexityMetrics data class"""
    
    def test_complexity_metrics_creation(self):
        """Test creating complexity metrics"""
        metrics = ComplexityMetrics(
            cyclomatic_complexity=5,
            cognitive_complexity=7,
            lines_of_code=100,
            function_count=10,
            class_count=2,
            max_nesting_depth=3
        )
        
        assert metrics.cyclomatic_complexity == 5
        assert metrics.cognitive_complexity == 7
        assert metrics.lines_of_code == 100
        assert metrics.function_count == 10
        assert metrics.class_count == 2
        assert metrics.max_nesting_depth == 3
    
    def test_complexity_metrics_to_dict(self):
        """Test converting complexity metrics to dictionary"""
        metrics = ComplexityMetrics(
            cyclomatic_complexity=5,
            cognitive_complexity=7,
            lines_of_code=100,
            function_count=10,
            class_count=2,
            max_nesting_depth=3
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict["cyclomatic_complexity"] == 5
        assert metrics_dict["cognitive_complexity"] == 7
        assert metrics_dict["lines_of_code"] == 100
        assert metrics_dict["function_count"] == 10
        assert metrics_dict["class_count"] == 2
        assert metrics_dict["max_nesting_depth"] == 3

if __name__ == "__main__":
    pytest.main([__file__])