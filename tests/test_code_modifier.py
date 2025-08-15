"""
Unit tests for code modification engine
"""

import pytest
import tempfile
import shutil
import os
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.code_modifier import (
    CodeModifier, FileModification, ModificationPlan, ValidationResult
)


class TestFileModification:
    """Test cases for FileModification model"""
    
    def test_file_modification_creation(self):
        """Test creating a FileModification"""
        modification = FileModification(
            file_path="test.py",
            original_content="print('hello')",
            modified_content="print('world')",
            modification_type="update"
        )
        
        assert modification.file_path == "test.py"
        assert modification.original_content == "print('hello')"
        assert modification.modified_content == "print('world')"
        assert modification.modification_type == "update"
        assert modification.backup_path is None
        assert modification.checksum_original is None
        assert modification.checksum_modified is None


class TestModificationPlan:
    """Test cases for ModificationPlan model"""
    
    def test_modification_plan_creation(self):
        """Test creating a ModificationPlan"""
        modification = FileModification(
            file_path="test.py",
            original_content="print('hello')",
            modified_content="print('world')",
            modification_type="update"
        )
        
        plan = ModificationPlan(
            id="test-plan-123",
            description="Test modification",
            modifications=[modification],
            created_at=datetime.now()
        )
        
        assert plan.id == "test-plan-123"
        assert plan.description == "Test modification"
        assert len(plan.modifications) == 1
        assert plan.modifications[0] == modification
        assert plan.status == "planned"
        assert plan.improvement_id is None
        assert plan.rollback_point is None


class TestValidationResult:
    """Test cases for ValidationResult model"""
    
    def test_validation_result_creation(self):
        """Test creating a ValidationResult"""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Warning message"],
            syntax_valid=True,
            imports_valid=True,
            tests_pass=True
        )
        
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == ["Warning message"]
        assert result.syntax_valid is True
        assert result.imports_valid is True
        assert result.tests_pass is True


class TestCodeModifier:
    """Test cases for CodeModifier class"""
    
    @pytest.fixture
    def temp_repo(self):
        """Create a temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def code_modifier(self, temp_repo):
        """Create CodeModifier instance with temp repo"""
        return CodeModifier(temp_repo)
    
    def test_init_creates_directories(self, temp_repo):
        """Test that initialization creates necessary directories"""
        code_modifier = CodeModifier(temp_repo)
        
        backup_dir = Path(temp_repo) / ".kiro" / "backups"
        plans_file = Path(temp_repo) / ".kiro" / "modification_plans.json"
        
        assert backup_dir.exists()
        assert plans_file.exists()
        assert plans_file.read_text() == "[]"
    
    def test_calculate_checksum(self, code_modifier):
        """Test checksum calculation"""
        content = "print('hello world')"
        checksum1 = code_modifier.calculate_checksum(content)
        checksum2 = code_modifier.calculate_checksum(content)
        
        assert checksum1 == checksum2
        assert len(checksum1) == 64  # SHA256 hex length
        
        # Different content should have different checksum
        different_content = "print('goodbye world')"
        checksum3 = code_modifier.calculate_checksum(different_content)
        assert checksum1 != checksum3
    
    def test_create_backup(self, code_modifier, temp_repo):
        """Test creating file backup"""
        # Create a test file
        test_file = Path(temp_repo) / "test.py"
        test_content = "print('test')"
        test_file.write_text(test_content)
        
        # Create backup
        backup_path = code_modifier.create_backup(str(test_file))
        
        assert backup_path is not None
        backup_file = Path(backup_path)
        assert backup_file.exists()
        assert backup_file.read_text() == test_content
        assert backup_file.parent == code_modifier.backup_dir
    
    def test_create_backup_nonexistent_file(self, code_modifier, temp_repo):
        """Test creating backup of non-existent file"""
        nonexistent_file = Path(temp_repo) / "nonexistent.py"
        
        backup_path = code_modifier.create_backup(str(nonexistent_file))
        
        assert backup_path is None
    
    def test_validate_python_syntax_valid(self, code_modifier):
        """Test validating valid Python syntax"""
        valid_content = """
def hello():
    print("Hello, world!")
    return True

if __name__ == "__main__":
    hello()
"""
        
        is_valid, errors = code_modifier.validate_python_syntax(valid_content, "test.py")
        
        assert is_valid is True
        assert errors == []
    
    def test_validate_python_syntax_invalid(self, code_modifier):
        """Test validating invalid Python syntax"""
        invalid_content = """
def hello(:
    print("Hello, world!")
    return True
"""
        
        is_valid, errors = code_modifier.validate_python_syntax(invalid_content, "test.py")
        
        assert is_valid is False
        assert len(errors) > 0
        assert "Syntax error" in errors[0]
    
    def test_validate_imports_valid(self, code_modifier):
        """Test validating valid imports"""
        content_with_imports = """
import os
import sys
from datetime import datetime

def test():
    return os.path.exists('.')
"""
        
        is_valid, issues = code_modifier.validate_imports(content_with_imports)
        
        assert is_valid is True
        # May have warnings but no errors
        assert all("Error" not in issue for issue in issues)
    
    def test_validate_imports_invalid(self, code_modifier):
        """Test validating invalid imports"""
        content_with_bad_imports = """
import nonexistent_module_12345
from another_nonexistent import something

def test():
    pass
"""
        
        is_valid, issues = code_modifier.validate_imports(content_with_bad_imports)
        
        # Should still be valid (warnings only for missing modules)
        assert is_valid is True
        assert len(issues) > 0
        assert any("may not be available" in issue for issue in issues)
    
    def test_validate_modification_python_file(self, code_modifier, temp_repo):
        """Test validating modification for Python file"""
        # Create a writable test file
        test_file = Path(temp_repo) / "test.py"
        test_file.write_text("print('hello')")
        
        modification = FileModification(
            file_path=str(test_file),
            original_content="print('hello')",
            modified_content="print('world')",
            modification_type="update"
        )
        
        result = code_modifier.validate_modification(modification)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.syntax_valid is True
        assert result.imports_valid is True
    
    def test_validate_modification_non_python_file(self, code_modifier):
        """Test validating modification for non-Python file"""
        modification = FileModification(
            file_path="test.txt",
            original_content="hello",
            modified_content="world",
            modification_type="update"
        )
        
        result = code_modifier.validate_modification(modification)
        
        assert result.is_valid is True
        assert result.syntax_valid is True
        assert result.imports_valid is True
    
    def test_validate_modification_syntax_error(self, code_modifier):
        """Test validating modification with syntax error"""
        modification = FileModification(
            file_path="test.py",
            original_content="print('hello')",
            modified_content="def invalid_syntax(:",
            modification_type="update"
        )
        
        result = code_modifier.validate_modification(modification)
        
        assert result.is_valid is False
        assert result.syntax_valid is False
        assert len(result.errors) > 0
    
    def test_create_modification_plan(self, code_modifier):
        """Test creating modification plan"""
        modifications = [
            FileModification(
                file_path="test1.py",
                original_content="print('hello')",
                modified_content="print('world')",
                modification_type="update"
            ),
            FileModification(
                file_path="test2.py",
                original_content="",
                modified_content="print('new file')",
                modification_type="create"
            )
        ]
        
        plan = code_modifier.create_modification_plan(
            modifications, 
            "Test modification plan",
            "improvement-123"
        )
        
        assert isinstance(plan, ModificationPlan)
        assert plan.description == "Test modification plan"
        assert plan.improvement_id == "improvement-123"
        assert len(plan.modifications) == 2
        assert plan.status == "planned"
        assert len(plan.id) == 12  # Generated ID length
        
        # Check that checksums were calculated
        for mod in plan.modifications:
            if mod.modification_type in ['create', 'update']:
                assert mod.checksum_modified is not None
    
    def test_save_and_load_modification_plan(self, code_modifier):
        """Test saving and loading modification plan"""
        modification = FileModification(
            file_path="test.py",
            original_content="print('hello')",
            modified_content="print('world')",
            modification_type="update"
        )
        
        plan = ModificationPlan(
            id="test-plan-123",
            description="Test plan",
            modifications=[modification],
            created_at=datetime.now()
        )
        
        # Save plan
        code_modifier.save_modification_plan(plan)
        
        # Load plan
        loaded_plan = code_modifier.load_modification_plan("test-plan-123")
        
        assert loaded_plan is not None
        assert loaded_plan.id == plan.id
        assert loaded_plan.description == plan.description
        assert len(loaded_plan.modifications) == 1
        assert loaded_plan.modifications[0].file_path == modification.file_path
    
    def test_load_nonexistent_plan(self, code_modifier):
        """Test loading non-existent modification plan"""
        loaded_plan = code_modifier.load_modification_plan("nonexistent-plan")
        
        assert loaded_plan is None
    
    def test_apply_single_modification_create(self, code_modifier, temp_repo):
        """Test applying single file creation modification"""
        modification = FileModification(
            file_path=str(Path(temp_repo) / "new_file.py"),
            original_content="",
            modified_content="print('new file')",
            modification_type="create"
        )
        
        result = code_modifier.apply_single_modification(modification)
        
        assert result is True
        
        # Check file was created
        new_file = Path(modification.file_path)
        assert new_file.exists()
        assert new_file.read_text() == "print('new file')"
    
    def test_apply_single_modification_update(self, code_modifier, temp_repo):
        """Test applying single file update modification"""
        # Create existing file
        test_file = Path(temp_repo) / "existing.py"
        original_content = "print('original')"
        test_file.write_text(original_content)
        
        modification = FileModification(
            file_path=str(test_file),
            original_content=original_content,
            modified_content="print('updated')",
            modification_type="update"
        )
        
        result = code_modifier.apply_single_modification(modification)
        
        assert result is True
        
        # Check file was updated
        assert test_file.read_text() == "print('updated')"
        
        # Check backup was created
        assert modification.backup_path is not None
        backup_file = Path(modification.backup_path)
        assert backup_file.exists()
        assert backup_file.read_text() == original_content
    
    def test_apply_single_modification_delete(self, code_modifier, temp_repo):
        """Test applying single file deletion modification"""
        # Create existing file
        test_file = Path(temp_repo) / "to_delete.py"
        original_content = "print('to be deleted')"
        test_file.write_text(original_content)
        
        modification = FileModification(
            file_path=str(test_file),
            original_content=original_content,
            modified_content="",
            modification_type="delete"
        )
        
        result = code_modifier.apply_single_modification(modification)
        
        assert result is True
        
        # Check file was deleted
        assert not test_file.exists()
        
        # Check backup was created
        assert modification.backup_path is not None
        backup_file = Path(modification.backup_path)
        assert backup_file.exists()
        assert backup_file.read_text() == original_content
    
    def test_rollback_single_modification_create(self, code_modifier, temp_repo):
        """Test rolling back file creation"""
        # Create file first
        test_file = Path(temp_repo) / "created.py"
        test_file.write_text("print('created')")
        
        modification = FileModification(
            file_path=str(test_file),
            original_content="",
            modified_content="print('created')",
            modification_type="create"
        )
        
        result = code_modifier.rollback_single_modification(modification)
        
        assert result is True
        assert not test_file.exists()
    
    def test_rollback_single_modification_update(self, code_modifier, temp_repo):
        """Test rolling back file update"""
        # Create file and backup
        test_file = Path(temp_repo) / "updated.py"
        original_content = "print('original')"
        updated_content = "print('updated')"
        
        test_file.write_text(updated_content)
        
        # Create backup
        backup_path = code_modifier.create_backup(str(test_file))
        # Manually set backup to original content for test
        Path(backup_path).write_text(original_content)
        
        modification = FileModification(
            file_path=str(test_file),
            original_content=original_content,
            modified_content=updated_content,
            modification_type="update",
            backup_path=backup_path
        )
        
        result = code_modifier.rollback_single_modification(modification)
        
        assert result is True
        assert test_file.read_text() == original_content
    
    def test_rollback_single_modification_delete(self, code_modifier, temp_repo):
        """Test rolling back file deletion"""
        # Create backup file
        original_content = "print('deleted file')"
        backup_path = code_modifier.backup_dir / "deleted.py.backup"
        backup_path.write_text(original_content)
        
        test_file = Path(temp_repo) / "deleted.py"
        
        modification = FileModification(
            file_path=str(test_file),
            original_content=original_content,
            modified_content="",
            modification_type="delete",
            backup_path=str(backup_path)
        )
        
        result = code_modifier.rollback_single_modification(modification)
        
        assert result is True
        assert test_file.exists()
        assert test_file.read_text() == original_content
    
    def test_verify_integrity_success(self, code_modifier, temp_repo):
        """Test successful integrity verification"""
        # Create test file
        test_file = Path(temp_repo) / "test.py"
        content = "print('test')"
        test_file.write_text(content)
        
        modification = FileModification(
            file_path=str(test_file),
            original_content="",
            modified_content=content,
            modification_type="create",
            checksum_modified=code_modifier.calculate_checksum(content)
        )
        
        plan = ModificationPlan(
            id="test-plan",
            description="Test plan",
            modifications=[modification],
            created_at=datetime.now(),
            status="applied"
        )
        
        result = code_modifier.verify_integrity(plan)
        
        assert result is True
    
    def test_verify_integrity_failure(self, code_modifier, temp_repo):
        """Test integrity verification failure"""
        # Create test file with different content than expected
        test_file = Path(temp_repo) / "test.py"
        actual_content = "print('actual')"
        expected_content = "print('expected')"
        test_file.write_text(actual_content)
        
        modification = FileModification(
            file_path=str(test_file),
            original_content="",
            modified_content=expected_content,
            modification_type="create",
            checksum_modified=code_modifier.calculate_checksum(expected_content)
        )
        
        plan = ModificationPlan(
            id="test-plan",
            description="Test plan",
            modifications=[modification],
            created_at=datetime.now(),
            status="applied"
        )
        
        result = code_modifier.verify_integrity(plan)
        
        assert result is False
    
    def test_cleanup_old_backups(self, code_modifier, temp_repo):
        """Test cleaning up old backup files"""
        # Create some backup files
        old_backup = code_modifier.backup_dir / "old.py.backup"
        recent_backup = code_modifier.backup_dir / "recent.py.backup"
        
        old_backup.write_text("old content")
        recent_backup.write_text("recent content")
        
        # Manually set old modification time (30+ days ago)
        old_time = datetime.now().timestamp() - (35 * 24 * 3600)
        os.utime(old_backup, (old_time, old_time))
        
        # Run cleanup
        code_modifier.cleanup_old_backups(keep_days=30)
        
        # Check results
        assert not old_backup.exists()
        assert recent_backup.exists()
    
    def test_get_modification_plans(self, code_modifier):
        """Test getting modification plans"""
        # Create and save multiple plans
        plans = []
        for i in range(3):
            modification = FileModification(
                file_path=f"test{i}.py",
                original_content="",
                modified_content=f"print('test{i}')",
                modification_type="create"
            )
            
            plan = ModificationPlan(
                id=f"plan-{i}",
                description=f"Plan {i}",
                modifications=[modification],
                created_at=datetime.now(),
                status="applied" if i % 2 == 0 else "planned"
            )
            
            plans.append(plan)
            code_modifier.save_modification_plan(plan)
        
        # Get all plans
        all_plans = code_modifier.get_modification_plans()
        assert len(all_plans) == 3
        
        # Get plans by status
        applied_plans = code_modifier.get_modification_plans(status="applied")
        assert len(applied_plans) == 2
        
        planned_plans = code_modifier.get_modification_plans(status="planned")
        assert len(planned_plans) == 1
    
    def test_apply_modification_plan_success(self, code_modifier, temp_repo):
        """Test successful modification plan application"""
        # Mock git integration
        with patch.object(code_modifier.git_integration, 'create_rollback_point', return_value="rollback-commit-123"), \
             patch.object(code_modifier.git_integration, 'get_current_branch', return_value="main"), \
             patch.object(code_modifier.git_integration, 'get_current_commit_hash', return_value="current-commit-456"), \
             patch.object(code_modifier.git_integration, 'generate_change_id', return_value="change-789"), \
             patch.object(code_modifier.git_integration, 'log_change'):
            
            # Create modification plan
            modification = FileModification(
            file_path=str(Path(temp_repo) / "test.py"),
            original_content="",
            modified_content="print('test')",
            modification_type="create"
        )
        
            plan = code_modifier.create_modification_plan(
                [modification],
                "Test plan",
                "improvement-123"
            )
        
            # Apply plan
            result = code_modifier.apply_modification_plan(plan)
            
            assert result is True
            
            # Check file was created
            test_file = Path(modification.file_path)
            assert test_file.exists()
            assert test_file.read_text() == "print('test')"
            
            # Check plan status was updated
            updated_plan = code_modifier.load_modification_plan(plan.id)
            assert updated_plan.status == "applied"
            assert updated_plan.rollback_point == "rollback-commit-123"
    
    def test_apply_modification_plan_validation_failure(self, code_modifier):
        """Test modification plan application with validation failure"""
        # Create plan with invalid syntax
        modification = FileModification(
            file_path="test.py",
            original_content="",
            modified_content="def invalid_syntax(:",
            modification_type="create"
        )
        
        plan = code_modifier.create_modification_plan(
            [modification],
            "Invalid plan"
        )
        
        # Apply plan
        result = code_modifier.apply_modification_plan(plan)
        
        assert result is False
        
        # Check plan status
        updated_plan = code_modifier.load_modification_plan(plan.id)
        assert updated_plan.status == "failed"
    
    def test_rollback_plan_success(self, code_modifier):
        """Test successful plan rollback"""
        # Mock git integration
        with patch.object(code_modifier.git_integration, 'rollback_to_commit', return_value=True), \
             patch.object(code_modifier.git_integration, 'get_current_branch', return_value="main"), \
             patch.object(code_modifier.git_integration, 'get_current_commit_hash', return_value="current-commit"), \
             patch.object(code_modifier.git_integration, 'generate_change_id', return_value="rollback-change-id"), \
             patch.object(code_modifier.git_integration, 'log_change'):
            
            # Create and save applied plan
            modification = FileModification(
            file_path="test.py",
            original_content="",
            modified_content="print('test')",
            modification_type="create"
        )
        
            plan = ModificationPlan(
                id="test-plan",
                description="Test plan",
                modifications=[modification],
                created_at=datetime.now(),
                status="applied",
                rollback_point="rollback-commit-123"
            )
        
            code_modifier.save_modification_plan(plan)
            
            # Rollback plan
            result = code_modifier.rollback_plan("test-plan")
            
            assert result is True
            
            # Check plan status
            updated_plan = code_modifier.load_modification_plan("test-plan")
            assert updated_plan.status == "rolled_back"
    
    def test_rollback_plan_not_applied(self, code_modifier):
        """Test rollback of plan that is not applied"""
        # Create and save planned (not applied) plan
        modification = FileModification(
            file_path="test.py",
            original_content="",
            modified_content="print('test')",
            modification_type="create"
        )
        
        plan = ModificationPlan(
            id="test-plan",
            description="Test plan",
            modifications=[modification],
            created_at=datetime.now(),
            status="planned"
        )
        
        code_modifier.save_modification_plan(plan)
        
        # Try to rollback plan
        result = code_modifier.rollback_plan("test-plan")
        
        assert result is False
        
        # Verify no rollback occurred
        assert result is False