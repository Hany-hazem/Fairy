"""
Code Modification Engine for Safe File Modifications

This module provides safe code modification capabilities with validation,
rollback functionality, and integrity checking for the self-improvement system.
"""

import os
import shutil
import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import ast
import py_compile
import tempfile
import json

from pydantic import BaseModel

from app.version_control import GitIntegration, ChangeRecord


logger = logging.getLogger(__name__)


class FileModification(BaseModel):
    """Represents a single file modification"""
    file_path: str
    original_content: str
    modified_content: str
    modification_type: str  # 'create', 'update', 'delete'
    backup_path: Optional[str] = None
    checksum_original: Optional[str] = None
    checksum_modified: Optional[str] = None


class ModificationPlan(BaseModel):
    """Plan for a set of file modifications"""
    id: str
    description: str
    modifications: List[FileModification]
    improvement_id: Optional[str] = None
    created_at: datetime
    rollback_point: Optional[str] = None
    status: str = "planned"  # 'planned', 'applying', 'applied', 'failed', 'rolled_back'


class ValidationResult(BaseModel):
    """Result of code validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    syntax_valid: bool
    imports_valid: bool
    tests_pass: bool


class CodeModifier:
    """Safe code modification engine with rollback capabilities"""
    
    def __init__(self, repo_path: str = ".", backup_dir: str = ".kiro/backups"):
        self.repo_path = Path(repo_path).resolve()
        self.backup_dir = self.repo_path / backup_dir
        self.git_integration = GitIntegration(repo_path)
        self.plans_file = self.repo_path / ".kiro" / "modification_plans.json"
        
        # Ensure directories exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.plans_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.plans_file.exists():
            self.plans_file.write_text("[]")
    
    def calculate_checksum(self, content: str) -> str:
        """Calculate SHA256 checksum of content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def create_backup(self, file_path: str) -> Optional[str]:
        """Create backup of a file before modification"""
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                return None
                
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{source_path.name}.{timestamp}.backup"
            backup_path = self.backup_dir / backup_filename
            
            # Copy file to backup location
            shutil.copy2(source_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to create backup for {file_path}: {e}")
            return None
    
    def validate_python_syntax(self, content: str, file_path: str) -> Tuple[bool, List[str]]:
        """Validate Python syntax of content"""
        errors = []
        
        try:
            # Parse the content as Python AST
            ast.parse(content)
            
            # Try to compile the content
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(content)
                temp_file.flush()
                
                try:
                    py_compile.compile(temp_file.name, doraise=True)
                    return True, []
                except py_compile.PyCompileError as e:
                    errors.append(f"Compilation error: {e}")
                finally:
                    os.unlink(temp_file.name)
                    
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            errors.append(f"Validation error: {e}")
            
        return False, errors
    
    def validate_imports(self, content: str) -> Tuple[bool, List[str]]:
        """Validate that all imports in the content are available"""
        errors = []
        warnings = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        try:
                            __import__(alias.name)
                        except ImportError:
                            warnings.append(f"Import '{alias.name}' may not be available")
                        except Exception as e:
                            errors.append(f"Error checking import '{alias.name}': {e}")
                            
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        try:
                            __import__(node.module)
                        except ImportError:
                            warnings.append(f"Module '{node.module}' may not be available")
                        except Exception as e:
                            errors.append(f"Error checking module '{node.module}': {e}")
            
            return len(errors) == 0, errors + warnings
            
        except Exception as e:
            errors.append(f"Import validation error: {e}")
            return False, errors
    
    def validate_modification(self, modification: FileModification) -> ValidationResult:
        """Validate a single file modification"""
        errors = []
        warnings = []
        syntax_valid = True
        imports_valid = True
        tests_pass = True  # Will be implemented with test runner integration
        
        file_path = Path(modification.file_path)
        
        # Skip validation for non-Python files
        if file_path.suffix != '.py':
            return ValidationResult(
                is_valid=True,
                errors=[],
                warnings=[],
                syntax_valid=True,
                imports_valid=True,
                tests_pass=True
            )
        
        # Validate syntax
        if modification.modification_type in ['create', 'update']:
            syntax_valid, syntax_errors = self.validate_python_syntax(
                modification.modified_content, 
                modification.file_path
            )
            errors.extend(syntax_errors)
            
            # Validate imports
            if syntax_valid:
                imports_valid, import_issues = self.validate_imports(modification.modified_content)
                if not imports_valid:
                    errors.extend([issue for issue in import_issues if "Error" in issue])
                    warnings.extend([issue for issue in import_issues if "may not be available" in issue])
        
        # Check file permissions
        if modification.modification_type == 'update':
            if not os.access(file_path, os.W_OK):
                errors.append(f"File {modification.file_path} is not writable")
        
        # Verify checksums
        if modification.modification_type == 'update':
            if file_path.exists():
                current_content = file_path.read_text()
                current_checksum = self.calculate_checksum(current_content)
                
                if modification.checksum_original and current_checksum != modification.checksum_original:
                    errors.append(f"File {modification.file_path} has been modified since plan creation")
        
        is_valid = len(errors) == 0 and syntax_valid and imports_valid
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            syntax_valid=syntax_valid,
            imports_valid=imports_valid,
            tests_pass=tests_pass
        )
    
    def create_modification_plan(self, 
                                modifications: List[FileModification], 
                                description: str,
                                improvement_id: Optional[str] = None) -> ModificationPlan:
        """Create a modification plan"""
        plan_id = hashlib.md5(f"{description}:{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        # Calculate checksums for existing files
        for modification in modifications:
            file_path = Path(modification.file_path)
            if file_path.exists() and modification.modification_type in ['update', 'delete']:
                current_content = file_path.read_text()
                modification.checksum_original = self.calculate_checksum(current_content)
                modification.original_content = current_content
            
            if modification.modification_type in ['create', 'update']:
                modification.checksum_modified = self.calculate_checksum(modification.modified_content)
        
        plan = ModificationPlan(
            id=plan_id,
            description=description,
            modifications=modifications,
            improvement_id=improvement_id,
            created_at=datetime.now()
        )
        
        # Save plan
        self.save_modification_plan(plan)
        
        return plan
    
    def save_modification_plan(self, plan: ModificationPlan) -> None:
        """Save modification plan to file"""
        try:
            # Read existing plans
            if self.plans_file.exists():
                with open(self.plans_file, 'r') as f:
                    plans_data = json.load(f)
            else:
                plans_data = []
            
            # Add or update plan
            plan_dict = plan.model_dump()
            
            # Find existing plan and update, or add new
            updated = False
            for i, existing_plan in enumerate(plans_data):
                if existing_plan['id'] == plan.id:
                    plans_data[i] = plan_dict
                    updated = True
                    break
            
            if not updated:
                plans_data.append(plan_dict)
            
            # Write back to file
            with open(self.plans_file, 'w') as f:
                json.dump(plans_data, f, indent=2, default=str)
                
            logger.info(f"Saved modification plan: {plan.id}")
            
        except Exception as e:
            logger.error(f"Failed to save modification plan: {e}")
    
    def load_modification_plan(self, plan_id: str) -> Optional[ModificationPlan]:
        """Load modification plan by ID"""
        try:
            if not self.plans_file.exists():
                return None
                
            with open(self.plans_file, 'r') as f:
                plans_data = json.load(f)
            
            for plan_data in plans_data:
                if plan_data['id'] == plan_id:
                    # Convert timestamp strings back to datetime
                    if isinstance(plan_data['created_at'], str):
                        plan_data['created_at'] = datetime.fromisoformat(plan_data['created_at'])
                    
                    return ModificationPlan(**plan_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load modification plan {plan_id}: {e}")
            return None
    
    def apply_modification_plan(self, plan: ModificationPlan) -> bool:
        """Apply a modification plan with rollback capability"""
        try:
            # Update plan status
            plan.status = "applying"
            self.save_modification_plan(plan)
            
            # Create rollback point
            rollback_commit = self.git_integration.create_rollback_point(
                f"Before applying modification plan: {plan.description}"
            )
            
            if rollback_commit:
                plan.rollback_point = rollback_commit
                self.save_modification_plan(plan)
            
            # Validate all modifications first
            validation_errors = []
            for modification in plan.modifications:
                validation_result = self.validate_modification(modification)
                if not validation_result.is_valid:
                    validation_errors.extend(validation_result.errors)
            
            if validation_errors:
                logger.error(f"Validation failed for plan {plan.id}: {validation_errors}")
                plan.status = "failed"
                self.save_modification_plan(plan)
                return False
            
            # Create backups and apply modifications
            applied_modifications = []
            
            try:
                for modification in plan.modifications:
                    if self.apply_single_modification(modification):
                        applied_modifications.append(modification)
                    else:
                        raise Exception(f"Failed to apply modification to {modification.file_path}")
                
                # All modifications applied successfully
                plan.status = "applied"
                self.save_modification_plan(plan)
                
                # Log the change
                change_record = ChangeRecord(
                    id=self.git_integration.generate_change_id(
                        plan.improvement_id or plan.id,
                        [mod.file_path for mod in plan.modifications]
                    ),
                    timestamp=datetime.now(),
                    branch_name=self.git_integration.get_current_branch() or "unknown",
                    commit_hash=self.git_integration.get_current_commit_hash() or "unknown",
                    files_modified=[mod.file_path for mod in plan.modifications],
                    description=plan.description,
                    improvement_id=plan.improvement_id,
                    rollback_point=plan.rollback_point or "unknown",
                    status="applied"
                )
                
                self.git_integration.log_change(change_record)
                
                logger.info(f"Successfully applied modification plan: {plan.id}")
                return True
                
            except Exception as e:
                logger.error(f"Error applying modifications: {e}")
                
                # Rollback applied modifications
                self.rollback_modifications(applied_modifications)
                
                plan.status = "failed"
                self.save_modification_plan(plan)
                return False
                
        except Exception as e:
            logger.error(f"Failed to apply modification plan {plan.id}: {e}")
            plan.status = "failed"
            self.save_modification_plan(plan)
            return False
    
    def apply_single_modification(self, modification: FileModification) -> bool:
        """Apply a single file modification"""
        try:
            file_path = Path(modification.file_path)
            
            if modification.modification_type == "create":
                # Create new file
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(modification.modified_content)
                logger.info(f"Created file: {file_path}")
                
            elif modification.modification_type == "update":
                # Create backup first
                backup_path = self.create_backup(str(file_path))
                modification.backup_path = backup_path
                
                # Update file
                file_path.write_text(modification.modified_content)
                logger.info(f"Updated file: {file_path}")
                
            elif modification.modification_type == "delete":
                # Create backup first
                backup_path = self.create_backup(str(file_path))
                modification.backup_path = backup_path
                
                # Delete file
                file_path.unlink()
                logger.info(f"Deleted file: {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply modification to {modification.file_path}: {e}")
            return False
    
    def rollback_modifications(self, modifications: List[FileModification]) -> bool:
        """Rollback a list of modifications"""
        try:
            for modification in reversed(modifications):
                self.rollback_single_modification(modification)
            
            logger.info("Successfully rolled back modifications")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback modifications: {e}")
            return False
    
    def rollback_single_modification(self, modification: FileModification) -> bool:
        """Rollback a single modification"""
        try:
            file_path = Path(modification.file_path)
            
            if modification.modification_type == "create":
                # Remove created file
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Removed created file: {file_path}")
                    
            elif modification.modification_type == "update":
                # Restore from backup
                if modification.backup_path and Path(modification.backup_path).exists():
                    shutil.copy2(modification.backup_path, file_path)
                    logger.info(f"Restored file from backup: {file_path}")
                elif modification.original_content:
                    file_path.write_text(modification.original_content)
                    logger.info(f"Restored file from original content: {file_path}")
                    
            elif modification.modification_type == "delete":
                # Restore from backup
                if modification.backup_path and Path(modification.backup_path).exists():
                    shutil.copy2(modification.backup_path, file_path)
                    logger.info(f"Restored deleted file: {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback modification for {modification.file_path}: {e}")
            return False
    
    def rollback_plan(self, plan_id: str) -> bool:
        """Rollback a modification plan using Git"""
        try:
            plan = self.load_modification_plan(plan_id)
            if not plan:
                logger.error(f"Plan {plan_id} not found")
                return False
            
            if plan.status != "applied":
                logger.error(f"Plan {plan_id} is not in applied state")
                return False
            
            if not plan.rollback_point:
                logger.error(f"No rollback point available for plan {plan_id}")
                return False
            
            # Rollback using Git
            if self.git_integration.rollback_to_commit(plan.rollback_point):
                plan.status = "rolled_back"
                self.save_modification_plan(plan)
                
                # Log the rollback
                change_record = ChangeRecord(
                    id=self.git_integration.generate_change_id(
                        f"rollback-{plan.id}",
                        [mod.file_path for mod in plan.modifications]
                    ),
                    timestamp=datetime.now(),
                    branch_name=self.git_integration.get_current_branch() or "unknown",
                    commit_hash=self.git_integration.get_current_commit_hash() or "unknown",
                    files_modified=[mod.file_path for mod in plan.modifications],
                    description=f"Rollback of plan: {plan.description}",
                    improvement_id=plan.improvement_id,
                    rollback_point=plan.rollback_point,
                    status="rolled_back"
                )
                
                self.git_integration.log_change(change_record)
                
                logger.info(f"Successfully rolled back plan: {plan_id}")
                return True
            else:
                logger.error(f"Failed to rollback plan {plan_id} using Git")
                return False
                
        except Exception as e:
            logger.error(f"Failed to rollback plan {plan_id}: {e}")
            return False
    
    def verify_integrity(self, plan: ModificationPlan) -> bool:
        """Verify integrity of applied modifications"""
        try:
            for modification in plan.modifications:
                file_path = Path(modification.file_path)
                
                if modification.modification_type == "create":
                    if not file_path.exists():
                        logger.error(f"Created file {file_path} does not exist")
                        return False
                        
                    current_content = file_path.read_text()
                    current_checksum = self.calculate_checksum(current_content)
                    
                    if current_checksum != modification.checksum_modified:
                        logger.error(f"Checksum mismatch for created file {file_path}")
                        return False
                        
                elif modification.modification_type == "update":
                    if not file_path.exists():
                        logger.error(f"Updated file {file_path} does not exist")
                        return False
                        
                    current_content = file_path.read_text()
                    current_checksum = self.calculate_checksum(current_content)
                    
                    if current_checksum != modification.checksum_modified:
                        logger.error(f"Checksum mismatch for updated file {file_path}")
                        return False
                        
                elif modification.modification_type == "delete":
                    if file_path.exists():
                        logger.error(f"Deleted file {file_path} still exists")
                        return False
            
            logger.info(f"Integrity verification passed for plan: {plan.id}")
            return True
            
        except Exception as e:
            logger.error(f"Integrity verification failed for plan {plan.id}: {e}")
            return False
    
    def cleanup_old_backups(self, keep_days: int = 30) -> None:
        """Clean up old backup files"""
        try:
            current_time = datetime.now().timestamp()
            
            for backup_file in self.backup_dir.glob("*.backup"):
                file_age = current_time - backup_file.stat().st_mtime
                age_days = file_age / (24 * 3600)
                
                if age_days > keep_days:
                    backup_file.unlink()
                    logger.info(f"Cleaned up old backup: {backup_file}")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")
    
    def get_modification_plans(self, status: Optional[str] = None) -> List[ModificationPlan]:
        """Get all modification plans, optionally filtered by status"""
        try:
            if not self.plans_file.exists():
                return []
                
            with open(self.plans_file, 'r') as f:
                plans_data = json.load(f)
            
            plans = []
            for plan_data in plans_data:
                # Convert timestamp strings back to datetime
                if isinstance(plan_data['created_at'], str):
                    plan_data['created_at'] = datetime.fromisoformat(plan_data['created_at'])
                
                plan = ModificationPlan(**plan_data)
                
                if status is None or plan.status == status:
                    plans.append(plan)
            
            return plans
            
        except Exception as e:
            logger.error(f"Failed to get modification plans: {e}")
            return []