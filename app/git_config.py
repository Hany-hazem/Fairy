# app/git_config.py
"""
Git Workflow Configuration Management

This module provides comprehensive configuration management for Git workflow
automation, branch management, and task integration with validation.
"""

import os
import subprocess
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field, validator
from pathlib import Path


class GitWorkflowMode(str, Enum):
    """Git workflow operation modes"""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    HYBRID = "hybrid"


class BranchType(str, Enum):
    """Git branch types"""
    FEATURE = "feature"
    TASK = "task"
    HOTFIX = "hotfix"
    BUGFIX = "bugfix"
    RELEASE = "release"


class MergeStrategy(str, Enum):
    """Git merge strategies"""
    RECURSIVE = "recursive"
    OCTOPUS = "octopus"
    OURS = "ours"
    SUBTREE = "subtree"
    RESOLVE = "resolve"


class ConflictResolutionStrategy(str, Enum):
    """Conflict resolution strategies"""
    MANUAL = "manual"
    AUTO_MERGE = "auto_merge"
    PREFER_OURS = "prefer_ours"
    PREFER_THEIRS = "prefer_theirs"
    ABORT = "abort"


@dataclass
class GitRepositoryConfig:
    """Git repository configuration"""
    repo_path: str = "."
    default_branch: str = "main"
    remote_name: str = "origin"
    remote_url: Optional[str] = None
    auto_fetch: bool = True
    fetch_interval: int = 300  # 5 minutes
    
    def __post_init__(self):
        """Validate repository configuration"""
        repo_path = Path(self.repo_path)
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {self.repo_path}")
        
        git_dir = repo_path / ".git"
        if not git_dir.exists():
            raise ValueError(f"Not a Git repository: {self.repo_path}")


@dataclass
class GitBranchConfig:
    """Git branch management configuration"""
    prefix_feature: str = "feature"
    prefix_task: str = "task"
    prefix_hotfix: str = "hotfix"
    prefix_bugfix: str = "bugfix"
    prefix_release: str = "release"
    cleanup_enabled: bool = True
    cleanup_days: int = 30
    auto_delete_merged: bool = True
    protect_main_branch: bool = True
    
    def get_branch_prefix(self, branch_type: BranchType) -> str:
        """Get prefix for branch type"""
        prefix_map = {
            BranchType.FEATURE: self.prefix_feature,
            BranchType.TASK: self.prefix_task,
            BranchType.HOTFIX: self.prefix_hotfix,
            BranchType.BUGFIX: self.prefix_bugfix,
            BranchType.RELEASE: self.prefix_release,
        }
        return prefix_map.get(branch_type, "feature")


@dataclass
class GitCommitConfig:
    """Git commit configuration"""
    message_template: str = "{type}({scope}): {description}\n\n{body}\n\nTask-ID: {task_id}\nRequirements: {requirements}"
    max_subject_length: int = 72
    max_body_length: int = 500
    sign_commits: bool = False
    gpg_key: Optional[str] = None
    auto_commit: bool = True
    commit_on_save: bool = False
    
    def __post_init__(self):
        """Validate commit configuration"""
        if not 10 <= self.max_subject_length <= 100:
            raise ValueError("Commit subject length must be between 10 and 100 characters")
        if self.max_body_length <= 0:
            raise ValueError("Commit body length must be positive")


@dataclass
class GitUserConfig:
    """Git user configuration"""
    name: Optional[str] = None
    email: Optional[str] = None
    signing_key: Optional[str] = None
    
    def __post_init__(self):
        """Validate and set user configuration"""
        if not self.name:
            try:
                result = subprocess.run(
                    ["git", "config", "--global", "user.name"],
                    capture_output=True, text=True, check=True
                )
                self.name = result.stdout.strip()
            except subprocess.CalledProcessError:
                pass
        
        if not self.email:
            try:
                result = subprocess.run(
                    ["git", "config", "--global", "user.email"],
                    capture_output=True, text=True, check=True
                )
                self.email = result.stdout.strip()
            except subprocess.CalledProcessError:
                pass


@dataclass
class GitMergeConfig:
    """Git merge configuration"""
    strategy: MergeStrategy = MergeStrategy.RECURSIVE
    conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.MANUAL
    auto_resolve_conflicts: bool = False
    conflict_timeout: int = 300  # 5 minutes
    merge_commit_template: str = "Merge {source_branch} into {target_branch}\n\nCompleted tasks: {task_ids}"
    squash_feature_branches: bool = False
    
    def __post_init__(self):
        """Validate merge configuration"""
        if self.conflict_timeout <= 0:
            raise ValueError("Conflict timeout must be positive")


@dataclass
class GitTaskConfig:
    """Git task integration configuration"""
    tracking_enabled: bool = True
    branch_mapping_file: str = ".git/task_branch_mapping.json"
    auto_merge_on_completion: bool = False
    dependency_check: bool = True
    task_id_in_commits: bool = True
    requirement_tracking: bool = True
    
    def __post_init__(self):
        """Validate task configuration"""
        if self.tracking_enabled and not self.branch_mapping_file:
            raise ValueError("Branch mapping file required when tracking is enabled")


@dataclass
class GitPerformanceConfig:
    """Git performance configuration"""
    operation_timeout: int = 60
    large_file_threshold: int = 104857600  # 100MB
    history_limit: int = 1000
    status_check_interval: int = 30
    parallel_operations: bool = True
    max_parallel_jobs: int = 4
    
    def __post_init__(self):
        """Validate performance configuration"""
        if self.operation_timeout <= 0:
            raise ValueError("Operation timeout must be positive")
        if self.large_file_threshold <= 0:
            raise ValueError("Large file threshold must be positive")


@dataclass
class GitSecurityConfig:
    """Git security configuration"""
    require_signed_commits: bool = False
    verify_signatures: bool = False
    allowed_signers_file: Optional[str] = None
    hook_verification: bool = True
    safe_directory_check: bool = True
    
    def __post_init__(self):
        """Validate security configuration"""
        if self.require_signed_commits and not self.verify_signatures:
            # Auto-enable signature verification if signing is required
            self.verify_signatures = True


class GitWorkflowConfig(BaseModel):
    """Comprehensive Git workflow configuration with validation"""
    
    repository: GitRepositoryConfig = Field(default_factory=GitRepositoryConfig)
    branch: GitBranchConfig = Field(default_factory=GitBranchConfig)
    commit: GitCommitConfig = Field(default_factory=GitCommitConfig)
    user: GitUserConfig = Field(default_factory=GitUserConfig)
    merge: GitMergeConfig = Field(default_factory=GitMergeConfig)
    task: GitTaskConfig = Field(default_factory=GitTaskConfig)
    performance: GitPerformanceConfig = Field(default_factory=GitPerformanceConfig)
    security: GitSecurityConfig = Field(default_factory=GitSecurityConfig)
    
    workflow_mode: GitWorkflowMode = GitWorkflowMode.AUTOMATIC
    auto_push: bool = False
    
    @classmethod
    def from_env(cls, prefix: str = "GIT_") -> "GitWorkflowConfig":
        """Create configuration from environment variables"""
        config_data = {}
        
        # Repository configuration
        repo_config = {}
        if os.getenv(f"{prefix}REPO_PATH"):
            repo_config["repo_path"] = os.getenv(f"{prefix}REPO_PATH")
        if os.getenv(f"{prefix}DEFAULT_BRANCH"):
            repo_config["default_branch"] = os.getenv(f"{prefix}DEFAULT_BRANCH")
        if os.getenv(f"{prefix}REMOTE_NAME"):
            repo_config["remote_name"] = os.getenv(f"{prefix}REMOTE_NAME")
        if os.getenv(f"{prefix}REMOTE_URL"):
            repo_config["remote_url"] = os.getenv(f"{prefix}REMOTE_URL")
        
        if repo_config:
            config_data["repository"] = GitRepositoryConfig(**repo_config)
        
        # Branch configuration
        branch_config = {}
        if os.getenv(f"{prefix}BRANCH_PREFIX_FEATURE"):
            branch_config["prefix_feature"] = os.getenv(f"{prefix}BRANCH_PREFIX_FEATURE")
        if os.getenv(f"{prefix}BRANCH_PREFIX_TASK"):
            branch_config["prefix_task"] = os.getenv(f"{prefix}BRANCH_PREFIX_TASK")
        if os.getenv(f"{prefix}BRANCH_CLEANUP_ENABLED"):
            branch_config["cleanup_enabled"] = os.getenv(f"{prefix}BRANCH_CLEANUP_ENABLED").lower() == "true"
        
        if branch_config:
            config_data["branch"] = GitBranchConfig(**branch_config)
        
        # Commit configuration
        commit_config = {}
        if os.getenv(f"{prefix}COMMIT_MESSAGE_TEMPLATE"):
            commit_config["message_template"] = os.getenv(f"{prefix}COMMIT_MESSAGE_TEMPLATE")
        if os.getenv(f"{prefix}COMMIT_MAX_SUBJECT_LENGTH"):
            commit_config["max_subject_length"] = int(os.getenv(f"{prefix}COMMIT_MAX_SUBJECT_LENGTH"))
        if os.getenv(f"{prefix}COMMIT_SIGN"):
            commit_config["sign_commits"] = os.getenv(f"{prefix}COMMIT_SIGN").lower() == "true"
        if os.getenv(f"{prefix}COMMIT_GPG_KEY"):
            commit_config["gpg_key"] = os.getenv(f"{prefix}COMMIT_GPG_KEY")
        
        if commit_config:
            config_data["commit"] = GitCommitConfig(**commit_config)
        
        # User configuration
        user_config = {}
        if os.getenv(f"{prefix}USER_NAME"):
            user_config["name"] = os.getenv(f"{prefix}USER_NAME")
        if os.getenv(f"{prefix}USER_EMAIL"):
            user_config["email"] = os.getenv(f"{prefix}USER_EMAIL")
        
        if user_config:
            config_data["user"] = GitUserConfig(**user_config)
        
        # Workflow mode
        if os.getenv(f"{prefix}WORKFLOW_MODE"):
            config_data["workflow_mode"] = GitWorkflowMode(os.getenv(f"{prefix}WORKFLOW_MODE"))
        
        if os.getenv(f"{prefix}AUTO_PUSH"):
            config_data["auto_push"] = os.getenv(f"{prefix}AUTO_PUSH").lower() == "true"
        
        return cls(**config_data)
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "GitWorkflowConfig":
        """Load configuration from JSON or YAML file"""
        import json
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.json':
                config_data = json.load(f)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                try:
                    import yaml
                    config_data = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML is required to load YAML configuration files")
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        return cls(**config_data)
    
    def to_file(self, config_path: Union[str, Path], format: str = "json") -> None:
        """Save configuration to file"""
        import json
        
        config_path = Path(config_path)
        config_data = self.dict()
        
        with open(config_path, 'w') as f:
            if format.lower() == 'json':
                json.dump(config_data, f, indent=2, default=str)
            elif format.lower() in ['yml', 'yaml']:
                try:
                    import yaml
                    yaml.dump(config_data, f, default_flow_style=False)
                except ImportError:
                    raise ImportError("PyYAML is required to save YAML configuration files")
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def validate_git_environment(self) -> List[str]:
        """Validate Git environment and configuration"""
        issues = []
        
        # Check if Git is installed
        try:
            subprocess.run(["git", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            issues.append("Git is not installed or not in PATH")
            return issues
        
        # Check repository
        try:
            repo_path = Path(self.repository.repo_path)
            if not (repo_path / ".git").exists():
                issues.append(f"Not a Git repository: {self.repository.repo_path}")
        except Exception as e:
            issues.append(f"Repository validation error: {e}")
        
        # Check user configuration
        if not self.user.name:
            issues.append("Git user name not configured")
        if not self.user.email:
            issues.append("Git user email not configured")
        
        # Check GPG configuration if signing is enabled
        if self.commit.sign_commits:
            try:
                subprocess.run(["gpg", "--version"], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                issues.append("GPG not available but commit signing is enabled")
        
        return issues
    
    def setup_git_config(self) -> bool:
        """Setup Git configuration based on settings"""
        try:
            repo_path = Path(self.repository.repo_path)
            
            # Set user configuration
            if self.user.name:
                subprocess.run(
                    ["git", "config", "user.name", self.user.name],
                    cwd=repo_path, check=True
                )
            
            if self.user.email:
                subprocess.run(
                    ["git", "config", "user.email", self.user.email],
                    cwd=repo_path, check=True
                )
            
            # Set signing configuration
            if self.commit.sign_commits:
                subprocess.run(
                    ["git", "config", "commit.gpgsign", "true"],
                    cwd=repo_path, check=True
                )
                
                if self.commit.gpg_key:
                    subprocess.run(
                        ["git", "config", "user.signingkey", self.commit.gpg_key],
                        cwd=repo_path, check=True
                    )
            
            return True
        except subprocess.CalledProcessError as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to setup Git configuration: {e}")
            return False


# Default Git workflow configuration instance
default_git_config = GitWorkflowConfig()


def get_git_config() -> GitWorkflowConfig:
    """Get Git workflow configuration from environment or default"""
    try:
        return GitWorkflowConfig.from_env()
    except Exception:
        return default_git_config


def validate_git_config(config: GitWorkflowConfig) -> bool:
    """Validate Git workflow configuration and log any issues"""
    import logging
    
    logger = logging.getLogger(__name__)
    issues = config.validate_git_environment()
    
    if issues:
        logger.warning("Git workflow configuration issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    
    logger.info("Git workflow configuration validation passed")
    return True