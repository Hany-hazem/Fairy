#!/usr/bin/env python3
"""
Configuration Validation Script

This script validates MCP and Git integration configurations and provides
detailed reports on configuration status and issues.
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, Any

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.config_manager import get_config_manager, initialize_configuration
from app.mcp_config import get_mcp_config, validate_mcp_config
from app.git_config import get_git_config, validate_git_config
from app.redis_config import get_redis_config, validate_redis_config


def print_section(title: str) -> None:
    """Print a section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_subsection(title: str) -> None:
    """Print a subsection header"""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")


def validate_individual_configs() -> Dict[str, bool]:
    """Validate individual configuration components"""
    results = {}
    
    print_section("Individual Configuration Validation")
    
    # Validate MCP configuration
    print_subsection("MCP Configuration")
    try:
        mcp_config = get_mcp_config()
        mcp_valid = validate_mcp_config(mcp_config)
        results['mcp'] = mcp_valid
        
        print(f"✓ MCP configuration loaded successfully")
        print(f"  Server: {mcp_config.server.host}:{mcp_config.server.port}")
        print(f"  Mode: {mcp_config.server.mode.value}")
        print(f"  Max Connections: {mcp_config.server.max_connections}")
        print(f"  Validation: {'PASSED' if mcp_valid else 'FAILED'}")
        
    except Exception as e:
        print(f"✗ MCP configuration failed: {e}")
        results['mcp'] = False
    
    # Validate Git configuration
    print_subsection("Git Configuration")
    try:
        git_config = get_git_config()
        git_valid = validate_git_config(git_config)
        results['git'] = git_valid
        
        print(f"✓ Git configuration loaded successfully")
        print(f"  Repository: {git_config.repository.repo_path}")
        print(f"  Default Branch: {git_config.repository.default_branch}")
        print(f"  Workflow Mode: {git_config.workflow_mode.value}")
        print(f"  Auto Commit: {git_config.commit.auto_commit}")
        print(f"  Validation: {'PASSED' if git_valid else 'FAILED'}")
        
    except Exception as e:
        print(f"✗ Git configuration failed: {e}")
        results['git'] = False
    
    # Validate Redis configuration
    print_subsection("Redis Configuration")
    try:
        redis_config = get_redis_config()
        redis_valid = validate_redis_config(redis_config)
        results['redis'] = redis_valid
        
        print(f"✓ Redis configuration loaded successfully")
        print(f"  Host: {redis_config.connection.host}:{redis_config.connection.port}")
        print(f"  Database: {redis_config.connection.db}")
        print(f"  Security Mode: {redis_config.security.mode.value}")
        print(f"  Max Connections: {redis_config.connection.max_connections}")
        print(f"  Validation: {'PASSED' if redis_valid else 'FAILED'}")
        
    except Exception as e:
        print(f"✗ Redis configuration failed: {e}")
        results['redis'] = False
    
    return results


def validate_integrated_config(config_dir: str = None) -> bool:
    """Validate integrated configuration using configuration manager"""
    print_section("Integrated Configuration Validation")
    
    try:
        config_manager = get_config_manager(config_dir)
        status = config_manager.validate_all_configurations()
        
        print(f"Configuration Status: {'VALID' if status.is_valid else 'INVALID'}")
        print(f"Last Validated: {status.last_validated}")
        
        if status.issues:
            print(f"\nIssues Found ({len(status.issues)}):")
            for i, issue in enumerate(status.issues, 1):
                print(f"  {i}. {issue}")
        
        if status.warnings:
            print(f"\nWarnings ({len(status.warnings)}):")
            for i, warning in enumerate(status.warnings, 1):
                print(f"  {i}. {warning}")
        
        # Print configuration summary
        print_subsection("Configuration Summary")
        summary = config_manager.get_configuration_summary()
        
        print("MCP Configuration:")
        for key, value in summary['mcp'].items():
            print(f"  {key}: {value}")
        
        print("\nGit Configuration:")
        for key, value in summary['git'].items():
            print(f"  {key}: {value}")
        
        print("\nRedis Configuration:")
        for key, value in summary['redis'].items():
            print(f"  {key}: {value}")
        
        return status.is_valid
        
    except Exception as e:
        print(f"✗ Integrated configuration validation failed: {e}")
        return False


def check_environment_variables() -> None:
    """Check for relevant environment variables"""
    print_section("Environment Variables")
    
    env_vars = [
        # MCP variables
        'MCP_SERVER_HOST', 'MCP_SERVER_PORT', 'MCP_SERVER_MODE',
        'MCP_SERVER_MAX_CONNECTIONS', 'MCP_MESSAGE_MAX_SIZE',
        
        # Git variables
        'GIT_REPO_PATH', 'GIT_DEFAULT_BRANCH', 'GIT_WORKFLOW_MODE',
        'GIT_AUTO_COMMIT', 'GIT_USER_NAME', 'GIT_USER_EMAIL',
        
        # Redis variables
        'REDIS_URL', 'REDIS_HOST', 'REDIS_PORT', 'REDIS_PASSWORD',
        'REDIS_MAX_CONNECTIONS', 'REDIS_SECURITY_MODE',
    ]
    
    found_vars = []
    for var in env_vars:
        value = os.getenv(var)
        if value is not None:
            found_vars.append((var, value))
    
    if found_vars:
        print(f"Found {len(found_vars)} configuration environment variables:")
        for var, value in found_vars:
            # Mask sensitive values
            if 'PASSWORD' in var or 'KEY' in var:
                display_value = '*' * len(value) if value else 'None'
            else:
                display_value = value
            print(f"  {var} = {display_value}")
    else:
        print("No configuration environment variables found")
        print("Using default configuration values")


def generate_config_report(output_file: str = None) -> None:
    """Generate detailed configuration report"""
    print_section("Generating Configuration Report")
    
    try:
        config_manager = get_config_manager()
        status = config_manager.validate_all_configurations()
        
        report = {
            "timestamp": status.last_validated.isoformat(),
            "overall_status": {
                "is_valid": status.is_valid,
                "issues_count": len(status.issues),
                "warnings_count": len(status.warnings)
            },
            "issues": status.issues,
            "warnings": status.warnings,
            "configuration_summary": config_manager.get_configuration_summary(),
            "detailed_configs": {
                "mcp": config_manager.mcp_config.dict(),
                "git": config_manager.git_config.dict(),
                "redis": config_manager.redis_config.dict()
            }
        }
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"✓ Configuration report saved to: {output_path}")
        else:
            print("Configuration Report:")
            print(json.dumps(report, indent=2, default=str))
        
    except Exception as e:
        print(f"✗ Failed to generate configuration report: {e}")


def main():
    """Main validation function"""
    parser = argparse.ArgumentParser(
        description="Validate MCP and Git integration configurations"
    )
    parser.add_argument(
        '--config-dir', 
        help='Configuration directory path'
    )
    parser.add_argument(
        '--report', 
        help='Generate detailed report to file'
    )
    parser.add_argument(
        '--individual-only', 
        action='store_true',
        help='Only validate individual configurations'
    )
    parser.add_argument(
        '--integrated-only', 
        action='store_true',
        help='Only validate integrated configuration'
    )
    parser.add_argument(
        '--no-env-check', 
        action='store_true',
        help='Skip environment variable check'
    )
    
    args = parser.parse_args()
    
    print("MCP and Git Integration Configuration Validator")
    print("=" * 60)
    
    # Check environment variables
    if not args.no_env_check:
        check_environment_variables()
    
    all_valid = True
    
    # Validate individual configurations
    if not args.integrated_only:
        individual_results = validate_individual_configs()
        if not all(individual_results.values()):
            all_valid = False
    
    # Validate integrated configuration
    if not args.individual_only:
        integrated_valid = validate_integrated_config(args.config_dir)
        if not integrated_valid:
            all_valid = False
    
    # Generate report if requested
    if args.report:
        generate_config_report(args.report)
    
    # Final status
    print_section("Validation Summary")
    if all_valid:
        print("✓ All configurations are valid and ready for use")
        sys.exit(0)
    else:
        print("✗ Configuration validation failed")
        print("Please review the issues above and update your configuration")
        sys.exit(1)


if __name__ == "__main__":
    main()