#!/usr/bin/env python3
"""
Maintenance Script for MCP and Git Integration

This script provides maintenance operations for the MCP and Git integration
system including health monitoring, log rotation, cleanup, and diagnostics.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.config_manager import get_config_manager
from app.health_endpoints import HealthChecker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MaintenanceManager:
    """
    Comprehensive maintenance manager for MCP and Git integration
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize maintenance manager"""
        self.config_manager = get_config_manager(config_dir)
        self.health_checker = HealthChecker()
        
        # Paths
        self.project_dir = Path.cwd()
        self.logs_dir = self.project_dir / "logs"
        self.backups_dir = self.project_dir / "backups"
        self.temp_dir = self.project_dir / "temp"
        
        # Ensure directories exist
        self.logs_dir.mkdir(exist_ok=True)
        self.backups_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check"""
        logger.info("Running health check...")
        
        health_status = await self.health_checker.check_overall_health()
        
        # Log results
        logger.info(f"Overall health: {health_status.status}")
        
        if health_status.status != "healthy":
            logger.warning("System is not healthy:")
            for component, check in health_status.checks.items():
                if check["status"] != "healthy":
                    logger.warning(f"  {component}: {check['status']} - {check['message']}")
        
        return health_status.dict()
    
    def rotate_logs(self, max_size_mb: int = 100, max_files: int = 5) -> None:
        """Rotate log files"""
        logger.info("Rotating log files...")
        
        rotated_count = 0
        
        for log_file in self.logs_dir.glob("*.log"):
            if log_file.stat().st_size > max_size_mb * 1024 * 1024:
                # Rotate the log file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                rotated_name = f"{log_file.stem}_{timestamp}.log"
                rotated_path = self.logs_dir / rotated_name
                
                # Move current log to rotated name
                shutil.move(log_file, rotated_path)
                
                # Create new empty log file
                log_file.touch()
                
                rotated_count += 1
                logger.info(f"Rotated log file: {log_file.name}")
                
                # Clean up old rotated files
                self._cleanup_old_log_files(log_file.stem, max_files)
        
        logger.info(f"Rotated {rotated_count} log files")
    
    def _cleanup_old_log_files(self, log_stem: str, max_files: int) -> None:
        """Clean up old rotated log files"""
        pattern = f"{log_stem}_*.log"
        rotated_files = sorted(
            self.logs_dir.glob(pattern),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # Remove files beyond the limit
        for old_file in rotated_files[max_files:]:
            old_file.unlink()
            logger.info(f"Removed old log file: {old_file.name}")
    
    def cleanup_temp_files(self, max_age_hours: int = 24) -> None:
        """Clean up temporary files"""
        logger.info("Cleaning up temporary files...")
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        removed_count = 0
        
        for temp_file in self.temp_dir.rglob("*"):
            if temp_file.is_file():
                file_time = datetime.fromtimestamp(temp_file.stat().st_mtime)
                if file_time < cutoff_time:
                    temp_file.unlink()
                    removed_count += 1
        
        logger.info(f"Removed {removed_count} temporary files")
    
    def optimize_git_repository(self) -> None:
        """Optimize Git repository"""
        logger.info("Optimizing Git repository...")
        
        repo_path = Path(self.config_manager.git_config.repository.repo_path)
        
        try:
            # Run git gc to optimize repository
            subprocess.run(
                ["git", "-C", str(repo_path), "gc", "--aggressive", "--prune=now"],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("Git repository optimized")
            
            # Get repository size after optimization
            repo_size = sum(f.stat().st_size for f in repo_path.rglob('*') if f.is_file())
            logger.info(f"Repository size: {repo_size / (1024*1024):.1f} MB")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Git optimization failed: {e.stderr}")
    
    def check_disk_space(self, warning_threshold: float = 80.0, critical_threshold: float = 90.0) -> Dict[str, Any]:
        """Check disk space usage"""
        logger.info("Checking disk space...")
        
        total, used, free = shutil.disk_usage("/")
        usage_percent = (used / total) * 100
        
        status = "ok"
        if usage_percent >= critical_threshold:
            status = "critical"
            logger.error(f"Disk usage is critical: {usage_percent:.1f}%")
        elif usage_percent >= warning_threshold:
            status = "warning"
            logger.warning(f"Disk usage is high: {usage_percent:.1f}%")
        else:
            logger.info(f"Disk usage is normal: {usage_percent:.1f}%")
        
        return {
            "status": status,
            "usage_percent": usage_percent,
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": free
        }
    
    def check_docker_resources(self) -> Dict[str, Any]:
        """Check Docker resource usage"""
        logger.info("Checking Docker resources...")
        
        try:
            # Get Docker system info
            result = subprocess.run(
                ["docker", "system", "df", "--format", "json"],
                capture_output=True,
                text=True,
                check=True
            )
            
            docker_info = json.loads(result.stdout)
            
            # Calculate total space used
            total_size = 0
            for item in docker_info:
                if "Size" in item:
                    # Parse size string (e.g., "1.2GB" -> bytes)
                    size_str = item["Size"]
                    if size_str != "0B":
                        # This is a simplified parser - in production, use a proper size parser
                        total_size += 1024 * 1024 * 1024  # Placeholder
            
            logger.info(f"Docker is using approximately {total_size / (1024*1024*1024):.1f} GB")
            
            return {
                "status": "ok",
                "total_size_bytes": total_size,
                "details": docker_info
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to check Docker resources: {e}")
            return {"status": "error", "error": str(e)}
    
    def cleanup_docker_resources(self) -> None:
        """Clean up unused Docker resources"""
        logger.info("Cleaning up Docker resources...")
        
        try:
            # Remove unused containers
            subprocess.run(["docker", "container", "prune", "-f"], check=True)
            logger.info("Removed unused containers")
            
            # Remove unused images
            subprocess.run(["docker", "image", "prune", "-f"], check=True)
            logger.info("Removed unused images")
            
            # Remove unused volumes
            subprocess.run(["docker", "volume", "prune", "-f"], check=True)
            logger.info("Removed unused volumes")
            
            # Remove unused networks
            subprocess.run(["docker", "network", "prune", "-f"], check=True)
            logger.info("Removed unused networks")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker cleanup failed: {e}")
    
    def generate_system_report(self) -> Dict[str, Any]:
        """Generate comprehensive system report"""
        logger.info("Generating system report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {},
            "configuration": {},
            "health": {},
            "resources": {},
            "git": {},
            "docker": {}
        }
        
        try:
            # System information
            report["system_info"] = {
                "hostname": os.uname().nodename,
                "platform": os.uname().sysname,
                "python_version": sys.version,
                "working_directory": str(Path.cwd())
            }
            
            # Configuration summary
            report["configuration"] = self.config_manager.get_configuration_summary()
            
            # Health status
            import asyncio
            health_status = asyncio.run(self.run_health_check())
            report["health"] = health_status
            
            # Resource usage
            report["resources"]["disk"] = self.check_disk_space()
            report["resources"]["docker"] = self.check_docker_resources()
            
            # Git repository info
            repo_path = Path(self.config_manager.git_config.repository.repo_path)
            if repo_path.exists() and (repo_path / ".git").exists():
                try:
                    # Get current branch
                    branch_result = subprocess.run(
                        ["git", "-C", str(repo_path), "branch", "--show-current"],
                        capture_output=True, text=True, check=True
                    )
                    
                    # Get status
                    status_result = subprocess.run(
                        ["git", "-C", str(repo_path), "status", "--porcelain"],
                        capture_output=True, text=True, check=True
                    )
                    
                    report["git"] = {
                        "current_branch": branch_result.stdout.strip(),
                        "uncommitted_files": len(status_result.stdout.strip().split('\n')) if status_result.stdout.strip() else 0,
                        "repository_path": str(repo_path)
                    }
                    
                except subprocess.CalledProcessError:
                    report["git"] = {"error": "Failed to get Git information"}
            
        except Exception as e:
            logger.error(f"Error generating system report: {e}")
            report["error"] = str(e)
        
        return report
    
    def save_report(self, report: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """Save system report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"system_report_{timestamp}.json"
        
        report_path = self.logs_dir / filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"System report saved to: {report_path}")
        return report_path


def main():
    """Main maintenance function"""
    parser = argparse.ArgumentParser(
        description="Maintenance operations for MCP and Git integration"
    )
    parser.add_argument(
        '--config-dir',
        help='Configuration directory path'
    )
    parser.add_argument(
        '--operation',
        choices=['health', 'logs', 'cleanup', 'optimize', 'report', 'all'],
        default='all',
        help='Maintenance operation to perform'
    )
    parser.add_argument(
        '--output',
        help='Output file for reports'
    )
    
    args = parser.parse_args()
    
    # Initialize maintenance manager
    maintenance = MaintenanceManager(args.config_dir)
    
    logger.info(f"Starting maintenance operation: {args.operation}")
    
    try:
        if args.operation in ['health', 'all']:
            import asyncio
            asyncio.run(maintenance.run_health_check())
        
        if args.operation in ['logs', 'all']:
            maintenance.rotate_logs()
        
        if args.operation in ['cleanup', 'all']:
            maintenance.cleanup_temp_files()
            maintenance.cleanup_docker_resources()
        
        if args.operation in ['optimize', 'all']:
            maintenance.optimize_git_repository()
        
        if args.operation in ['report', 'all']:
            report = maintenance.generate_system_report()
            
            if args.output:
                maintenance.save_report(report, args.output)
            else:
                maintenance.save_report(report)
        
        logger.info("Maintenance operations completed successfully")
        
    except Exception as e:
        logger.error(f"Maintenance operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()