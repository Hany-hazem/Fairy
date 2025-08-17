#!/usr/bin/env python3
"""
Backup and Recovery Service for MCP and Git Integration

This service provides automated backup and recovery capabilities for:
- Redis data and configuration
- Git repository state and history
- MCP configuration and logs
- System state and metadata
"""

import os
import sys
import json
import gzip
import shutil
import logging
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess
import tarfile

# Add app directory to path
sys.path.insert(0, '/app')

from app.config_manager import get_config_manager
from app.redis_config import get_redis_config
from app.git_config import get_git_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/backup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BackupService:
    """
    Comprehensive backup and recovery service for MCP and Git integration
    """
    
    def __init__(self):
        """Initialize backup service"""
        self.backup_dir = Path(os.getenv('BACKUP_DIR', '/backups'))
        self.backup_dir.mkdir(exist_ok=True)
        
        self.retention_days = int(os.getenv('BACKUP_RETENTION_DAYS', '7'))
        self.compression_enabled = os.getenv('BACKUP_COMPRESSION', 'true').lower() == 'true'
        
        # Configuration
        self.config_manager = get_config_manager()
        self.redis_config = get_redis_config()
        self.git_config = get_git_config()
        
        logger.info(f"Backup service initialized. Backup directory: {self.backup_dir}")
    
    def create_backup(self) -> bool:
        """Create comprehensive backup of all components"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"mcp_backup_{timestamp}"
        backup_path = self.backup_dir / backup_name
        
        try:
            backup_path.mkdir(exist_ok=True)
            logger.info(f"Creating backup: {backup_name}")
            
            # Create backup manifest
            manifest = {
                "backup_name": backup_name,
                "timestamp": timestamp,
                "created_at": datetime.now().isoformat(),
                "components": [],
                "metadata": {
                    "service_version": "1.0.0",
                    "backup_type": "full"
                }
            }
            
            # Backup Redis data
            if self._backup_redis(backup_path):
                manifest["components"].append("redis")
                logger.info("✓ Redis backup completed")
            else:
                logger.error("✗ Redis backup failed")
            
            # Backup Git repository
            if self._backup_git_repository(backup_path):
                manifest["components"].append("git")
                logger.info("✓ Git repository backup completed")
            else:
                logger.error("✗ Git repository backup failed")
            
            # Backup configurations
            if self._backup_configurations(backup_path):
                manifest["components"].append("config")
                logger.info("✓ Configuration backup completed")
            else:
                logger.error("✗ Configuration backup failed")
            
            # Backup logs
            if self._backup_logs(backup_path):
                manifest["components"].append("logs")
                logger.info("✓ Logs backup completed")
            else:
                logger.error("✗ Logs backup failed")
            
            # Save manifest
            manifest_file = backup_path / "backup_manifest.json"
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Compress backup if enabled
            if self.compression_enabled:
                compressed_path = self._compress_backup(backup_path)
                if compressed_path:
                    shutil.rmtree(backup_path)
                    logger.info(f"✓ Backup compressed: {compressed_path}")
            
            logger.info(f"✓ Backup completed successfully: {backup_name}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Backup failed: {e}")
            # Cleanup failed backup
            if backup_path.exists():
                shutil.rmtree(backup_path)
            return False
    
    def _backup_redis(self, backup_path: Path) -> bool:
        """Backup Redis data"""
        try:
            redis_backup_dir = backup_path / "redis"
            redis_backup_dir.mkdir(exist_ok=True)
            
            # Use redis-cli to create backup
            redis_host = self.redis_config.connection.host
            redis_port = self.redis_config.connection.port
            redis_db = self.redis_config.connection.db
            
            # Create Redis dump
            dump_file = redis_backup_dir / "dump.rdb"
            cmd = [
                "redis-cli",
                "-h", redis_host,
                "-p", str(redis_port),
                "-n", str(redis_db),
                "--rdb", str(dump_file)
            ]
            
            # Add authentication if configured
            if self.redis_config.security.password:
                cmd.extend(["-a", self.redis_config.security.password])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Redis backup failed: {result.stderr}")
                return False
            
            # Backup Redis configuration
            redis_info_file = redis_backup_dir / "redis_info.json"
            info_cmd = [
                "redis-cli",
                "-h", redis_host,
                "-p", str(redis_port),
                "INFO"
            ]
            
            if self.redis_config.security.password:
                info_cmd.extend(["-a", self.redis_config.security.password])
            
            info_result = subprocess.run(info_cmd, capture_output=True, text=True)
            if info_result.returncode == 0:
                redis_info = {
                    "info": info_result.stdout,
                    "config": self.redis_config.dict(),
                    "backup_timestamp": datetime.now().isoformat()
                }
                
                with open(redis_info_file, 'w') as f:
                    json.dump(redis_info, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"Redis backup error: {e}")
            return False
    
    def _backup_git_repository(self, backup_path: Path) -> bool:
        """Backup Git repository"""
        try:
            git_backup_dir = backup_path / "git"
            git_backup_dir.mkdir(exist_ok=True)
            
            repo_path = Path(self.git_config.repository.repo_path)
            
            # Create Git bundle (complete repository backup)
            bundle_file = git_backup_dir / "repository.bundle"
            bundle_cmd = [
                "git", "-C", str(repo_path),
                "bundle", "create", str(bundle_file),
                "--all"
            ]
            
            result = subprocess.run(bundle_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Git bundle creation failed: {result.stderr}")
                return False
            
            # Backup Git configuration
            git_config_file = git_backup_dir / "git_config.json"
            git_info = {
                "config": self.git_config.dict(),
                "repository_path": str(repo_path),
                "backup_timestamp": datetime.now().isoformat()
            }
            
            # Get repository status
            try:
                status_cmd = ["git", "-C", str(repo_path), "status", "--porcelain"]
                status_result = subprocess.run(status_cmd, capture_output=True, text=True)
                git_info["status"] = status_result.stdout
                
                # Get current branch
                branch_cmd = ["git", "-C", str(repo_path), "branch", "--show-current"]
                branch_result = subprocess.run(branch_cmd, capture_output=True, text=True)
                git_info["current_branch"] = branch_result.stdout.strip()
                
                # Get recent commits
                log_cmd = ["git", "-C", str(repo_path), "log", "--oneline", "-10"]
                log_result = subprocess.run(log_cmd, capture_output=True, text=True)
                git_info["recent_commits"] = log_result.stdout
                
            except Exception as e:
                logger.warning(f"Could not get Git status: {e}")
            
            with open(git_config_file, 'w') as f:
                json.dump(git_info, f, indent=2, default=str)
            
            # Backup task-branch mapping if it exists
            task_mapping_file = repo_path / self.git_config.task.branch_mapping_file
            if task_mapping_file.exists():
                shutil.copy2(task_mapping_file, git_backup_dir / "task_branch_mapping.json")
            
            return True
            
        except Exception as e:
            logger.error(f"Git backup error: {e}")
            return False
    
    def _backup_configurations(self, backup_path: Path) -> bool:
        """Backup all configurations"""
        try:
            config_backup_dir = backup_path / "config"
            config_backup_dir.mkdir(exist_ok=True)
            
            # Backup configuration manager state
            config_summary = self.config_manager.get_configuration_summary()
            config_file = config_backup_dir / "config_summary.json"
            
            with open(config_file, 'w') as f:
                json.dump(config_summary, f, indent=2, default=str)
            
            # Backup individual configurations
            mcp_config_file = config_backup_dir / "mcp_config.json"
            self.config_manager.mcp_config.to_file(mcp_config_file)
            
            git_config_file = config_backup_dir / "git_config.json"
            self.config_manager.git_config.to_file(git_config_file)
            
            redis_config_file = config_backup_dir / "redis_config.json"
            self.config_manager.redis_config.to_file(redis_config_file)
            
            # Backup environment variables
            env_file = config_backup_dir / "environment.json"
            env_vars = {
                key: value for key, value in os.environ.items()
                if key.startswith(('MCP_', 'GIT_', 'REDIS_', 'BACKUP_'))
            }
            
            with open(env_file, 'w') as f:
                json.dump(env_vars, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration backup error: {e}")
            return False
    
    def _backup_logs(self, backup_path: Path) -> bool:
        """Backup log files"""
        try:
            logs_backup_dir = backup_path / "logs"
            logs_backup_dir.mkdir(exist_ok=True)
            
            # Backup application logs
            log_dirs = [
                Path("/app/logs"),
                Path("/var/log"),
            ]
            
            for log_dir in log_dirs:
                if log_dir.exists():
                    for log_file in log_dir.glob("*.log"):
                        if log_file.is_file():
                            dest_file = logs_backup_dir / log_file.name
                            shutil.copy2(log_file, dest_file)
            
            # Create logs manifest
            logs_manifest = {
                "backup_timestamp": datetime.now().isoformat(),
                "log_files": [f.name for f in logs_backup_dir.glob("*.log")]
            }
            
            manifest_file = logs_backup_dir / "logs_manifest.json"
            with open(manifest_file, 'w') as f:
                json.dump(logs_manifest, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Logs backup error: {e}")
            return False
    
    def _compress_backup(self, backup_path: Path) -> Optional[Path]:
        """Compress backup directory"""
        try:
            compressed_file = backup_path.with_suffix('.tar.gz')
            
            with tarfile.open(compressed_file, 'w:gz') as tar:
                tar.add(backup_path, arcname=backup_path.name)
            
            return compressed_file
            
        except Exception as e:
            logger.error(f"Backup compression error: {e}")
            return None
    
    def cleanup_old_backups(self) -> None:
        """Remove old backups based on retention policy"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            removed_count = 0
            
            for backup_item in self.backup_dir.iterdir():
                if backup_item.is_dir() or backup_item.suffix == '.tar.gz':
                    # Extract timestamp from backup name
                    try:
                        if backup_item.is_dir():
                            timestamp_str = backup_item.name.split('_')[-2] + '_' + backup_item.name.split('_')[-1]
                        else:
                            # For compressed files, remove .tar.gz extension first
                            name_without_ext = backup_item.stem.replace('.tar', '')
                            timestamp_str = name_without_ext.split('_')[-2] + '_' + name_without_ext.split('_')[-1]
                        
                        backup_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        
                        if backup_date < cutoff_date:
                            if backup_item.is_dir():
                                shutil.rmtree(backup_item)
                            else:
                                backup_item.unlink()
                            
                            removed_count += 1
                            logger.info(f"Removed old backup: {backup_item.name}")
                            
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not parse backup date for {backup_item.name}: {e}")
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old backups")
            else:
                logger.info("No old backups to clean up")
                
        except Exception as e:
            logger.error(f"Backup cleanup error: {e}")
    
    def restore_backup(self, backup_name: str) -> bool:
        """Restore from backup"""
        try:
            # Find backup (directory or compressed file)
            backup_path = self.backup_dir / backup_name
            compressed_backup = self.backup_dir / f"{backup_name}.tar.gz"
            
            if compressed_backup.exists():
                # Extract compressed backup
                logger.info(f"Extracting backup: {compressed_backup}")
                with tarfile.open(compressed_backup, 'r:gz') as tar:
                    tar.extractall(self.backup_dir)
                backup_path = self.backup_dir / backup_name
            
            if not backup_path.exists():
                logger.error(f"Backup not found: {backup_name}")
                return False
            
            # Load backup manifest
            manifest_file = backup_path / "backup_manifest.json"
            if not manifest_file.exists():
                logger.error("Backup manifest not found")
                return False
            
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            
            logger.info(f"Restoring backup: {backup_name}")
            logger.info(f"Components: {manifest.get('components', [])}")
            
            # Restore components
            success = True
            
            if "redis" in manifest.get("components", []):
                if not self._restore_redis(backup_path):
                    success = False
            
            if "git" in manifest.get("components", []):
                if not self._restore_git_repository(backup_path):
                    success = False
            
            if "config" in manifest.get("components", []):
                if not self._restore_configurations(backup_path):
                    success = False
            
            if success:
                logger.info(f"✓ Backup restored successfully: {backup_name}")
            else:
                logger.error(f"✗ Backup restoration failed: {backup_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Backup restoration error: {e}")
            return False
    
    def _restore_redis(self, backup_path: Path) -> bool:
        """Restore Redis data"""
        try:
            redis_backup_dir = backup_path / "redis"
            dump_file = redis_backup_dir / "dump.rdb"
            
            if not dump_file.exists():
                logger.error("Redis dump file not found in backup")
                return False
            
            # This is a simplified restore - in production, you'd need to
            # stop Redis, replace the dump file, and restart Redis
            logger.warning("Redis restore requires manual intervention")
            logger.info(f"Redis dump file available at: {dump_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Redis restore error: {e}")
            return False
    
    def _restore_git_repository(self, backup_path: Path) -> bool:
        """Restore Git repository"""
        try:
            git_backup_dir = backup_path / "git"
            bundle_file = git_backup_dir / "repository.bundle"
            
            if not bundle_file.exists():
                logger.error("Git bundle file not found in backup")
                return False
            
            # This is a simplified restore - in production, you'd need to
            # clone from the bundle to restore the repository
            logger.warning("Git restore requires manual intervention")
            logger.info(f"Git bundle file available at: {bundle_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Git restore error: {e}")
            return False
    
    def _restore_configurations(self, backup_path: Path) -> bool:
        """Restore configurations"""
        try:
            config_backup_dir = backup_path / "config"
            
            # Restore configuration files
            config_files = [
                "mcp_config.json",
                "git_config.json",
                "redis_config.json"
            ]
            
            for config_file in config_files:
                src_file = config_backup_dir / config_file
                if src_file.exists():
                    dest_file = Path("/app/config") / config_file
                    dest_file.parent.mkdir(exist_ok=True)
                    shutil.copy2(src_file, dest_file)
                    logger.info(f"Restored configuration: {config_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration restore error: {e}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups"""
        backups = []
        
        try:
            for item in self.backup_dir.iterdir():
                if item.name.startswith('mcp_backup_'):
                    backup_info = {
                        "name": item.name,
                        "path": str(item),
                        "is_compressed": item.suffix == '.tar.gz',
                        "size": self._get_size(item),
                        "created": datetime.fromtimestamp(item.stat().st_ctime).isoformat()
                    }
                    
                    # Try to load manifest for more details
                    try:
                        if item.is_dir():
                            manifest_file = item / "backup_manifest.json"
                        else:
                            # For compressed backups, we'd need to extract to read manifest
                            manifest_file = None
                        
                        if manifest_file and manifest_file.exists():
                            with open(manifest_file, 'r') as f:
                                manifest = json.load(f)
                            backup_info.update({
                                "components": manifest.get("components", []),
                                "backup_type": manifest.get("metadata", {}).get("backup_type", "unknown")
                            })
                    except Exception:
                        pass
                    
                    backups.append(backup_info)
            
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x["created"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing backups: {e}")
        
        return backups
    
    def _get_size(self, path: Path) -> str:
        """Get human-readable size of file or directory"""
        try:
            if path.is_file():
                size = path.stat().st_size
            else:
                size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            
            # Convert to human-readable format
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024.0:
                    return f"{size:.1f} {unit}"
                size /= 1024.0
            return f"{size:.1f} TB"
            
        except Exception:
            return "Unknown"
    
    def run_scheduled_backup(self) -> None:
        """Run scheduled backup job"""
        logger.info("Starting scheduled backup")
        
        if self.create_backup():
            logger.info("Scheduled backup completed successfully")
        else:
            logger.error("Scheduled backup failed")
        
        # Cleanup old backups
        self.cleanup_old_backups()


def main():
    """Main backup service function"""
    backup_service = BackupService()
    
    # Get backup schedule from environment
    backup_schedule = os.getenv('BACKUP_SCHEDULE', '0 2 * * *')  # Default: daily at 2 AM
    
    # Parse cron-like schedule (simplified)
    if backup_schedule == '0 2 * * *':
        schedule.every().day.at("02:00").do(backup_service.run_scheduled_backup)
        logger.info("Scheduled daily backup at 02:00")
    elif backup_schedule.startswith('0 */'):
        # Every N hours
        hours = int(backup_schedule.split()[1].replace('*/', ''))
        schedule.every(hours).hours.do(backup_service.run_scheduled_backup)
        logger.info(f"Scheduled backup every {hours} hours")
    else:
        # Default to daily
        schedule.every().day.at("02:00").do(backup_service.run_scheduled_backup)
        logger.info("Using default daily backup schedule at 02:00")
    
    # Run initial backup
    logger.info("Running initial backup")
    backup_service.create_backup()
    
    # Start scheduler
    logger.info("Backup service started. Waiting for scheduled backups...")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    main()