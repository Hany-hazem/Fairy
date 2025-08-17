"""
File System Manager

This module provides secure file system access and management with user permission controls,
integrating with the existing FileContentAnalyzer and FileOrganizationManager.
"""

import os
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from .file_content_analyzer import FileContentAnalyzer, ContentAnalysis
from .file_organization_manager import FileOrganizationManager, OrganizationPlan, DuplicateGroup
from .privacy_security_manager import PrivacySecurityManager, PermissionType, DataCategory
from .user_context_manager import UserContextManager
from .personal_assistant_models import UserContext, Interaction, InteractionType

logger = logging.getLogger(__name__)


class FileOperation(Enum):
    """Types of file operations"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    MOVE = "move"
    COPY = "copy"
    LIST = "list"
    SEARCH = "search"
    ANALYZE = "analyze"
    ORGANIZE = "organize"


class AccessLevel(Enum):
    """File access levels"""
    NONE = "none"
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    FULL_ACCESS = "full_access"


@dataclass
class FileAccessRequest:
    """Request for file access"""
    user_id: str
    operation: FileOperation
    file_path: str
    requested_at: datetime = field(default_factory=datetime.now)
    justification: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileAccessResult:
    """Result of file access operation"""
    success: bool
    operation: FileOperation
    file_path: str
    content: Optional[str] = None
    analysis: Optional[ContentAnalysis] = None
    error_message: Optional[str] = None
    permission_required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileSystemStats:
    """File system statistics"""
    total_files_accessed: int = 0
    total_bytes_read: int = 0
    total_bytes_written: int = 0
    operations_by_type: Dict[FileOperation, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


class FileSystemManager:
    """Manages secure file system access with user permission controls"""
    
    def __init__(self, 
                 privacy_manager: PrivacySecurityManager,
                 context_manager: UserContextManager,
                 content_analyzer: Optional[FileContentAnalyzer] = None,
                 organization_manager: Optional[FileOrganizationManager] = None):
        self.privacy_manager = privacy_manager
        self.context_manager = context_manager
        self.content_analyzer = content_analyzer or FileContentAnalyzer()
        self.organization_manager = organization_manager or FileOrganizationManager(self.content_analyzer)
        
        # File access statistics
        self._stats: Dict[str, FileSystemStats] = {}
        
        # Restricted paths that require special permission
        self._restricted_paths = {
            # System directories
            "/etc", "/sys", "/proc", "/dev", "/boot",
            # Windows system directories
            "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)",
            # User sensitive directories
            "/.ssh", "/.gnupg", "/keychain",
            # Common sensitive file patterns
            "*.key", "*.pem", "*.p12", "*.pfx", "password*", "secret*"
        }
        
        # Allowed file extensions for reading
        self._safe_read_extensions = {
            '.txt', '.md', '.json', '.xml', '.csv', '.log',
            '.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.h',
            '.pdf', '.doc', '.docx', '.rtf', '.odt',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'
        }
        
        # Maximum file size for operations (100MB)
        self._max_file_size = 100 * 1024 * 1024
    
    async def read_file(self, user_id: str, file_path: str, 
                       justification: Optional[str] = None) -> FileAccessResult:
        """Read file content with permission checks"""
        try:
            # Create access request
            request = FileAccessRequest(
                user_id=user_id,
                operation=FileOperation.READ,
                file_path=file_path,
                justification=justification
            )
            
            # Check permissions
            permission_check = await self._check_file_access_permission(request)
            if not permission_check.success:
                return permission_check
            
            # Validate file path and safety
            validation_result = await self._validate_file_access(file_path, FileOperation.READ)
            if not validation_result.success:
                return validation_result
            
            # Read file content
            path = Path(file_path).resolve()
            if not path.exists():
                return FileAccessResult(
                    success=False,
                    operation=FileOperation.READ,
                    file_path=file_path,
                    error_message="File not found"
                )
            
            if not path.is_file():
                return FileAccessResult(
                    success=False,
                    operation=FileOperation.READ,
                    file_path=file_path,
                    error_message="Path is not a file"
                )
            
            # Check file size
            file_size = path.stat().st_size
            if file_size > self._max_file_size:
                return FileAccessResult(
                    success=False,
                    operation=FileOperation.READ,
                    file_path=file_path,
                    error_message=f"File too large ({file_size} bytes, max {self._max_file_size})"
                )
            
            # Read content
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try binary mode for non-text files
                with open(path, 'rb') as f:
                    binary_content = f.read()
                    content = f"[Binary file: {len(binary_content)} bytes]"
            
            # Update statistics
            await self._update_stats(user_id, FileOperation.READ, file_size)
            
            # Log access
            try:
                await self._log_file_access(user_id, request, success=True)
            except Exception as log_error:
                logger.warning(f"Failed to log file access: {log_error}")
            
            # Update user context
            try:
                await self._update_user_context(user_id, file_path, FileOperation.READ)
            except Exception as context_error:
                logger.warning(f"Failed to update user context: {context_error}")
            
            return FileAccessResult(
                success=True,
                operation=FileOperation.READ,
                file_path=file_path,
                content=content,
                metadata={'file_size': file_size, 'encoding': 'utf-8'}
            )
            
        except Exception as e:
            logger.error(f"Error reading file {file_path} for user {user_id}: {e}")
            try:
                await self._log_file_access(user_id, request, success=False, error=str(e))
            except Exception as log_error:
                logger.warning(f"Failed to log file access error: {log_error}")
            
            return FileAccessResult(
                success=False,
                operation=FileOperation.READ,
                file_path=file_path,
                error_message=f"Failed to read file: {str(e)}"
            )
    
    async def write_file(self, user_id: str, file_path: str, content: str,
                        create_dirs: bool = False, backup: bool = True,
                        justification: Optional[str] = None) -> FileAccessResult:
        """Write content to file with permission checks"""
        try:
            # Create access request
            request = FileAccessRequest(
                user_id=user_id,
                operation=FileOperation.WRITE,
                file_path=file_path,
                justification=justification,
                metadata={'create_dirs': create_dirs, 'backup': backup}
            )
            
            # Check permissions
            permission_check = await self._check_file_access_permission(request)
            if not permission_check.success:
                return permission_check
            
            # Validate file path and safety
            validation_result = await self._validate_file_access(file_path, FileOperation.WRITE)
            if not validation_result.success:
                return validation_result
            
            path = Path(file_path).resolve()
            
            # Create directories if requested
            if create_dirs and not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup if file exists and backup is requested
            if backup and path.exists():
                backup_path = path.with_suffix(path.suffix + '.backup')
                import shutil
                shutil.copy2(path, backup_path)
            
            # Write content
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Update statistics
            content_size = len(content.encode('utf-8'))
            await self._update_stats(user_id, FileOperation.WRITE, content_size)
            
            # Log access
            await self._log_file_access(user_id, request, success=True)
            
            # Update user context
            await self._update_user_context(user_id, file_path, FileOperation.WRITE)
            
            return FileAccessResult(
                success=True,
                operation=FileOperation.WRITE,
                file_path=file_path,
                metadata={'bytes_written': content_size, 'backup_created': backup and path.exists()}
            )
            
        except Exception as e:
            logger.error(f"Error writing file {file_path} for user {user_id}: {e}")
            await self._log_file_access(user_id, request, success=False, error=str(e))
            
            return FileAccessResult(
                success=False,
                operation=FileOperation.WRITE,
                file_path=file_path,
                error_message=f"Failed to write file: {str(e)}"
            )
    
    async def analyze_file_content(self, user_id: str, file_path: str,
                                 justification: Optional[str] = None) -> FileAccessResult:
        """Analyze file content using the content analyzer"""
        try:
            # First read the file
            read_result = await self.read_file(user_id, file_path, justification)
            if not read_result.success:
                return read_result
            
            # Analyze content
            analysis = await self.content_analyzer.analyze_file_content(file_path, read_result.content)
            
            # Update statistics
            await self._update_stats(user_id, FileOperation.ANALYZE, 0)
            
            return FileAccessResult(
                success=True,
                operation=FileOperation.ANALYZE,
                file_path=file_path,
                content=read_result.content,
                analysis=analysis,
                metadata={'analysis_completed': True}
            )
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path} for user {user_id}: {e}")
            
            return FileAccessResult(
                success=False,
                operation=FileOperation.ANALYZE,
                file_path=file_path,
                error_message=f"Failed to analyze file: {str(e)}"
            )
    
    async def list_directory(self, user_id: str, directory_path: str,
                           recursive: bool = False, include_hidden: bool = False,
                           justification: Optional[str] = None) -> FileAccessResult:
        """List directory contents with permission checks"""
        try:
            # Create access request
            request = FileAccessRequest(
                user_id=user_id,
                operation=FileOperation.LIST,
                file_path=directory_path,
                justification=justification,
                metadata={'recursive': recursive, 'include_hidden': include_hidden}
            )
            
            # Check permissions
            permission_check = await self._check_file_access_permission(request)
            if not permission_check.success:
                return permission_check
            
            # Validate directory path
            validation_result = await self._validate_file_access(directory_path, FileOperation.LIST)
            if not validation_result.success:
                return validation_result
            
            path = Path(directory_path).resolve()
            if not path.exists():
                return FileAccessResult(
                    success=False,
                    operation=FileOperation.LIST,
                    file_path=directory_path,
                    error_message="Directory not found"
                )
            
            if not path.is_dir():
                return FileAccessResult(
                    success=False,
                    operation=FileOperation.LIST,
                    file_path=directory_path,
                    error_message="Path is not a directory"
                )
            
            # List contents
            files = []
            directories = []
            
            if recursive:
                items = path.rglob('*')
            else:
                items = path.iterdir()
            
            for item in items:
                # Skip hidden files unless requested
                if not include_hidden and item.name.startswith('.'):
                    continue
                
                try:
                    stat = item.stat()
                    item_info = {
                        'name': item.name,
                        'path': str(item),
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'is_file': item.is_file(),
                        'is_dir': item.is_dir()
                    }
                    
                    if item.is_file():
                        files.append(item_info)
                    elif item.is_dir():
                        directories.append(item_info)
                        
                except (PermissionError, OSError):
                    # Skip items we can't access
                    continue
            
            # Sort results
            files.sort(key=lambda x: x['name'])
            directories.sort(key=lambda x: x['name'])
            
            result_content = {
                'directories': directories,
                'files': files,
                'total_directories': len(directories),
                'total_files': len(files)
            }
            
            # Update statistics
            await self._update_stats(user_id, FileOperation.LIST, 0)
            
            # Log access
            await self._log_file_access(user_id, request, success=True)
            
            return FileAccessResult(
                success=True,
                operation=FileOperation.LIST,
                file_path=directory_path,
                content=str(result_content),
                metadata=result_content
            )
            
        except Exception as e:
            logger.error(f"Error listing directory {directory_path} for user {user_id}: {e}")
            await self._log_file_access(user_id, request, success=False, error=str(e))
            
            return FileAccessResult(
                success=False,
                operation=FileOperation.LIST,
                file_path=directory_path,
                error_message=f"Failed to list directory: {str(e)}"
            )
    
    async def search_files(self, user_id: str, search_query: str, 
                          search_directory: Optional[str] = None,
                          file_types: Optional[List[str]] = None,
                          justification: Optional[str] = None) -> FileAccessResult:
        """Search for files using the organization manager"""
        try:
            # Create access request
            request = FileAccessRequest(
                user_id=user_id,
                operation=FileOperation.SEARCH,
                file_path=search_directory or ".",
                justification=justification,
                metadata={'query': search_query, 'file_types': file_types}
            )
            
            # Check permissions
            permission_check = await self._check_file_access_permission(request)
            if not permission_check.success:
                return permission_check
            
            # Use organization manager for search
            search_results = await self.organization_manager.search_files(
                query=search_query,
                directory=search_directory,
                file_types=file_types
            )
            
            # Convert results to serializable format
            results = []
            for metadata in search_results:
                results.append({
                    'path': metadata.path,
                    'name': metadata.name,
                    'size': metadata.size,
                    'modified': metadata.modified_at.isoformat(),
                    'file_type': metadata.file_type,
                    'category': metadata.category,
                    'tags': metadata.tags
                })
            
            # Update statistics
            await self._update_stats(user_id, FileOperation.SEARCH, 0)
            
            # Log access
            await self._log_file_access(user_id, request, success=True)
            
            return FileAccessResult(
                success=True,
                operation=FileOperation.SEARCH,
                file_path=search_directory or ".",
                content=f"Found {len(results)} files matching '{search_query}'",
                metadata={'results': results, 'query': search_query, 'count': len(results)}
            )
            
        except Exception as e:
            logger.error(f"Error searching files for user {user_id}: {e}")
            await self._log_file_access(user_id, request, success=False, error=str(e))
            
            return FileAccessResult(
                success=False,
                operation=FileOperation.SEARCH,
                file_path=search_directory or ".",
                error_message=f"Failed to search files: {str(e)}"
            )
    
    async def organize_files(self, user_id: str, directory_path: str,
                           strategy: str = "by_type", dry_run: bool = True,
                           justification: Optional[str] = None) -> FileAccessResult:
        """Organize files using the organization manager"""
        try:
            # Create access request
            request = FileAccessRequest(
                user_id=user_id,
                operation=FileOperation.ORGANIZE,
                file_path=directory_path,
                justification=justification,
                metadata={'strategy': strategy, 'dry_run': dry_run}
            )
            
            # Check permissions (requires write access for actual organization)
            if not dry_run:
                permission_check = await self._check_file_access_permission(request)
                if not permission_check.success:
                    return permission_check
            
            # Import the strategy enum
            from .file_organization_manager import OrganizationStrategy
            strategy_map = {
                'by_type': OrganizationStrategy.BY_TYPE,
                'by_date': OrganizationStrategy.BY_DATE,
                'by_size': OrganizationStrategy.BY_SIZE
            }
            
            org_strategy = strategy_map.get(strategy, OrganizationStrategy.BY_TYPE)
            
            # Create organization plan
            plan = await self.organization_manager.create_organization_plan(
                directory_path, org_strategy
            )
            
            # Execute plan if not dry run
            if not dry_run:
                execution_result = await self.organization_manager.execute_organization_plan(
                    plan, dry_run=False
                )
            else:
                execution_result = {'dry_run': True, 'plan_created': True}
            
            # Update statistics
            await self._update_stats(user_id, FileOperation.ORGANIZE, 0)
            
            # Log access
            await self._log_file_access(user_id, request, success=True)
            
            return FileAccessResult(
                success=True,
                operation=FileOperation.ORGANIZE,
                file_path=directory_path,
                content=f"Organization plan created with {len(plan.moves)} moves",
                metadata={
                    'plan': {
                        'strategy': strategy,
                        'moves': len(plan.moves),
                        'creates': len(plan.creates),
                        'estimated_time': plan.estimated_time,
                        'benefits': plan.benefits
                    },
                    'execution': execution_result
                }
            )
            
        except Exception as e:
            logger.error(f"Error organizing files for user {user_id}: {e}")
            await self._log_file_access(user_id, request, success=False, error=str(e))
            
            return FileAccessResult(
                success=False,
                operation=FileOperation.ORGANIZE,
                file_path=directory_path,
                error_message=f"Failed to organize files: {str(e)}"
            )
    
    async def get_file_stats(self, user_id: str) -> Dict[str, Any]:
        """Get file access statistics for user"""
        if user_id not in self._stats:
            return {
                'total_files_accessed': 0,
                'total_bytes_read': 0,
                'total_bytes_written': 0,
                'operations_by_type': {},
                'last_updated': None
            }
        
        stats = self._stats[user_id]
        return {
            'total_files_accessed': stats.total_files_accessed,
            'total_bytes_read': stats.total_bytes_read,
            'total_bytes_written': stats.total_bytes_written,
            'operations_by_type': {op.value: count for op, count in stats.operations_by_type.items()},
            'last_updated': stats.last_updated.isoformat()
        }
    
    async def _check_file_access_permission(self, request: FileAccessRequest) -> FileAccessResult:
        """Check if user has permission for file access"""
        # Determine required permission based on operation
        if request.operation in [FileOperation.READ, FileOperation.LIST, 
                               FileOperation.SEARCH, FileOperation.ANALYZE]:
            # Check read permission
            has_permission = await self.privacy_manager.check_permission(
                request.user_id, PermissionType.FILE_READ
            )
            
            if not has_permission:
                # Request permission
                granted = await self.privacy_manager.request_permission(
                    request.user_id, PermissionType.FILE_READ,
                    scope={'operation': request.operation.value, 'path': request.file_path}
                )
                
                if not granted:
                    return FileAccessResult(
                        success=False,
                        operation=request.operation,
                        file_path=request.file_path,
                        error_message="File read permission denied",
                        permission_required=True
                    )
        
        elif request.operation in [FileOperation.WRITE, FileOperation.DELETE, 
                                 FileOperation.MOVE, FileOperation.COPY, FileOperation.ORGANIZE]:
            # Check write permission
            has_permission = await self.privacy_manager.check_permission(
                request.user_id, PermissionType.FILE_WRITE
            )
            
            if not has_permission:
                # Request permission
                granted = await self.privacy_manager.request_permission(
                    request.user_id, PermissionType.FILE_WRITE,
                    scope={'operation': request.operation.value, 'path': request.file_path}
                )
                
                if not granted:
                    return FileAccessResult(
                        success=False,
                        operation=request.operation,
                        file_path=request.file_path,
                        error_message="File write permission denied",
                        permission_required=True
                    )
        
        return FileAccessResult(success=True, operation=request.operation, file_path=request.file_path)
    
    async def _validate_file_access(self, file_path: str, operation: FileOperation) -> FileAccessResult:
        """Validate file access for security"""
        try:
            # Check for path traversal before resolving
            if '..' in file_path:
                return FileAccessResult(
                    success=False,
                    operation=operation,
                    file_path=file_path,
                    error_message="Path traversal detected"
                )
            
            # Try to resolve the path
            try:
                path = Path(file_path).resolve()
            except (OSError, ValueError) as e:
                return FileAccessResult(
                    success=False,
                    operation=operation,
                    file_path=file_path,
                    error_message=f"Invalid path: {str(e)}"
                )
            
            # Check for restricted paths
            path_str = str(path)
            for restricted in self._restricted_paths:
                if restricted.startswith('/') or restricted.startswith('C:'):
                    # Absolute path restriction
                    if path_str.startswith(restricted):
                        return FileAccessResult(
                            success=False,
                            operation=operation,
                            file_path=file_path,
                            error_message=f"Access to restricted path denied: {restricted}"
                        )
                else:
                    # Pattern-based restriction
                    import fnmatch
                    if fnmatch.fnmatch(path.name, restricted):
                        return FileAccessResult(
                            success=False,
                            operation=operation,
                            file_path=file_path,
                            error_message=f"Access to restricted file pattern denied: {restricted}"
                        )
            
            # Check file extension for read operations
            if operation == FileOperation.READ and path.suffix:
                if path.suffix.lower() not in self._safe_read_extensions:
                    return FileAccessResult(
                        success=False,
                        operation=operation,
                        file_path=file_path,
                        error_message=f"File type not allowed for reading: {path.suffix}"
                    )
            
            return FileAccessResult(success=True, operation=operation, file_path=file_path)
            
        except Exception as e:
            return FileAccessResult(
                success=False,
                operation=operation,
                file_path=file_path,
                error_message=f"Path validation failed: {str(e)}"
            )
    
    async def _update_stats(self, user_id: str, operation: FileOperation, bytes_processed: int):
        """Update file access statistics"""
        if user_id not in self._stats:
            self._stats[user_id] = FileSystemStats()
        
        stats = self._stats[user_id]
        stats.total_files_accessed += 1
        stats.last_updated = datetime.now()
        
        if operation in [FileOperation.READ, FileOperation.ANALYZE]:
            stats.total_bytes_read += bytes_processed
        elif operation == FileOperation.WRITE:
            stats.total_bytes_written += bytes_processed
        
        if operation not in stats.operations_by_type:
            stats.operations_by_type[operation] = 0
        stats.operations_by_type[operation] += 1
    
    async def _log_file_access(self, user_id: str, request: FileAccessRequest, 
                             success: bool, error: Optional[str] = None):
        """Log file access for audit purposes"""
        # Create interaction record
        interaction = Interaction(
            user_id=user_id,
            interaction_type=InteractionType.FILE_ACCESS,
            content=f"{request.operation.value}: {request.file_path}",
            response="Success" if success else f"Failed: {error}",
            context_data={
                'operation': request.operation.value,
                'file_path': request.file_path,
                'justification': request.justification,
                'success': success,
                'error': error,
                'metadata': request.metadata
            }
        )
        
        # Add to user context
        await self.context_manager.add_interaction(user_id, interaction)
    
    async def _update_user_context(self, user_id: str, file_path: str, operation: FileOperation):
        """Update user context with file access information"""
        context = await self.context_manager.get_user_context(user_id)
        
        # Add to recent files if it's a read or write operation
        if operation in [FileOperation.READ, FileOperation.WRITE, FileOperation.ANALYZE]:
            if file_path not in context.current_files:
                context.current_files.append(file_path)
                # Keep only recent files (last 20)
                if len(context.current_files) > 20:
                    context.current_files = context.current_files[-20:]
            
            # Also add to task context recent files
            if file_path not in context.task_context.recent_files:
                context.task_context.recent_files.append(file_path)
                # Keep only recent files (last 50)
                if len(context.task_context.recent_files) > 50:
                    context.task_context.recent_files = context.task_context.recent_files[-50:]
        
        # Update context
        await self.context_manager.update_user_context(context)