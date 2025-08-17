"""
File Organization and Management Tools

This module provides intelligent file organization, duplicate detection,
search capabilities, and metadata indexing.
"""

import os
import hashlib
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio
from collections import defaultdict, Counter
import mimetypes

from .file_content_analyzer import FileContentAnalyzer, ContentAnalysis


class OrganizationStrategy(Enum):
    """File organization strategies"""
    BY_TYPE = "by_type"
    BY_DATE = "by_date"
    BY_PROJECT = "by_project"
    BY_SIZE = "by_size"
    BY_CONTENT = "by_content"
    CUSTOM = "custom"


class DuplicateDetectionMethod(Enum):
    """Methods for duplicate detection"""
    HASH = "hash"
    NAME = "name"
    CONTENT = "content"
    SIZE_AND_NAME = "size_and_name"


@dataclass
class FileMetadata:
    """Extended file metadata"""
    path: str
    name: str
    size: int
    created_at: datetime
    modified_at: datetime
    accessed_at: datetime
    file_type: str
    mime_type: Optional[str]
    hash_md5: Optional[str] = None
    hash_sha256: Optional[str] = None
    content_hash: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    description: Optional[str] = None
    project: Optional[str] = None
    category: Optional[str] = None
    importance: int = 0  # 0-10 scale
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    indexed_at: datetime = field(default_factory=datetime.now)


@dataclass
class DuplicateGroup:
    """Group of duplicate files"""
    files: List[FileMetadata]
    detection_method: DuplicateDetectionMethod
    confidence: float
    total_size: int
    potential_savings: int
    recommended_action: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrganizationPlan:
    """File organization plan"""
    strategy: OrganizationStrategy
    moves: List[Dict[str, str]]  # source -> destination mappings
    creates: List[str]  # directories to create
    estimated_time: float
    benefits: List[str]
    warnings: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileIndex:
    """File index for fast searching"""
    files: Dict[str, FileMetadata] = field(default_factory=dict)
    content_index: Dict[str, Set[str]] = field(default_factory=dict)  # word -> file paths
    tag_index: Dict[str, Set[str]] = field(default_factory=dict)  # tag -> file paths
    type_index: Dict[str, Set[str]] = field(default_factory=dict)  # type -> file paths
    size_index: Dict[str, Set[str]] = field(default_factory=dict)  # size range -> file paths
    date_index: Dict[str, Set[str]] = field(default_factory=dict)  # date -> file paths
    last_updated: datetime = field(default_factory=datetime.now)


class FileOrganizationManager:
    """Manages file organization, duplicate detection, and indexing"""
    
    def __init__(self, content_analyzer: FileContentAnalyzer):
        self.content_analyzer = content_analyzer
        self.logger = logging.getLogger(__name__)
        self.file_index = FileIndex()
        
        # Organization rules
        self.type_categories = {
            'documents': {'.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt'},
            'images': {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'},
            'videos': {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'},
            'audio': {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma'},
            'archives': {'.zip', '.rar', '.7z', '.tar', '.gz', '.bz2'},
            'code': {'.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.h', '.php'},
            'data': {'.json', '.xml', '.csv', '.xlsx', '.xls', '.db', '.sql'},
            'executables': {'.exe', '.msi', '.deb', '.rpm', '.dmg', '.app'}
        }
        
        # Size categories (in bytes)
        self.size_categories = {
            'tiny': (0, 1024),  # < 1KB
            'small': (1024, 1024 * 1024),  # 1KB - 1MB
            'medium': (1024 * 1024, 100 * 1024 * 1024),  # 1MB - 100MB
            'large': (100 * 1024 * 1024, 1024 * 1024 * 1024),  # 100MB - 1GB
            'huge': (1024 * 1024 * 1024, float('inf'))  # > 1GB
        }
    
    async def index_directory(self, directory: str, recursive: bool = True, update_existing: bool = False) -> int:
        """Index files in a directory for fast searching"""
        try:
            dir_path = Path(directory).resolve()
            if not dir_path.exists() or not dir_path.is_dir():
                return 0
            
            indexed_count = 0
            
            # Get all files
            if recursive:
                files = dir_path.rglob('*')
            else:
                files = dir_path.iterdir()
            
            for file_path in files:
                if not file_path.is_file():
                    continue
                
                file_str = str(file_path)
                
                # Skip if already indexed and not updating
                if not update_existing and file_str in self.file_index.files:
                    continue
                
                try:
                    metadata = await self._extract_file_metadata(file_path)
                    self.file_index.files[file_str] = metadata
                    
                    # Update indexes
                    await self._update_indexes(metadata)
                    indexed_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error indexing file {file_path}: {e}")
                    continue
            
            self.file_index.last_updated = datetime.now()
            self.logger.info(f"Indexed {indexed_count} files in {directory}")
            return indexed_count
            
        except Exception as e:
            self.logger.error(f"Error indexing directory {directory}: {e}")
            return 0
    
    async def _extract_file_metadata(self, file_path: Path) -> FileMetadata:
        """Extract comprehensive metadata from a file"""
        stat = file_path.stat()
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        # Calculate file hashes
        hash_md5 = await self._calculate_file_hash(file_path, 'md5')
        
        # Determine category
        category = self._categorize_file(file_path)
        
        metadata = FileMetadata(
            path=str(file_path),
            name=file_path.name,
            size=stat.st_size,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            accessed_at=datetime.fromtimestamp(stat.st_atime),
            file_type=file_path.suffix.lower(),
            mime_type=mime_type,
            hash_md5=hash_md5,
            category=category
        )
        
        return metadata
    
    async def _calculate_file_hash(self, file_path: Path, algorithm: str = 'md5') -> str:
        """Calculate file hash"""
        try:
            hash_obj = hashlib.new(algorithm)
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
            
        except Exception as e:
            self.logger.warning(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def _categorize_file(self, file_path: Path) -> str:
        """Categorize file based on extension"""
        extension = file_path.suffix.lower()
        
        for category, extensions in self.type_categories.items():
            if extension in extensions:
                return category
        
        return 'other'
    
    async def _update_indexes(self, metadata: FileMetadata):
        """Update search indexes with file metadata"""
        file_path = metadata.path
        
        # Type index
        if metadata.category not in self.file_index.type_index:
            self.file_index.type_index[metadata.category] = set()
        self.file_index.type_index[metadata.category].add(file_path)
        
        # Size index
        size_category = self._get_size_category(metadata.size)
        if size_category not in self.file_index.size_index:
            self.file_index.size_index[size_category] = set()
        self.file_index.size_index[size_category].add(file_path)
        
        # Date index (by year-month)
        date_key = metadata.modified_at.strftime('%Y-%m')
        if date_key not in self.file_index.date_index:
            self.file_index.date_index[date_key] = set()
        self.file_index.date_index[date_key].add(file_path)
        
        # Tag index
        for tag in metadata.tags:
            if tag not in self.file_index.tag_index:
                self.file_index.tag_index[tag] = set()
            self.file_index.tag_index[tag].add(file_path)
    
    def _get_size_category(self, size: int) -> str:
        """Get size category for a file size"""
        for category, (min_size, max_size) in self.size_categories.items():
            if min_size <= size < max_size:
                return category
        return 'unknown'
    
    async def find_duplicates(
        self, 
        directory: str, 
        method: DuplicateDetectionMethod = DuplicateDetectionMethod.HASH,
        min_size: int = 0
    ) -> List[DuplicateGroup]:
        """Find duplicate files in a directory"""
        try:
            # First, index the directory if not already done
            await self.index_directory(directory, recursive=True, update_existing=True)
            
            duplicates = []
            
            if method == DuplicateDetectionMethod.HASH:
                duplicates = await self._find_duplicates_by_hash(directory, min_size)
            elif method == DuplicateDetectionMethod.NAME:
                duplicates = await self._find_duplicates_by_name(directory, min_size)
            elif method == DuplicateDetectionMethod.SIZE_AND_NAME:
                duplicates = await self._find_duplicates_by_size_and_name(directory, min_size)
            
            return duplicates
            
        except Exception as e:
            self.logger.error(f"Error finding duplicates in {directory}: {e}")
            return []
    
    async def _find_duplicates_by_hash(self, directory: str, min_size: int) -> List[DuplicateGroup]:
        """Find duplicates by file hash"""
        hash_groups = defaultdict(list)
        
        # Group files by hash
        for file_path, metadata in self.file_index.files.items():
            if not file_path.startswith(directory):
                continue
            
            if metadata.size < min_size:
                continue
            
            if metadata.hash_md5:
                hash_groups[metadata.hash_md5].append(metadata)
        
        # Find groups with multiple files
        duplicate_groups = []
        for hash_value, files in hash_groups.items():
            if len(files) > 1:
                total_size = sum(f.size for f in files)
                potential_savings = total_size - files[0].size  # Keep one copy
                
                group = DuplicateGroup(
                    files=files,
                    detection_method=DuplicateDetectionMethod.HASH,
                    confidence=1.0,  # Hash matching is 100% confident
                    total_size=total_size,
                    potential_savings=potential_savings,
                    recommended_action="Keep newest, delete others"
                )
                duplicate_groups.append(group)
        
        return duplicate_groups
    
    async def _find_duplicates_by_name(self, directory: str, min_size: int) -> List[DuplicateGroup]:
        """Find duplicates by file name"""
        name_groups = defaultdict(list)
        
        # Group files by name
        for file_path, metadata in self.file_index.files.items():
            if not file_path.startswith(directory):
                continue
            
            if metadata.size < min_size:
                continue
            
            name_groups[metadata.name.lower()].append(metadata)
        
        # Find groups with multiple files
        duplicate_groups = []
        for name, files in name_groups.items():
            if len(files) > 1:
                total_size = sum(f.size for f in files)
                potential_savings = total_size - max(f.size for f in files)  # Keep largest
                
                group = DuplicateGroup(
                    files=files,
                    detection_method=DuplicateDetectionMethod.NAME,
                    confidence=0.7,  # Name matching is less confident
                    total_size=total_size,
                    potential_savings=potential_savings,
                    recommended_action="Review manually - same name but may be different content"
                )
                duplicate_groups.append(group)
        
        return duplicate_groups
    
    async def _find_duplicates_by_size_and_name(self, directory: str, min_size: int) -> List[DuplicateGroup]:
        """Find duplicates by size and name combination"""
        size_name_groups = defaultdict(list)
        
        # Group files by size and name
        for file_path, metadata in self.file_index.files.items():
            if not file_path.startswith(directory):
                continue
            
            if metadata.size < min_size:
                continue
            
            key = (metadata.size, metadata.name.lower())
            size_name_groups[key].append(metadata)
        
        # Find groups with multiple files
        duplicate_groups = []
        for (size, name), files in size_name_groups.items():
            if len(files) > 1:
                total_size = sum(f.size for f in files)
                potential_savings = total_size - files[0].size  # Keep one copy
                
                group = DuplicateGroup(
                    files=files,
                    detection_method=DuplicateDetectionMethod.SIZE_AND_NAME,
                    confidence=0.9,  # Size + name is quite confident
                    total_size=total_size,
                    potential_savings=potential_savings,
                    recommended_action="Likely duplicates - verify before deletion"
                )
                duplicate_groups.append(group)
        
        return duplicate_groups
    
    async def create_organization_plan(
        self, 
        directory: str, 
        strategy: OrganizationStrategy,
        custom_rules: Optional[Dict[str, Any]] = None
    ) -> OrganizationPlan:
        """Create a file organization plan"""
        try:
            # Index directory first
            await self.index_directory(directory, recursive=True, update_existing=True)
            
            moves = []
            creates = []
            warnings = []
            benefits = []
            
            if strategy == OrganizationStrategy.BY_TYPE:
                moves, creates = await self._plan_organization_by_type(directory)
                benefits.append("Files grouped by type for easy access")
                benefits.append("Reduced clutter in main directory")
            
            elif strategy == OrganizationStrategy.BY_DATE:
                moves, creates = await self._plan_organization_by_date(directory)
                benefits.append("Files organized chronologically")
                benefits.append("Easy to find files by time period")
            
            elif strategy == OrganizationStrategy.BY_SIZE:
                moves, creates = await self._plan_organization_by_size(directory)
                benefits.append("Large files separated from small ones")
                benefits.append("Better storage management")
            
            # Estimate time (rough calculation)
            estimated_time = len(moves) * 0.1  # 0.1 seconds per move
            
            plan = OrganizationPlan(
                strategy=strategy,
                moves=moves,
                creates=creates,
                estimated_time=estimated_time,
                benefits=benefits,
                warnings=warnings
            )
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Error creating organization plan for {directory}: {e}")
            return OrganizationPlan(
                strategy=strategy,
                moves=[],
                creates=[],
                estimated_time=0,
                benefits=[],
                warnings=[f"Error: {str(e)}"]
            )
    
    async def _plan_organization_by_type(self, directory: str) -> Tuple[List[Dict[str, str]], List[str]]:
        """Plan organization by file type"""
        moves = []
        creates = set()
        
        base_path = Path(directory)
        
        for file_path, metadata in self.file_index.files.items():
            if not file_path.startswith(directory):
                continue
            
            current_path = Path(file_path)
            category = metadata.category
            
            # Skip if already in correct category folder
            if current_path.parent.name.lower() == category:
                continue
            
            target_dir = base_path / category
            target_path = target_dir / current_path.name
            
            # Handle name conflicts
            counter = 1
            while str(target_path) in [move['destination'] for move in moves]:
                name_parts = current_path.stem, counter, current_path.suffix
                target_path = target_dir / f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
                counter += 1
            
            moves.append({
                'source': file_path,
                'destination': str(target_path),
                'reason': f'Organize by type: {category}'
            })
            
            creates.add(str(target_dir))
        
        return moves, list(creates)
    
    async def _plan_organization_by_date(self, directory: str) -> Tuple[List[Dict[str, str]], List[str]]:
        """Plan organization by date"""
        moves = []
        creates = set()
        
        base_path = Path(directory)
        
        for file_path, metadata in self.file_index.files.items():
            if not file_path.startswith(directory):
                continue
            
            current_path = Path(file_path)
            year = metadata.modified_at.strftime('%Y')
            month = metadata.modified_at.strftime('%m-%B')
            
            target_dir = base_path / year / month
            target_path = target_dir / current_path.name
            
            # Handle name conflicts
            counter = 1
            while str(target_path) in [move['destination'] for move in moves]:
                name_parts = current_path.stem, counter, current_path.suffix
                target_path = target_dir / f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
                counter += 1
            
            moves.append({
                'source': file_path,
                'destination': str(target_path),
                'reason': f'Organize by date: {year}/{month}'
            })
            
            creates.add(str(target_dir))
        
        return moves, list(creates)
    
    async def _plan_organization_by_size(self, directory: str) -> Tuple[List[Dict[str, str]], List[str]]:
        """Plan organization by file size"""
        moves = []
        creates = set()
        
        base_path = Path(directory)
        
        for file_path, metadata in self.file_index.files.items():
            if not file_path.startswith(directory):
                continue
            
            current_path = Path(file_path)
            size_category = self._get_size_category(metadata.size)
            
            target_dir = base_path / f"{size_category}_files"
            target_path = target_dir / current_path.name
            
            # Handle name conflicts
            counter = 1
            while str(target_path) in [move['destination'] for move in moves]:
                name_parts = current_path.stem, counter, current_path.suffix
                target_path = target_dir / f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
                counter += 1
            
            moves.append({
                'source': file_path,
                'destination': str(target_path),
                'reason': f'Organize by size: {size_category}'
            })
            
            creates.add(str(target_dir))
        
        return moves, list(creates)
    
    async def execute_organization_plan(self, plan: OrganizationPlan, dry_run: bool = True) -> Dict[str, Any]:
        """Execute a file organization plan"""
        results = {
            'success': True,
            'moves_completed': 0,
            'directories_created': 0,
            'errors': [],
            'warnings': []
        }
        
        try:
            if dry_run:
                self.logger.info("Dry run mode - no actual file operations performed")
                results['dry_run'] = True
                return results
            
            # Create directories first
            for dir_path in plan.creates:
                try:
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                    results['directories_created'] += 1
                except Exception as e:
                    results['errors'].append(f"Failed to create directory {dir_path}: {e}")
            
            # Execute moves
            for move in plan.moves:
                try:
                    source = Path(move['source'])
                    destination = Path(move['destination'])
                    
                    if not source.exists():
                        results['warnings'].append(f"Source file not found: {source}")
                        continue
                    
                    if destination.exists():
                        results['warnings'].append(f"Destination already exists: {destination}")
                        continue
                    
                    shutil.move(str(source), str(destination))
                    results['moves_completed'] += 1
                    
                except Exception as e:
                    results['errors'].append(f"Failed to move {move['source']}: {e}")
            
            return results
            
        except Exception as e:
            results['success'] = False
            results['errors'].append(f"Organization execution failed: {e}")
            return results
    
    async def search_files(
        self,
        query: str,
        directory: Optional[str] = None,
        file_types: Optional[List[str]] = None,
        size_range: Optional[Tuple[int, int]] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        tags: Optional[List[str]] = None
    ) -> List[FileMetadata]:
        """Search files using various criteria"""
        try:
            results = []
            
            # Search in all indexed files or specific directory
            search_files = self.file_index.files
            if directory:
                search_files = {
                    path: metadata for path, metadata in search_files.items()
                    if path.startswith(directory)
                }
            
            for file_path, metadata in search_files.items():
                # Apply filters
                if file_types and metadata.file_type not in file_types:
                    continue
                
                if size_range and not (size_range[0] <= metadata.size <= size_range[1]):
                    continue
                
                if date_range and not (date_range[0] <= metadata.modified_at <= date_range[1]):
                    continue
                
                if tags and not any(tag in metadata.tags for tag in tags):
                    continue
                
                # Text search in filename and metadata
                search_text = f"{metadata.name} {metadata.description or ''} {' '.join(metadata.tags)}"
                if query.lower() in search_text.lower():
                    results.append(metadata)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching files: {e}")
            return []