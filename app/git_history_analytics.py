"""
Git History Analytics for Task-Based Development

This module provides comprehensive analytics and insights based on Git history
and task completion patterns.
"""

import logging
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
import re

from .task_git_models import GitCommit, TaskGitMetrics, TaskStatus
from .task_git_bridge import TaskGitBridge


logger = logging.getLogger(__name__)


@dataclass
class DeveloperMetrics:
    """Metrics for a specific developer"""
    name: str
    email: str
    total_commits: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0
    lines_added: int = 0
    lines_deleted: int = 0
    avg_commit_size: float = 0.0
    most_active_hours: List[int] = field(default_factory=list)
    preferred_file_types: Dict[str, int] = field(default_factory=dict)


@dataclass
class ProjectMetrics:
    """Overall project metrics"""
    total_commits: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0
    active_tasks: int = 0
    total_files_modified: int = 0
    total_lines_added: int = 0
    total_lines_deleted: int = 0
    avg_task_duration: float = 0.0  # hours
    completion_rate: float = 0.0
    most_active_files: Dict[str, int] = field(default_factory=dict)
    task_completion_trend: List[Tuple[str, int]] = field(default_factory=list)  # (date, count)


@dataclass
class TaskAnalytics:
    """Analytics for task patterns"""
    task_id: str
    complexity_score: float = 0.0
    estimated_vs_actual_duration: Tuple[float, float] = (0.0, 0.0)
    code_churn_ratio: float = 0.0  # lines_deleted / lines_added
    file_impact_score: float = 0.0
    collaboration_score: float = 0.0  # number of different authors
    quality_indicators: Dict[str, Any] = field(default_factory=dict)


class GitHistoryAnalytics:
    """Provides comprehensive Git history analytics for task-based development"""
    
    def __init__(self, task_git_bridge: TaskGitBridge, repo_path: str = "."):
        self.task_git_bridge = task_git_bridge
        self.repo_path = repo_path
        self._commit_cache: Dict[str, GitCommit] = {}
    
    async def get_project_metrics(self, days_back: int = 30) -> ProjectMetrics:
        """Get comprehensive project metrics"""
        try:
            since_date = datetime.now() - timedelta(days=days_back)
            
            # Get all commits since the specified date
            commits = await self._get_commits_since(since_date)
            
            # Get task mappings
            mappings = self.task_git_bridge.get_all_mappings()
            
            # Calculate metrics
            metrics = ProjectMetrics()
            metrics.total_commits = len(commits)
            metrics.total_tasks = len(mappings)
            metrics.completed_tasks = sum(1 for m in mappings.values() if m.status == TaskStatus.COMPLETED)
            metrics.active_tasks = sum(1 for m in mappings.values() if m.status == TaskStatus.IN_PROGRESS)
            
            # Calculate file and line metrics
            file_changes = defaultdict(int)
            for commit in commits:
                metrics.total_lines_added += await self._get_commit_lines_added(commit.hash)
                metrics.total_lines_deleted += await self._get_commit_lines_deleted(commit.hash)
                
                for file_path in commit.files_changed:
                    file_changes[file_path] += 1
            
            metrics.total_files_modified = len(file_changes)
            metrics.most_active_files = dict(sorted(file_changes.items(), key=lambda x: x[1], reverse=True)[:10])
            
            # Calculate completion rate
            if metrics.total_tasks > 0:
                metrics.completion_rate = metrics.completed_tasks / metrics.total_tasks
            
            # Calculate average task duration
            completed_mappings = [m for m in mappings.values() if m.status == TaskStatus.COMPLETED and m.completed_at]
            if completed_mappings:
                total_duration = sum(
                    (m.completed_at - m.created_at).total_seconds() / 3600
                    for m in completed_mappings
                )
                metrics.avg_task_duration = total_duration / len(completed_mappings)
            
            # Calculate completion trend
            metrics.task_completion_trend = await self._calculate_completion_trend(days_back)
            
            logger.info(f"Generated project metrics for {days_back} days")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get project metrics: {e}")
            return ProjectMetrics()
    
    async def get_developer_metrics(self, days_back: int = 30) -> Dict[str, DeveloperMetrics]:
        """Get metrics for all developers"""
        try:
            since_date = datetime.now() - timedelta(days=days_back)
            commits = await self._get_commits_since(since_date)
            
            developer_metrics = defaultdict(lambda: DeveloperMetrics(name="", email=""))
            
            for commit in commits:
                author_key = f"{commit.author}"
                metrics = developer_metrics[author_key]
                
                if not metrics.name:
                    # Parse author name and email
                    if '<' in commit.author and '>' in commit.author:
                        name_part, email_part = commit.author.rsplit('<', 1)
                        metrics.name = name_part.strip()
                        metrics.email = email_part.rstrip('>').strip()
                    else:
                        metrics.name = commit.author
                        metrics.email = ""
                
                metrics.total_commits += 1
                metrics.lines_added += await self._get_commit_lines_added(commit.hash)
                metrics.lines_deleted += await self._get_commit_lines_deleted(commit.hash)
                
                # Track file types
                for file_path in commit.files_changed:
                    ext = self._get_file_extension(file_path)
                    if ext:
                        metrics.preferred_file_types[ext] = metrics.preferred_file_types.get(ext, 0) + 1
                
                # Track commit hours
                hour = commit.timestamp.hour
                if hour not in metrics.most_active_hours:
                    metrics.most_active_hours.append(hour)
            
            # Calculate average commit size and task counts
            mappings = self.task_git_bridge.get_all_mappings()
            for author_key, metrics in developer_metrics.items():
                if metrics.total_commits > 0:
                    metrics.avg_commit_size = (metrics.lines_added + metrics.lines_deleted) / metrics.total_commits
                
                # Count tasks by author (approximate based on commits)
                author_tasks = set()
                for mapping in mappings.values():
                    for commit_hash in mapping.commits:
                        commit = await self._get_commit_details(commit_hash)
                        if commit and commit.author == author_key:
                            author_tasks.add(mapping.task_id)
                
                metrics.total_tasks = len(author_tasks)
                metrics.completed_tasks = sum(
                    1 for mapping in mappings.values()
                    if mapping.task_id in author_tasks and mapping.status == TaskStatus.COMPLETED
                )
                
                # Sort most active hours
                metrics.most_active_hours = sorted(set(metrics.most_active_hours))
            
            logger.info(f"Generated metrics for {len(developer_metrics)} developers")
            return dict(developer_metrics)
            
        except Exception as e:
            logger.error(f"Failed to get developer metrics: {e}")
            return {}
    
    async def analyze_task_complexity(self, task_id: str) -> TaskAnalytics:
        """Analyze complexity and quality metrics for a specific task"""
        try:
            mapping = self.task_git_bridge.get_task_mapping(task_id)
            if not mapping:
                raise ValueError(f"Task {task_id} not found")
            
            analytics = TaskAnalytics(task_id=task_id)
            
            # Get all commits for this task
            commits = []
            for commit_hash in mapping.commits:
                commit = await self._get_commit_details(commit_hash)
                if commit:
                    commits.append(commit)
            
            if not commits:
                return analytics
            
            # Calculate complexity score based on multiple factors
            complexity_factors = []
            
            # Factor 1: Number of files modified
            all_files = set()
            for commit in commits:
                all_files.update(commit.files_changed)
            file_count_score = min(len(all_files) / 10.0, 1.0)  # Normalize to 0-1
            complexity_factors.append(file_count_score)
            
            # Factor 2: Total lines changed
            total_lines_added = sum(await self._get_commit_lines_added(c.hash) for c in commits)
            total_lines_deleted = sum(await self._get_commit_lines_deleted(c.hash) for c in commits)
            total_lines_changed = total_lines_added + total_lines_deleted
            lines_score = min(total_lines_changed / 1000.0, 1.0)  # Normalize to 0-1
            complexity_factors.append(lines_score)
            
            # Factor 3: Number of commits (more commits might indicate complexity)
            commit_count_score = min(len(commits) / 20.0, 1.0)  # Normalize to 0-1
            complexity_factors.append(commit_count_score)
            
            # Factor 4: Duration (longer tasks might be more complex)
            if mapping.created_at and mapping.completed_at:
                duration_hours = (mapping.completed_at - mapping.created_at).total_seconds() / 3600
                duration_score = min(duration_hours / 168.0, 1.0)  # Normalize to 0-1 (1 week max)
                complexity_factors.append(duration_score)
            
            # Calculate overall complexity score
            analytics.complexity_score = sum(complexity_factors) / len(complexity_factors)
            
            # Calculate code churn ratio
            if total_lines_added > 0:
                analytics.code_churn_ratio = total_lines_deleted / total_lines_added
            
            # Calculate file impact score (based on file types and importance)
            file_impact_scores = []
            for file_path in all_files:
                impact = self._calculate_file_impact(file_path)
                file_impact_scores.append(impact)
            
            if file_impact_scores:
                analytics.file_impact_score = sum(file_impact_scores) / len(file_impact_scores)
            
            # Calculate collaboration score
            unique_authors = set(commit.author for commit in commits)
            analytics.collaboration_score = len(unique_authors)
            
            # Calculate quality indicators
            analytics.quality_indicators = {
                "commit_message_quality": self._assess_commit_message_quality(commits),
                "test_coverage_impact": await self._assess_test_coverage_impact(all_files),
                "documentation_updates": self._count_documentation_updates(all_files),
                "refactoring_ratio": self._calculate_refactoring_ratio(commits)
            }
            
            logger.info(f"Analyzed complexity for task {task_id}")
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to analyze task complexity for {task_id}: {e}")
            return TaskAnalytics(task_id=task_id)
    
    async def get_task_timeline(self, task_id: str) -> List[Dict[str, Any]]:
        """Get detailed timeline for a specific task"""
        try:
            mapping = self.task_git_bridge.get_task_mapping(task_id)
            if not mapping:
                return []
            
            timeline = []
            
            # Add task start event
            timeline.append({
                "timestamp": mapping.created_at.isoformat(),
                "event_type": "task_started",
                "description": f"Task {task_id} started",
                "branch": mapping.branch_name
            })
            
            # Add commit events
            for commit_hash in mapping.commits:
                commit = await self._get_commit_details(commit_hash)
                if commit:
                    timeline.append({
                        "timestamp": commit.timestamp.isoformat(),
                        "event_type": "commit",
                        "description": commit.message,
                        "commit_hash": commit.hash,
                        "files_changed": len(commit.files_changed),
                        "author": commit.author
                    })
            
            # Add completion event
            if mapping.completed_at:
                timeline.append({
                    "timestamp": mapping.completed_at.isoformat(),
                    "event_type": "task_completed",
                    "description": f"Task {task_id} completed",
                    "status": mapping.status.value
                })
            
            # Sort by timestamp
            timeline.sort(key=lambda x: x["timestamp"])
            
            return timeline
            
        except Exception as e:
            logger.error(f"Failed to get task timeline for {task_id}: {e}")
            return []
    
    async def generate_productivity_report(self, days_back: int = 30) -> Dict[str, Any]:
        """Generate comprehensive productivity report"""
        try:
            project_metrics = await self.get_project_metrics(days_back)
            developer_metrics = await self.get_developer_metrics(days_back)
            
            # Calculate productivity indicators
            productivity_score = 0.0
            if project_metrics.total_tasks > 0:
                productivity_score = (
                    project_metrics.completion_rate * 0.4 +
                    min(project_metrics.total_commits / (days_back * 2), 1.0) * 0.3 +
                    min(project_metrics.avg_task_duration / 24.0, 1.0) * 0.3  # Prefer shorter tasks
                )
            
            # Identify top performers
            top_developers = sorted(
                developer_metrics.values(),
                key=lambda d: d.completed_tasks + (d.total_commits * 0.1),
                reverse=True
            )[:5]
            
            # Identify bottlenecks
            bottlenecks = []
            if project_metrics.avg_task_duration > 48:  # More than 2 days
                bottlenecks.append("Long average task duration")
            
            if project_metrics.completion_rate < 0.7:
                bottlenecks.append("Low task completion rate")
            
            active_to_total_ratio = project_metrics.active_tasks / max(project_metrics.total_tasks, 1)
            if active_to_total_ratio > 0.5:
                bottlenecks.append("Too many active tasks (potential context switching)")
            
            report = {
                "period_days": days_back,
                "generated_at": datetime.now().isoformat(),
                "productivity_score": productivity_score,
                "project_metrics": project_metrics.__dict__,
                "developer_count": len(developer_metrics),
                "top_developers": [
                    {
                        "name": dev.name,
                        "completed_tasks": dev.completed_tasks,
                        "total_commits": dev.total_commits,
                        "avg_commit_size": dev.avg_commit_size
                    }
                    for dev in top_developers
                ],
                "bottlenecks": bottlenecks,
                "recommendations": self._generate_recommendations(project_metrics, developer_metrics)
            }
            
            logger.info(f"Generated productivity report for {days_back} days")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate productivity report: {e}")
            return {}
    
    # Private helper methods
    
    async def _get_commits_since(self, since_date: datetime) -> List[GitCommit]:
        """Get all commits since a specific date"""
        try:
            since_str = since_date.strftime("%Y-%m-%d")
            result = subprocess.run([
                'git', 'log', '--since', since_str, '--format=%H|%an|%ai|%s', '--name-only'
            ], capture_output=True, text=True, check=True, cwd=self.repo_path)
            
            commits = []
            current_commit = None
            
            for line in result.stdout.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if '|' in line:
                    # This is a commit header
                    if current_commit:
                        commits.append(current_commit)
                    
                    parts = line.split('|', 3)
                    if len(parts) >= 4:
                        current_commit = GitCommit(
                            hash=parts[0],
                            author=parts[1],
                            timestamp=datetime.fromisoformat(parts[2].replace(' ', 'T', 1)),
                            message=parts[3],
                            files_changed=[]
                        )
                else:
                    # This is a file name
                    if current_commit:
                        current_commit.files_changed.append(line)
            
            if current_commit:
                commits.append(current_commit)
            
            return commits
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get commits since {since_date}: {e}")
            return []
    
    async def _get_commit_details(self, commit_hash: str) -> Optional[GitCommit]:
        """Get detailed information about a specific commit"""
        if commit_hash in self._commit_cache:
            return self._commit_cache[commit_hash]
        
        try:
            result = subprocess.run([
                'git', 'show', '--format=%H|%an|%ai|%s', '--name-only', commit_hash
            ], capture_output=True, text=True, check=True, cwd=self.repo_path)
            
            lines = result.stdout.strip().split('\n')
            if not lines:
                return None
            
            # Parse header
            header_parts = lines[0].split('|', 3)
            if len(header_parts) < 4:
                return None
            
            # Get changed files
            files_changed = [line for line in lines[1:] if line.strip()]
            
            commit = GitCommit(
                hash=header_parts[0],
                author=header_parts[1],
                timestamp=datetime.fromisoformat(header_parts[2].replace(' ', 'T', 1)),
                message=header_parts[3],
                files_changed=files_changed
            )
            
            self._commit_cache[commit_hash] = commit
            return commit
            
        except subprocess.CalledProcessError:
            return None
    
    async def _get_commit_lines_added(self, commit_hash: str) -> int:
        """Get number of lines added in a commit"""
        try:
            result = subprocess.run([
                'git', 'show', '--stat', '--format=', commit_hash
            ], capture_output=True, text=True, check=True, cwd=self.repo_path)
            
            for line in result.stdout.split('\n'):
                if 'insertion' in line:
                    match = re.search(r'(\d+) insertion', line)
                    if match:
                        return int(match.group(1))
            
            return 0
            
        except subprocess.CalledProcessError:
            return 0
    
    async def _get_commit_lines_deleted(self, commit_hash: str) -> int:
        """Get number of lines deleted in a commit"""
        try:
            result = subprocess.run([
                'git', 'show', '--stat', '--format=', commit_hash
            ], capture_output=True, text=True, check=True, cwd=self.repo_path)
            
            for line in result.stdout.split('\n'):
                if 'deletion' in line:
                    match = re.search(r'(\d+) deletion', line)
                    if match:
                        return int(match.group(1))
            
            return 0
            
        except subprocess.CalledProcessError:
            return 0
    
    async def _calculate_completion_trend(self, days_back: int) -> List[Tuple[str, int]]:
        """Calculate task completion trend over time"""
        try:
            mappings = self.task_git_bridge.get_all_mappings()
            completed_tasks = [
                m for m in mappings.values()
                if m.status == TaskStatus.COMPLETED and m.completed_at
            ]
            
            # Group by date
            completion_by_date = defaultdict(int)
            for mapping in completed_tasks:
                date_str = mapping.completed_at.strftime("%Y-%m-%d")
                completion_by_date[date_str] += 1
            
            # Create trend data for the last N days
            trend = []
            for i in range(days_back):
                date = datetime.now() - timedelta(days=i)
                date_str = date.strftime("%Y-%m-%d")
                count = completion_by_date.get(date_str, 0)
                trend.append((date_str, count))
            
            trend.reverse()  # Chronological order
            return trend
            
        except Exception as e:
            logger.error(f"Failed to calculate completion trend: {e}")
            return []
    
    def _get_file_extension(self, file_path: str) -> str:
        """Get file extension from path"""
        if '.' in file_path:
            return file_path.split('.')[-1].lower()
        return ""
    
    def _calculate_file_impact(self, file_path: str) -> float:
        """Calculate impact score for a file based on its type and location"""
        impact_scores = {
            'py': 1.0,
            'js': 1.0,
            'ts': 1.0,
            'java': 1.0,
            'cpp': 1.0,
            'c': 1.0,
            'go': 1.0,
            'rs': 1.0,
            'md': 0.3,
            'txt': 0.2,
            'json': 0.5,
            'yaml': 0.5,
            'yml': 0.5,
            'xml': 0.4,
            'html': 0.6,
            'css': 0.4,
            'sql': 0.7
        }
        
        ext = self._get_file_extension(file_path)
        base_score = impact_scores.get(ext, 0.5)
        
        # Adjust based on file location
        if 'test' in file_path.lower():
            base_score *= 0.8  # Tests are important but less impactful
        elif 'main' in file_path.lower() or 'index' in file_path.lower():
            base_score *= 1.2  # Main files are more impactful
        elif 'config' in file_path.lower():
            base_score *= 1.1  # Config files are important
        
        return min(base_score, 1.0)
    
    def _assess_commit_message_quality(self, commits: List[GitCommit]) -> float:
        """Assess the quality of commit messages"""
        if not commits:
            return 0.0
        
        quality_scores = []
        for commit in commits:
            score = 0.0
            message = commit.message.strip()
            
            # Length check
            if 10 <= len(message) <= 100:
                score += 0.3
            
            # Starts with capital letter
            if message and message[0].isupper():
                score += 0.2
            
            # Contains task reference
            if re.search(r'[Tt]ask\s+\d+', message):
                score += 0.3
            
            # Not too generic
            generic_words = ['fix', 'update', 'change', 'modify']
            if not any(word in message.lower() for word in generic_words):
                score += 0.2
            
            quality_scores.append(score)
        
        return sum(quality_scores) / len(quality_scores)
    
    async def _assess_test_coverage_impact(self, files: set) -> float:
        """Assess impact on test coverage"""
        test_files = sum(1 for f in files if 'test' in f.lower())
        total_files = len(files)
        
        if total_files == 0:
            return 0.0
        
        return test_files / total_files
    
    def _count_documentation_updates(self, files: set) -> int:
        """Count documentation file updates"""
        doc_extensions = {'.md', '.rst', '.txt', '.doc', '.docx'}
        return sum(1 for f in files if any(f.lower().endswith(ext) for ext in doc_extensions))
    
    def _calculate_refactoring_ratio(self, commits: List[GitCommit]) -> float:
        """Calculate ratio of refactoring commits"""
        if not commits:
            return 0.0
        
        refactoring_keywords = ['refactor', 'cleanup', 'reorganize', 'restructure', 'optimize']
        refactoring_commits = sum(
            1 for commit in commits
            if any(keyword in commit.message.lower() for keyword in refactoring_keywords)
        )
        
        return refactoring_commits / len(commits)
    
    def _generate_recommendations(self, project_metrics: ProjectMetrics, 
                                developer_metrics: Dict[str, DeveloperMetrics]) -> List[str]:
        """Generate recommendations based on metrics"""
        recommendations = []
        
        if project_metrics.completion_rate < 0.7:
            recommendations.append("Consider breaking down large tasks into smaller, more manageable pieces")
        
        if project_metrics.avg_task_duration > 48:
            recommendations.append("Tasks are taking longer than expected - review task estimation process")
        
        if project_metrics.active_tasks > project_metrics.completed_tasks:
            recommendations.append("Too many active tasks - focus on completing existing tasks before starting new ones")
        
        # Check developer distribution
        if len(developer_metrics) > 1:
            commit_counts = [dev.total_commits for dev in developer_metrics.values()]
            if max(commit_counts) > 3 * min(commit_counts):
                recommendations.append("Uneven workload distribution - consider better task allocation")
        
        if not recommendations:
            recommendations.append("Project metrics look healthy - keep up the good work!")
        
        return recommendations