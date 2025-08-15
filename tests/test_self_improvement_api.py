# tests/test_self_improvement_api.py
"""
Integration tests for self-improvement API endpoints
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import json
from datetime import datetime

from app.main import app
from app.self_improvement_engine import (
    SelfImprovementEngine,
    ImprovementCycle,
    ImprovementCycleStatus,
    SafetyLevel
)
from app.improvement_engine import Improvement, ImprovementType, ImprovementPriority, RiskLevel


class TestSelfImprovementAPI:
    """Test cases for self-improvement API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_engine(self):
        """Mock self-improvement engine"""
        with patch('app.main.get_self_improvement_engine') as mock_get_engine:
            engine = Mock(spec=SelfImprovementEngine)
            mock_get_engine.return_value = engine
            
            # Configure default mock responses
            engine.get_current_status.return_value = {
                "is_running": False,
                "current_cycle": None,
                "safety_level": "conservative",
                "scheduler_running": False,
                "last_performance_check": datetime.utcnow().isoformat(),
                "total_cycles": 0,
                "successful_cycles": 0
            }
            
            engine.get_cycle_history.return_value = []
            engine.get_cycle_details.return_value = None
            engine.trigger_improvement_cycle = AsyncMock(return_value="cycle_123")
            engine.start_scheduler = AsyncMock()
            engine.stop_scheduler = AsyncMock()
            engine.emergency_stop = AsyncMock(return_value=True)
            engine.update_safety_level = Mock()
            engine.safety_level = SafetyLevel.CONSERVATIVE
            
            # Mock improvement engine
            engine.improvement_engine = Mock()
            engine.improvement_engine.analyze_and_suggest_improvements.return_value = [
                Improvement(
                    id="test_improvement_1",
                    type=ImprovementType.PERFORMANCE,
                    priority=ImprovementPriority.HIGH,
                    risk_level=RiskLevel.LOW,
                    title="Test Improvement",
                    description="Test improvement description",
                    affected_files=["test_file.py"],
                    impact_score=8.5,
                    confidence_score=9.0
                )
            ]
            
            engine._filter_improvements_by_safety = Mock(side_effect=lambda x: x)
            engine._select_improvements = Mock(side_effect=lambda x, max_count: x[:max_count])
            
            yield engine
    
    def test_trigger_improvement_cycle(self, client, mock_engine):
        """Test triggering an improvement cycle"""
        response = client.post("/improvement/trigger", json={"trigger": "manual"})
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["cycle_id"] == "cycle_123"
        assert data["trigger"] == "manual"
        assert data["status"] == "started"
        assert "timestamp" in data
        
        mock_engine.trigger_improvement_cycle.assert_called_once_with("manual")
    
    def test_trigger_improvement_cycle_conflict(self, client, mock_engine):
        """Test triggering cycle when one is already running"""
        mock_engine.trigger_improvement_cycle.side_effect = ValueError("Another improvement cycle is already running")
        
        response = client.post("/improvement/trigger", json={"trigger": "manual"})
        
        assert response.status_code == 409
        assert "Another improvement cycle is already running" in response.json()["detail"]
    
    def test_get_improvement_status(self, client, mock_engine):
        """Test getting improvement engine status"""
        response = client.get("/improvement/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert data["status"]["is_running"] is False
        assert data["status"]["safety_level"] == "conservative"
        assert "timestamp" in data
        
        mock_engine.get_current_status.assert_called_once()
    
    def test_get_improvement_cycles(self, client, mock_engine):
        """Test getting improvement cycle history"""
        mock_cycles = [
            {
                "id": "cycle_1",
                "status": "completed",
                "started_at": datetime.utcnow().isoformat(),
                "success_rate": 100.0
            },
            {
                "id": "cycle_2",
                "status": "failed",
                "started_at": datetime.utcnow().isoformat(),
                "success_rate": 0.0
            }
        ]
        mock_engine.get_cycle_history.return_value = mock_cycles
        
        response = client.get("/improvement/cycles?limit=5")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["cycles"] == mock_cycles
        assert data["count"] == 2
        assert data["limit"] == 5
        assert "timestamp" in data
        
        mock_engine.get_cycle_history.assert_called_once_with(5)
    
    def test_get_improvement_cycle_details(self, client, mock_engine):
        """Test getting detailed cycle information"""
        mock_details = {
            "id": "cycle_123",
            "status": "completed",
            "started_at": datetime.utcnow().isoformat(),
            "improvements": [],
            "test_results": {},
            "modification_plans": []
        }
        mock_engine.get_cycle_details.return_value = mock_details
        
        response = client.get("/improvement/cycles/cycle_123")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["cycle"] == mock_details
        assert "timestamp" in data
        
        mock_engine.get_cycle_details.assert_called_once_with("cycle_123")
    
    def test_get_improvement_cycle_details_not_found(self, client, mock_engine):
        """Test getting details for non-existent cycle"""
        mock_engine.get_cycle_details.return_value = None
        
        response = client.get("/improvement/cycles/nonexistent")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_update_safety_level(self, client, mock_engine):
        """Test updating safety level"""
        response = client.post("/improvement/safety-level", json={"safety_level": "moderate"})
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["safety_level"] == "moderate"
        assert "updated" in data["message"]
        assert "timestamp" in data
        
        mock_engine.update_safety_level.assert_called_once()
    
    def test_update_safety_level_invalid(self, client, mock_engine):
        """Test updating safety level with invalid value"""
        response = client.post("/improvement/safety-level", json={"safety_level": "invalid"})
        
        assert response.status_code == 400
        assert "Invalid safety level" in response.json()["detail"]
    
    def test_start_improvement_scheduler(self, client, mock_engine):
        """Test starting the improvement scheduler"""
        response = client.post("/improvement/scheduler/start")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "running"
        assert "started" in data["message"]
        assert "timestamp" in data
        
        mock_engine.start_scheduler.assert_called_once()
    
    def test_stop_improvement_scheduler(self, client, mock_engine):
        """Test stopping the improvement scheduler"""
        response = client.post("/improvement/scheduler/stop")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "stopped"
        assert "stopped" in data["message"]
        assert "timestamp" in data
        
        mock_engine.stop_scheduler.assert_called_once()
    
    def test_emergency_stop_improvements(self, client, mock_engine):
        """Test emergency stop functionality"""
        response = client.post("/improvement/emergency-stop")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "stopped"
        assert data["rollback_performed"] is True
        assert "Emergency stop executed" in data["message"]
        assert "timestamp" in data
        
        mock_engine.emergency_stop.assert_called_once()
    
    def test_emergency_stop_failure(self, client, mock_engine):
        """Test emergency stop failure"""
        mock_engine.emergency_stop.return_value = False
        
        response = client.post("/improvement/emergency-stop")
        
        assert response.status_code == 500
        assert "Emergency stop failed" in response.json()["detail"]
    
    def test_get_improvement_suggestions(self, client, mock_engine):
        """Test getting improvement suggestions"""
        response = client.get("/improvement/suggestions?limit=5")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "suggestions" in data
        assert len(data["suggestions"]) == 1
        assert data["suggestions"][0]["id"] == "test_improvement_1"
        assert data["safety_level"] == "conservative"
        assert "timestamp" in data
        
        mock_engine.improvement_engine.analyze_and_suggest_improvements.assert_called_once()
        mock_engine._filter_improvements_by_safety.assert_called_once()
        mock_engine._select_improvements.assert_called_once()
    
    def test_approve_improvement(self, client, mock_engine):
        """Test approving an improvement"""
        response = client.post("/improvement/approve", json={
            "improvement_id": "test_improvement_1",
            "apply_immediately": True
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["improvement_id"] == "test_improvement_1"
        assert data["status"] == "approved"
        assert data["apply_immediately"] is True
        assert "timestamp" in data
    
    def test_get_rollback_points(self, client, mock_engine):
        """Test getting rollback points"""
        mock_cycles = [
            {
                "id": "cycle_1",
                "status": "completed",
                "started_at": datetime.utcnow().isoformat(),
                "rollback_points": ["commit_abc123"],
                "applied_improvements": ["improvement_1"]
            }
        ]
        mock_engine.get_cycle_history.return_value = mock_cycles
        
        response = client.get("/improvement/rollback-points")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "rollback_points" in data
        assert len(data["rollback_points"]) == 1
        assert data["rollback_points"][0]["cycle_id"] == "cycle_1"
        assert data["rollback_points"][0]["rollback_point"] == "commit_abc123"
        assert "timestamp" in data
    
    def test_rollback_improvement_cycle(self, client, mock_engine):
        """Test rolling back an improvement cycle"""
        mock_details = {
            "id": "cycle_123",
            "status": "completed",
            "started_at": datetime.utcnow().isoformat()
        }
        mock_engine.get_cycle_details.return_value = mock_details
        
        response = client.post("/improvement/rollback", json={"cycle_id": "cycle_123"})
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["cycle_id"] == "cycle_123"
        assert data["status"] == "rollback_initiated"
        assert "timestamp" in data
        
        mock_engine.get_cycle_details.assert_called_once_with("cycle_123")
    
    def test_rollback_improvement_cycle_not_found(self, client, mock_engine):
        """Test rolling back non-existent cycle"""
        mock_engine.get_cycle_details.return_value = None
        
        response = client.post("/improvement/rollback", json={"cycle_id": "nonexistent"})
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_rollback_improvement_cycle_invalid_status(self, client, mock_engine):
        """Test rolling back cycle with invalid status"""
        mock_details = {
            "id": "cycle_123",
            "status": "running",
            "started_at": datetime.utcnow().isoformat()
        }
        mock_engine.get_cycle_details.return_value = mock_details
        
        response = client.post("/improvement/rollback", json={"cycle_id": "cycle_123"})
        
        assert response.status_code == 400
        assert "Cannot rollback cycle in status" in response.json()["detail"]
    
    def test_api_error_handling(self, client):
        """Test API error handling when engine is not available"""
        with patch('app.main.get_self_improvement_engine', side_effect=Exception("Engine error")):
            response = client.get("/improvement/status")
            
            assert response.status_code == 500
            assert "Engine error" in response.json()["detail"]
    
    def test_api_validation(self, client, mock_engine):
        """Test API request validation"""
        # Test invalid JSON
        response = client.post("/improvement/trigger", json={"invalid_field": "value"})
        assert response.status_code == 200  # Should use default trigger value
        
        # Test missing required fields
        response = client.post("/improvement/rollback", json={})
        assert response.status_code == 422  # Validation error
        
        # Test invalid parameter types
        response = client.get("/improvement/cycles?limit=invalid")
        assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__])