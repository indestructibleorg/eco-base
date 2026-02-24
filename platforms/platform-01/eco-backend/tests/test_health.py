# =============================================================================
# Health Check Tests
# =============================================================================

import pytest
from fastapi import status


class TestHealth:
    """健康檢查相關測試"""
    
    def test_root_endpoint(self, client):
        """測試根端點"""
        response = client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "service" in data
        assert "version" in data
    
    def test_health_endpoint(self, client):
        """測試健康檢查端點"""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data
        assert "version" in data
    
    def test_ready_endpoint(self, client):
        """測試就緒檢查端點"""
        response = client.get("/ready")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "ready"
    
    def test_api_health_endpoint(self, client):
        """測試API健康檢查端點"""
        response = client.get("/api/health/system")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["data"]["status"] == "healthy"
    
    def test_database_health_endpoint(self, client):
        """測試數據庫健康檢查端點"""
        response = client.get("/api/health/database")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "status" in data["data"]
    
    def test_providers_health_endpoint(self, client):
        """測試提供者健康檢查端點"""
        response = client.get("/api/providers/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert isinstance(data["data"], list)
    
    def test_metrics_endpoint(self, client):
        """測試指標端點"""
        response = client.get("/metrics")
        
        assert response.status_code == status.HTTP_200_OK
        # Prometheus 格式指標
        assert "# HELP" in response.text or "http_requests_total" in response.text
