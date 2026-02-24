# =============================================================================
# Provider Tests
# =============================================================================

import pytest
from fastapi import status


class TestProviders:
    """提供者相關測試"""
    
    def test_list_providers(self, client):
        """測試列出提供者"""
        response = client.get("/api/providers")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        # 響應結構可能是直接的列表或包含 items 的對象
        if isinstance(data["data"], dict):
            assert "items" in data["data"] or "provider_id" in data["data"]
        elif isinstance(data["data"], list):
            assert len(data["data"]) > 0
    
    def test_list_providers_with_domain_filter(self, client):
        """測試按領域過濾提供者"""
        response = client.get("/api/providers?domain=COGNITIVE_COMPUTE")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        
        # 獲取提供者列表
        providers = data["data"]
        if isinstance(providers, dict) and "items" in providers:
            providers = providers["items"]
        elif not isinstance(providers, list):
            providers = [providers]
        
        # 驗證所有返回的提供者都屬於認知計算領域
        for provider in providers:
            if isinstance(provider, dict):
                assert provider.get("domain") == "COGNITIVE_COMPUTE"
    
    def test_get_provider_detail(self, client):
        """測試獲取提供者詳情"""
        response = client.get("/api/providers/gamma-cognitive")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["data"]["provider_id"] == "gamma-cognitive"
        assert "capabilities" in data["data"]
    
    def test_get_nonexistent_provider(self, client):
        """測試獲取不存在的提供者"""
        response = client.get("/api/providers/nonexistent")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert data["success"] is False
    
    def test_check_providers_health(self, client):
        """測試檢查提供者健康狀態"""
        response = client.get("/api/providers/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        
        # 響應可能是列表或單個對象
        health_data = data["data"]
        if isinstance(health_data, list):
            # 驗證每個健康狀態都有必要的字段
            for health in health_data:
                assert "provider_id" in health or "healthy" in health
        elif isinstance(health_data, dict):
            assert "provider_id" in health_data or "healthy" in health_data
    
    def test_check_single_provider_health(self, client):
        """測試檢查單個提供者健康狀態"""
        response = client.get("/api/providers/gamma-cognitive/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["data"]["provider_id"] == "gamma-cognitive"
