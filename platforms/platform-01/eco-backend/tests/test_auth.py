# =============================================================================
# Authentication Tests
# =============================================================================

import pytest
from fastapi import status


class TestAuth:
    """認證相關測試"""
    
    def test_register_success(self, client, test_user_data):
        """測試成功註冊"""
        response = client.post("/api/auth/register", json=test_user_data)
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["success"] is True
        assert data["data"]["email"] == test_user_data["email"]
        assert data["data"]["username"] == test_user_data["username"]
        assert "hashed_password" not in data["data"]
    
    def test_register_duplicate_email(self, client, test_user_data):
        """測試重複郵箱註冊"""
        # 第一次註冊
        response1 = client.post("/api/auth/register", json=test_user_data)
        assert response1.status_code == status.HTTP_201_CREATED
        
        # 第二次註冊應該失敗
        response2 = client.post("/api/auth/register", json=test_user_data)
        
        # 可能返回 400 或 422，取決於錯誤處理
        assert response2.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_422_UNPROCESSABLE_ENTITY]
        data = response2.json()
        assert data["success"] is False
    
    def test_login_success(self, client, test_user_data, test_login_data):
        """測試成功登錄"""
        # 先註冊
        response1 = client.post("/api/auth/register", json=test_user_data)
        assert response1.status_code == status.HTTP_201_CREATED
        
        # 登錄
        response = client.post("/api/auth/login", json=test_login_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "access_token" in data["data"]
        assert "refresh_token" in data["data"]
        assert data["data"]["token_type"] == "bearer"
    
    def test_login_invalid_credentials(self, client, test_user_data):
        """測試無效憑證登錄"""
        response = client.post("/api/auth/login", json={
            "username": "wrong@example.com",
            "password": "wrongpassword",
        })
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        data = response.json()
        assert data["success"] is False
    
    def test_get_me_unauthorized(self, client):
        """測試未授權獲取用戶信息"""
        response = client.get("/api/users/me")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_get_me_authorized(self, client, test_user_data, test_login_data):
        """測試授權獲取用戶信息"""
        # 註冊並登錄
        response1 = client.post("/api/auth/register", json=test_user_data)
        assert response1.status_code == status.HTTP_201_CREATED
        
        login_response = client.post("/api/auth/login", json=test_login_data)
        assert login_response.status_code == status.HTTP_200_OK
        token = login_response.json()["data"]["access_token"]
        
        # 獲取用戶信息
        response = client.get(
            "/api/users/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["data"]["email"] == test_user_data["email"]
