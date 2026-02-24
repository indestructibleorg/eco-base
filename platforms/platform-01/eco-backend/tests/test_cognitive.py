# =============================================================================
# Cognitive Computing Tests
# =============================================================================

import pytest
from fastapi import status


class TestCognitive:
    """認知計算相關測試"""
    
    def test_generate_text_success(self, client):
        """測試文本生成"""
        response = client.post("/api/cognitive/generate", json={
            "prompt": "Hello, world!",
            "provider": "gamma-cognitive",
        })
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "content" in data["data"]
        assert data["data"]["provider"] == "gamma-cognitive"
    
    def test_generate_text_with_parameters(self, client):
        """測試帶參數的文本生成"""
        response = client.post("/api/cognitive/generate", json={
            "prompt": "Explain Python",
            "parameters": {
                "temperature": 0.7,
                "max_tokens": 100,
            },
            "provider": "gamma-cognitive",
        })
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
    
    def test_generate_text_stream(self, client):
        """測試流式文本生成"""
        response = client.post("/api/cognitive/generate/stream", json={
            "prompt": "Hello",
            "provider": "gamma-cognitive",
        })
        
        assert response.status_code == status.HTTP_200_OK
        # 流式響應可能有多種 content-type 格式
        content_type = response.headers.get("content-type", "")
        assert "text/event-stream" in content_type or "application/json" in content_type
    
    def test_embed_texts(self, client):
        """測試文本嵌入"""
        response = client.post("/api/cognitive/embed", json={
            "texts": ["Hello", "World"],
            "provider": "gamma-cognitive",
        })
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "embeddings" in data["data"]
        assert len(data["data"]["embeddings"]) == 2
    
    def test_function_call(self, client):
        """測試函數調用"""
        response = client.post("/api/cognitive/function-call", json={
            "prompt": "What's the weather in Tokyo?",
            "functions": [
                {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                }
            ],
            "provider": "gamma-cognitive",
        })
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "function_call" in data["data"]
