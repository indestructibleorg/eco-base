# =============================================================================
# Visual Design Adapters
# =============================================================================
# 視覺設計適配器實現
# 封裝設計工具服務調用
# =============================================================================

import json
import base64
from typing import Any, Dict, List, Optional
from datetime import datetime
import httpx

from core.interfaces import (
    IVisualDesignProvider, CapabilityContext, OperationResult,
    DesignAsset, DesignSystem, CapabilityDomain
)


class LambdaVisualAdapter(IVisualDesignProvider):
    """
    Lambda 視覺設計適配器
    雲端協作設計工具
    """
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._token = config.get('token')
        self._base_url = 'https://api.lambda.dev/v1'
    
    @property
    def domain(self) -> CapabilityDomain:
        return CapabilityDomain.VISUAL_DESIGN
    
    @property
    def provider_id(self) -> str:
        return "lambda-visual"
    
    async def health_check(self) -> OperationResult:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self._base_url}/me',
                    headers={'X-Figma-Token': self._token}
                )
                return OperationResult(success=response.status_code == 200)
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def get_capabilities(self) -> List[str]:
        return ['components', 'export', 'generate', 'inspect', 'prototype']
    
    async def get_components(self, system_id: Optional[str], ctx: CapabilityContext) -> OperationResult:
        """獲取組件庫"""
        try:
            async with httpx.AsyncClient() as client:
                if system_id:
                    response = await client.get(
                        f'{self._base_url}/files/{system_id}/components',
                        headers={'X-Figma-Token': self._token}
                    )
                else:
                    # 獲取用戶所有組件
                    response = await client.get(
                        f'{self._base_url}/me/components',
                        headers={'X-Figma-Token': self._token}
                    )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'components': data.get('components', []),
                        'styles': data.get('styles', [])
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def export_asset(self, asset_id: str, format: str, ctx: CapabilityContext) -> OperationResult:
        """導出設計資源"""
        try:
            async with httpx.AsyncClient() as client:
                # 請求導出
                export_response = await client.get(
                    f'{self._base_url}/images/{asset_id}',
                    headers={'X-Figma-Token': self._token},
                    params={'format': format, 'scale': 2}
                )
                
                data = export_response.json()
                image_url = data.get('images', {}).get(asset_id)
                
                if image_url:
                    # 下載圖像
                    image_response = await client.get(image_url)
                    return OperationResult(
                        success=image_response.status_code == 200,
                        data={
                            'content': base64.b64encode(image_response.content).decode(),
                            'format': format,
                            'url': image_url
                        }
                    )
                else:
                    return OperationResult(success=False, error_message='Export failed')
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def generate_from_description(self, description: str, asset_type: str, ctx: CapabilityContext) -> OperationResult:
        """根據描述生成設計 (AI功能)"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/ai/generate',
                    headers={'X-Figma-Token': self._token},
                    json={
                        'prompt': description,
                        'type': asset_type
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'design_id': data.get('file_key'),
                        'preview_url': data.get('preview_url')
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def inspect_design(self, design_id: str, ctx: CapabilityContext) -> OperationResult:
        """設計檢視 (獲取CSS/樣式)"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self._base_url}/files/{design_id}/nodes',
                    headers={'X-Figma-Token': self._token}
                )
                
                data = response.json()
                nodes = data.get('nodes', {})
                
                # 提取 CSS 屬性
                css_properties = self._extract_css_properties(nodes)
                
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'css': css_properties,
                        'styles': data.get('styles', {})
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def create_prototype(self, screens: List[str], transitions: List[Dict], ctx: CapabilityContext) -> OperationResult:
        """創建交互原型"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/prototypes',
                    headers={'X-Figma-Token': self._token},
                    json={
                        'screens': screens,
                        'transitions': transitions
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 201,
                    data={
                        'prototype_id': data.get('id'),
                        'prototype_url': data.get('url')
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    def _extract_css_properties(self, nodes: Dict) -> Dict[str, Any]:
        """從節點提取 CSS 屬性"""
        css = {}
        for node_id, node_data in nodes.items():
            document = node_data.get('document', {})
            css[node_id] = {
                'width': document.get('absoluteBoundingBox', {}).get('width'),
                'height': document.get('absoluteBoundingBox', {}).get('height'),
                'background': document.get('fills', []),
                'typography': document.get('style', {})
            }
        return css


class MuVisualAdapter(IVisualDesignProvider):
    """
    Mu 視覺設計適配器
    macOS 原生設計工具
    """
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._token = config.get('token')
        self._base_url = 'https://api.mu.dev/v1'
    
    @property
    def domain(self) -> CapabilityDomain:
        return CapabilityDomain.VISUAL_DESIGN
    
    @property
    def provider_id(self) -> str:
        return "mu-visual"
    
    async def health_check(self) -> OperationResult:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self._base_url}/health',
                    headers={'Authorization': f'Bearer {self._token}'}
                )
                return OperationResult(success=response.status_code == 200)
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def get_capabilities(self) -> List[str]:
        return ['components', 'export', 'inspect']
    
    async def get_components(self, system_id: Optional[str], ctx: CapabilityContext) -> OperationResult:
        """獲取組件庫 (Symbol)"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self._base_url}/documents/{system_id}/symbols',
                    headers={'Authorization': f'Bearer {self._token}'}
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={'symbols': data.get('symbols', [])}
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def export_asset(self, asset_id: str, format: str, ctx: CapabilityContext) -> OperationResult:
        """導出設計資源"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/exports',
                    headers={'Authorization': f'Bearer {self._token}'},
                    json={
                        'layer_id': asset_id,
                        'format': format
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 201,
                    data={
                        'export_id': data.get('id'),
                        'download_url': data.get('url')
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def generate_from_description(self, description: str, asset_type: str, ctx: CapabilityContext) -> OperationResult:
        """根據描述生成設計 (不支持)"""
        return OperationResult(
            success=False,
            error_message='AI generation not supported in Mu adapter'
        )
    
    async def inspect_design(self, design_id: str, ctx: CapabilityContext) -> OperationResult:
        """設計檢視"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self._base_url}/documents/{design_id}/layers',
                    headers={'Authorization': f'Bearer {self._token}'}
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={'layers': data.get('layers', [])}
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def create_prototype(self, screens: List[str], transitions: List[Dict], ctx: CapabilityContext) -> OperationResult:
        """創建交互原型 (不支持)"""
        return OperationResult(
            success=False,
            error_message='Prototype creation not supported in Mu adapter'
        )
