# =============================================================================
# Knowledge Management Adapters
# =============================================================================
# 知識管理適配器實現
# 封裝文檔、知識庫服務調用
# =============================================================================

import json
from typing import Any, Dict, List, Optional
from datetime import datetime
import httpx

from core.interfaces import (
    IKnowledgeManagementProvider, CapabilityContext, OperationResult,
    DocumentSpec, KnowledgeQuery, CapabilityDomain
)


class NuKnowledgeAdapter(IKnowledgeManagementProvider):
    """
    Nu 知識管理適配器
    多功能筆記與知識庫平台
    """
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._token = config.get('token')
        self._base_url = 'https://api.nu.dev/v1'
    
    @property
    def domain(self) -> CapabilityDomain:
        return CapabilityDomain.KNOWLEDGE_MGMT
    
    @property
    def provider_id(self) -> str:
        return "nu-knowledge"
    
    async def health_check(self) -> OperationResult:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self._base_url}/users/me',
                    headers={'Authorization': f'Bearer {self._token}'}
                )
                return OperationResult(success=response.status_code == 200)
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def get_capabilities(self) -> List[str]:
        return ['create', 'update', 'query', 'sync', 'export']
    
    async def create_document(self, spec: DocumentSpec, ctx: CapabilityContext) -> OperationResult:
        """創建文檔"""
        try:
            async with httpx.AsyncClient() as client:
                # Notion API 使用 blocks 結構
                blocks = self._convert_to_blocks(spec.content)
                
                response = await client.post(
                    f'{self._base_url}/pages',
                    headers={
                        'Authorization': f'Bearer {self._token}',
                        'Notion-Version': '2022-06-28'
                    },
                    json={
                        'parent': {'database_id': spec.parent_id} if spec.parent_id else {'page_id': spec.parent_id},
                        'properties': {
                            'title': {'title': [{'text': {'content': spec.title}}]}
                        },
                        'children': blocks
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={'document_id': data.get('id'), 'url': data.get('url')}
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def update_document(self, doc_id: str, spec: DocumentSpec, ctx: CapabilityContext) -> OperationResult:
        """更新文檔"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    f'{self._base_url}/pages/{doc_id}',
                    headers={
                        'Authorization': f'Bearer {self._token}',
                        'Notion-Version': '2022-06-28'
                    },
                    json={
                        'properties': {
                            'title': {'title': [{'text': {'content': spec.title}}]}
                        }
                    }
                )
                
                return OperationResult(success=response.status_code == 200)
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def query_knowledge(self, query: KnowledgeQuery, ctx: CapabilityContext) -> OperationResult:
        """知識查詢"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/databases/{query.filters.get("database_id")}/query',
                    headers={
                        'Authorization': f'Bearer {self._token}',
                        'Notion-Version': '2022-06-28'
                    },
                    json={
                        'filter': {
                            'property': 'title',
                            'title': {'contains': query.query}
                        },
                        'page_size': query.top_k
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'results': data.get('results', []),
                        'total': len(data.get('results', []))
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def sync_from_git(self, repo_url: str, branch: str, ctx: CapabilityContext) -> OperationResult:
        """從 Git 同步文檔 (不支持)"""
        return OperationResult(
            success=False,
            error_message='Git sync not supported in Nu adapter'
        )
    
    async def export_to_format(self, doc_id: str, target_format: str, ctx: CapabilityContext) -> OperationResult:
        """導出為其他格式"""
        try:
            async with httpx.AsyncClient() as client:
                # Notion 支持導出為 Markdown 和 HTML
                response = await client.get(
                    f'{self._base_url}/blocks/{doc_id}/children',
                    headers={
                        'Authorization': f'Bearer {self._token}',
                        'Notion-Version': '2022-06-28'
                    }
                )
                
                data = response.json()
                content = self._convert_to_markdown(data.get('results', []))
                
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'content': content,
                        'format': 'markdown'
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    def _convert_to_blocks(self, content: str) -> List[Dict]:
        """將 Markdown 轉換為 Notion blocks"""
        blocks = []
        lines = content.split('\n')
        
        for line in lines:
            if line.startswith('# '):
                blocks.append({
                    'object': 'block',
                    'type': 'heading_1',
                    'heading_1': {'rich_text': [{'type': 'text', 'text': {'content': line[2:]}}]}
                })
            elif line.startswith('## '):
                blocks.append({
                    'object': 'block',
                    'type': 'heading_2',
                    'heading_2': {'rich_text': [{'type': 'text', 'text': {'content': line[3:]}}]}
                })
            elif line.strip():
                blocks.append({
                    'object': 'block',
                    'type': 'paragraph',
                    'paragraph': {'rich_text': [{'type': 'text', 'text': {'content': line}}]}
                })
        
        return blocks
    
    def _convert_to_markdown(self, blocks: List[Dict]) -> str:
        """將 Notion blocks 轉換為 Markdown"""
        markdown = []
        
        for block in blocks:
            block_type = block.get('type')
            
            if block_type == 'paragraph':
                text = self._extract_text(block.get('paragraph', {}).get('rich_text', []))
                markdown.append(text)
            elif block_type == 'heading_1':
                text = self._extract_text(block.get('heading_1', {}).get('rich_text', []))
                markdown.append(f'# {text}')
            elif block_type == 'heading_2':
                text = self._extract_text(block.get('heading_2', {}).get('rich_text', []))
                markdown.append(f'## {text}')
            elif block_type == 'bulleted_list_item':
                text = self._extract_text(block.get('bulleted_list_item', {}).get('rich_text', []))
                markdown.append(f'- {text}')
        
        return '\n\n'.join(markdown)
    
    def _extract_text(self, rich_text: List[Dict]) -> str:
        """從 rich_text 提取純文本"""
        return ''.join([t.get('text', {}).get('content', '') for t in rich_text])


class XiKnowledgeAdapter(IKnowledgeManagementProvider):
    """
    Xi 知識管理適配器
    Git 驅動的文檔平台
    """
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._token = config.get('token')
        self._base_url = 'https://api.xi.dev/v1'
    
    @property
    def domain(self) -> CapabilityDomain:
        return CapabilityDomain.KNOWLEDGE_MGMT
    
    @property
    def provider_id(self) -> str:
        return "xi-knowledge"
    
    async def health_check(self) -> OperationResult:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self._base_url}/user',
                    headers={'Authorization': f'Bearer {self._token}'}
                )
                return OperationResult(success=response.status_code == 200)
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def get_capabilities(self) -> List[str]:
        return ['create', 'update', 'sync', 'export']
    
    async def create_document(self, spec: DocumentSpec, ctx: CapabilityContext) -> OperationResult:
        """創建文檔"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/collections/{spec.parent_id}/pages',
                    headers={'Authorization': f'Bearer {self._token}'},
                    json={
                        'title': spec.title,
                        'content': spec.content
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 201,
                    data={'document_id': data.get('id')}
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def update_document(self, doc_id: str, spec: DocumentSpec, ctx: CapabilityContext) -> OperationResult:
        """更新文檔"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f'{self._base_url}/pages/{doc_id}',
                    headers={'Authorization': f'Bearer {self._token}'},
                    json={
                        'title': spec.title,
                        'content': spec.content
                    }
                )
                
                return OperationResult(success=response.status_code == 200)
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def query_knowledge(self, query: KnowledgeQuery, ctx: CapabilityContext) -> OperationResult:
        """知識查詢"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self._base_url}/search',
                    headers={'Authorization': f'Bearer {self._token}'},
                    params={'q': query.query}
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={'results': data.get('results', [])}
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def sync_from_git(self, repo_url: str, branch: str, ctx: CapabilityContext) -> OperationResult:
        """從 Git 同步文檔 (核心功能)"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/sync',
                    headers={'Authorization': f'Bearer {self._token}'},
                    json={
                        'repo_url': repo_url,
                        'branch': branch
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={'sync_id': data.get('id')}
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def export_to_format(self, doc_id: str, target_format: str, ctx: CapabilityContext) -> OperationResult:
        """導出為其他格式 (PDF/EPUB)"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/pages/{doc_id}/export',
                    headers={'Authorization': f'Bearer {self._token}'},
                    json={'format': target_format}
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'download_url': data.get('url'),
                        'format': target_format
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
