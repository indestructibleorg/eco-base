# =============================================================================
# Collaboration Adapters
# =============================================================================
# 協作通信適配器實現
# 封裝團隊協作、即時通訊服務調用
# =============================================================================

import json
from typing import Any, Dict, List, Optional
from datetime import datetime
import httpx

from core.interfaces import (
    ICollaborationProvider, CapabilityContext, OperationResult,
    MessagePayload, WorkflowTrigger, CapabilityDomain
)


class IotaCollaborationAdapter(ICollaborationProvider):
    """
    Iota 協作通信適配器
    團隊即時通訊與自動化平台
    """
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._bot_token = config.get('bot_token')
        self._base_url = 'https://api.iota.com/v1'
    
    @property
    def domain(self) -> CapabilityDomain:
        return CapabilityDomain.COLLABORATION
    
    @property
    def provider_id(self) -> str:
        return "iota-collaboration"
    
    async def health_check(self) -> OperationResult:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self._base_url}/auth.test',
                    headers={'Authorization': f'Bearer {self._bot_token}'}
                )
                return OperationResult(success=response.status_code == 200)
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def get_capabilities(self) -> List[str]:
        return ['message', 'channel', 'summarize', 'workflow', 'search']
    
    async def send_message(self, payload: MessagePayload, ctx: CapabilityContext) -> OperationResult:
        """發送消息"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/chat.postMessage',
                    headers={'Authorization': f'Bearer {self._bot_token}'},
                    json={
                        'channel': payload.channel,
                        'text': payload.content,
                        'thread_ts': payload.thread_id,
                        'attachments': payload.attachments
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=data.get('ok', False),
                    data={'message_id': data.get('ts')}
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def create_channel(self, name: str, members: List[str], ctx: CapabilityContext) -> OperationResult:
        """創建頻道"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/conversations.create',
                    headers={'Authorization': f'Bearer {self._bot_token}'},
                    json={
                        'name': name,
                        'user_ids': members
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=data.get('ok', False),
                    data={'channel_id': data.get('channel', {}).get('id')}
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def summarize_conversation(self, channel: str, since: datetime, ctx: CapabilityContext) -> OperationResult:
        """對話摘要 (AI功能)"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/ai.summarize',
                    headers={'Authorization': f'Bearer {self._bot_token}'},
                    json={
                        'channel': channel,
                        'since': since.isoformat()
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=data.get('ok', False),
                    data={
                        'summary': data.get('summary'),
                        'key_points': data.get('key_points', [])
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def setup_workflow(self, trigger: WorkflowTrigger, ctx: CapabilityContext) -> OperationResult:
        """設置自動化工作流"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/workflows.create',
                    headers={'Authorization': f'Bearer {self._bot_token}'},
                    json={
                        'event_type': trigger.event_type,
                        'conditions': trigger.conditions,
                        'actions': trigger.actions
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=data.get('ok', False),
                    data={'workflow_id': data.get('workflow_id')}
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def search_knowledge(self, query: str, ctx: CapabilityContext) -> OperationResult:
        """企業知識搜索"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/search.messages',
                    headers={'Authorization': f'Bearer {self._bot_token}'},
                    json={'query': query}
                )
                
                data = response.json()
                return OperationResult(
                    success=data.get('ok', False),
                    data={
                        'results': data.get('messages', {}).get('matches', []),
                        'total': data.get('messages', {}).get('total', 0)
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))


class KappaCollaborationAdapter(ICollaborationProvider):
    """
    Kappa 協作通信適配器
    代碼託管與開發協作平台
    """
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._token = config.get('token')
        self._base_url = 'https://api.kappa.dev/v1'
    
    @property
    def domain(self) -> CapabilityDomain:
        return CapabilityDomain.COLLABORATION
    
    @property
    def provider_id(self) -> str:
        return "kappa-collaboration"
    
    async def health_check(self) -> OperationResult:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self._base_url}/user',
                    headers={'Authorization': f'token {self._token}'}
                )
                return OperationResult(success=response.status_code == 200)
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def get_capabilities(self) -> List[str]:
        return ['message', 'workflow', 'search', 'pr', 'issue']
    
    async def send_message(self, payload: MessagePayload, ctx: CapabilityContext) -> OperationResult:
        """發送消息 (Issue/PR 評論)"""
        try:
            async with httpx.AsyncClient() as client:
                # 解析 channel 格式: owner/repo/issue/123
                parts = payload.channel.split('/')
                if len(parts) >= 4:
                    owner, repo, _, issue_number = parts
                    
                    response = await client.post(
                        f'{self._base_url}/repos/{owner}/{repo}/issues/{issue_number}/comments',
                        headers={'Authorization': f'token {self._token}'},
                        json={'body': payload.content}
                    )
                    
                    data = response.json()
                    return OperationResult(
                        success=response.status_code == 201,
                        data={'comment_id': data.get('id')}
                    )
                else:
                    return OperationResult(success=False, error_message='Invalid channel format')
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def create_channel(self, name: str, members: List[str], ctx: CapabilityContext) -> OperationResult:
        """創建頻道 (創建 Issue)"""
        try:
            async with httpx.AsyncClient() as client:
                # name 格式: owner/repo
                parts = name.split('/')
                if len(parts) == 2:
                    owner, repo = parts
                    
                    response = await client.post(
                        f'{self._base_url}/repos/{owner}/{repo}/issues',
                        headers={'Authorization': f'token {self._token}'},
                        json={
                            'title': 'Discussion Channel',
                            'body': f'Collaboration channel for: {", ".join(members)}'
                        }
                    )
                    
                    data = response.json()
                    return OperationResult(
                        success=response.status_code == 201,
                        data={'issue_id': data.get('number')}
                    )
                else:
                    return OperationResult(success=False, error_message='Invalid name format')
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def summarize_conversation(self, channel: str, since: datetime, ctx: CapabilityContext) -> OperationResult:
        """對話摘要 (不支持)"""
        return OperationResult(
            success=False,
            error_message='Summarize not supported in Kappa adapter'
        )
    
    async def setup_workflow(self, trigger: WorkflowTrigger, ctx: CapabilityContext) -> OperationResult:
        """設置自動化工作流 (GitHub Actions)"""
        try:
            async with httpx.AsyncClient() as client:
                # 創建 workflow 文件
                workflow_content = self._generate_workflow_yaml(trigger)
                
                response = await client.put(
                    f'{self._base_url}/repos/{trigger.conditions.get("repo")}/contents/.github/workflows/auto-workflow.yml',
                    headers={'Authorization': f'token {self._token}'},
                    json={
                        'message': 'Add automated workflow',
                        'content': workflow_content
                    }
                )
                
                return OperationResult(success=response.status_code in [200, 201])
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def search_knowledge(self, query: str, ctx: CapabilityContext) -> OperationResult:
        """企業知識搜索 (代碼搜索)"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self._base_url}/search/code',
                    headers={'Authorization': f'token {self._token}'},
                    params={'q': query}
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'results': data.get('items', []),
                        'total': data.get('total_count', 0)
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    def _generate_workflow_yaml(self, trigger: WorkflowTrigger) -> str:
        """生成 GitHub Actions YAML"""
        import base64
        
        yaml_content = f"""
name: Automated Workflow
on:
  {trigger.event_type}:
    {trigger.conditions}

jobs:
  automate:
    runs-on: ubuntu-latest
    steps:
      {trigger.actions}
"""
        return base64.b64encode(yaml_content.encode()).decode()
